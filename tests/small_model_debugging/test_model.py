# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import json
import argparse
import torch
import deepspeed
from torch.utils.data.distributed import DistributedSampler
from deepspeed.utils.utility import ForkedPdb
import deepspeed.comm as dist
from deepspeed.utils import (
    safe_get_full_fp32_param, 
    safe_get_full_grad, 
    safe_get_full_optimizer_state, 
    safe_set_full_fp32_param, 
    safe_set_full_optimizer_state,
    # get_local_fp32_param,
    )
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3




class SubModel(torch.nn.Module):

    def __init__(self, hidden_dim, empty_grad=False):
        super(SubModel, self).__init__()
        self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim*2, bias=True)
        self.linear2 = torch.nn.Linear(hidden_dim*2, hidden_dim*3, bias=False)
        self.linear3 = torch.nn.Linear(hidden_dim*3, hidden_dim, bias=False)

    def forward(self, x):
        hidden = x
        hidden1 = self.linear1(hidden)
        hidden2 = self.linear2(hidden1)
        hidden3 = self.linear3(hidden2)
        return hidden3

class NewCompositeModel(torch.nn.Module):
    def __init__(self, hidden_dim, empty_grad=False):
        super(NewCompositeModel, self).__init__()
        self.sub1 = SubModel(hidden_dim=hidden_dim)
        self.sub2 = SubModel(hidden_dim=hidden_dim)
        self.sub3 = SubModel(hidden_dim=hidden_dim)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, x, y):
        hidden = x
        hidden1 = self.sub1(hidden)
        hidden2 = self.sub2(hidden1)
        hidden3 = self.sub3(hidden2)
        return self.cross_entropy_loss(hidden3, y)


class SimpleModel(torch.nn.Module):

    def __init__(self, hidden_dim, empty_grad=False):
        super(SimpleModel, self).__init__()
        self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim*2, bias=True)
        self.linear2 = torch.nn.Linear(hidden_dim*2, hidden_dim*3, bias=False)
        self.linear3 = torch.nn.Linear(hidden_dim*3, hidden_dim, bias=False)
        if empty_grad:
            self.layers2 = torch.nn.ModuleList([torch.nn.Linear(hidden_dim,
                                                                hidden_dim)])  #QuantizeLinear(hidden_dim, hidden_dim)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        hidden = x
        hidden1 = self.linear1(hidden)
        hidden2 = self.linear2(hidden1)
        hidden3 = self.linear3(hidden2)
        return self.cross_entropy_loss(hidden3, y)


def create_config_from_dict(tmpdir, config_dict):
    config_path = os.path.join(tmpdir, 'temp_config.json')
    with open(config_path, 'w') as fd:
        json.dump(config_dict, fd)
    return config_path


def get_data_loader(model, total_samples, hidden_dim, device):
    batch_size = model.train_micro_batch_size_per_gpu()
    train_data = torch.randn(total_samples, hidden_dim, device=device, dtype=torch.half)
    train_label = torch.empty(total_samples, dtype=torch.long, device=device).random_(hidden_dim)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    return train_loader


def get_args(tmpdir, config_dict):
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--zero', type=int, default=3)
    parser.add_argument('--zero_hpz_partition_size', type=int, default=1)
    args = parser.parse_args()  #args=''

    config_dict["zero_optimization"]["stage"] = args.zero
    config_dict["zero_optimization"]["zero_hpz_partition_size"] = args.zero_hpz_partition_size
    print('config_dict["zero_optimization"]', config_dict["zero_optimization"])
    config_path = create_config_from_dict(tmpdir, config_dict)

    args.deepspeed_config = config_path
    return args


def print0(msg):
    if dist.get_rank() == 0:
        print(msg, flush=True)
        
def print_with_rank(msg):
    _rank = dist.get_rank()
    print(f"[rank: {_rank}] {msg}")



def get_id(model):
    for name, param in model.named_parameters():
        print_with_rank(f"name: {name}, param: {param.ds_summary()['id']}")

    for name, module in model.named_modules():
        print_with_rank(f"name: {name}, module: {module.id}")
    

def check_params_status(model, stage=""):
    for name, param in model.named_parameters():
        print_with_rank(f"[stage: {stage}]name: {name}, module ds status: {param.ds_status}")
        
rank = int(os.environ['RANK'])
print('seed:', 2222 + rank)
torch.random.manual_seed(2222 + rank)

config_dict = {
    "train_batch_size": 256,
    "steps_per_print": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.00015,
        }
    },
    "fp16": {
        "enabled": True,
        "initial_scale_power": 8
    },
    "zero_optimization": {
        "stage": 0,
        "reduce_bucket_size": 20,
        "zero_hpz_partition_size": 1,
        "sub_group_size": 2000*4000,
        "reduce_scatter": True,
        "zero_quantized_weights": False,
        "zero_quantized_gradients": False
    }
}
#        "initial_scale_power": 15
args = get_args('/tmp/', config_dict)
hidden_dim = 4 * 100

model = SimpleModel(hidden_dim, empty_grad=False)

from copy import deepcopy

model_bk = deepcopy(model)

model_parameters = model.parameters()

# model_parameters_cp = deepcopy(model_parameters)

model, _, _, _ = deepspeed.initialize(args=args,
                                      model=model,
                                      model_parameters=model.parameters(),
                                      dist_init_required=True)




print(model.linear1.weight)

# ForkedPdb().set_trace()

def print_function_name(func):
    def wrapper(*args, **kwargs):
        print(f"Calling function: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper


def print_params(tag, model):
    if dist.get_rank() == 0:
        for n, p in model.named_parameters():
            print0("{} {}:{}".format(tag, n, p))

def handle_model_weights(model, stage=""):
    for n, lp in model.named_parameters():
        hp_grad = safe_get_full_grad(lp)

        # 2. fp32 and optim states can probably be called anywhere in the training loop, but will be updated after `step`
        hp = safe_get_full_fp32_param(lp)
        exp_avg = safe_get_full_optimizer_state(lp, "exp_avg")
        exp_avg_sq = safe_get_full_optimizer_state(lp, "exp_avg_sq")
        # mask = hp * hp_grad
        print_with_rank(f"[stage:{stage}] name: {n}, lp shape: {lp.shape}, lp type: {type(hp)}, lp numel: {lp.numel()}, lp size: {lp.size()}")
        # print_with_rank(f"name: {n}, mask shape: {mask.shape}")
        print_with_rank(f"[stage:{stage}] name: {n}, hp_grad shape: {hp_grad.shape if hasattr(hp_grad, 'shape') else None}")
        print_with_rank(f"[stage:{stage}] name: {n}, hp shape: {hp.shape if hasattr(hp, 'shape') else None}")
        
        # assign the model's weight to zeros
        zero_tensor = torch.zeros_like(hp)
        print_with_rank(f"[stage:{stage}] name: {n}, zero_tensor shape: {zero_tensor.shape}, zero_tensor type: {type(zero_tensor)}, zero_tensor numel: {zero_tensor.numel()}, zero_tensor size: {zero_tensor.size()}")
        hp = safe_set_full_fp32_param(lp, zero_tensor)
        print_with_rank(f"[stage:{stage}] update the params {n}")
        # print_with_rank(mask)
        hp = safe_get_full_fp32_param(lp)
        print_with_rank(f"[stage:{stage}] [get again ]name: {n}, lp shape: {lp.shape}, lp type: {type(hp)}, lp numel: {lp.numel()}, lp size: {lp.size()}")




        

data_loader = get_data_loader(model=model, total_samples=256*4, hidden_dim=hidden_dim, device=model.device)
#print_params('pre-train', model)


def ds_fetch_param(param):
    ds_fetch_result = param._z3_optimizer.get_local_fp32_param(param)
    print_with_rank(f"ds fetch result shape, {ds_fetch_result.shape}")
    return ds_fetch_result
    # from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    # assert hasattr(param, 'ds_id'), "assume the params is used by z3 optimizer ..."
    # if (param.ds_status == ZeroParamStatus.AVAILABLE):
    #     print_with_rank("Current params are AVAILABLE, return it directly..")
    #     #TODO(Yi) convert full param to local?
    #     return param
    # else:
    #     print_with_rank("Current params are NOT AVAILABLE, fetch aggregate it..")
    #     if hasattr(param, 'ds_id'):
    #         # ForkedPdb().set_trace()
    #         _params = param._z3_optimizer._get_fp32_opt_state_partition(param, optim_state_key=False)
    #         return _params

def ds_fetch_param_grad(param):
    _z3_optimizer: DeepSpeedZeroOptimizer_Stage3 = param._z3_optimizer
    fp32_grad = _z3_optimizer.get_local_fp32_grad_for_param(param)
    return fp32_grad

def ds_assign_param(param, new_value):
    _z3_optimizer: DeepSpeedZeroOptimizer_Stage3 = param._z3_optimizer
    _z3_optimizer.set_local_hp_param(new_value, param)

# for name, param in model.named_parameters():
#     if 'weight' in name and param.requires_grad:
#         _param = ds_fetch_param(param)
#         _param = _param.detach().cpu()[:160000]
#         linear1_weight = model_bk.linear1.weight.data.flatten()
#         print(torch.sum(torch.abs(linear1_weight[:160000] - _param)))

pruner_mask = {}
pruner_score = {}
def criterion_score(weight, weight_grad):
    return weight * weight_grad /weight.numel()

def update_weight(weight, mask):
    new_weight = weight * mask
    return new_weight

def _update_mask(pre_mask, score):
    new_mask = pre_mask * [score > 0.1]
    return new_mask

@print_function_name
def pruner_update_mask(model, step=0):
    if step == 0:
        return 
    for name, param in model.named_parameters():
        print_with_rank(f"start to update mask for param : {name}")
        score = pruner_score[name]
        if name not in pre_mask:
            pre_mask[name] = torch.ones_like(score)
        pre_mask = pre_mask[name]
        new_mask = _update_mask(pre_mask, score)
        print_with_rank(f"update mask for {name}; pre mask shape {pre_mask.shape}, new mask shape: {new_mask.shape}, score shape: {score.shape}")
        pruner_mask[name] = new_mask

@print_function_name
def pruner_handle_score(model):
    for name, param in model.named_parameters():
        print_with_rank(f"start to handle score for param : {name}")
        if 'weight' in name and param.requires_grad:
            weight = ds_fetch_param(param)
            weight_grad = ds_fetch_param_grad(param)
            print_with_rank(f"get weight: {weight.shape} and grad : {weight_grad.shape}")
            # ForkedPdb().set_trace()
            pruner_score[name] = criterion_score(weight, weight_grad)
            print_with_rank(f"update score for {name}; weight shape {weight.shape}, weight_grad shape: {weight_grad.shape}")


@print_function_name
def pruner_update_weight(model):
    for name, param in model.named_parameters():
        print_with_rank(f"start to update weight for param : {name}")
        weight = ds_fetch_param(param)
        if name not in pruner_mask:
            pruner_mask[name] = torch.ones_like(weight)
        mask = pruner_mask[name]
        new_weight = update_weight(weight=weight, mask=mask)
        ds_assign_param(param, new_weight)
        print_with_rank(f"update weight for {name}; param ds_shape {param.ds_shape}, weight_grad shape: {new_weight.shape}")



for n, batch in enumerate(data_loader):
    # <- pre_step
    #       mask = fetch_mask(mask_dict)
    #       mask_dict.update_mask(mask, weights(local))
    print_with_rank(f"start to deal with {n}-batch data ================= ")
    pruner_update_mask(model)
    check_params_status(model, stage="Pre-forward")
    loss = model(batch[0], batch[1])
    check_params_status(model, stage="Post-forward")
    get_id(model)
    if dist.get_rank() == 0:
        print("LOSS:", loss.item())
    model.backward(loss)
    check_params_status(model, stage="Post-backward")
    # <- post_backward
    #       criterion_score(weight, weights.grad)  # pruner_handle_score
    #handle_model_weights(model, "After backward, before step")
    pruner_handle_score(model)
    model.step()
    check_params_status(model, stage="Post-optimizer step")
    # <- post step
    #       model.weights = update_weight(mask, model.weights)  # pruner_update_weight
    # handle_model_weights(model, "After step")
    pruner_update_weight(model)
    print_params('step={}'.format(n), model)
    
    #if n == 5: break
