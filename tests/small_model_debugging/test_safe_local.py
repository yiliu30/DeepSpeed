# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# import pytest
import deepspeed.comm as dist
import torch

# from unit.common import DistributedTest
# from .unit.simple_model import random_dataloader, SimpleModel
# from unit.util import bf16_required_version_check

import deepspeed
from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_grad, safe_get_full_optimizer_state
from deepspeed.utils import safe_set_full_fp32_param, safe_set_full_optimizer_state, safe_set_local_fp32_param, safe_get_local_fp32_param
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.ops.aio import AsyncIOBuilder

WEIGHT_KEY = 'weight'
FIRST_ORDER_KEY = 'exp_avg'
SECOND_ORDER_KEY = 'exp_avg_sq'


class SimpleModel(torch.nn.Module):

    def __init__(self, hidden_dim, empty_grad=False, nlayers=1):
        super(SimpleModel, self).__init__()
        self.linears = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for i in range(nlayers)])
        if empty_grad:
            self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.empty_grad = empty_grad

    def forward(self, x, y):
        if len(self.linears) == 1:
            x = self.linears[0](x)
        else:
            for i, l in enumerate(self.linears):
                x = self.linears[i // 2](x) + l(x)
        return self.cross_entropy_loss(x, y)


def validate_full_tensors(model):
    for _, lp in model.named_parameters():
        hp = safe_get_full_fp32_param(lp)
        exp_avg = safe_get_full_optimizer_state(lp, 'exp_avg')
        exp_avg_sq = safe_get_full_optimizer_state(lp, 'exp_avg_sq')
        hp_grad = safe_get_full_grad(lp)
        param_list = [hp, hp_grad, exp_avg, exp_avg_sq]
        if lp.requires_grad:
            assert all([p is not None for p in param_list])
        else:
            assert all([p is None for p in param_list])


class MyModel(torch.nn.Module):

    def __init__(self, hidden_dim, frozen_weights):
        super(MyModel, self).__init__()
        self.act = torch.nn.ReLU()
        self.cel = torch.nn.CrossEntropyLoss()
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dim, 1),
             torch.nn.Linear(1, 1),
             torch.nn.Linear(1, hidden_dim)])
        if frozen_weights:
            self.linears[0].weight.requires_grad = False
            self.linears[0].bias.requires_grad = False

    def forward(self, x, y):
        for l in self.linears:
            x = l(x)
            x = self.act(x)
        loss = self.cel(x, y)
        val = (x, loss)
        return val


from deepspeed.utils.utility import ForkedPdb


def print_with_rank(msg):
    _rank = dist.get_rank()
    print(f"[rank: {_rank}] {msg}")

def create_random_values(model, key_list, group):
    param_values = {}
    for n, lp in model.named_parameters():
        # ForkedPdb().set_trace()
        param_shape = lp.ds_tensor.shape
        param_values[n] = {}
        for key in key_list:
            rand_value = torch.rand(param_shape, dtype=torch.float32, device=model.device)
            # dist.broadcast(rand_value, src=0, group=group)
            param_values[n][key] = rand_value
            print_with_rank(f"Create rand val for module {n}'s weight with shape {rand_value.shape}")
            print_with_rank(rand_value)
    return param_values


def set_param_values_with_dict(model, value_dict):
    for n, lp in model.named_parameters():
        for key, value_tensor in value_dict[n].items():
            if key == WEIGHT_KEY:
                safe_set_full_fp32_param(lp, value_tensor)
            else:
                safe_set_full_optimizer_state(lp, value_tensor, key)


def set_local_param_values_with_dict(model, value_dict):
    for n, lp in model.named_parameters():
        for key, value_tensor in value_dict[n].items():
            if key == WEIGHT_KEY:
                safe_set_local_fp32_param(lp, value_tensor)

def validate_param_values_with_dict(model, value_dict):
    for n, lp in model.named_parameters():
        for key, expected_tensor in value_dict[n].items():
            if key == WEIGHT_KEY:
                actual_tensor = safe_get_full_fp32_param(lp)
            else:
                actual_tensor = safe_get_full_optimizer_state(lp, key)
            assert torch.equal(expected_tensor, actual_tensor)
            print(f"validated {n} DONE!")
    print(f"validate_param_values_with_dict DONE!")



def validate_param_values_with_dict(model, value_dict):
    for n, lp in model.named_parameters():
        for key, expected_tensor in value_dict[n].items():
            if key == WEIGHT_KEY:
                actual_tensor = safe_get_local_fp32_param(lp)
            else:
                raise NotImplemented
            assert torch.equal(expected_tensor, actual_tensor)
            print(f"validated {n} DONE!")
    print(f"validate_param_values_with_dict DONE!")

zero_stage = 3

config_dict = {
    "train_micro_batch_size_per_gpu": 1,
    "steps_per_print": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-6
        }
    },
    "zero_optimization": {
        "stage": zero_stage,
    }
}

# if offload_device == OffloadDeviceEnum.cpu:
#     config_dict["zero_optimization"]["offload_optimizer"] = {"device": offload_device}
# elif offload_device == OffloadDeviceEnum.nvme:
#     config_dict["zero_optimization"]["offload_optimizer"] = {
#         "device": offload_device,
#         "nvme_path": str(tmpdir)
#     }

# if dtype == torch.float16:
#     config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
# elif dtype == torch.bfloat16:
#     config_dict["bf16"] = {"enabled": True}

hidden_dim = 128
if zero_stage == 3:
    config_dict["zero_optimization"]["param_persistence_threshold"] = hidden_dim
    with deepspeed.zero.Init(config_dict_or_path=config_dict):
        model = SimpleModel(hidden_dim, nlayers=4)
else:
    model = SimpleModel(hidden_dim, nlayers=4)

model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
world = dist.get_world_size()
group = dist.new_group(ranks=list(range(world)))

dist.barrier()
optim_keys = [WEIGHT_KEY]
optim_state_values = create_random_values(model, optim_keys, group)
# set_param_values_with_dict(model, optim_state_values)
# validate_param_values_with_dict(model, optim_state_values)

set_local_param_values_with_dict(model, optim_state_values)
validate_param_values_with_dict(model, optim_state_values)

# Needed in ZeRO 3. Not doing so can leak memory.
model.destroy()
