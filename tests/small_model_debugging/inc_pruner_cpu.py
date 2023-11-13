import torch
from neural_compressor.config import WeightPruningConfig
from neural_compressor.training import prepare_compression
from neural_compressor.utils import logger

def test_conv1_prunig():
    local_config = [
        {
            "op_names": ["conv1.*"],
            "target_sparsity": 0.6,
            "pattern": "4x1",
            "pruning_type": "snip",
            "pruning_scope": "global",
        },
        {
            "op_names": ["conv2.*"], 
            "target_sparsity": 0.5, 
            "pattern": "2:4", 
            "pruning_scope": "global"},
    ]

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv1d(4, 4, 2)
            self.act = torch.nn.ReLU()
            self.conv2 = torch.nn.Conv1d(4, 4, 2)
            self.linear = torch.nn.Linear(32, 3)

        def forward(self, x):
            out = self.conv1(x)
            out = self.act(out)
            out = self.conv2(out)
            out = out.view(1, -1)
            out = self.linear(out)
            return out

    model = Model()
    # from hook_demo import model_wrapper
    # model = model_wrapper(model)
    data = torch.rand((1, 4, 10))
    output = model(data)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    config = WeightPruningConfig(local_config, target_sparsity=0.8, start_step=1, end_step=10)
    compression_manager = prepare_compression(model=model, confs=config)
    compression_manager.callbacks.on_train_begin()
    logger.info("========== Start the tuning process ============")
    for epoch in range(2):
        logger.info(f"[EPOCH: {epoch}][PRE EPOCH] ============")
        logger.info(f"[EPOCH] ======== {epoch} ============")
        model.train()
        compression_manager.callbacks.on_epoch_begin(epoch)
        local_step = 0
        for i in range(3):
            logger.info(f"[EPOCH: {epoch}][Batch: {i+1}][PRE BATCH] ============")
            data, target = torch.rand((1, 4, 10), requires_grad=True), torch.empty(1, dtype=torch.long).random_(3)
            compression_manager.callbacks.on_step_begin(local_step)
            logger.info(f"[EPOCH: {epoch}][Batch: {i+1}][PRE_FORWARD]")
            output = model(data)
            logger.info(f"[EPOCH: {epoch}][Batch: {i+1}][AFTER_FORWARD]")
            loss = criterion(output, target)
            logger.info(f"[EPOCH: {epoch}][Batch: {i+1}][CALCULATED LOSS]")
            optimizer.zero_grad()
            logger.info(f"[EPOCH: {epoch}][Batch: {i+1}][PRE_BACKWARD]")
            loss.backward()
            logger.info(f"[EPOCH: {epoch}][Batch: {i+1}][AFTER_BACKWARD]")
            compression_manager.callbacks.on_before_optimizer_step()
            logger.info(f"[EPOCH: {epoch}][Batch: {i+1}][PRE_OPTIMIZER_STEP]")
            optimizer.step()
            logger.info(f"[EPOCH: {epoch}][Batch: {i+1}][AFTER_OPTIMIZER_STEP]")
            compression_manager.callbacks.on_after_optimizer_step()
            compression_manager.callbacks.on_step_end()
            local_step += 1
            logger.info(f"[EPOCH: {epoch}][Batch: {i+1}][END BATCH] ============")
        logger.info(f"[EPOCH: {epoch}][Batch: {i+1}][END EPOCH] ============")
            

        compression_manager.callbacks.on_epoch_end()
    compression_manager.callbacks.on_train_end()
    compression_manager.callbacks.on_before_eval()
    compression_manager.callbacks.on_after_eval()
    
test_conv1_prunig()



def _pre_ds_hook(modules):
    # covert the ds properites
    for module in modules:
        # parameter
        param = module.weight
        param.shape = param.numel()
        param.grad = safe_get_grad(param)
        param.data = safe_get_data(param)
        
        # param.data = updated_weight ?


def _post_ds_hook(modules):
    # revert the ds prooerties to original
    pass