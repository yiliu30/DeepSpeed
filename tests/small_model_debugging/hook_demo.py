import torch

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
    
hidden_dim = 5
model = SubModel(hidden_dim=hidden_dim)
print(model)


# 1. convert the linear's param into 1-D 
# 2. register a pre-forward hook to revert change 
# 3. reigster a post-forward hook to convert the linear's param into 1-D


        # def forward_pre_hook(m, input):
        #     return torch.nn.functional.relu(input[0])

def _pre_forward_hook(module, input):
    weight_param = module.weight
    original_shape = module.__original_weight_shape
    print(f"Revert param's shape from {weight_param.data.shape} into {original_shape}")
    weight_param.data = weight_param.data.reshape(original_shape)

        # def forward_hook(m, input, output):
        #     return -output
        
def _post_forward_hook(module, input, output):
    weight_param = module.weight
    original_shape = module.__original_weight_shape
    weight_param.data = weight_param.data.flatten()
    print(f"Covert param's shape from {original_shape} into {weight_param.data.shape}")
    


def model_wrapper(model):
    linears_lst = []
    supported_module_lst = [torch.nn.Linear, torch.nn.Conv1d, torch.nn.Linear]
    for name, module in model.named_modules():
        if isinstance(module, tuple(supported_module_lst)):
            linears_lst.append(module)
            weight_param = module.weight
            module.__original_weight_shape = weight_param.shape
            weight_param.data = weight_param.data.flatten()
            module.register_forward_pre_hook(_pre_forward_hook)
            module.register_forward_hook(_post_forward_hook)
    return model
            

# for name, param in model.named_parameters():
#     print(f"name: {name}, param shape: {param.shape}")
    
# total_samples = 10
# train_data = torch.randn(total_samples, hidden_dim)
# model.eval()
# out = model(train_data)
# print(out.shape)