import torch

class SubModel(torch.nn.Module):

    def __init__(self, hidden_dim, empty_grad=False):
        super(SubModel, self).__init__()
        self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim*2, bias=True)
        self.linear2 = torch.nn.Linear(hidden_dim*2, hidden_dim*3, bias=False)
        self.linear3 = torch.nn.Linear(hidden_dim*3, hidden_dim, bias=False)

    def forward(self, x, y):
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
