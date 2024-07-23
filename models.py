import torch
from torch import nn, Tensor
import math

def evaluator(criterion):
    def evaluate(model, data):
        x, y = data
        logits = model(x)
        return criterion(logits, y)
    return evaluate

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
        
def mse_evaluator(model, data):
    x, y = data
    o = model(x)
    return nn.functional.mse_loss(o, y)

def rmse_evaluator(model, data): 
    x, y = data
    o = model(x) 
    return torch.sqrt(nn.functional.mse_loss(o, y))

def encoder():
    return nn.Sequential(
        nn.Linear(1, 32),
        nn.ReLU(),
        nn.Linear(32, 5),
        nn.ReLU(),
    )

def predictor():
    return nn.Sequential(
        nn.Linear(5, 64),
        nn.ReLU(),
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, 1024),
        nn.ReLU(),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )

def weightor():
    return nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 20),
    )

class MyModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.encoders = nn.ModuleList([encoder() for _ in range(input_dim)])
        self.predictor = predictor()
        self.weightor = weightor()
        self.weight = nn.Parameter(torch.ones(input_dim))
    def forward(self, input):
        inputs = torch.split(input, 1, dim=1)
        codes =[self.encoders[i](inputs[i]) for i in range(self.input_dim)]
        codes = torch.stack(codes, dim=1)
        if self.training:
            weights = self.weightor(self.weight)
        else:
            weight = self.weightor(self.weight)
            abs_arr = torch.abs(weight)
            topk_values, topk_indices = torch.topk(abs_arr, k=1)
            weights = torch.zeros_like(weight)
            weights[topk_indices] = weight[topk_indices]
        summary = torch.einsum('bij,i->bj', codes, weights)
        return self.predictor(summary)


