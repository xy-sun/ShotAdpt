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

def rmse_evaluator1(model, data):
    x, y = data
    o = model(x)
    return torch.sqrt(nn.functional.mse_loss(o, y))
def rmse_evaluator2(model, data):
    x, y = data
    o = model(x)
    return torch.sqrt(nn.functional.mse_loss(o, y))
def rmse_evaluator3(model, data):
    x, y = data
    o = model(x)
    return torch.sqrt(nn.functional.mse_loss(o, y))



def mean_squared_relative_error(o: Tensor, y: Tensor):
    return torch.mean(((o - y) ** 2).sum(list(range(1, y.dim())))
                      / (y ** 2).sum(list(range(1, y.dim()))))


MSRE = mean_squared_relative_error


def mean_log_squared_relative_error(o: Tensor, y: Tensor):
    return torch.mean(torch.log(((o - y) ** 2).sum(list(range(1, y.dim())))
                                / (y ** 2).sum(list(range(1, y.dim())))))


MLSRE = mean_log_squared_relative_error


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

def mixed_gaussian(mean=math.log10(50), deviation=1, epsilon=1, c=1):
    def _mixed_gaussian(x):
        return torch.exp(- (x - mean) ** 2 / (2 * deviation ** 2)) * c + epsilon

    return _mixed_gaussian


def weighted_mse_evaluator(weighting=mixed_gaussian()):
    def _evaluator(model, data):
        x, y = data
        o = model(x)
        weight = weighting(x)
        return (weight * ((y - o) ** 2)).mean()

    return _evaluator


def weighted_mse(weighting=mixed_gaussian()):
    def _evaluator(x, y):
        x, y = data
        o = model(x)
        weight = weighting(x)
        return (weight * ((y - o) ** 2)).mean()

    return _evaluator


def encoder():
    return nn.Sequential(
        nn.Linear(1, 32),
        nn.ReLU(),
        nn.Linear(32, 5),
        nn.ReLU(),
    )


def wnn():
    return nn.Sequential(
        nn.Linear(115, 230),
        nn.ReLU(),
        nn.Linear(230, 115),
    )

def predictor():
    return nn.Sequential(
        nn.Linear(5, 64),
        nn.ReLU(),
        nn.Linear(64, 256),     
        nn.ReLU(),             
        nn.Linear(256, 64),     
        nn.ReLU(),             
        nn.Linear(64, 2),
        nn.Softmax(),
    )

class MyModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.encoders = nn.ModuleList([encoder() for _ in range(input_dim)])
        self.predictor = predictor()
        self.wnn = wnn()
        self.weight = nn.Parameter(torch.ones(input_dim))
    def forward(self, input):
        inputs = torch.split(input, 1, dim=1)
        codes =[self.encoders[i](inputs[i]) for i in range(self.input_dim)]
        codes = torch.stack(codes, dim=1)
        if self.training:
            weights = self.wnn(self.weight)
        else:
            weights = self.wnn(self.weight)
            weights_sgn=torch.sgn(weights)
            weights_abs = torch.abs(weights)
            m=torch.mean(weights_abs)
            m=m.item()
            TH = torch.nn.Threshold(m, 0)
            weights_TH = TH(weights_abs)
            weights=torch.mul(weights_TH,weights_sgn)
        summary = torch.einsum('bij,i->bj', codes, weights)
        return self.predictor(summary)
