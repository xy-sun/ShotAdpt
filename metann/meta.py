import torch
from torch import nn
from .proto import tensor_copy, mimo_functional,MimoModel
from .utils.containers import MultipleList
from metann import ProtoModule
from metann.utils.containers import DefaultList
import numpy as np

is_tensor = np.vectorize(lambda x: isinstance(x, torch.Tensor))


def active_indices(lst):
    indices = []
    for k, v in enumerate(lst):
        if isinstance(v, torch.Tensor):
            indices.append(k)
    return indices


def default_evaluator_classification(model, data, criterion=nn.CrossEntropyLoss()):
    x, y = data
    logits = model(x)
    loss = criterion(logits, y)
    return loss


def mamlpp_evaluator(mimo, data, steps, evaluator, gamma=0.6):
    weights = [1*gamma**i for i in range(steps+1)]
    weights = list(reversed(weights))
    evaluators = [evaluator] * (steps+1)
    data = [data] * (steps+1)
    loss = mimo(data, evaluators)
    return sum(i[0] * i[1] for i in zip(loss, weights)), loss[-1]

def maml_evaluator(mimo, data, steps, evaluator, gamma=0.6):
    weights = [1*gamma**i for i in range(steps+1)]
    weights = list(reversed(weights))
    evaluators = [evaluator] * (steps+1)
    data = [data] * (steps+1)
    loss = mimo(data, evaluators)
    return sum(i[0] * i[1] for i in zip(loss, weights)), loss[-1]


class Learner(nn.Module):
    def __init__(self):
        super(Learner, self).__init__()

    def forward(self, *args, inplace=False, **kwargs):
        if inplace:
            return self.forward_inplace(*args, **kwargs)
        else:
            return self.forward_pure(*args, **kwargs)

    def forward_pure(self, model, data):
        raise NotImplementedError

    def forward_inplace(self, model, data):
        raise NotImplementedError


class GDLearner(Learner):
    def __init__(self, steps, lr, create_graph=True, evaluator=default_evaluator_classification):
        super(GDLearner, self).__init__()
        self.steps = steps
        self.sgd = SequentialGDLearner(lr, momentum=0, create_graph=create_graph, evaluator=evaluator)

    def forward(self, model, data, inplace=False, **kwargs):
        kwargs['model'] = model
        kwargs['data'] = [data, ]*self.steps
        kwargs['inplace'] = inplace
        return self.sgd(**kwargs)


class SequentialGDLearner(Learner):
    def __init__(self, lr, momentum=0.5, create_graph=True, evaluator=default_evaluator_classification):
        super(SequentialGDLearner, self).__init__()
        self.lr = lr
        self.momentum = momentum
        self.create_graph = create_graph
        self.evaluator = evaluator

    def forward_pure(self, model, data, evaluator=None, mimo=False):
        evaluator = self.evaluator if evaluator is None else evaluator
        model = ProtoModule(model)
        model.train()
        if mimo:
            fast_weights_lst = [MultipleList(list(model.parameters()))]
            velocities = DefaultList(lambda: 0)
            actives = active_indices(fast_weights_lst[-1])
            for batch in data:
                fast_weights = tensor_copy(fast_weights_lst[-1])
                fast_loss = evaluator(model.functional(fast_weights), batch)
                grads = torch.autograd.grad(fast_loss, fast_weights[actives],
                create_graph=self.create_graph)
                velocities = [grad + velocity*self.momentum for (grad, velocity) in zip(grads, velocities)]
                fast_weights[actives] = [w - self.lr * g for (w, g) in zip(fast_weights[actives], velocities)]
                fast_weights_lst.append(fast_weights)
            return MimoModel(model, fast_weights_lst)
        else:
            fast_weights = MultipleList(list(model.parameters()))
            velocities = DefaultList(lambda: 0)
            actives = active_indices(fast_weights)
            for batch in data:
                fast_loss = evaluator(model.functional(fast_weights), batch)
                grads = torch.autograd.grad(fast_loss, fast_weights[actives],
                                            create_graph=self.create_graph)
                velocities = [grad + velocity*self.momentum for (grad, velocity) in zip(grads, velocities)]
                fast_weights[actives] = [w - self.lr * g for (w, g) in zip(fast_weights[actives], velocities)]
            return model.functional(fast_weights)

    def forward_inplace(self, model, data, evaluator=None):
        evaluator = self.evaluator if evaluator is None else evaluator
        optim = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)
        for batch in data:
            optim.zero_grad()
            loss = evaluator(model, batch)
            loss.backward()
            optim.step()
        return model


class MAML(nn.Module):
    def __init__(self, model, steps_train, steps_eval, lr,
                 evaluator=default_evaluator_classification, first_order=False):
        super(MAML, self).__init__()
        self.model = model
        self.steps_train = steps_train
        self.steps_eval = steps_eval
        self.lr = lr
        self.evaluator = evaluator
        self.first_order = first_order

    def forward(self, data):
        if self.training:
            steps = self.steps_train
        else:
            steps = self.steps_eval
        learner = GDLearner(self.steps_train, self.lr, create_graph=not self.first_order)
        return learner(self.model, data, evaluator=self.evaluator)


class MAMLpp(nn.Module):
    def __init__(self, model, steps_train, steps_eval, lr,
                 evaluator=default_evaluator_classification, first_order=False):
        super(MAMLpp, self).__init__()
        self.model = model
        self.steps_train = steps_train
        self.steps_eval = steps_eval
        self.lr = lr
        self.evaluator = evaluator
        self.first_order = first_order

    def forward(self, data):
        if self.training:
            steps = self.steps_train
        else:
            steps = self.steps_eval
        learner = GDLearner(steps, self.lr, create_graph=not self.first_order)

        if self.training:
            return learner(self.model, data, evaluator=self.evaluator, mimo=True) 
        else:
            return learner(self.model, data, evaluator=self.evaluator, mimo=True)
