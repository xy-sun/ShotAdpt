import torch
from torch import nn
from metann.meta import *
from metann.dependentmodule import *
from metann.proto import *
from metann.utils.containers import *
from metann.utils.numpy import *
import data
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from models import *
from models import rmse_evaluator
import numpy as np


rmse = rmse_evaluator
TRANSFORMER = lambda x:x 
TRANSFORM = None
PARSER = lambda x:x
train_update=5
test_update=10
device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
model = MyModel(20).to(device)
maml = MAMLpp(model, train_update,test_update, lr=1e-5, evaluator=rmse)  
batch_size= 5 
iterations = ITERATIONS*batch_size
single_epochs = 1
epochs = iterations//single_epochs
writer = SummaryWriter()
optimizer = optim.Adam(maml.parameters(), lr=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs,
                                                 eta_min=1e-5)
data_provider = data.MetaZDataProvider(root=DATA_PATH, batch_size=SUPPORT_SET,
                                       batch_size_eval=QUERY_SET, length=iterations, parser=PARSER, 
                                       items=None)
it = iter(data_provider)
itn = zip(*[it for _ in range(batch_size)])

for i, datasets in enumerate(itn):
    n = i*batch_size
    running_loss = 0
    loss = 0
    optimizer.zero_grad()
    maml.train()  
    for dataset in datasets: 
        (x, y), (x_test, y_test) = dataset
        x = x.to(device)
        y = y.to(device)
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        model = maml((x, y))
        model.eval() 
        _running_loss, _loss =  mamlpp_evaluator(model, (x_test, y_test), steps=train_update,
                                               evaluator=rmse,
                                               gamma=max(0.,1-n/(0.6*iterations)))      
        loss += _loss 
        running_loss += _running_loss
    running_loss = running_loss/batch_size  
    loss = loss/batch_size
    running_loss.backward()
    torch.nn.utils.clip_grad_value_(maml.parameters(), 1)
    optimizer.step()

