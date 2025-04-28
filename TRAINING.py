import torch
from torch import nn
from torch.utils.data._utils.collate import default_collate as collate
from metann2.meta import *
import torch.optim as optim
from TextDataset import TextFolder
from utils.data import TaskLoader
from models import *

import random
seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def classification_accuracy_evaluator(model, data):
    x, y = data
    y_pred = model(x)
    corrects = (y_pred.argmax(dim=1) == y).sum().item()
    return corrects / len(y)


ITERATIONS = 5000
EVAL = default_evaluator_classification
K_SHOTS = 1
K_QUERY = 1
train_updata=5
test_updata=10
batch_size= 5
batch_size_test=1

device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
net = MyModel(115).to(device)
maml = MAMLpp(net,  train_updata,test_updata, lr=1e-4, 
              evaluator=EVAL,
              )
iterations = ITERATIONS
single_epochs = 1
epochs = iterations//single_epochs
# writer = SummaryWriter()
optimizer = optim.Adam(maml.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs,
                                                 eta_min=1e-5)
train_dataset = TextFolder(root1)
taskset = TaskLoader(train_dataset, 2, K_SHOTS, K_QUERY, ITERATIONS, batch_size)
test_dataset = TextFolder(root2)
test_taskset = TaskLoader(test_dataset, 2, K_SHOTS, K_QUERY,1 , batch_size_test)
for i, task in enumerate(taskset):
    data = [[x.to(device) for x in collate(a) + collate(b)] for a, b in task]
    n = i*batch_size
    running_loss = 0
    loss = 0
    acc=0
    accs_all_test=[]
    optimizer.zero_grad()
    maml.train()
    # print("Train")
    for dataset in data: 
        x_support, y_support, x_query, y_query = dataset
        model = maml((x_support, y_support))
        model.eval()   
        _running_loss, _loss = mamlpp_evaluator(model, (x_query, y_query), steps=train_updata,
                                            evaluator=EVAL,
                                            gamma=max(0., 1 - n / (0.6 * iterations))))
        running_loss += _running_loss
        loss += _loss
        with torch.no_grad():
            _, _acc = mamlpp_evaluator(
                model, (x_query, y_query), steps=train_updata,
                evaluator=classification_accuracy_evaluator,
                gamma=0)
        acc += _acc
    loss = loss/batch_size 
    running_loss = running_loss/batch_size
    acc=acc/batch_size
    running_loss.backward()
    torch.nn.utils.clip_grad_value_(maml.parameters(), 1)
    optimizer.step()

    if i % 1 == 0:
        maml.eval()
        for j, task in enumerate(test_taskset):
            data = [[x.to(device) for x in collate(a) + collate(b)] for a, b in task]
            m = j*batch_size_test
            testloss = 0
            testrunning_loss = 0
            testacc=0
            for dataset in data: 
                maml.train()
                x_support, y_support, x_query, y_query = dataset
                model = maml((x_support, y_support))
                model.eval()   
                with torch.no_grad():
                    _, test_acc = mamlpp_evaluator(
                        model, (x_query, y_query), steps=test_updata,
                        evaluator=classification_accuracy_evaluator,
                        gamma=0)
                test_running_loss, test_loss = mamlpp_evaluator(model, (x_query, y_query), steps=test_updata,
                                                    evaluator=EVAL,
                                                    gamma=max(0., 1 - n / (0.6 * iterations)))
                                                    # gamma=0)
            
                testrunning_loss += test_running_loss
                testloss += test_loss
                testacc +=test_acc
            testloss = testloss/batch_size_test 
            testrunning_loss = testrunning_loss/batch_size_test
            testacc=testacc/batch_size_test
            optimizer.zero_grad()
            maml.train()

    if n%single_epochs == 0:
        epoch = n//single_epochs
        scheduler.step()            