import random

import torch
import numpy as np
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import Module
import torch.nn as nn


def generate_data(num):
    r = np.random.rand(num)
    r = r*6
    theta = r * np.pi
    x = np.reshape(r * np.cos(theta), (-1, 1))
    y = np.reshape(r * np.sin(theta), (-1, 1))

    dataset1 = np.concatenate((x, y, np.ones((num, 1))), axis=1)
    dataset0 = np.concatenate((-x, -y, np.zeros((num, 1))), axis=1)
    dataset = np.concatenate((dataset0,dataset1),axis=0)

    np.random.shuffle(dataset)
    dataset = np.float32(dataset)
    return dataset



class MLQP1(nn.Module):
    def __init__(self):
        super(MLQP1, self).__init__()

        self.FC1 = nn.Linear(in_features=4,out_features=8,bias=True)
        self.FC2 = nn.Linear(in_features=8,out_features=1,bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_Q = torch.pow(x,2)
        input = torch.cat((x,x_Q),dim=1)
        output = self.FC1(input)
        output = self.sigmoid(output)
        output = self.FC2(output)
        output = self.sigmoid(output)
        return output

class MLQP2(nn.Module):
    def __init__(self):
        super(MLQP2, self).__init__()

        self.FC1 = nn.Linear(in_features=4,out_features=8,bias=True)
        self.FC2 = nn.Linear(in_features=16,out_features=1,bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_Q = torch.pow(x,2)
        input = torch.cat((x,x_Q),dim=1)
        output = self.FC1(input)
        output = self.sigmoid(output)

        output_Q = torch.pow(output,2)
        output = torch.cat((output,output_Q),dim=1)

        output = self.FC2(output)
        output = self.sigmoid(output)
        return output

def fit_model(model, train_loader, test_loader, opt, loss_fn, epochs):
    optimizer = opt
    train_loss_hist = []
    train_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            y_batch = torch.reshape(y_batch, (-1, 1))
            optimizer.zero_grad()
            output = model(X_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()
            output = int(output>=0.5)
            train_acc += np.mean(output==int(y_batch))
        train_loss /= batch_idx + 1
        train_acc /= batch_idx + 1

        model.eval()
        test_loss = 0
        test_acc = 0
        for batch_idx, (X_batch, y_batch) in enumerate(test_loader):
            y_batch = torch.reshape(y_batch,(-1,1))
            output = model(X_batch)
            loss = loss_fn(output, y_batch)
            test_loss += loss.detach().item()
            output = int(output >= 0.5)
            test_acc += np.mean(output == int(y_batch))
        test_loss /= batch_idx + 1
        test_acc /= batch_idx + 1
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)
        if epoch % 5 == 0:
            print(
                'Epoch : {}, Training loss = {:.2f}, Training Acc = {:.2f}, Validation loss = {:.2f}, Validation Acc = {:.2f}'.format(
                    epoch, train_loss, train_acc, test_loss, test_acc))

    return train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist


def min_structrue(model1,model2,model3,x):
    x1 = model1(x)
    x2 = model2(x)
    x3 = model3(x)

    [output,_] = torch.min( torch.cat((x1, x2, x3), dim=1), dim = 1)
    return output

def min_max_structure(mutimodel,x):
    x = torch.reshape(x,(-1,2))
    x1 = mutimodel[0](x)
    x2 = mutimodel[1](x)
    x3 = mutimodel[2](x)
    x4 = mutimodel[3](x)
    x5 = mutimodel[4](x)
    x6 = mutimodel[5](x)
    x7 = mutimodel[6](x)
    x8 = mutimodel[7](x)
    x9 = mutimodel[8](x)

    # temp = torch.min( torch.cat((x1,x2,x3),dim=1), dim = 1)

    [x_1,_] = torch.min( torch.cat((x1,x2,x3),dim=1), dim = 1)
    [x_2,_] = torch.min( torch.cat((x4,x5,x6),dim=1), dim = 1)
    [x_3,_] = torch.min(torch.cat((x7, x8, x9), dim=1), dim=1)

    x_1 = torch.reshape(x_1,(-1,1))
    x_2 = torch.reshape(x_2, (-1, 1))
    x_3= torch.reshape(x_3, (-1, 1))

    [output,_] = torch.max(torch.cat((x_1,x_2,x_3),dim=1),dim=1)
    return output









