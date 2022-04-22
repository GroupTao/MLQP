# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 11:08:29 2021

@author: Yuqiang (Ethan) Heng
"""

import math
import numpy as np
from numpy import unique,where
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from Model_define import *
from matplotlib import pyplot

lr = 0.03
batch_size = 1
epochs = 50
np.random.seed(7)
# number of probing beams (N_W)

#读取数据
train_data_address = './Dataset/two_spiral_train_data-1.txt'  # 'Rosslyn_ULA' or 'O1_28B_ULA' or 'I3_60_ULA' or 'O1_28_ULA'
test_data_address = './Dataset/two_spiral_test_data-1.txt'
plt_save_address = './1 model'

# train_data = np.loadtxt(train_data_address,dtype=np.float32)
train_data = generate_data(3000)
test_data = np.loadtxt(test_data_address,dtype=np.float32)

train_X,train_Y = torch.from_numpy(train_data[:,0:2]),torch.from_numpy(train_data[:,2])
test_X,test_Y = torch.from_numpy(test_data[:,0:2]),torch.from_numpy(test_data[:,2])
#数据封装
train = torch.utils.data.TensorDataset(train_X,train_Y)
test = torch.utils.data.TensorDataset(test_X,test_Y)
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)

#模型训练
model = MLQP2()
opt = optim.SGD(model.parameters(), lr=lr)

st = time.time()
train_loss_hist, train_acc_hist,test_loss_hist,test_acc_hist = fit_model(model=model, train_loader=train_loader,test_loader=test_loader,opt=opt,loss_fn=nn.MSELoss(),epochs=epochs)
st = time.time() - st
print('time cost{time} sec'.format(time=st))

plt.figure()
plt.plot(train_acc_hist, label='training acc')
plt.plot(test_acc_hist, label='validation acc')
plt.legend()
plt.title('Acc hist,lr = {}'.format(lr))
pyplot.savefig(plt_save_address + '/Acc hist.png')
plt.show()

plt.figure()
plt.plot(train_loss_hist, label='training loss')
plt.plot(test_loss_hist, label='validation loss')
plt.legend()
plt.title('Loss hist,lr = {}'.format(lr))
pyplot.savefig(plt_save_address + '/Loss hist.png')
plt.show()
# learned_codebooks.append(first_model.get_codebook())

#作黑白图
blank = torch.zeros((int(12/0.05),int(12/0.05),1))

x = torch.arange(-6,6,0.05)
x = torch.reshape(x,(-1,1,1))
x = blank + x

y = torch.arange(-6,6,0.05)
y = torch.reshape(y,(1,-1,1))
y = blank + y

axis = torch.cat((x,y),dim = 2)
axis = torch.reshape(axis,(240*240,2))

classify = model(axis)
classify = (classify>=0.5)

groups = unique(classify)
for group in groups:
    row_ix = where(classify == group)
    pyplot.scatter(axis[row_ix, 0], axis[row_ix, 1])
pyplot.savefig(plt_save_address + '/Classify.png')
pyplot.show()

temp = 0



