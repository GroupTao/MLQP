# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 11:08:29 2021

@author: Yuqiang (Ethan) Heng
"""

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

lr = 0.1
batch_size = 1
epochs = 200
np.random.seed(7)
# number of probing beams (N_W)

# dataset =


#读取数据
train_data_address = './Dataset/two_spiral_train_data-1.txt'  # 'Rosslyn_ULA' or 'O1_28B_ULA' or 'I3_60_ULA' or 'O1_28_ULA'
test_data_address = './Dataset/two_spiral_test_data-1.txt'
plt_save_address = './9 submodel'

# train_data = np.loadtxt(train_data_address,dtype=np.float32)
train_data = generate_data(3000)
np.random.shuffle(train_data)
row_ix0 = where(train_data[:,2]==0)
row_ix1 = where(train_data[:,2]==1)

train_data0 = train_data[row_ix0,:]
train_data1 = train_data[row_ix1,:]
train_data0 = np.reshape(train_data0,(3,-1,3))
train_data1 = np.reshape(train_data1,(3,-1,3))

#test数据集
test_data = np.loadtxt(test_data_address,dtype=np.float32)
test_X,test_Y = torch.from_numpy(test_data[:,0:2]),torch.from_numpy(test_data[:,2])
test = torch.utils.data.TensorDataset(test_X,test_Y)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)

# 黑白图坐标
blank = torch.zeros((int(12 / 0.05), int(12 / 0.05), 1))

x = torch.arange(-6, 6, 0.05)
x = torch.reshape(x, (-1, 1, 1))
x = blank + x

y = torch.arange(-6, 6, 0.05)
y = torch.reshape(y, (1, -1, 1))
y = blank + y

axis = torch.cat((x, y), dim=2)
axis = torch.reshape(axis, (240 * 240, 2))

#定义模型
num_model = 9
mutimodel = nn.ModuleList()
for _ in range(num_model):
    mutimodel.append(nn.Sequential(
        MLQP2(),
    ))
#模型分别训练与作图
for i in range(3):
    for ii in range(3):
        train_data = np.concatenate((train_data1[i,:,:],train_data0[ii,:,:]),axis=0)
        np.random.shuffle(train_data)
        train_X, train_Y = torch.from_numpy(train_data[:, 0:2]), torch.from_numpy(train_data[:, 2])
        train = torch.utils.data.TensorDataset(train_X, train_Y)
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)

        # 模型训练
        opt = optim.SGD(mutimodel[i*3+ii].parameters(), lr=lr)
        print('Start training, model_num = {num}'.format(num = i*3+ii))
        st = time.time()
        train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist = fit_model(model=mutimodel[i*3+ii],
                                                                                   train_loader=train_loader,
                                                                                   test_loader=test_loader, opt=opt,
                                                                                   loss_fn=nn.MSELoss(), epochs=epochs)
        st = time.time() - st
        print('time cost{time} sec'.format(time=st))


        # 作黑白图
        classify = mutimodel[i*3+ii](axis)
        classify = (classify >= 0.5)

        groups = unique(classify)
        for group in groups:
            row_ix = where(classify == group)
            pyplot.scatter(axis[row_ix, 0], axis[row_ix, 1])
        pyplot.savefig(plt_save_address + '/submodel{}.png'.format(i * 3 + ii))
        pyplot.show()


#作min结构的黑白图
for i in range(3):
    classify = min_structrue(mutimodel[i*3],mutimodel[i*3+1],mutimodel[i*3+2],axis)
    classify = (classify >= 0.5)
    groups = unique(classify)
    for group in groups:
        row_ix = where(classify == group)
        pyplot.scatter(axis[row_ix, 0], axis[row_ix, 1])
    pyplot.savefig(plt_save_address + '/MINstucture{}.png'.format(i))
    pyplot.show()

#作max-min结构的黑白图并计算准确率
test_acc = 0
for i in range(len(test_X)):
    output = min_max_structure(mutimodel,test_X[i,:])
    output = int(output >= 0.5)
    test_acc += (output == int(test_Y[i]))
test_acc = test_acc/ len(test_X)
print('test Acc = {Acc}'.format(Acc = test_acc))



classify = min_max_structure(mutimodel,axis)
classify = (classify >= 0.5)
groups = unique(classify)
for group in groups:
    row_ix = where(classify == group)
    pyplot.scatter(axis[row_ix, 0], axis[row_ix, 1])
pyplot.savefig(plt_save_address + '/MIN-MAX.png')
pyplot.show()


temp = 0



