# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 09:03:01 2019

@author: hall_ce
"""

# pytorch monte carlo dropout model, no physical guidance

# pytorch imports
import torch
import torch.utils.data
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

# other imports
import numpy as np
import copy

torch.manual_seed(42)    # reproducible

# functiion for custom weight initization


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1.0)
        m.bias.data.fill_(0)


class NN(torch.nn.Module):
    def __init__(self, input_layer, output_layer, hidden_layers=(300, 300,), droprate=0.0, batchnorm=False, activation='ReLU'):
        super().__init__()
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.hidden_layers = hidden_layers
        self.droprate = droprate
        self.batchnorm = batchnorm
        self.activation = activation
        self.build_model()

    def build_model(self):
        self.model = nn.Sequential()
        self.model.add_module('input', nn.Linear(
            self.input_layer, self.hidden_layers[0]))
        self.model.add_module('activation0', eval(f'nn.{self.activation}()'))
        if self.batchnorm == True:
            self.model.add_module(
                'batchnorm0', nn.BatchNorm1d(self.hidden_layers[0]))
        if len(self.hidden_layers) > 1:
            for i in range(len(self.hidden_layers)-1):
                self.model.add_module('dropout'+str(i+1),
                                      nn.Dropout(p=self.droprate))
                self.model.add_module(
                    'hidden'+str(i+1), nn.Linear(self.hidden_layers[i], self.hidden_layers[i+1]))
                self.model.add_module(
                    'activation'+str(i+1), eval(f'nn.{self.activation}()'))
                if self.batchnorm == True:
                    self.model.add_module(
                        'batchnorm'+str(i+1), nn.BatchNorm1d(self.hidden_layers[i]))
        else:
            i = -1
        self.model.add_module('dropout'+str(i+2), nn.Dropout(p=self.droprate))
        self.model.add_module('final', nn.Linear(
            self.hidden_layers[i+1], self.output_layer))

    def forward(self, x):
        output = self.model(x)
        return output


class bootstrap_ensemble:
    def __init__(self, input_layer, output_layer, hidden_layers=(300, 300,), droprate=0.0, max_epoch=500, lr=0.02, weight_decay=1e-3, batch_size=256, num_nets=10, sub_set_size=0.5, batchnorm=False, activation='ReLU', cuda=True):
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.droprate = droprate
        self.weight_decay = weight_decay
        self.max_epoch = max_epoch
        self.lr = lr
        self.batchnorm = batchnorm
        self.hidden_layers = hidden_layers
        self.criterion = nn.MSELoss()
        self.cuda = cuda

        self.num_nets = num_nets
        self.sub_set_size = sub_set_size  # size of subset for each net
        self.activation = activation

    def fit(self, X_train, y_train, verbose=False):
        X = Variable(torch.from_numpy(X_train).type(torch.FloatTensor))
        y = Variable(torch.from_numpy(y_train).type(torch.FloatTensor))

        self.nets = []
        for n in range(self.num_nets):
            self.model = NN(input_layer=self.input_layer, output_layer=self.output_layer, hidden_layers=self.hidden_layers,
                            droprate=self.droprate, batchnorm=self.batchnorm, activation=self.activation)

            optimizer = optim.Adam(self.model.parameters(
            ), lr=self.lr, weight_decay=self.weight_decay)

            sub_idx = np.random.choice(np.arange(0, len(X)), size=(
                int(len(X)*self.sub_set_size),), replace=True)

            x_sub, y_sub = X[sub_idx], y[sub_idx]

            loss_log = []
            if self.cuda == True:
                x_sub = x_sub.cuda()
                y_sub = y_sub.cuda()
                self.model = self.model.cuda()
                self.criterion = nn.MSELoss().cuda()
                torch.set_default_tensor_type(torch.cuda.FloatTensor)

            for epoch in range(self.max_epoch):
                optimizer.zero_grad()
                y1 = self.model(x_sub)
                loss = self.criterion(y1, y_sub)

                if verbose:
                    if epoch % 50 == 0:
                        print(f"Epoch {epoch+1} loss: {loss.item()}")

                loss_log.append(loss.item())
                loss.backward()
                optimizer.step()

                # Save best model
                if loss.item() <= min(loss_log):
                    self.best_model = self.model

            self.nets.append(copy.deepcopy(self.best_model))

        return self

    def predict(self, X, return_std=True, percentiles=(5.0, 50.0, 95.0)):
        X = Variable(torch.from_numpy(X).type(torch.FloatTensor))

        if self.cuda == True:
            X = X.cuda()

        values = []
        for net in self.nets:
            if self.cuda == True:
                net = net.cuda()

            preds = net(X).cpu().detach().numpy()
            values.append(preds[:, 0])
        values = np.array(values).reshape(self.num_nets, X.shape[0])
        y_pred = np.percentile(values, percentiles, axis=0).T

        y_median = y_pred[:, 1].reshape(-1, 1)
        y_lower_upper_quantil = np.concatenate(
            (y_pred[:, 1].reshape(-1, 1)-y_pred[:, 0].reshape(-1, 1), y_pred[:, 2].reshape(-1, 1)-y_pred[:, 1].reshape(-1, 1)), axis=1)

        # y_median=np.mean(tmp,axis=0)
        # y_lower_upper_quantil=np.std(tmp,axis=0)

        if return_std:
            return y_median, y_lower_upper_quantil
        else:
            return y_median
