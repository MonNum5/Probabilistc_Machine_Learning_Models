# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:05:51 2020

@author: hall_ce
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 09:03:01 2019

@author: hall_ce
"""

#pytorch monte carlo dropout model, wth physical guidance


#pytorch imports
import torch
import torch.utils.data
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

#other imports
import numpy as np

torch.manual_seed(42)    # reproducible
    
#functiion for custom weight initization
def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1.0)
        m.bias.data.fill_(0)

class NN(torch.nn.Module):
    def __init__(self, input_layer, output_layer, hidden_layers=(300,300,),droprate=0.5, batchnorm=False, activation='Sigmoid'):
        super().__init__()
        self.input_layer=input_layer
        self.output_layer=output_layer
        self.hidden_layers=hidden_layers
        self.droprate=droprate
        self.batchnorm=batchnorm
        self.activation=activation
        self.build_model()

    def build_model(self):
        self.model = nn.Sequential()
        self.model.add_module('input', nn.Linear(self.input_layer, self.hidden_layers[0]))
        self.model.add_module('activation0', eval(f'nn.{self.activation}()'))
        if self.batchnorm==True:
            self.model.add_module('batchnorm0', nn.BatchNorm1d(self.hidden_layers[0])) 
        if len(self.hidden_layers)>1:
            for i in range(len(self.hidden_layers)-1):
                self.model.add_module('dropout'+str(i+1), nn.Dropout(p=self.droprate))
                self.model.add_module('hidden'+str(i+1), nn.Linear(self.hidden_layers[i], self.hidden_layers[i+1]))
                self.model.add_module('activation'+str(i+1), eval(f'nn.{self.activation}()'))
                if self.batchnorm==True:
                    self.model.add_module('batchnorm'+str(i+1), nn.BatchNorm1d(self.hidden_layers[i]))
        else:
            i=-1
        
        self.model.add_module('dropout'+str(i+2), nn.Dropout(p=self.droprate))
        self.model.add_module('final', nn.Linear(self.hidden_layers[i+1], self.output_layer))


    def forward(self,x):
        output = self.model(x)
        return output
    
    
class MCDnet:
    def __init__(self,input_layer, output_layer, hidden_layers=(300,300,),droprate=0.2,max_epoch=500, lr=0.01, weight_decay=1e-6, batch_size=256, T=1000, batchnorm=False, activation='ReLU', cuda=True):
        self.max_epoch = max_epoch
        self.lr = lr
        self.batch_size=batch_size
        self.cuda=cuda
        self.criterion = nn.MSELoss()
        self.model = NN(input_layer=input_layer, output_layer=output_layer, hidden_layers=hidden_layers, droprate=droprate, batchnorm=batchnorm, activation=activation)
        #custom weight initialization
        #self.model.apply(weights_init)
        
        if self.cuda==True:
            self.model=self.model.cuda()
            self.criterion = nn.MSELoss().cuda()
        
        self.T=T
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        
    def fit(self, X_train, y_train, verbose=False):

        X = Variable(torch.from_numpy(X_train).type(torch.FloatTensor))
        y = Variable(torch.from_numpy(y_train).type(torch.FloatTensor))
        
        if self.cuda==True:
            X=X.cuda()
            y=y.cuda()
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        
        
        loss_log=[]
        loss_sum=1e6
                                          
        for epoch in range(self.max_epoch):
            
            self.optimizer.zero_grad()
            y1 = self.model(X)
            loss = self.criterion(y1, y) #mse  
            
            loss.backward()
            self.optimizer.step()
                      
            loss_sum=loss.item()
            #Save best model
            if not loss_log:
                self.best_model=self.model
            elif loss_sum<=min(loss_log):
                self.best_model=self.model
            loss_log.append(loss_sum)

            if verbose:
                if epoch % 50 == 0:
                    print(f'Epoch {epoch} loss {loss}')
                   
                
                
        return self
    
    def predict(self,X, return_std=True, percentiles=(2.5, 50.0, 97.5)):        
        
        model=self.best_model
        X = Variable(torch.from_numpy(X).type(torch.FloatTensor))
        if self.cuda==True:
            X=X.cuda()
            model=model.cuda()
            
        model=model.train()
        tmp = np.array([model(X).cpu().detach().numpy() for _ in range(self.T)]).squeeze()
        
        percentiles=percentiles
        y_pred = np.percentile(tmp, percentiles, axis=0).T
        
        if len(X)==1:
            y_median=y_pred[1].reshape(-1,1)
            y_lower_upper_quantil=np.concatenate((y_pred[1].reshape(-1,1)-y_pred[0].reshape(-1,1),y_pred[2].reshape(-1,1)-y_pred[1].reshape(-1,1)),axis=1)
        else:
            y_median=y_pred[:,1].reshape(-1,1)
            y_lower_upper_quantil=np.concatenate((y_pred[:,1].reshape(-1,1)-y_pred[:,0].reshape(-1,1),y_pred[:,2].reshape(-1,1)-y_pred[:,1].reshape(-1,1)),axis=1)
        #y_median=np.mean(tmp,axis=0)
        #y_lower_upper_quantil=np.std(tmp,axis=0)
        
        if return_std:
            return y_median, y_lower_upper_quantil
        else:
            return y_median