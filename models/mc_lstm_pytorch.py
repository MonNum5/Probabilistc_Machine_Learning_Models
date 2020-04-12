

import torch
import torch.utils.data
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

#other imports
from sklearn.metrics import r2_score
import numpy as np
from copy import deepcopy

torch.manual_seed(42)    # reproducible


def loss_if_below_zero(model,x,dep):
    '''
    x_train_new=torch.FloatTensor(0,x.shape[1])
    for i in range(x.shape[0]):
        for dep_i in dep:
            x_i=deepcopy(x[i].reshape(1,-1))
            x_i[:,-1]=dep_i  
            x_train_new=torch.cat((x_train_new,x_i),0)
    y_pred=model(x_train_new)
    below_zero=y_pred[y_pred<0]
    criterion=nn.MSELoss()
    below_zero_loss=criterion(below_zero,torch.zeros(size=below_zero.shape))
    if below_zero_loss!=below_zero_loss: #check if nan
        below_zero_loss=Variable(torch.tensor(0)).type(torch.float32)
    '''
    loss=Variable(torch.tensor(0)).type(torch.float32)
    for i in range(x.shape[0]):
        x_i_extend=torch.FloatTensor(0,x.shape[1])
        for i in range(len(dep)):
            dep_i=dep[i]
            x_i=deepcopy(x[i].reshape(1,-1))
            x_i[:,-1]=dep_i
            x_i_extend=torch.cat((x_i_extend,x_i),0)
        y_i_pred=model(x_i_extend)
        increase=y_i_pred[y_i_pred<0]
        criterion=nn.MSELoss()
        loss_i=criterion(increase,torch.zeros(size=increase.shape))
        if loss_i==loss_i: #check if not nan
            loss+=loss_i
    return loss

def loss_if_pred_increases_with_dep(model,x,dep):
    loss=Variable(torch.tensor(0)).type(torch.float32)
    for i in range(x.shape[0]):
        x_i_extend=torch.FloatTensor(0,x.shape[1])
        for i in range(len(dep)):
            dep_i=dep[i]
            x_i=deepcopy(x[i].reshape(1,-1))
            x_i[:,-1]=dep_i
            x_i_extend=torch.cat((x_i_extend,x_i),0)
        y_i_pred=model(x_i_extend)
        diff_i=y_i_pred[1:]-y_i_pred[:-1]
        increase=diff_i[diff_i>0]
        criterion=nn.MSELoss()
        loss_i=criterion(increase,torch.zeros(size=increase.shape))
        if loss_i==loss_i: #check if not nan
            loss+=loss_i
    return loss


def loss_if_pred_decrease_with_dep(model,x,dep):
    loss=Variable(torch.tensor(0)).type(torch.float32)
    for i in range(x.shape[0]):
        x_i_extend=torch.FloatTensor(0,x.shape[1])
        for i in range(len(dep)):
            dep_i=dep[i]
            x_i=deepcopy(x[i].reshape(1,-1))
            x_i[:,-1]=dep_i
            x_i_extend=torch.cat((x_i_extend,x_i),0)
        y_i_pred=model(x_i_extend)
                
        #above zero
        diff_i=(y_i_pred[1:]-y_i_pred[:-1])/(torch.tensor(dep)[1:].reshape(-1,1)-torch.tensor(dep)[:-1].reshape(-1,1))
        #print('dy',y_i_pred[1:]-y_i_pred[:-1])
        #print('dT',torch.tensor(dep)[1:].reshape(-1,1)-torch.tensor(dep)[:-1].reshape(-1,1))
        #print('diff',diff_i)
        decrease=diff_i[0>diff_i].reshape(-1,1)

        decrease=decrease[decrease>-0.01]
        #print('decrease',decrease)
        
        criterion=nn.L1Loss()
        loss_i=criterion(decrease,torch.zeros(size=decrease.shape))

        if loss_i==loss_i: #check if not nan
            loss+=loss_i
    return loss

from torch.nn import functional as F

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, nlayers, dropout):
        """"Constructor of the class"""
        super(LSTMCell, self).__init__()

        self.nlayers = nlayers
        self.dropout = nn.Dropout(p=dropout)

        ih, hh = [], []
        for i in range(nlayers):
            ih.append(nn.Linear(input_size, 4 * hidden_size))
            hh.append(nn.Linear(hidden_size, 4 * hidden_size))
        self.w_ih = nn.ModuleList(ih)
        self.w_hh = nn.ModuleList(hh)

    def forward(self, input, hidden):
        """"Defines the forward computation of the LSTMCell"""
        hy, cy = [], []
        for i in range(self.nlayers):
            hx, cx = hidden[0][i], hidden[1][i]
            gates = self.w_ih[i](input) + self.w_hh[i](hx)
            i_gate, f_gate, c_gate, o_gate = gates.chunk(4, 1)

            i_gate = F.sigmoid(i_gate)
            f_gate = F.sigmoid(f_gate)
            c_gate = F.tanh(c_gate)
            o_gate = F.sigmoid(o_gate)

            ncx = (f_gate * cx) + (i_gate * c_gate)
            nhx = o_gate * F.tanh(ncx)
            cy.append(ncx)
            hy.append(nhx)
            input = self.dropout(nhx)

        hy, cy = torch.stack(hy, 0), torch.stack(cy, 0)
        return hy, cy

#functiion for custom weight initization
def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1.0)
        m.bias.data.fill_(0)

class NN(torch.nn.Module):
    def __init__(self, input_layer, output_layer, lstm_layer, hidden_layers=(200,),droprate=0.1, batchnorm=False, activation='ReLU'):
        super().__init__()
        self.input_layer=input_layer
        self.output_layer=output_layer
        self.lstm_layer=lstm_layer
        self.hidden_layers=hidden_layers
        self.droprate=droprate
        self.batchnorm=batchnorm
        self.activation=activation
        self.build_model()

    def build_model(self):
        self.lstm=nn.LSTMCell(self.input_layer, self.lstm_layer)
        
        self.lstm=nn.RNN(self.input_layer, self.lstm_layer,1,batch_first=True)
        
        self.fc = nn.Sequential()

        if self.batchnorm==True:
            self.fc.add_module('batchnorm0', nn.BatchNorm1d(self.hidden_layers[0])) 
        self.fc.add_module('dropout0', nn.Dropout(p=self.droprate))
        
        self.fc.add_module('hidden0', nn.Linear(self.lstm_layer, self.hidden_layers[0]))
        self.fc.add_module('activation0', eval(f'nn.{self.activation}()'))
        
        if len(self.hidden_layers)>1:
            for i in range(len(self.hidden_layers)-1):
                self.fc.add_module('dropout'+str(i+1), nn.Dropout(p=self.droprate))
                self.fc.add_module('hidden'+str(i+1), nn.Linear(self.hidden_layers[i], self.hidden_layers[i+1]))
                self.fc.add_module('activation'+str(i+1), eval(f'nn.{self.activation}()'))
                if self.batchnorm==True:
                    self.fc.add_module('batchnorm'+str(i+1), nn.BatchNorm1d(self.hidden_layers[i]))
        else:
            i=-1
        self.fc.add_module('dropout'+str(i+2), nn.Dropout(p=self.droprate))
        self.fc.add_module('final', nn.Linear(self.hidden_layers[-1], self.output_layer))
                


    def forward(self,x, do_teacher_forcing=None, y=None):

        h_t = torch.zeros(x.shape[0], self.lstm_layer, dtype=torch.float32)
        c_t = torch.zeros(x.shape[0], self.lstm_layer, dtype=torch.float32)
        
        #h_t, c_t = self.lstm(x, (h_t, c_t))
        h_t=None
        h_t, c_t=self.lstm(x,h_t)
        
        output=self.fc(h_t)
        
        '''
        if do_teacher_forcing==True:
            if y is not None:# and random.random()>0.8:
                output=y
            h_t, c_t = self.lstm(x, (h_t, c_t))
            output=self.fc(h_t)
        '''
        
        #build in teacher training loop
        return output
    

    
class lstm_reg:
    def __init__(self,input_layer, output_layer, lstm_layer, hidden_layers=(200,),droprate=0.1,max_epoch=500, lr=0.02, weight_decay=1e-6, batch_size=256, T=500, batchnorm=False, activation='ReLU', cuda=True):
        self.max_epoch = max_epoch
        self.lr = lr
        self.batch_size=batch_size
        self.cuda=cuda
        self.criterion = nn.MSELoss()
        self.model = NN(input_layer=input_layer, lstm_layer=lstm_layer, output_layer=output_layer, hidden_layers=hidden_layers, droprate=droprate, batchnorm=batchnorm, activation=activation)
        
        
        if self.cuda==True:
            print('Using cuda')
            self.model=self.model.cuda()
            self.criterion = nn.MSELoss().cuda()
            
        self.T=T
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            
            
    def fit(self, X_train, y_train, do_teacher_forcing=None, verbose=False):
        X = Variable(torch.from_numpy(X_train).type(torch.FloatTensor))
        y = Variable(torch.from_numpy(y_train).type(torch.FloatTensor))
        
        if self.cuda==True:
            X=X.cuda()
            y=y.cuda()
        
        torch_dataset= torch.utils.data.TensorDataset(X,y)
        trainloader = torch.utils.data.DataLoader(torch_dataset, batch_size=self.batch_size,
                                          shuffle=True, num_workers=0)
        loss_log=[]
                
        for epoch in range(self.max_epoch):
            batch_loss=0
            for i, (batch_x, batch_y) in enumerate(trainloader, 0):
                self.optimizer.zero_grad()
                
                y1 = self.model(batch_x)
                
                
                
                loss_meas = self.criterion(y1, batch_y) #mse  
            
                dep=[-40,-20,0,20,40,100,140,180,220,300]
    
                loss_below_zero=loss_if_below_zero(self.model,X,dep)
                
                loss_incease=loss_if_pred_increases_with_dep(self.model,X,dep)
                
                dep=[40,80,100,140]
            
                loss_decease=loss_if_pred_decrease_with_dep(self.model,X,dep)
                
                loss=loss_meas+loss_below_zero+loss_incease-loss_decease
                '''
                loss = self.criterion(y1, batch_y) #mse    
                '''
                batch_loss+=loss.item()

                loss.backward()
                self.optimizer.step()
                      
            loss_sum=batch_loss
            #Save best model
            if not loss_log:
                self.best_model=self.model
            elif loss_sum<=min(loss_log):
                '''
                if not os.path.exists(cwd+'\\Probabilistic_Models/Pytorch_Best_Model/'):
                    os.makedirs(cwd+'\\Probabilistic_Models/Pytorch_Best_Model/')
                torch.save(self.model, cwd+'\\Probabilistic_Models/Pytorch_Best_Model/'+'best_pytorch_model.pth')
                '''
                self.best_model=self.model
            loss_log.append(loss_sum)

            if verbose:
                if epoch % 10 == 0:
                 
                    print('Epoch {} loss: {} loss measurements {} loss below zero {} loss incease {} loss decrease {}'.format(epoch+1, loss_sum, loss_meas, loss_below_zero, loss_incease, loss_decease))
                    
                    '''
                    print(f'Epoch {epoch} loss {loss}')
                    '''
        return self
    
    def predict(self,X, return_std=True, percentiles=(5, 50.0, 95)):        
        
        model=self.best_model
        X = Variable(torch.from_numpy(X).type(torch.FloatTensor))
        if self.cuda==True:
            X=X.cuda()
            model=model.cuda()
            
        model=model.train()
        tmp = np.array([model(X).cpu().detach().numpy() for _ in range(self.T)]).squeeze()
        
        percentiles=percentiles
        y_pred = np.percentile(tmp, percentiles, axis=0).T
        
        y_median=y_pred[:,1].reshape(-1,1)
        y_lower_upper_quantil=np.concatenate((y_pred[:,1].reshape(-1,1)-y_pred[:,0].reshape(-1,1),y_pred[:,2].reshape(-1,1)-y_pred[:,1].reshape(-1,1)),axis=1)
          
        #y_median=np.mean(tmp,axis=0)
        #y_lower_upper_quantil=np.std(tmp,axis=0)
        
        if return_std:
            return y_median, y_lower_upper_quantil
        else:
            return y_median
        
        
###############################################################################

#import data

from load_train_test_data_for_development import load_pickle_for_ml, load_test_data_for_ml

import os

import sys
root='D:\Python_Clemens\Probabilistic_Models'+'/'
#os.getcwd()+'/' #Uwe fragen wie das zu verallgemeinern ist

#Insert folders with used modules
sys.path.insert(0, root+'Database/')
sys.path.insert(0, root+'Machine_Learning/')
sys.path.insert(0, root+'Physical_Models/')

test_data_output_path='Data_Downloads/Raw_Data/'+'Paper_Test/'

#folder path raw data
preprocessed_data_output_path='Data_Downloads/Preprocessed_Data/'+'Paper_Train_transfer/'

#folder path trained models
output_path='Trained_Models/'+'Paper_Train_transfer/'

if not os.path.exists(root+output_path):
    os.makedirs(root+output_path)
    
pickle_list=['GCxGC-density.pickle',
             'GCxGC-viscosity_kinematic.pickle',
             'GCxGC-distillation.pickle',
             ]

file=pickle_list[1]

###############################################################################
#Training Data
x,y,scaler=load_pickle_for_ml(root+preprocessed_data_output_path+file)

df_dict=load_test_data_for_ml(root+test_data_output_path,file)

####################################################




import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


train_x, test_x, train_y, test_y=train_test_split(x,y)

from numpy import random

random.seed(42)

perm=random.permutation(x.shape[0])
x=x[perm]
y=y[perm]

model=lstm_reg(x.shape[1],y.shape[1], 176,hidden_layers=(176,), droprate=0.1, cuda=False,activation='ReLU', max_epoch=100, T=500)
model.fit(x, y, verbose=True)

for fuel in df_dict:
    test_x=df_dict[fuel]['To Predict']
    tmp = model.predict(test_x, return_std=True)
    y_pred = tmp[0]
    y_std = tmp[1]
    if y_std.shape[1]>1:
        y_lower=y_pred.flatten()-y_std[:,0]
        y_upper=y_pred.flatten()+y_std[:,1]
    else:
        y_lower=y_pred.flatten()-y_std.flatten()
        y_upper=y_pred.flatten()+y_std.flatten()
        
    plt.figure(figsize=(8, 6))
    plt.title('Prediction for {}'.format(fuel))
    plt.plot(test_x[:,-1],y_pred,'-')
    plt.fill_between(test_x[:,-1],y_lower.flatten(),y_upper.flatten(), alpha=0.5)

    #Measurements
    plt.plot(df_dict[fuel]['Measurements']['dep'],df_dict[fuel]['Measurements']['y'],'ko')

    plt.xlabel('temperature [°C]')
    plt.ylabel('viscosity_kinematic [mm2/s]')
    plt.ylim(-0.5,8.5)
    plt.show()