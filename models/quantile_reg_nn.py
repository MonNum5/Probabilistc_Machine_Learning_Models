#qunatile neural network regression

#https://github.com/ceshine/quantile-regression-tensorflow?source=post_page-----6fdbc26b2629----------------------

#pytorch imports
import torch
import torch.utils.data
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from itertools import chain
import numpy as np

torch.manual_seed(42)    # reproducible

def mse_loss(input, target):
    return ((input - target) ** 2).sum() / input.data.nelement()


class q_model(nn.Module):
    def __init__(self,input_layer, output_layer, quantiles, hidden_layers=[1024, 1024, 1024, 1024, 1024], droprate=0.2, batchnorm=False, activation='ReLU'):     
        super().__init__()
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        
        self.hidden_layers=hidden_layers
        self.droprate = droprate
        self.batchnorm = batchnorm
        self.activation=activation
        self.build_model()
        self.init_weights()
        

    def build_model(self): 
        activation=self.activation
        self.model = nn.Sequential()
        self.model.add_module('input', nn.Linear(self.input_layer, self.hidden_layers[0]))
        self.model.add_module('activation0', eval(f'nn.{activation}()'))
        if self.batchnorm==True:
            self.model.add_module('batchnorm0', nn.BatchNorm1d(self.hidden_layers[0])) 
        for i in range(len(self.hidden_layers)-1):
            self.model.add_module('dropout'+str(i+1), nn.Dropout(p=self.droprate))
            self.model.add_module('hidden'+str(i+1), nn.Linear(self.hidden_layers[i], self.hidden_layers[i+1]))
            self.model.add_module('activation'+str(i+1), eval(f'nn.{activation}()'))
            if self.batchnorm==True:
                self.model.add_module('batchnorm'+str(i+1), nn.BatchNorm1d(self.hidden_layers[i]))
        self.model.add_module('dropout'+str(i+2), nn.Dropout(p=self.droprate))
        final_layers = [
            nn.Linear(self.hidden_layers[i+1], self.output_layer) for _ in range(len(self.quantiles))
        ]
        self.final_layers = nn.ModuleList(final_layers)
        
    def init_weights(self):
        for m in chain(self.model, self.final_layers):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):      
        tmp_=self.model(x)       
        return torch.cat([layer(tmp_) for layer in self.final_layers], dim=1)


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(torch.max((q-1) * errors, q * errors).unsqueeze(1))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss

class q_reg_mlp:
    def __init__(self,input_layer, output_layer,quantiles, hidden_layers=[1024,1024, 1024, 1024, 1024],droprate=0.2,max_epoch=500, weight_decay=1e-3, batch_size=256, T=1000, batchnorm=False):
        self.input_layer=input_layer
        self.output_layer=output_layer
        self.quantiles=quantiles
        self.hidden_layers=hidden_layers
        self.droprate=droprate
        self.weight_decay=weight_decay
        self.max_epoch = max_epoch
        self.lr = lr
        self.batch_size=batch_size
        self.batchnorm=batchnorm
        self.model=q_model(self.input_layer,self.output_layer,self.quantiles,self.hidden_layers,self.droprate,batchnorm=self.batchnorm)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.loss_func=QuantileLoss(quantiles)
        self.T=T
 
    def fit(self, X_train, y_train, verbose=False):
        X = Variable(torch.from_numpy(X_train).type(torch.FloatTensor))
        y = Variable(torch.from_numpy(y_train).type(torch.FloatTensor))
        
        torch_dataset= torch.utils.data.TensorDataset(X,y)
        trainloader = torch.utils.data.DataLoader(torch_dataset, batch_size=self.batch_size,
                                          shuffle=True, num_workers=0)

        loss_log=[]
        for epoch in range(self.max_epoch):

            for i, (batch_x, batch_y) in enumerate(trainloader, 0):
                self.optimizer.zero_grad()
                y1 = self.model(batch_x)
                loss = self.loss_func(y1, batch_y)              
            
                loss_sum=loss
                loss_log.append(loss_sum)
   
                loss_sum.backward()
                self.optimizer.step()
            
            if verbose:
                if epoch % 50 == 0:
                    print('Epoch {} loss: {}'.format(epoch+1, loss_sum))
                
            #Save best model
            if loss_sum<=min(loss_log):
                '''
                if not os.path.exists(cwd+'\\Probabilistic_Models/Pytorch_Best_Model/'):
                    os.makedirs(cwd+'\\Probabilistic_Models/Pytorch_Best_Model/')
                torch.save(self.model, cwd+'\\Probabilistic_Models/Pytorch_Best_Model/'+'best_pytorch_model.pth')
                '''
                self.best_model=self.model
                
        return self
    
    def predict(self, x, return_std=True, mc=False, percentiles=(31.73, 50.0, 68.27)):
        model=self.best_model
        x = Variable(torch.from_numpy(x).type(torch.FloatTensor))
        if mc:
            model.train()
            tmp = np.array([model(x).detach().numpy()[:,1] for _ in range(self.T)]).squeeze()

            #calculation with percentile
            percentiles=percentiles
            y_pred = np.percentile(tmp, percentiles, axis=0).T
            
        else:
            model.eval()
            y_pred = model(x).detach().numpy()
            
        if return_std==False:
            y_median=y_pred[:,1].reshape(-1,1) #only return median
            
            return(y_median)
        else:
            y_median=y_pred[:,1].reshape(-1,1)
            y_lower_upper_quantil=np.concatenate((y_pred[:,1].reshape(-1,1)-y_pred[:,0].reshape(-1,1),y_pred[:,2].reshape(-1,1)-y_pred[:,1].reshape(-1,1)),axis=1)
            
            return y_median, y_lower_upper_quantil

'''
#example
def f(x):
    """The function to predict."""
    return x * np.sin(x)

#----------------------------------------------------------------------
#  First the noiseless case
X = np.atleast_2d(np.random.uniform(0, 10.0, size=100)).T
X = X.astype(np.float32)

# Observations
y = f(X).ravel()

dy = 1.5 + 1.0 * np.random.random(y.shape)
noise = np.random.normal(0, dy)
y += noise
y = y.astype(np.float32)

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
xx = np.atleast_2d(np.linspace(0, 10, 1000)).T
xx = xx.astype(np.float32)

X.shape, y.shape, xx.shape


quantiles = [.025, .5, .975]

model=q_reg_mlp(X.shape[1],1,quantiles,[100,100,100],droprate=0.1,max_epoch=1000,weight_decay=1e-6, batchnorm=True)

model.fit(X, y, verbose=True)


tmp = model.predict(xx, mc=False)
y_pred = tmp[0]
y_lower = tmp[0]-tmp[1][:,0].reshape(-1,1)
y_upper = tmp[0]+tmp[1][:,1].reshape(-1,1)

import matplotlib.pyplot as plt

# Plot the function, the prediction and the 90% confidence interval based on
# the MSE
fig = plt.figure()
plt.plot(xx, f(xx), 'g:', label=u'$f(x) = x\,\sin(x)$')
plt.plot(X, y, 'b.', markersize=10, label=u'Observations')
plt.plot(xx.flatten(), y_pred, 'r-', label=u'Prediction')
plt.plot(xx.flatten(), y_upper, 'k-')
plt.plot(xx.reshape(-1,1), y_lower, 'k-')
plt.fill(np.concatenate([xx, xx[::-1]]),
         np.concatenate([y_upper, y_lower[::-1]]),
         alpha=.5, fc='b', ec='None', label='90% prediction interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')
plt.show()


tmp = model.predict(xx, mc=True)
y_pred = tmp[0]
y_lower = tmp[0]-tmp[1][:,0].reshape(-1,1)
y_upper = tmp[0]+tmp[1][:,1].reshape(-1,1)

import matplotlib.pyplot as plt

# Plot the function, the prediction and the 90% confidence interval based on
# the MSE
fig = plt.figure()
plt.plot(xx, f(xx), 'g:', label=u'$f(x) = x\,\sin(x)$')
plt.plot(X, y, 'b.', markersize=10, label=u'Observations')
plt.plot(xx.flatten(), y_pred, 'r-', label=u'Prediction')
plt.plot(xx.flatten(), y_upper, 'k-')
plt.plot(xx.reshape(-1,1), y_lower, 'k-')
plt.fill(np.concatenate([xx, xx[::-1]]),
         np.concatenate([y_upper, y_lower[::-1]]),
         alpha=.5, fc='b', ec='None', label='90% prediction interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')
plt.show()
'''