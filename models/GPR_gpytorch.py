# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 09:05:34 2019

@author: hall_ce
"""

from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, LinearKernel
from gpytorch.distributions import MultivariateNormal
import gpytorch
import torch

def kernel_fun(rbf_var, rbf_lengthscale, lin_var):
    return(gpytorch.kernels.ScaleKernel(RBFKernel(lengthscale=torch.tensor(rbf_lengthscale)), outputscale=torch.tensor(rbf_var))+ScaleKernel(LinearKernel(), outputscale=torch.tensor(lin_var)))

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood(), rbf_var=5., rbf_lengthscale=10., lin_var=5):
        
        train_x=torch.tensor(train_x)
        train_y=torch.tensor(train_y)
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.likelihood=likelihood
        self.mean_module = ConstantMean()
        self.ard_num_dims=train_x.shape[0]
        kernel=kernel_fun(rbf_var, rbf_lengthscale, lin_var)
        self.covar_module =kernel
                

        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
class GPR_gpytorch:
    def __init__(self, train_x, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood, noise=1.,\
                 lr=1e-1, weight_decay=1e-4, max_epoch=500, rbf_var=5., rbf_lengthscale=10., lin_var=5):
        #print(train_x, train_y, likelihood, \
         #        lr, weight_decay, max_epoch, rbf_var, rbf_lengthscale, lin_var)
        self.likelihood=likelihood(noise=torch.tensor(noise))
        self.lr=lr
        self.weight_decay=weight_decay
        self.max_epoch=max_epoch
        self.model=GPRegressionModel(train_x=train_x, train_y=train_y.flatten(), likelihood=self.likelihood, rbf_var=rbf_var, rbf_lengthscale=rbf_lengthscale, lin_var=lin_var)
    def fit(self,train_x, train_y, verbose=False):
        train_y=train_y.flatten()
        train_x=Variable(torch.from_numpy(train_x).type(torch.FloatTensor))
        train_y=Variable(torch.from_numpy(train_y.flatten()).type(torch.FloatTensor))
        
        self.model.train()
        self.likelihood.train()
        
        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        

        for epoch in range(self.max_epoch):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)

            loss.backward()
                      
            if verbose:
                if epoch % 50 == 0:
                    print('Iter %d/%d - Loss: %.3f' % (epoch + 1, self.max_epoch, loss.item()))

            optimizer.step()
                
    def predict(self,test_x, return_std=False):
        test_x=Variable(torch.from_numpy(test_x).type(torch.FloatTensor))
        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad():
            preds = self.model(test_x)
                
        y_mean=preds.mean.detach().numpy()
        std=np.sqrt(preds.variance.detach().numpy())
  
        if return_std:
            return(y_mean, std)
        else:
            return(y_mean)
        
'''
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

train_x, test_x, train_y, test_y=train_test_split(x,y)

model=GPR_gpytorch(train_x, train_y)

model.fit(train_x, train_y, verbose=True)

tmp = model.predict(test_x, return_std=True)
y_pred = tmp[0]
y_std = tmp[1]

from sklearn.metrics import mean_squared_error
print('rmse {}'.format(np.sqrt(mean_squared_error(y_pred, test_y))))

model=GPR_gpytorch(x, y)

model.fit(x,y, verbose=True)

tmp = model.predict(test_x, return_std=True)
y_pred = tmp[0]
y_std = tmp[1]


for fuel in df_dict:
    test_x=df_dict[fuel]['To Predict']
    tmp = model.predict(test_x, return_std=True)
    y_pred = tmp[0]
    y_std = tmp[1]
    if len(y_std.shape)>1:
        y_lower=y_pred.flatten()-y_std[:,0]
        y_upper=y_pred.flatten()+y_std[:,1]
    else:
        y_lower=y_pred.flatten()-y_std.flatten()
        y_upper=y_pred.flatten()+y_std.flatten()
        
    plt.figure(figsize=(8, 6))
    plt.title('Prediction for {}'.format(fuel))
    plt.plot(test_x[:,-1],y_pred,'-o')
    plt.fill_between(test_x[:,-1],y_lower.flatten(),y_upper.flatten(), alpha=0.5)

    #Measurements
    plt.plot(df_dict[fuel]['Measurements']['dep'],df_dict[fuel]['Measurements']['y'],'ko')

    plt.xlabel('Dep')
    plt.ylabel('Property')
    plt.show()
'''



