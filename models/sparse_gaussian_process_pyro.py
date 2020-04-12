
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 10:48:26 2019

@author: hall_ce
"""
#gaussian process with pyro (pytorch based)
import torch

import pyro
import pyro.contrib.gp as gp

from torch.autograd import Variable
import os
import matplotlib.pyplot as plt
import torch

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import random
import numpy as np

def kernel_fun(input_dim, rbf_variance=5., lengthscale_rbf=10.):
    kernel= gp.kernels.RBF(input_dim=input_dim, variance=torch.tensor(rbf_variance),lengthscale=torch.tensor(lengthscale_rbf))
    return kernel

class GP_pyro_sparse:
    def __init__(self,input_dim,number_of_inducing_points,lr=0.1,epochs=3000, rbf_variance=5.0, lengthscale_rbf=10., variance_noise=1.0, weight_decay=1e-4):
        self.input_dim=input_dim
        self.lr=lr
        self.epochs=epochs
        self.rbf_variance=rbf_variance
        self.lengthscale_rbf=lengthscale_rbf
        self.variance_noise=variance_noise
        self.number_of_inducing_points=number_of_inducing_points #specifies the intervall, the sparse algorithm selects the initial points
        self.weight_decay=weight_decay
        self.kernel=kernel_fun(input_dim=self.input_dim, rbf_variance=self.rbf_variance, lengthscale_rbf=self.lengthscale_rbf)
        
         
    def fit(self, X_train, y_train, verbose=False):   
        train_x=Variable(torch.from_numpy(X_train).float())
        train_y=Variable(torch.from_numpy(y_train.flatten()).float())
        #select inducing points
        #random selection

        rand_select=random.sample(range(X_train.shape[0]),self.number_of_inducing_points)
        Xu=train_x[rand_select]        

        #likelihood = gp.likelihoods.Gaussian()
        #self.gpr = gp.models.VariationalSparseGP(train_x, train_y, kernel=self.kernel, Xu=Xu, likelihood=likelihood, whiten=True)
        
        self.gpr = gp.models.SparseGPRegression(train_x, train_y, self.kernel, Xu=Xu, noise=torch.tensor(self.variance_noise), jitter=1.0e-5)
        
        optimizer = torch.optim.Adam(self.gpr.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        losses = []

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            loss = loss_fn(self.gpr.model , self.gpr.guide)
            loss.backward()
            
            if verbose:
                if epoch % 100 == 0:
                    print('Epoch {} loss: {}'.format(epoch+1, loss.item()))
            
            optimizer.step()
            losses.append(loss.item())
            
        
                    
    def predict(self,X, return_std=False): 
        x=Variable(torch.from_numpy(X).float())
        with torch.no_grad():
            mean, cov = self.gpr(x, noiseless=False)
        y_mean=mean.numpy()
        y_std=cov.diag().sqrt().numpy()
        if return_std==True:
            return y_mean, y_std
        else:
            return y_mean

'''
import matplotlib.pyplot as plt
import numpy as np
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


x_train=x[:int(x.shape[0]*0.8),:]
y_train=y[:int(x.shape[0]*0.8),:].flatten()
test_x=x[int(x.shape[0]*0.8):,:]
test_y=y[int(x.shape[0]*0.8):,:].flatten()

model=GP_pyro_sparse(input_dim=x.shape[1], number_of_inducing_points=int(x.shape[0]*0.8*0.5))

model.fit(x_train, y_train, verbose=True)


y_pred, y_std = model.predict(test_x, return_std=True)
plt.errorbar(test_y,y_pred,yerr=y_std,marker='o',fmt='none')


y_upper=y_pred+y_std
y_lower=y_pred-y_std

fig = plt.figure()
plt.plot(xx, f(xx), 'g:', label=u'$f(x) = x\,\sin(x)$')
plt.plot(X, y, 'b.', markersize=10, label=u'Observations')
plt.plot(xx.flatten(), y_pred, 'r-', label=u'Prediction')
plt.plot(xx.flatten(), y_lower, 'k-')
plt.plot(xx.reshape(-1,1), y_upper, 'k-')
plt.fill(np.concatenate([xx, xx[::-1]]),
         np.concatenate([y_upper, y_lower[::-1]]),
         alpha=.5, fc='b', ec='None', label='90% prediction interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')
plt.show()
'''
     