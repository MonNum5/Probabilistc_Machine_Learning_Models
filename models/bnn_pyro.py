# Bayesian Neural Network for Regression using the Pyro library

# torch imports
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import sys
import os
from load_train_test_data_for_development import load_pickle_for_ml, load_test_data_for_ml
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import pyro
from torch.autograd import Variable

# pyro imports
import pyro
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO
import numpy as np

softplus = torch.nn.Softplus()


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1.0)
        m.bias.data.fill_(0)


class NN:
    def __init__(self, input_layer, output_layer, hidden_layers=(200, 200,), droprate=0, batchnorm=False, activation='ReLU'):
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.hidden_layers = hidden_layers
        self.droprate = droprate
        self.batchnorm = batchnorm
        self.activation = activation
        self.build_model()

    def build_model(self):
        self.model = torch.nn.Sequential()
        self.model.add_module('input', torch.nn.Linear(
            self.input_layer, self.hidden_layers[0]))
        self.model.add_module('activation0', eval(
            f'torch.nn.{self.activation}()'))
        if self.batchnorm == True:
            self.model.add_module(
                'batchnorm0', torch.nn.BatchNorm1d(self.hidden_layers[0]))
        if len(self.hidden_layers) > 1:
            for i in range(len(self.hidden_layers)-1):
                self.model.add_module('dropout'+str(i+1),
                                      torch.nn.Dropout(p=self.droprate))
                self.model.add_module(
                    'hidden'+str(i+1), torch.nn.Linear(self.hidden_layers[i], self.hidden_layers[i+1]))
                self.model.add_module(
                    'activation'+str(i+1), eval(f'torch.nn.{self.activation}()'))
                if self.batchnorm == True:
                    self.model.add_module(
                        'batchnorm'+str(i+1), torch.nn.BatchNorm1d(self.hidden_layers[i]))
        else:
            i = -1
        self.model.add_module('dropout'+str(i+2),
                              torch.nn.Dropout(p=self.droprate))
        self.model.add_module('final', torch.nn.Linear(
            self.hidden_layers[i+1], self.output_layer))

    def forward(self, x):
        output = self.model(x)
        return output


class bnn_pyro:
    def __init__(self, X, Y, hidden_layers=(200,), droprate=0, batchnorm=False, activation='ReLU', epochs=5000, lr=2e-3, weight_decay=1e-6, T=100, batch_size=256):
        self.net = NN(input_layer=X.shape[1], output_layer=Y.shape[1], hidden_layers=hidden_layers,
                      droprate=droprate, batchnorm=batchnorm, activation=activation)
        # custom weight initialization
        self.net.model.apply(weights_init)
        self.X = Variable(torch.from_numpy(X).type(torch.FloatTensor))
        self.Y = Variable(torch.from_numpy(Y).type(torch.FloatTensor))
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.T = T
        self.batch_size = batch_size

    def model(self, data):
        x_data = data[:, :-1]
        y_data = data[:, -1].flatten()
        modules = self.net.model._modules
        priors = {}
        priors['input.weight'] = Normal(loc=Variable(torch.zeros_like(modules['input'].weight)).type_as(
            data), scale=Variable(torch.ones_like(modules['input'].weight)).type_as(data))
        priors['input.bias'] = Normal(loc=Variable(torch.zeros_like(modules['input'].bias)).type_as(
            data), scale=Variable(torch.ones_like(modules['input'].bias)).type_as(data))

        for i in range(1, len(self.net.hidden_layers)):
            priors[f'hidden{i}.weight'] = Normal(loc=Variable(torch.zeros_like(modules[f'hidden{i}'].weight)).type_as(
                data), scale=Variable(torch.ones_like(modules[f'hidden{i}'].weight)).type_as(data))
            priors[f'hidden{i}.bias'] = Normal(loc=Variable(torch.zeros_like(modules[f'hidden{i}'].bias)).type_as(
                data), scale=Variable(torch.ones_like(modules[f'hidden{i}'].bias)).type_as(data))

        priors['final.weight'] = Normal(loc=Variable(torch.zeros_like(modules['final'].weight)).type_as(
            data), scale=Variable(torch.ones_like(modules['final'].weight)).type_as(data))
        priors['final.bias'] = Normal(loc=Variable(torch.zeros_like(modules['final'].bias)).type_as(
            data), scale=Variable(torch.ones_like(modules['final'].bias)).type_as(data))

        # lift module parameters to random variables sampled from the priors
        lifted_module = pyro.random_module("module", self.net.model, priors)

        # sample a regressor (which also samples w and b)
        lifted_reg_model = lifted_module()

        # run the regressor forward conditioned on inputs
        prediction_mean = lifted_reg_model(x_data).squeeze()
        pyro.sample('obs', Normal(prediction_mean, Variable(torch.ones(data.size(0))).type_as(data)),
                    obs=y_data.squeeze())

    def guide(self, data):
        modules = self.net.model._modules
        priors = {}
        params = {}
        dist = {}

        # First layer weight distribution priors
        priors['input_mu_w'] = Variable(torch.randn_like(
            modules['input'].weight).type_as(data.data), requires_grad=True)
        priors['input_sigma_w'] = Variable(torch.randn_like(
            modules['input'].weight).type_as(data.data), requires_grad=True)
        params['input_mu_param_w'] = pyro.param(
            'input_mu_w', priors['input_sigma_w'])
        params['input_sigma_param_w'] = softplus(pyro.param(
            "input_sigma_param_w", priors['input_sigma_w']))
        dist['input.weight'] = Normal(
            params['input_mu_param_w'], params['input_sigma_param_w'])

        # First layer bias distribution priors
        priors['input_mu_b'] = Variable(torch.randn_like(
            modules['input'].bias).type_as(data.data), requires_grad=True)
        priors['input_sigma_b'] = Variable(torch.randn_like(
            modules['input'].bias).type_as(data.data), requires_grad=True)
        params['input_mu_param_b'] = pyro.param(
            'input_mu_b', priors['input_mu_b'])
        params['input_sigma_param_b'] = softplus(pyro.param(
            "input_sigma_param_b", priors['input_sigma_b']))
        dist['input.bias'] = Normal(
            params['input_mu_param_b'], params['input_sigma_param_b'])

        for i in range(1, len(self.net.hidden_layers)):
            # Hidden layers weights
            priors[f'hidden{i}_mu_w'] = Variable(torch.randn_like(
                modules[f'hidden{i}'].weight).type_as(data.data), requires_grad=True)
            priors[f'hidden{i}_sigma_w'] = Variable(torch.randn_like(
                modules[f'hidden{i}'].weight).type_as(data.data), requires_grad=True)
            params[f'hidden{i}_mu_param_w'] = pyro.param(
                'hidden{i}_mu_w', priors[f'hidden{i}_mu_w'])
            params[f'hidden{i}_sigma_param_w'] = softplus(pyro.param(
                f"hidden{i}_sigma_param_w", priors[f'hidden{i}_sigma_w']))
            dist[f'hidden{i}.weight'] = Normal(
                params[f'hidden{i}_mu_param_w'], params[f'hidden{i}_sigma_param_w'])

            # Hidden layers bias
            priors[f'hidden{i}_mu_b'] = Variable(torch.randn_like(
                modules[f'hidden{i}'].bias).type_as(data.data), requires_grad=True)
            priors[f'hidden{i}_sigma_b'] = Variable(torch.randn_like(
                modules[f'hidden{i}'].bias).type_as(data.data), requires_grad=True)
            params[f'hidden{i}_mu_param_b'] = pyro.param(
                f'hidden{i}_mu_b', priors[f'hidden{i}_mu_b'])
            params[f'hidden{i}_sigma_param_b'] = softplus(pyro.param(
                f'hidden{i}_sigma_param_b', priors[f'hidden{i}_sigma_b']))
            dist[f'hidden{i}.bias'] = Normal(
                params[f'hidden{i}_mu_param_b'], params[f'hidden{i}_sigma_param_b'])

        # First layer weight distribution priors
        priors['final_mu_w'] = Variable(torch.randn_like(
            modules['final'].weight).type_as(data.data), requires_grad=True)
        priors['final_sigma_w'] = Variable(torch.randn_like(
            modules['final'].weight).type_as(data.data), requires_grad=True)
        params['final_mu_param_w'] = pyro.param(
            'final_mu_w', priors['final_mu_w'])
        params['final_sigma_param_w'] = softplus(pyro.param(
            "final_sigma_param_w", priors['final_sigma_w']))
        dist['final.weight'] = Normal(
            params['final_mu_param_w'], params['final_sigma_param_w'])

        # First layer bias distribution priors
        priors['final_mu_b'] = Variable(torch.randn_like(
            modules['final'].bias).type_as(data.data), requires_grad=True)
        priors['final_sigma_b'] = Variable(torch.randn_like(
            modules['final'].bias).type_as(data.data), requires_grad=True)
        params['final_mu_param_b'] = pyro.param(
            'final_mu_b', priors['final_mu_b'])
        params['final_sigma_param_b'] = softplus(pyro.param(
            "final_sigma_param_b", priors['final_sigma_b']))
        dist['final.bias'] = Normal(
            params['final_mu_param_b'], params['final_sigma_param_b'])

        lifted_module = pyro.random_module("module", self.net.model, dist)

        return lifted_module()

    def fit(self, X, Y, verbose=False):
        optim = Adam({"lr": self.weight_decay,
                      "weight_decay": self.weight_decay})
        svi = SVI(self.model, self.guide, optim, loss=Trace_ELBO())

        data = np.concatenate((X, Y), axis=1)
        data = Variable(torch.from_numpy(data).type(torch.FloatTensor))

        loss_log = []
        for epoch in range(self.epochs):

            loss = svi.step(data)
            loss_log.append(loss)

            if verbose:
                if epoch % 100 == 0:
                    print('Epoch {} loss: {}'.format(epoch+1, loss))

            # Save best model
            if loss <= min(loss_log):
                self.best_model = self.guide

    def predict(self, X_test, return_std=True, percentiles=(31.73, 50.0, 68.27)):
        X_test = Variable(torch.from_numpy(X_test).type(torch.FloatTensor))

        preds = []
        '''
        for i in range(100):
            sampled_reg_model = self.best_model(X_test)
            pred = sampled_reg_model(X_test).data.numpy().flatten()
            preds.append(pred)
        return preds
        '''

        for i in range(self.T):
            sampled_reg_model = self.best_model(X_test)
            tmp = sampled_reg_model(X_test).data.numpy().flatten()
            preds.append(tmp)

        y_pred = np.percentile(preds, percentiles, axis=0).T

        y_median = y_pred[:, 1].reshape(-1, 1)
        y_lower_upper_quantil = np.concatenate(
            (y_pred[:, 1].reshape(-1, 1)-y_pred[:, 0].reshape(-1, 1), y_pred[:, 2].reshape(-1, 1)-y_pred[:, 1].reshape(-1, 1)), axis=1)

        '''
        y_median=np.mean(preds,axis=0)
        y_lower_upper_quantil=np.std(preds,axis=0)
        '''

        if return_std:
            return y_median, y_lower_upper_quantil, preds
        else:
            return y_median

###############################################################################

#import data


root = 'D:\Python_Clemens\Probabilistic_Models'+'/'
# os.getcwd()+'/' #Uwe fragen wie das zu verallgemeinern ist

# Insert folders with used modules
sys.path.insert(0, root+'Database/')
sys.path.insert(0, root+'Machine_Learning/')
sys.path.insert(0, root+'Physical_Models/')

test_data_output_path = 'Data_Downloads/Raw_Data/'+'Paper_Test/'

# folder path raw data
preprocessed_data_output_path = 'Data_Downloads/Preprocessed_Data/' + \
    'Paper_Train_Transfer/'

# folder path trained models
output_path = 'Trained_Models/'+'Paper_Train_transfer/'

if not os.path.exists(root+output_path):
    os.makedirs(root+output_path)

pickle_list = ['GCxGC-density.pickle',
               'GCxGC-viscosity_kinematic.pickle',
               'GCxGC-distillation.pickle',
               ]

file = pickle_list[1]

###############################################################################
# Training Data
x, y, scaler = load_pickle_for_ml(root+preprocessed_data_output_path+file)

df_dict = load_test_data_for_ml(root+test_data_output_path, file)

####################################################

scaler = StandardScaler().fit_transform(x)
train_x, test_x, train_y, test_y = train_test_split(x, y)


pyro.clear_param_store()
model = bnn_pyro(train_x, train_y, hidden_layers=(
    200,), droprate=0, batch_size=x.shape[0])
model.fit(train_x, train_y, verbose=True)
pred = model.predict(test_x, return_std=True)
print('rmse {}'.format(np.sqrt(mean_squared_error(pred[0], test_y))))

# unit plot
plt.figure(figsize=(8, 6))
plt.errorbar(test_y, pred[0], yerr=pred[1][:, 0], fmt='o')
plt.plot(test_y, test_y, 'k-')
plt.title('Prediction for {}'.format(file))
plt.xlabel('Dep')
plt.ylabel('Property')
plt.show()

pyro.clear_param_store()
model = bnn_pyro(x, y, hidden_layers=(200,), T=100, batch_size=x.shape[0])
model.fit(x, y, verbose=True)

for fuel in df_dict:
    test_x = df_dict[fuel]['To Predict']
    tmp = model.predict(test_x, return_std=True)
    y_pred = tmp[0]
    y_std = tmp[1]
    if y_std.shape[1] > 1:
        y_lower = y_pred.flatten()-y_std[:, 0]
        y_upper = y_pred.flatten()+y_std[:, 1]
    else:
        y_lower = y_pred.flatten()-y_std.flatten()
        y_upper = y_pred.flatten()+y_std.flatten()

    plt.figure(figsize=(8, 6))
    plt.title('Prediction for {}'.format(fuel))
    plt.plot(test_x[:, -1], y_pred, '-')
    plt.fill_between(test_x[:, -1], y_lower.flatten(),
                     y_upper.flatten(), alpha=0.5)

    # Measurements
    plt.plot(df_dict[fuel]['Measurements']['dep'],
             df_dict[fuel]['Measurements']['y'], 'ko')

    plt.xlabel('Dep')
    plt.ylabel('Property')
    plt.show()
