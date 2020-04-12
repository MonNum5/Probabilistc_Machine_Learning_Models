# Probabilistc Machine Learning Models

# Introduction
This Repository contains various probabilistic Machine Learning models and with their libraries for regression tasks. A short summary is given in the following table. Goal of this is to collect possible algorithms and compare them. All the algorithms have a similar object oriented structure and use pytorch as main library

Algorithm | Description | File 
--- | --- | --- 
Bayesian Neural Network | Bayesian neural network using the [Pyro](pyro.ai) library | bnn_pyro
Bootstrap Neural Network | Bootstrap neural network with cuda option | bootstrap_net_pytorch_cuda
Monte Carlo Dropout Neural Network | Monte Carlo dropout neural network with cuda option | mc_net_pytorch_cuda
Monte Carlo LSTM Neural Network | Monte Carlo Long Short Term Memory neural network | mc_lstm_pytorch
Deep Ensemble Monte Carlo Droupout Neural Network | Deep Ensemble dropout neural network | pytorch_mc_deep_ensemble
Quantile Regression Neural Network | Quantile Regression Neural Network | quantile_reg_nn
Gaussian Process | Gaussian Process using [Pyro](pyro.ai) library | gaussian_process_pyro
Gaussian Process | Gaussian Process using [GPytorch](https://gpytorch.ai/) library | GPR_gpytorch_pyro
Sparse Gaussian Process | Gaussian Process using [Pyro](pyro.ai) library | sparse_gaussian_process_pytro
Sparse Gaussian Process | Gaussian Process using [GPytorch](https://gpytorch.ai/) library | sparse_gpytorch


# File and folder description
File/Folder| Description 
--- | ---
comparison.ipynb | Jupyter notebook with the comparison of the algorithm
moedels | Folder containing the compared models

# Installation 
```bash
pip install -r requirements.txt
```

## If you can want to run the file in a new enviroment:
- Make sure conda is installed (Best practice, set up with virtualenv is not tested)
- Open a terminal or a anaconda prompt
- If desired make new enviroment: conda create -n name_of_enviroment python
- Activate enviroment conda activate: conda create name_of_enviroment
- Install dependencies: pip install requirements.txt
- If the new enviroment / kernel is supposed to be used in Jupyter, install kernel:
    python -m ipykernel install --name name_of_enviroment
- Open your Jupyter Notebook it should work now


