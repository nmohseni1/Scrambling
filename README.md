# Deep Learning and Scrambling

This repository contains the code to generate the results presented in the paper [Link to paper](https://arxiv.org/abs/2302.04621).

## Notebooks

### 1. OTOC.ipynb
This notebook demonstrates how to calculate Out-of-Time-Ordered Correlators (OTOC) to reproduce Figure 6 from the paper.

### 2. LSTM.ipynb
This notebook illustrates the training of the LSTM neural network for the homogeneous model, where $\theta^i_p = \theta^j_p$. Using this notebook,  Figure 3 of the paper can be reproduced.

### 3. CONVLSTM.ipynb
This notebook showcases the training of the CONVLSTM neural network for the inhomogeneous model, where $\theta^i_p \neq \theta^j_p$. Using this notebook,  Figure 4 of the paper can be reproduced.

## Python Modules

### 1. Simulating_Dynamics_of_random_circuits.py
This module contains the code for simulating the dynamics of random circuits to generate data for training purposes.
