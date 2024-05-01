
# Deep Learning and  scrambling

This repository contains the code to generate the results of the paper  https://arxiv.org/abs/2302.04621.

## Notebooks:

### 1. Random_parameters_generation.ipynb
This notebook demonstrates how to generate random parameters $\theta_p^i$of our circuit by applying a random Gaussian Process. 

### 2. OTOC.ipynb
This notebook demonstrates how to calculate OTOC for our random circuits and reproduce Fig. 6.

### 2. Training.ipynb
This notebook focuses on training the neural network. It includes:
- Network training process and architecture details.
- Evaluation of the network's generalization power.
- Extrapolation tests on different driving fields: fresh Gaussian, quench, and periodic fields. Extrapolation power in time.

## Python Modules:

### 1. Simulating_Dynamics_of_random_circuits.py
This module includes the code for simulating the dynamics of random circuits to generate data for training. 




