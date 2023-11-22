
# Deep Learning for Random Driving

This repository contains the code to generate the results of the paper  https://quantum-journal.org/papers/q-2022-05-17-714/ for the Transverse Field Ising (TFI) model.

## Notebooks:

### 1. Random_Field.ipynb
This notebook demonstrates how to generate random Gaussian fields. 

### 2. Training_TFI.ipynb
This notebook focuses on training the neural network on random Gaussian fields for the TFI model. It includes:
- Network training process and architecture details.
- Evaluation of the network's generalization power.
- Extrapolation tests on different driving fields: fresh Gaussian, quench, and periodic fields. Extrapolation power in time.

## Python Modules:

### 1. Simulating_Dynamics_Random_Gaussian.py
This module provides code for simulating dynamics using randomly generated Gaussian fields. 
It includes methods for generating training data for the neural network.

### 2. Simulating_Dynamics_Random_Periodic.py
Similar to the Random Gaussian module, this module simulates the dynamics of the driven system with random periodic fields for generating data for testing the neural network.


