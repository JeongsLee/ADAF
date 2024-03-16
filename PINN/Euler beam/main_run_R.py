import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="100"

import pickle
import tensorflow as tf
tf.keras.utils.disable_interactive_logging()

import numpy as np
import matplotlib.pyplot as plt

from time import time
from pinn_utils import *
from multiprocessing import Pool

def run(trial, condition):
    # Model keyword
    key = 'R' # or 'M2' or 'R'
    num_hidden_layers=condition[0]
    num_neurons_per_layer=condition[1]
    ## Time Stepping Hyperparameters
    time_stepping_number = 1
    time_marching_constant = 1
    # Material Properties
    L = 1. # Length
    F = 1.
    # Set boundary
    xmin = 0.
    xmax = L
    # Properties_dict
    properties = {
        'L':L,
        'xmin':xmin,
        'xmax':L,
        }
    DTYPE = 'float32'
    # Set number of data points
    N_r = 20
    #Model construction
    lb = tf.constant([xmin], dtype=DTYPE)
    ub = tf.constant([xmax], dtype=DTYPE)
    pinn = Build_PINN(lb, ub, properties, num_hidden_layers, num_neurons_per_layer, key)
    pinn.model.summary()
    #Solver
    solver = Solver_PINN(pinn, properties, N_r=N_r)
    #Train Adam
    ref_time = time()
    solver.train_adam(200)
    time1 = time()-ref_time
    print('\nComputation time: {} seconds'.format(time()-ref_time))
    #Train lbfgs
    ref_time = time()
    solver.ScipyOptimizer(method='L-BFGS-B', 
        options={'maxiter': 4000, 
            'maxfun': 50000, 
            'maxcor': 50, 
            'maxls': 50, 
            'ftol': np.finfo(float).eps,
            'gtol': np.finfo(float).eps,            
            'factr':np.finfo(float).eps,
            'iprint':50})
    time2 = time()-ref_time
    print('\nComputation time: {} seconds'.format(time()-ref_time))
    solver.save_results(trial, (time1,time2))   
    del solver

def run_upper(trial):
    condition_set = [(2,10),(2,20),(4,10),(4,20)]
    for condition in condition_set:
        run(trial, condition)    
    

if __name__== '__main__':
    p = Pool(processes=10)
    p.map(run_upper, range(10))