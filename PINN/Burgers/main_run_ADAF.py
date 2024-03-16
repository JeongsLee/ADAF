import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="100"

import pickle
import tensorflow as tf
#tf.keras.utils.disable_interactive_logging()

import numpy as np
import matplotlib.pyplot as plt

from time import time
from pinn_utils import *
from multiprocessing import Pool


def run(trial, condition):
    # Model keyword
    key = 'ADAF'
    num_hidden_layers=condition[0]
    num_neurons_per_layer=condition[1]
    
    DTYPE = 'float32'
    loss_dict = {
        'loss_BC_coeff': tf.constant([1e0, 1e0]),
        'loss_PDE_coeff': tf.constant(1e0),
        'loss_IC_coeff': tf.constant(1e0)
        }

    ## Time Stepping Hyperparameters
    time_stepping_number = 1
    time_marching_constant = 1

    # Material Properties
    viscosity = .01/np.pi
    time_concern = 1.
    L = 1.

    # Set boundary
    tmin = 0.
    tmax = time_concern
    xmin = -L
    xmax = L

    # Properties_dict
    properties = {
        'viscosity':viscosity,  
        'L':L,
        'time_concern':time_concern,
        'time_stepping_number':time_stepping_number,
        'time_marching_constant':time_marching_constant,
        'tmin':tmin,
        'tmax':time_concern,
        'xmin':xmin,
        'xmax':L,
        }

    # Set number of data points
    N_0 = 200
    N_b = 200
    N_r = 10000

    #Model construction
    lb = tf.constant([tmin, xmin], dtype=DTYPE)
    ub = tf.constant([tmax/time_marching_constant, xmax], dtype=DTYPE)
    pinn = Build_PINN(lb, ub, num_hidden_layers, num_neurons_per_layer, key)
    pinn.model.summary()
    tf.keras.utils.disable_interactive_logging()


    #Solver
    solver = Solver_PINN(pinn, properties, loss_dict, N_0=N_0, N_b=N_b, N_r=N_r)
    #Train
    ref_time = time()
    #Train Adam
    solver.train_adam(200)
    time1 = time()-ref_time
    print('\nComputation time: {} seconds'.format(time()-ref_time))
    #Train lbfgs
    ref_time = time()
    solver.ScipyOptimizer(method='L-BFGS-B', 
        options={'maxiter': 40000, 
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
    condition_set = [(4,60)]
    for condition in condition_set:
        run(trial, condition)    
    

if __name__== '__main__':
    p = Pool(processes=10)
    p.map(run_upper, range(10))