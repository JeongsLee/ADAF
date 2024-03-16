import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from adaf_utils import *


import scipy.optimize

from drawnow import drawnow

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

gamma = 0.2
t_step = 0.1
N_p = 10
N_m = 10
dtype = 'float64'

solver_set=[]
init2 = [1.0, 1.0, 1.0]

for i in range(200):
    t_train = np.linspace(i*t_step, (i+1)*t_step, 51)
    solver = Lorenz_Solver(t_train, gamma=gamma, t_step=t_step, N_p = N_p, N_m = N_m, init2 = init2, dtype=dtype)
    if os.path.exists('./weights_%s.txt' % i):
        weights = np.loadtxt('./weights_%s.txt' % i, delimiter=',').astype(dtype)
        solver.X.W_i = tf.Variable(weights[:,0])
        solver.Y.W_i = tf.Variable(weights[:,1])
        solver.Z.W_i = tf.Variable(weights[:,2])    
    else:    
        solver.ScipyOptimizer(method='L-BFGS-B', 
            options={'maxiter': 4000, 
                'maxfun': 50000, 
                'maxcor': 50, 
                'maxls': 50, 
                'ftol': np.finfo(float).eps,
                'gtol': np.finfo(float).eps,            
                'factr':np.finfo(float).eps,
                'iprint':50})
        weights = np.concatenate(list(map(lambda x:np.expand_dims(x,axis=-1), (solver.X.W_i, solver.Y.W_i, solver.Z.W_i))),axis=-1)
        np.savetxt('./weights_%s.txt' % i,weights,delimiter=',')

    X_cur = solver.X.out_g_x_2(solver.t_train).numpy()
    Y_cur = solver.Y.out_g_x_2(solver.t_train).numpy()
    Z_cur = solver.Z.out_g_x_2(solver.t_train).numpy()    
    if i == 0:
        predict = np.concatenate(list(map(lambda x:np.expand_dims(x,axis=-1), (solver.t_orig,X_cur,Y_cur,Z_cur))),axis=-1)
    else:
        dummy = np.concatenate(list(map(lambda x:np.expand_dims(x,axis=-1), (solver.t_orig,X_cur,Y_cur,Z_cur))),axis=-1)
        predict = np.concatenate((predict,dummy[1:,:]),axis=0)
    np.savetxt('predict.txt', predict, delimiter=',')

    solver_set.append(solver)
    init2 = [solver.X.out_g_x_2(np.array(gamma)).numpy(), solver.Y.out_g_x_2(np.array(gamma)).numpy(), solver.Z.out_g_x_2(np.array(gamma)).numpy()]