import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mlp
import os

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.kernel_ridge import KernelRidge

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import BayesianRidge, ARDRegression


from adaf_utils import *

L = 1.0
N_p = 40
N_m = 40
dtype = 'float32'

N_data = 100
N_test = 201

lb = -5
ub = -lb


if os.path.exists('./data.txt'):
    data = np.loadtxt('./data.txt',delimiter=',')
    x_orig = data[:,0]
    y_train = data[:,1]
else:  
    x_orig = np.linspace(lb,ub,N_data)
    y_train = sample_function(x_orig) + 0.2*np.random.normal(size=(N_data,))
    data = np.concatenate((x_orig.reshape(N_data,1),y_train.reshape(N_data,1)),axis=-1)
    np.savetxt('data.txt',data,delimiter=',')

x_test= np.linspace(lb,ub,N_test)
y_test = sample_function(x_test)

solver = Reg_Solver(x_orig, y_train, x_test, y_test, N_p = N_p, N_m = N_m, L=L, dtype=dtype)    

if os.path.exists('./weights_%s.txt' % N_p):
    weights = np.loadtxt('./weights_%s.txt' % N_p, delimiter=',').astype(dtype)
    W_i = weights[:-1]
    const_real = weights[-1]
    solver.Y.W_i = tf.Variable(W_i)
    solver.const = tf.Variable(const_real)
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
    weights = np.concatenate((solver.Y.W_i.numpy(),[solver.const]))        
    np.savetxt('weights_%s.txt' % N_p ,weights,delimiter=',')
                
pred = solver.predict(x_test)


x_scaler = MinMaxScaler()
x_scaler.fit(x_orig.reshape(-1,1))
y_scaler = MinMaxScaler()
y_scaler.fit(y_train.reshape(-1,1))

svr_rbf = SVR()
svr_rbf.fit(x_scaler.transform(x_orig.reshape(-1,1)), y_scaler.transform(y_train.reshape(-1,1)).reshape(-1))
y_svr_predict = svr_rbf.predict(x_scaler.transform(x_test.reshape(-1,1)))
y_svr_predict = y_scaler.inverse_transform(y_svr_predict.reshape(-1,1))


gaussian_process = GaussianProcessRegressor()
gaussian_process.fit(x_scaler.transform(x_orig.reshape(-1,1)), y_scaler.transform(y_train.reshape(-1,1)).reshape(-1))
gaussian_process.kernel_
y_gpr_predict = gaussian_process.predict(x_scaler.transform(x_test.reshape(-1,1)))
y_gpr_predict = y_scaler.inverse_transform(y_gpr_predict.reshape(-1,1))

brr_reg = BayesianRidge()
brr_reg.fit(x_scaler.transform(x_orig.reshape(-1,1)), y_scaler.transform(y_train.reshape(-1,1)).reshape(-1))
brr_reg_predict = brr_reg.predict(x_scaler.transform(x_test.reshape(-1,1)))
brr_reg_predict = y_scaler.inverse_transform(brr_reg_predict.reshape(-1,1))




plt.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(5,4))
plt.plot(x_test, y_svr_predict, linestyle='-', color='green')
plt.plot(x_test, y_gpr_predict, linestyle='-', color='grey')
plt.plot(x_test, brr_reg_predict, linestyle='-', color='yellow')

plt.plot(x_orig, y_train, marker='o',markersize=4,linestyle='', color='blue')
plt.plot(x_test, pred, 'r-')
plt.plot(x_test, y_test, 'k--')
plt.xlim([-5.,5.])
plt.xticks([-5,-2.5,0,2.5,5],fontsize=15)
plt.ylim([-2,7])
plt.yticks([-2,0,2,4,6],fontsize=15)
plt.xlabel('x', fontsize=20, fontstyle='italic')
plt.ylabel('y', fontsize=20, fontstyle='italic')
plt.tight_layout()
plt.savefig('ADAF.jpg', dpi=400)
plt.show()



pred_int1 = solver.predict_integration(x_test,1)
pred_int2 = solver.predict_integration(x_test,2)
pred_int3 = solver.predict_integration(x_test,3)

y_test_int1 = np.array(pred_analytic(x_test,1)[1])
y_test_int2 = np.array(pred_analytic(x_test,2)[1])
y_test_int3 = np.array(pred_analytic(x_test,3)[1])


pred_int1 = (pred_int1 - np.min(y_test_int1))/(np.max(y_test_int1)-np.min(y_test_int1))
y_test_int1 = (y_test_int1 - np.min(y_test_int1))/(np.max(y_test_int1)-np.min(y_test_int1))
pred_int2 = (pred_int2 - np.min(y_test_int2))/(np.max(y_test_int2)-np.min(y_test_int2))
y_test_int2 = (y_test_int2 - np.min(y_test_int2))/(np.max(y_test_int2)-np.min(y_test_int2))
pred_int3 = (pred_int3 - np.min(y_test_int3))/(np.max(y_test_int3)-np.min(y_test_int3))
y_test_int3 = (y_test_int3 - np.min(y_test_int3))/(np.max(y_test_int3)-np.min(y_test_int3))

plt.figure(figsize=(5,4))
plt.plot(x_test, pred_int1, 'r-')
plt.plot(x_test, y_test_int1, 'k--')
plt.plot(x_test, pred_int2, 'b-')
plt.plot(x_test, y_test_int2, 'k--')
plt.plot(x_test, pred_int3, 'g-')
plt.plot(x_test, y_test_int3, 'k--')
plt.xlim([-5.,5.])
plt.xticks([-5,-2.5,0,2.5,5],fontsize=15)
plt.ylim([-0.05,1.05])
plt.yticks([0,0.25,0.5,0.75,1],fontsize=15)
plt.xlabel('x', fontsize=20, fontstyle='italic')
plt.ylabel('Scaled anti-derivatives', fontsize=17)
plt.tight_layout()
plt.savefig('ADAF_antiderivatives.jpg', dpi=400)
#plt.show()



