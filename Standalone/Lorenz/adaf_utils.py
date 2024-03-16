import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.optimize

from scipy.integrate import odeint
from drawnow import drawnow

def f(state, t, rho=28.0, sigma=10., beta=8.0/3.0, ):
    x, y, z = state
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives

def lorenz(state0 =[1.0,1.0,1.0], t_max = 20.0, rho=28.0, sigma=10., beta=8.0/3.0, t_step=0.01):
    state0 = [1.0, 1.0, 1.0]
    t = np.arange(0.0, t_max, 0.01)
    states = odeint(f, state0, t)
    return t, states
    

class ADA_F():
    def __init__(self, init1=None, init2=None, init3=None, init4=None, N_p = 20, N_m = 20, L = 1.0, gamma=1.0, dtype = 'float32'):
        self.dtype = dtype
        self.L = L        
        self.N_p = N_p
        self.N_m = N_m                
        if gamma == 1.0:
            self.x_i = np.linspace(0, self.L, N_p+1).astype(dtype)
        else:
            self.x_i = np.concatenate((np.linspace(0, gamma, N_p-1)[:-1],np.linspace(gamma,1,3)),axis=0).astype(dtype)
        self.W_i = np.random.uniform(-1.0, 1.0, self.N_p).astype(dtype)
        self.W_i = tf.Variable(self.W_i)                           
        self.init1 = init1
        self.init2 = init2
        self.init3 = init3
        self.init4 = init4
        return
    def out_bn(self, n, x_1, x_2):                
        sum_1 = -tf.math.cos(n*np.pi/self.L* x_1)
        sum_2 = tf.math.cos(n*np.pi/self.L* x_2)
        
        b_n = self.W_i *(sum_1 + sum_2)
        b_n = tf.reduce_sum(b_n)
        b_n = (2./ (n*np.pi))*b_n
        return b_n
    def out_an(self, n, x_1, x_2):
        if n == 0:
            a_n = tf.reduce_sum(self.W_i)
            a_n = a_n/self.N_p
        else:
            sum_1 = tf.math.sin(n*np.pi/self.L* x_1)
            sum_2 = -tf.math.sin(n*np.pi/self.L* x_2)
            a_n = self.W_i * (sum_1 + sum_2)
            a_n = tf.reduce_sum(a_n)
            a_n = (2./(n*np.pi)) * a_n
        return a_n        
    def out_g_x_1(self, x):
        if x.dtype == self.dtype:
            pass
        else:        
            x = tf.cast(x,self.dtype)            
        x_1 = self.x_i[1:]
        x_2 = self.x_i[:-1]      
        
        g_x = tf.cast(0., self.dtype)
        g_x += 0.5*self.out_an(0, x_1, x_2)*x
        for n in range(1,self.N_m+1):
            factor = self.L/(n*np.pi)
            factor = tf.constant(factor,self.dtype)            
            g_x += 0.5*(-factor*self.out_bn(n, x_1, x_2)) * ( tf.math.cos(x/factor) - 1 )                        
            g_x += 0.5*(factor*self.out_an(n,x_1, x_2)) * (tf.math.sin(x/factor))
        g_x += self.init1
        return g_x
    def out_g_x_2(self, x):
        if x.dtype == self.dtype:
            pass
        else:        
            x = tf.cast(x,self.dtype)            
        x_1 = self.x_i[1:]
        x_2 = self.x_i[:-1]                
        g_x_2 = tf.cast(0., self.dtype)
        g_x_2 += self.out_an(0, x_1, x_2)/4. * tf.math.square(x)
        for n in range(1,self.N_m+1):
            factor = self.L/(n*np.pi)
            factor = tf.constant(factor,self.dtype)            
            g_x_2 += 0.5*(-factor*self.out_bn(n, x_1, x_2)) * ( (factor)*tf.math.sin(x/factor) - x ) 
            g_x_2 += 0.5*tf.math.square(factor)*self.out_an(n, x_1, x_2)*(1.-tf.math.cos(x/factor))
        g_x_2 += self.init1*x + self.init2
        return g_x_2
    def out_g_x_3(self, x):
        if x.dtype == self.dtype:
            pass
        else:        
            x = tf.cast(x,self.dtype)        
        x_1 = self.x_i[1:]
        x_2 = self.x_i[:-1]
        g_x_3 = tf.cast(0., self.dtype)
        g_x_3 += self.out_an(0, x_1, x_2)/12. * tf.math.pow(x,3)
        for n in range(1,self.N_m+1):
            factor = self.L/(n*np.pi)
            factor = tf.constant(factor,self.dtype)            
            g_x_3 += 0.5*(factor*self.out_bn(n, x_1, x_2))*(0.5*tf.math.pow(x,2) + tf.math.pow(factor,2)*(tf.math.cos(x/factor)-1.))
            g_x_3 += 0.5*tf.math.square(factor)*self.out_an(n, x_1, x_2)*(x-factor*tf.math.sin(x/factor))
        g_x_3 += 0.5*self.init1*tf.math.square(x) + self.init2*x + self.init3        
        return g_x_3
    def out_g_x_4(self, x):
        if x.dtype == self.dtype:
            pass
        else:        
            x = tf.cast(x,self.dtype)        
        x_1 = self.x_i[1:]
        x_2 = self.x_i[:-1]
        g_x_4 = tf.cast(0., self.dtype)
        g_x_4 += self.out_an(0, x_1, x_2)/48. * tf.math.pow(x,4)
        for n in range(1,self.N_m+1):
            factor = self.L/(n*np.pi)
            factor = tf.constant(factor,self.dtype)            
            g_x_4 += 0.5*(factor*self.out_bn(n, x_1, x_2))*(tf.math.pow(x,3)/6.-tf.math.square(factor)*x+tf.math.pow(factor,3)*tf.math.sin(x/factor))
            g_x_4 += 0.5*tf.math.square(factor)*self.out_an(n, x_1, x_2)*(0.5*tf.math.square(x)+tf.math.square(factor)*(tf.math.cos(x/factor)-1.))
        g_x_4 += self.init1*tf.math.pow(x,3)/6. + self.init2*tf.math.square(x)/2. + self.init3*x + self.init4        
        return g_x_4        

class Lorenz_Solver:
    def __init__(self, t_train, rho = 28.0, sigma = 10.0, beta = 8.0/3.0, init2 = [1.0, 1.0, 1.0], L=1., gamma=.2, t_step = 0.2, dtype = 'float32', N_p= 20, N_m=20):
        self.rho = rho
        self.sigma = sigma
        self.beta = beta
        self.t_orig = t_train        
        self.t_train = (self.t_orig - np.min(self.t_orig))
        self.t_train = (self.t_train/np.max(self.t_train))*L*gamma
        self.t_train = self.t_train.astype(dtype)
        self.dtype = dtype
        
        self.t, self.states = lorenz()

        self.L = L
        self.gamma = gamma
        self.t_step = t_step

        self.lbfgs_step = 0        
        self.init2 = init2
        self.init1 = [self.sigma*(init2[1]-init2[0])*(self.t_step/self.gamma), (init2[0]*(self.rho-init2[2])-init2[1])*(self.t_step/self.gamma), (init2[0]*init2[1]-self.beta*init2[2])*(self.t_step/self.gamma)]
        
        self.X = ADA_F(init1 = self.init1[0], init2 = self.init2[0], N_p=N_p, N_m=N_m, L=L, gamma=gamma, dtype=dtype)
        self.Y = ADA_F(init1 = self.init1[1], init2 = self.init2[1], N_p=N_p, N_m=N_m, L=L, gamma=gamma, dtype=dtype)
        self.Z = ADA_F(init1 = self.init1[2], init2 = self.init2[2], N_p=N_p, N_m=N_m, L=L, gamma=gamma, dtype=dtype)               
        return
    def get_gradient(self):
        with tf.GradientTape() as tape:
            tape.watch(self.X.W_i)
            tape.watch(self.Y.W_i)
            tape.watch(self.Z.W_i)
            
            X_cur = self.X.out_g_x_2(self.t_train)
            Y_cur = self.Y.out_g_x_2(self.t_train)
            Z_cur = self.Z.out_g_x_2(self.t_train) 
            
            dX_dt_cur = self.X.out_g_x_1(self.t_train)
            dY_dt_cur = self.Y.out_g_x_1(self.t_train)
            dZ_dt_cur = self.Z.out_g_x_1(self.t_train)
                                    
            X_res = -dX_dt_cur*(self.gamma/self.t_step) + self.sigma * (Y_cur-X_cur)
            X_res = tf.reduce_mean(tf.square(X_res))
            
            Y_res = -dY_dt_cur*(self.gamma/self.t_step) + X_cur * (self.rho - Z_cur) - Y_cur
            Y_res = tf.reduce_mean(tf.square(Y_res))
            
            Z_res = -dZ_dt_cur*(self.gamma/self.t_step) + X_cur*Y_cur - self.beta*Z_cur
            Z_res = tf.reduce_mean(tf.square(Z_res))  
                        
            loss = X_res + Y_res + Z_res
        g = tape.gradient(loss, [self.X.W_i,  self.Y.W_i, self.Z.W_i])
        return g, loss       
    def train(self, epochs):        
        for i in range(epochs):    
            g, loss = self.get_gradient()
            self.opt.apply_gradients(zip(g,[self.X.W_i,  self.Y.W_i, self.Z.W_i]))
            if i%50 == 0:
                print('Epochs: %s, Loss: %s' % (i, loss.numpy()))                
                if True:
                    drawnow(self.plot_iteration)
            else:
                pass
    def plot_iteration(self):
        X_cur = self.X.out_g_x_2(self.t_train)
        Y_cur = self.Y.out_g_x_2(self.t_train)
        Z_cur = self.Z.out_g_x_2(self.t_train)

        plt.plot(self.t_orig, X_cur,'r-')
        plt.plot(self.t_orig, Y_cur,'b-')
        plt.plot(self.t_orig, Z_cur,'g-')    
    
        condit = (self.t<=np.max(self.t_orig)) & (self.t>=np.min(self.t_orig))
        plt.plot(self.t[condit], self.states[:,0][condit],'k--')
        plt.plot(self.t[condit], self.states[:,1][condit],'k--')
        plt.plot(self.t[condit], self.states[:,2][condit],'k--')
    def callback(self, xr=None):
        if self.lbfgs_step % 50 == 0:
            if True:
                drawnow(self.plot_iteration)
        self.lbfgs_step+=1
    def ScipyOptimizer(self,method='L-BFGS-B', **kwargs):    
        def get_weight_tensor():
            weight_list = []
            shape_list = []
            
            for v in [self.X.W_i,  self.Y.W_i, self.Z.W_i]:
                shape_list.append(v.shape)
                weight_list.extend(v.numpy().flatten())
            weight_list = tf.convert_to_tensor(weight_list)
            
            return weight_list, shape_list    
        x0, shape_list = get_weight_tensor()
        def set_weight_tensor(weight_list):        
            idx=0
            for v in [self.X.W_i,  self.Y.W_i, self.Z.W_i]:
                vs = v.shape
                
                if len(vs) == 2:
                    sw = vs[0]*vs[1]
                    new_val = tf.reshape(weight_list[idx:idx+sw], (vs[0],vs[1]))
                    idx += sw
                elif len(vs) == 1:
                    new_val = weight_list[idx:idx+vs[0]]
                    idx+=vs[0]
                elif len(vs) ==0:
                    new_val = weight_list[idx]
                    idx+=1
                elif len(vs) ==3:
                    sw = vs[0]*vs[1]*vs[2]
                    new_val = tf.reshape(weight_list[idx:idx+sw], (vs[0],vs[1],vs[2]))                    
                    idx += sw
                elif len(vs) == 4:
                    sw = vs[0]*vs[1]*vs[2]*vs[3]
                    new_val = tf.reshape(weight_list[idx:idx+sw], (vs[0],vs[1],vs[2],vs[3]))                    
                    idx += sw                    
                v.assign(tf.cast(new_val, self.dtype))   
        
        def get_loss_and_grad(w):
            set_weight_tensor(w)
            grad, loss = self.get_gradient()
            loss = loss.numpy().astype(np.float64)
            grad_flat=[]
            for g in grad:
                grad_flat.extend(g.numpy().flatten())
            
            grad_flat = np.array(grad_flat, dtype=np.float64)
            return loss, grad_flat

        return scipy.optimize.minimize(fun=get_loss_and_grad,
                                    x0 = x0,
                                    jac = True,
                                    callback = self.callback,
                                    method=method,
                                    **kwargs)        