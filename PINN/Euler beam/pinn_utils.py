import numpy as np
import tensorflow as tf
import scipy.optimize
from drawnow import drawnow
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import numpy as np
import sympy
import os

def solution(x):
    return -np.power(x,4)/24. + np.power(x,3)/6. - np.power(x,2)/4.

def get_Xr(lb, ub, N_r, DTYPE='float32'):    
    X_r = tf.random.uniform((N_r,1), lb[0], ub[0], dtype=DTYPE)
    return X_r


class ADAF(tf.keras.layers.Layer):
    def __init__(self,  N_p = 5, N_m = 5, L=1.,  DTYPE='float32', kernel_regularizer=None):
        super(ADAF, self).__init__()        
        self.N_p = N_p
        self.N_m = N_m
        self.L = L
        self.x_i = tf.cast(tf.linspace(0., L, N_p+1),dtype=DTYPE)
        self.DTYPE = DTYPE
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    def build(self, input_shape):        
        self.init1 = self.add_weight('init1', shape=(), initializer='random_normal', regularizer = self.kernel_regularizer, trainable=True, dtype=self.DTYPE)
        self.init2 = self.add_weight('init2', shape=(), initializer='random_normal', regularizer = self.kernel_regularizer, trainable=True, dtype=self.DTYPE)        
        self.init3 = self.add_weight('init3', shape=(), initializer='random_normal', regularizer = self.kernel_regularizer, trainable=True, dtype=self.DTYPE)
        self.init4 = self.add_weight('init4', shape=(), initializer='random_normal', regularizer = self.kernel_regularizer, trainable=True, dtype=self.DTYPE)                                       
        self.W_i = self.add_weight('W_i', shape=(self.N_p,), initializer='random_normal', regularizer = self.kernel_regularizer, trainable=True, dtype=self.DTYPE)               
    def out_an(self, n, x_1, x_2, W_i):
        if n == 0:
            a_n = tf.reduce_sum(W_i)
            a_n = a_n/self.N_p
        else:
            sum_1 = tf.math.sin(n*np.pi/self.L* x_1)
            sum_2 = -tf.math.sin(n*np.pi/self.L* x_2)
            a_n = W_i * (sum_1 + sum_2)
            a_n = tf.reduce_sum(a_n)
            a_n = (2./(n*np.pi)) * a_n
        return a_n 
    def out_bn(self, n, x_1, x_2, W_i):                
        sum_1 = -tf.math.cos(n*np.pi/self.L* x_1)
        sum_2 = tf.math.cos(n*np.pi/self.L* x_2)        
        b_n = W_i *(sum_1 + sum_2)
        b_n = tf.reduce_sum(b_n)
        b_n = (2./ (n*np.pi))*b_n
        return b_n        
    def out_g_x_3(self, x):        
        x_1 = self.x_i[1:]
        x_2 = self.x_i[:-1]
        g_x = tf.cast(0., self.DTYPE)
        g_x += self.out_an(0, x_1, x_2, self.W_i)/24. * tf.math.pow(x,4)
        for n in range(1,self.N_m+1):
            factor = self.L/(n*np.pi)
            factor = tf.constant(factor,self.DTYPE)            
            g_x += tf.math.square(factor)*self.out_an(n, x_1, x_2, self.W_i)*(0.5*tf.math.square(x)+tf.math.square(factor)*(tf.math.cos(x/factor)-1.))
        g_x += self.init1*tf.math.pow(x,3)/6. + self.init2*tf.math.square(x)/2. + self.init3*x + self.init4        
        return g_x
    def call(self, inputs):
        return self.out_g_x_3(inputs)

                  
class Build_PINN():
    def __init__(self, lb, ub, properties, 
        num_hidden_layers=2, 
        num_neurons_per_layer=10, 
        key = 'R'):        
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_per_layer = num_neurons_per_layer
        self.lb = lb
        self.ub = ub
        self.key = key
        self.properties = properties
        if key == 'R':
            self.model = self.init_model_VAN()  
        elif key == 'ADAF':
            self.model = self.init_model_ADAF()         
        else:
            pass
    def init_model_VAN(self):
        X_in =tf.keras.Input(1)
        hiddens = tf.keras.layers.Lambda(lambda x: 2.0*(x-self.lb)/(self.ub-self.lb) -1.0)(X_in)        
        for _ in range(self.num_hidden_layers):
            hiddens = tf.keras.layers.Dense(self.num_neurons_per_layer,
                activation=tf.keras.activations.get('tanh'),
                kernel_initializer='glorot_normal')(hiddens)
        prediction = tf.keras.layers.Dense(1)(hiddens)
        model = tf.keras.Model(X_in, prediction)
        return model
    def init_model_ADAF(self):
        X_in =tf.keras.Input(1)
        hiddens = tf.keras.layers.Lambda(lambda x: 2.0*(x-self.lb)/(self.ub-self.lb) -1.0)(X_in)               
        for _ in range(self.num_hidden_layers-1):
            hiddens = tf.keras.layers.Dense(self.num_neurons_per_layer,
                    kernel_initializer='glorot_normal', 
                    #activation='tanh',
                    )(hiddens)                            
        hiddens = tf.keras.layers.Dense(self.num_neurons_per_layer,
                kernel_initializer='glorot_normal', 
                )(hiddens)
        hiddens = ADAF(N_p = 3, N_m = 3)(hiddens)
        prediction = tf.keras.layers.Dense(1, kernel_initializer='glorot_normal')(hiddens)
        model = tf.keras.Model(X_in, prediction)
        return model         

class Solver_PINN():
    def __init__(self, pinn, properties, N_r=100, show=False, DTYPE='float32'):
                                
        self.lbfgs_step = 0        
        
        self.cur_pinn = pinn
        self.properties = properties
        self.show = show
        self.DTYPE = DTYPE

        self.N_r = N_r
        self.X_r = get_Xr(self.cur_pinn.lb, self.cur_pinn.ub, self.N_r)       
        self.lr = None
        self.optim = None
        self.build_optimizer()        
        self.x_exam = np.linspace(0,self.properties['L'],100)
        self.path = './results/%s_%s/%s/' % (self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer, self.cur_pinn.key)
        self.path2 = './results/'
        os.makedirs(self.path, exist_ok=True)
        
        self.loss_history = []
        self.accuracy_history =[]            
    def save_results(self, trial, times, num_hidden_layers=2, num_neurons_per_layer=10):
        #self.accuracy_update()
        #self.loss_history.append(self.loss)
        self.cur_pinn.model.save_weights('./checkpoints/%s_%s/%s/ckpt_lbfgs_%s' % (self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer, self.cur_pinn.key, trial))        
        np.savetxt('./results/loss_hist_%s_%s_%s_%s.txt' % (self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer, self.cur_pinn.key, trial), np.array(self.loss_history), delimiter=',')
        np.savetxt('./results/acc_hist_%s_%s_%s_%s.txt' % (self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer, self.cur_pinn.key, trial), np.array(self.accuracy_history), delimiter=',') 
        np.savetxt('./results/cal_time_%s_%s_%s_%s.txt' % (self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer, self.cur_pinn.key, trial), np.array(times), delimiter=',') 
    def plot_iteration(self):
        plt.plot(self.x_exam, solution(self.x_exam), 'k--')
        plt.plot(self.x_exam, self.cur_pinn.model.predict(self.x_exam), 'r-')                    
    def build_optimizer(self):
        del self.lr
        del self.optim
        self.lr = 1e-2
        self.optim = tf.keras.optimizers.Adam(learning_rate=self.lr)
    
    def get_X0(self):
        with tf.GradientTape() as tape:
            x = tf.reshape(self.cur_pinn.lb[0],(-1,1))
            tape.watch(x)
            Y = self.cur_pinn.model(x)
        Y_x = tape.gradient(Y,x)
        del tape
        return Y, Y_x
    def get_XL(self):
        with tf.GradientTape(persistent=True) as tape:
            x = tf.reshape(self.cur_pinn.ub[0],(-1,1))
            tape.watch(x)
            Y = self.cur_pinn.model(x)
            Y_x = tape.gradient(Y,x)
            Y_xx = tape.gradient(Y_x,x)
        Y_xxx = tape.gradient(Y_xx,x)
        del tape
        return Y_xx, Y_xxx        
    def fun_r(self, Y_xxxx):
        return Y_xxxx + 1.
    def get_r(self, X_r):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X_r)
            Y = self.cur_pinn.model(X_r)
            Y_x = tape.gradient(Y,X_r)
            Y_xx = tape.gradient(Y_x,X_r)
            Y_xxx = tape.gradient(Y_xx,X_r)        
        Y_xxxx = tape.gradient(Y_xxx,X_r)
        del tape
        return self.fun_r(Y_xxxx)
    def compute_loss(self):        
        #PDE Residual
        r = self.get_r(self.X_r)
        Phi_r = tf.reduce_mean(tf.square(r))
        #B0
        Y, Y_x =self.get_X0()
        B0 = tf.reduce_mean(tf.square(Y)) + tf.reduce_mean(tf.square(Y_x))
        #BL
        Y_xx, Y_xxx =self.get_XL()
        BL = tf.reduce_mean(tf.square(Y_xx)) + tf.reduce_mean(tf.square(Y_xxx))         
        total_loss = Phi_r + B0 + BL 
        return total_loss
    @tf.function    
    def get_grad(self):
        with tf.GradientTape() as tape:
            tape.watch(self.cur_pinn.model.trainable_weights)
            total_loss = self.compute_loss()
        g = tape.gradient(total_loss, self.cur_pinn.model.trainable_weights)
        del tape
        return g, total_loss
    def train_step(self):
        grad_theta, loss = self.get_grad()
        self.loss = loss
        self.loss_history.append(self.loss)
        self.optim.apply_gradients(zip(grad_theta, self.cur_pinn.model.trainable_weights))
        return 
    def train_adam(self, N=5000):
        for num_step in range(N):
            self.train_step()
            self.accuracy_update()
            if num_step%50 == 0:
                print('Iter {:05d}: loss = {:10.8e}'.format(num_step, self.loss))                
                if self.show:
                    drawnow(self.plot_iteration)
    def accuracy_update(self):
        prediction = self.cur_pinn.model.predict(self.x_exam).reshape(-1)
        exact = solution(self.x_exam).reshape(-1)
        l1_absolute = np.mean(np.abs(prediction-exact))
        l2_relative = np.linalg.norm(prediction-exact,2)/np.linalg.norm(exact,2)
        #print('     l1_absolute_error:   ', l1_absolute)   
        #print('     l2_relative_error:   ', l2_relative)
        self.accuracy_element = np.array([l1_absolute, l2_relative])
        self.accuracy_history.append(self.accuracy_element)    
    def callback(self, xr=None):
        self.accuracy_update()
        self.loss_history.append(self.loss)
        if self.lbfgs_step % 50 == 0:
            if self.show:
                drawnow(self.plot_iteration)  
        self.lbfgs_step+=1
            
    def ScipyOptimizer(self, method='L-BFGS-B', **kwargs):    
        def get_weight_tensor():
            weight_list = []
            shape_list = []
            
            for v in self.cur_pinn.model.variables:
                shape_list.append(v.shape)
                weight_list.extend(v.numpy().flatten())
            weight_list = tf.convert_to_tensor(weight_list)
            
            return weight_list, shape_list    
        x0, shape_list = get_weight_tensor()
        def set_weight_tensor(weight_list):        
            idx=0
            for v in self.cur_pinn.model.variables:
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
                v.assign(tf.cast(new_val, self.DTYPE))   
        
        def get_loss_and_grad(w):
            set_weight_tensor(w)
            grad, loss = self.get_grad()
            loss = loss.numpy().astype(np.float64)
            grad_flat=[]
            for g in grad:
                grad_flat.extend(g.numpy().flatten())
            
            grad_flat = np.array(grad_flat, dtype=np.float64)
            self.loss = loss
            return loss, grad_flat

        return scipy.optimize.minimize(fun=get_loss_and_grad,
                                    x0 = x0,
                                    jac = True,
                                    callback=self.callback,
                                    method=method,
                                    **kwargs)
                                            