import numpy as np
import tensorflow as tf
import scipy.optimize
from drawnow import drawnow
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import numpy as np
import sympy
import os

Re = 40
nu = 1 / Re
l = 1 / (2 * nu) - np.sqrt(1 / (4 * nu ** 2) + 4 * np.pi ** 2)

def solution(x):
    return (u_func(x), v_func(x), p_func(x))

def u_func(x):
    return 1 - tf.math.exp(l * x[:, 0:1]) * tf.math.cos(2 * np.pi * x[:, 1:2])

def v_func(x):
    return l / (2 * np.pi) * tf.math.exp(l * x[:, 0:1]) * tf.math.sin(2 * np.pi * x[:, 1:2])

def p_func(x):
    return 1 / 2 * (1 - tf.math.exp(2 * l * x[:, 0:1]))


def get_XB(lb, ub, N_b, DTYPE='float32'):    
    x_b = tf.random.uniform((N_b,1), lb[0], ub[0], dtype=DTYPE)
    y_b = tf.random.uniform((N_b,1), lb[1], ub[1], dtype=DTYPE)
    
    x_0 = tf.ones((N_b,1),dtype=DTYPE)*lb[0]
    x_L = tf.ones((N_b,1),dtype=DTYPE)*ub[0]
    y_0 = tf.ones((N_b,1),dtype=DTYPE)*lb[1]
    y_L = tf.ones((N_b,1),dtype=DTYPE)*ub[1]

    X_b_0 = tf.concat([x_0, y_b], axis=1)
    X_b_L = tf.concat([x_L, y_b], axis=1)
    Y_b_0 = tf.concat([x_b, y_0], axis=1)
    Y_b_L = tf.concat([x_b, y_L], axis=1)    
    return X_b_0, X_b_L, Y_b_0, Y_b_L

def get_Xr(lb, ub, N_r, DTYPE='float32'):    
    x_r = tf.random.uniform((N_r,1), lb[0], ub[0], dtype=DTYPE)
    y_r = tf.random.uniform((N_r,1), lb[1], ub[1], dtype=DTYPE)
    XY_r = tf.concat([x_r, y_r], axis=1)
    return XY_r

class Custom_Normal(tf.keras.layers.Layer):
    def __init__(self):
        super(Custom_Normal, self).__init__() 
    def call(self, inputs):  
        max_ = tf.math.reduce_max(inputs)
        min_ = tf.math.reduce_min(inputs)
        return (inputs - min_)/(max_ - min_)
        
        
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
        self.W_i = self.add_weight('W_i', shape=(self.N_p,), initializer='random_normal', regularizer = self.kernel_regularizer, trainable=True, dtype=self.DTYPE)               
        self.w = self.add_weight('w', shape=(), initializer='random_normal', regularizer = self.kernel_regularizer, trainable=True, dtype=self.DTYPE)        

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
    def out_g_x_1(self, x):           
        x_1 = self.x_i[1:]
        x_2 = self.x_i[:-1]                
                
        g_x = tf.cast(0., self.DTYPE)
        g_x += self.out_an(0, x_1, x_2, self.W_i)/2. * tf.math.square(x)
        for n in range(1,self.N_m+1):
            factor = self.L/(n*np.pi)
            factor = tf.constant(factor,self.DTYPE)            
            g_x += tf.math.square(factor)*self.out_an(n, x_1, x_2, self.W_i)*(1.-tf.math.cos(x/factor))
        return g_x
    def call(self, inputs):
        return self.w*self.out_g_x_1(inputs)


                  
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
        if key == 'ADAF':
            self.model = self.init_model_ADAF()      
        elif key == 'R':
            self.model = self.init_model_VAN()  
        else:
            pass
    def init_model_VAN(self):
        X_in =tf.keras.Input(2)
        hiddens = tf.keras.layers.Lambda(lambda x: 2.0*(x-self.lb)/(self.ub-self.lb) -1.0)(X_in)        
        for _ in range(self.num_hidden_layers):
            hiddens = tf.keras.layers.Dense(self.num_neurons_per_layer,
                activation=tf.keras.activations.get('tanh'),
                kernel_initializer='glorot_normal')(hiddens)
        prediction = tf.keras.layers.Dense(3)(hiddens)
        model = tf.keras.Model(X_in, prediction)
        return model
    def init_model_ADAF(self):
        X_in =tf.keras.Input(2)
        hiddens = tf.keras.layers.Lambda(lambda x: 2.0*(x-self.lb)/(self.ub-self.lb) -1.0)(X_in)               
        hiddens = tf.keras.layers.Dense(self.num_neurons_per_layer,
                kernel_initializer='glorot_normal', 
                activation='tanh',
                )(hiddens)
        for _ in range(self.num_hidden_layers-2):
            hiddens = tf.keras.layers.Dense(self.num_neurons_per_layer,
                    kernel_initializer='glorot_normal', 
                    activation='tanh',
                    )(hiddens)
        hiddens = tf.keras.layers.Dense(self.num_neurons_per_layer,
                kernel_initializer='glorot_normal', 
                )(hiddens)
        hiddens = ADAF(3,3)(hiddens)
        hiddens = tf.math.tanh(hiddens)
        prediction = tf.keras.layers.Dense(3)(hiddens)
        model = tf.keras.Model(X_in, prediction)
        return model        

class Solver_PINN():
    def __init__(self, pinn, properties, N_b=150, N_r=2500, show=False, DTYPE='float32'):
        self.ref_pinn = None
        self.loss_element = None                
                
        self.lbfgs_step = 0        
        self.loss_history = []
        self.cur_pinn = pinn
        self.properties = properties
        self.show = show
        self.DTYPE = DTYPE
        self.N_b = N_b
        self.N_r = N_r                
        
        self.X_b_0, self.X_b_L, self.Y_b_0, self.Y_b_L, self.XY_r = self.data_sampling()        
        
        self.lr = None
        self.optim = None
        self.build_optimizer()                
        self.call_examset()
                
        self.path = './results/%s_%s/%s/' % (self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer, self.cur_pinn.key)
        self.path2 = './results/'
        os.makedirs(self.path, exist_ok=True)

        self.accuracy_history =[]
    def data_sampling(self):    
        X_b_0, X_b_L, Y_b_0, Y_b_L = get_XB(self.cur_pinn.lb, self.cur_pinn.ub, self.N_b)
        XY_r = get_Xr(self.cur_pinn.lb, self.cur_pinn.ub, self.N_r)
        return X_b_0, X_b_L, Y_b_0, Y_b_L, XY_r
    def call_examset(self):
        x = np.linspace(self.cur_pinn.lb[0],self.cur_pinn.ub[0],10)
        y = np.linspace(self.cur_pinn.lb[1],self.cur_pinn.ub[1],10)
        xx, yy = np.meshgrid(x,y)
        self.XY_test = np.stack((xx.flatten(), yy.flatten()), axis=1)    
    def save_results(self, trial, times, num_hidden_layers=2, num_neurons_per_layer=10):
        self.accuracy_update()
        self.loss_history.append(self.loss)
        self.cur_pinn.model.save_weights('./checkpoints/%s_%s/%s/ckpt_lbfgs_%s' % (self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer, self.cur_pinn.key, trial))        
        np.savetxt('./results/loss_hist_%s_%s_%s_%s.txt' % (self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer, self.cur_pinn.key, trial), np.array(self.loss_history), delimiter=',')
        np.savetxt('./results/acc_hist_%s_%s_%s_%s.txt' % (self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer, self.cur_pinn.key, trial), np.array(self.accuracy_history), delimiter=',') 
        np.savetxt('./results/cal_time_%s_%s_%s_%s.txt' % (self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer, self.cur_pinn.key, trial), np.array(times), delimiter=',') 
    def plot_iteration(self):
        color = cm.Reds(np.linspace(0.1,1,10))
        pred = self.cur_pinn.model.predict(self.XY_test)
        u_pred = pred[:,0]
        v_pred = pred[:,1]
        p_pred = pred[:,2]
        u,v,p = solution(self.XY_test) 
        for i in range(10):
            plt.plot(self.XY_test[:10,0], u_pred[i*10:(i+1)*10],c=color[len(color)-1-i])
            plt.plot(self.XY_test[:10,0], u[i*10:(i+1)*10],c=color[len(color)-1-i], marker='o', markersize=2, linestyle='-')                  
    def build_optimizer(self):
        del self.lr
        del self.optim
        self.lr = 1e-2
        self.optim = tf.keras.optimizers.Adam(learning_rate=self.lr) 
    
    def get_B(self, X):
        u, v, p = solution(X)
        pred = self.cur_pinn.model(X)
        u_pred, v_pred, p_pred = tf.split(pred, 3, axis=1)
        return tf.reduce_mean(tf.square(u-u_pred))+tf.reduce_mean(tf.square(v-v_pred))+tf.reduce_mean(tf.square(p-p_pred))
    def get_r(self, X_r):
        with tf.GradientTape(persistent=True) as tape:
            x, y = tf.split(X_r, 2, axis=1)
            tape.watch(x)
            tape.watch(y)
            
            pred = self.cur_pinn.model(tf.stack([x[:,0], y[:,0]], axis=1))            
            u, v, p = tf.split(pred,3,axis=1)
            u_x, v_x, p_x = map(lambda x:tf.squeeze(x,axis=-1),tf.split(tape.batch_jacobian(pred,x),3,axis=1))
            u_y, v_y, p_y = map(lambda x:tf.squeeze(x,axis=-1),tf.split(tape.batch_jacobian(pred,y),3,axis=1))
        u_xx = tape.gradient(u_x,x)
        u_yy = tape.gradient(u_y,y)
        v_xx = tape.gradient(v_x,x)
        v_yy = tape.gradient(v_y,y)        
        del tape
        x_mom = tf.math.multiply(u, u_x)+ tf.math.multiply(v, u_y) + p_x - (1/Re)*(u_xx+u_yy)
        y_mom = tf.math.multiply(u, v_x)+ tf.math.multiply(v, v_y) + p_y - (1/Re)*(v_xx+v_yy)
        contin = u_x + v_y
        return x_mom, y_mom, contin
    def compute_loss(self):        
        #PDE Residual
        x_mom, y_mom, contin = self.get_r(self.XY_r)
        Phi_r = tf.reduce_mean(tf.square(x_mom))+tf.reduce_mean(tf.square(y_mom))+tf.reduce_mean(tf.square(contin))
        
        BX0 = self.get_B(self.X_b_0)
        BXL = self.get_B(self.X_b_L)
        BY0 = self.get_B(self.Y_b_0)
        BYL = self.get_B(self.Y_b_L)        
        #Total Loss    
        total_loss = Phi_r + BX0 + BXL + BY0 + BYL
        #self.loss_element = (lambda x: np.array(x))((total_loss, Phi_r, BX0, BXL, BY0, BYL))
        #self.loss_history.append(self.loss_element)
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
            if num_step%50 == 0:
                self.accuracy_update()
                print('Iter {:05d}: loss = {:10.8e}'.format(num_step, self.loss))                
                if self.show:
                    drawnow(self.plot_iteration)
    def accuracy_update(self):
        prediction = self.cur_pinn.model.predict(self.XY_test)
        u_pred, v_pred, p_pred = prediction[:,0], prediction[:,1], prediction[:,2]
        u, v, p = list(map(lambda x:x.numpy().reshape(-1), solution(self.XY_test)))
        l1_absolute_u = np.mean(np.abs(u_pred-u))
        l2_relative_u = np.linalg.norm(u_pred-u,2)/np.linalg.norm(u,2)
        l1_absolute_v = np.mean(np.abs(v_pred-v))
        l2_relative_v = np.linalg.norm(v_pred-v,2)/np.linalg.norm(v,2)        
        l1_absolute_p = np.mean(np.abs(p_pred-p))
        l2_relative_p = np.linalg.norm(p_pred-p,2)/np.linalg.norm(p,2)                  
        #print('     l1_absolute_error_u:   ', l1_absolute_u)   
        #print('     l1_absolute_error_v:   ', l1_absolute_v)   
        #print('     l1_absolute_error_p:   ', l1_absolute_p)   
        #print('     l2_relative_error_u:   ', l2_relative_u)
        #print('     l2_relative_error_v:   ', l2_relative_v)
        #print('     l2_relative_error_p:   ', l2_relative_p)        
        self.accuracy_element = np.array([l1_absolute_u, l1_absolute_v, l1_absolute_p, l2_relative_u, l2_relative_v, l2_relative_p])
        self.accuracy_history.append(self.accuracy_element)    
    def callback(self, xr=None):       
        self.loss_history.append(self.loss)
        if self.lbfgs_step % 50 == 0:
            self.accuracy_update()
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