import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.optimize
import sympy as sp

from drawnow import drawnow

def pred_analytic(x, n_int=1):
    f, g = sympy_integration(n_int)
    y_pred = list(map(lambda x_orig: float(f.subs('x',x_orig).evalf()), x))
    y_pred_int = list(map(lambda x_orig: float(g.subs('x',x_orig).evalf()), x))
    return y_pred, y_pred_int
    
def sympy_integration(n_int=1):
    x = sp.symbols('x')
    f = sp.sin(x) + sp.sin(2*x) + sp.cos(x)*0.5 + sp.cos(6*x)+ sp.cos(x**2) + sp.sqrt(x+5)
    g = f
    for i in range(n_int):
        g = sp.integrate(g,x)
        g -= g.subs('x', -5)
    return f, g

def sample_function(x_data):
    return np.sin(x_data) + np.sin(2*x_data) + np.cos(x_data)*0.5 + np.cos(6*x_data)+ np.cos(np.power(x_data,2)) + np.sqrt(x_data+5)

    
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
    def out_g_x_0(self, x):
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
    def out_g_x_1(self, x):
        if x.dtype == self.dtype:
            pass
        else:        
            x = tf.cast(x,self.dtype)            
        x_1 = self.x_i[1:]
        x_2 = self.x_i[:-1]                
        g_x = tf.cast(0., self.dtype)
        g_x += self.out_an(0, x_1, x_2)/4. * tf.math.square(x)
        for n in range(1,self.N_m+1):
            factor = self.L/(n*np.pi)
            factor = tf.constant(factor,self.dtype)            
            g_x += 0.5*(-factor*self.out_bn(n, x_1, x_2)) * ( (factor)*tf.math.sin(x/factor) - x ) 
            g_x += 0.5*tf.math.square(factor)*self.out_an(n, x_1, x_2)*(1.-tf.math.cos(x/factor))
        g_x += self.init1*x + self.init2
        return g_x
    def out_g_x_2(self, x):
        if x.dtype == self.dtype:
            pass
        else:        
            x = tf.cast(x,self.dtype)        
        x_1 = self.x_i[1:]
        x_2 = self.x_i[:-1]
        g_x = tf.cast(0., self.dtype)
        g_x += self.out_an(0, x_1, x_2)/12. * tf.math.pow(x,3)
        for n in range(1,self.N_m+1):
            factor = self.L/(n*np.pi)
            factor = tf.constant(factor,self.dtype)            
            g_x += 0.5*(factor*self.out_bn(n, x_1, x_2))*(0.5*tf.math.square(x) + tf.math.pow(factor,2)*(tf.math.cos(x/factor)-1.))
            g_x += 0.5*tf.math.square(factor)*self.out_an(n, x_1, x_2)*(x-factor*tf.math.sin(x/factor))
        g_x += 0.5*self.init1*tf.math.square(x) + self.init2*x + self.init3        
        return g_x
    def out_g_x_3(self, x):
        if x.dtype == self.dtype:
            pass
        else:        
            x = tf.cast(x,self.dtype)        
        x_1 = self.x_i[1:]
        x_2 = self.x_i[:-1]
        g_x = tf.cast(0., self.dtype)
        g_x += self.out_an(0, x_1, x_2)/48. * tf.math.pow(x,4)
        for n in range(1,self.N_m+1):
            factor = self.L/(n*np.pi)
            factor = tf.constant(factor,self.dtype)            
            g_x += 0.5*(factor*self.out_bn(n, x_1, x_2))*(tf.math.pow(x,3)/6.-tf.math.square(factor)*x+tf.math.pow(factor,3)*tf.math.sin(x/factor))
            g_x += 0.5*tf.math.square(factor)*self.out_an(n, x_1, x_2)*(0.5*tf.math.square(x)+tf.math.square(factor)*(tf.math.cos(x/factor)-1.))
        g_x += self.init1*tf.math.pow(x,3)/6. + self.init2*tf.math.square(x)/2. + self.init3*x + self.init4        
        return g_x        



class Reg_Solver():
    def __init__(self, x_train, y_train, x_test, y_test, N_p = 40, N_m = 40, L=1., dtype = 'float32'):
        self.x_orig = x_train
        self.lb = np.min(x_train)
        self.ub = np.max(x_train)
        self.L = L
        self.dtype = dtype
        
        self.x_train = (self.x_orig - np.min(self.x_orig))
        self.x_train = (self.x_train/np.max(self.x_train))*L
        
        self.x_test = x_test
        self.y_test = y_test
        
        self.y_train = y_train
        self.const = tf.Variable(tf.cast(0.,self.dtype))
        
        self.Y = ADA_F(init1=0, init2=0, init3=0, init4=0, N_p=N_p, N_m=N_m, L=L, dtype=dtype)
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.04)
        self.lbfgs_step = 0
    def predict_integration(self, x_test, n_int=1):
        fac = self.ub - self.lb
        x_test = x_test - self.lb
        x_test = (x_test/(self.ub-self.lb))*self.L
        self.Y.init1 = self.const
        if n_int == 1:
            return (fac**n_int)*self.Y.out_g_x_1(x_test)
        if n_int == 2:
            return (fac**n_int)*self.Y.out_g_x_2(x_test)
        if n_int == 3:
            return (fac**n_int)*self.Y.out_g_x_3(x_test)           
    def predict(self, x_test):
        x_test = x_test - self.lb
        x_test = (x_test/(self.ub-self.lb))*self.L
        return self.Y.out_g_x_0(x_test) + self.const
    def plot_iteration(self):
        y_pred = self.predict(self.x_test)
        plt.plot(self.x_orig, self.y_train, 'b.')
        plt.plot(self.x_test, y_pred, 'r-')
        plt.plot(self.x_test, self.y_test, 'k--')       
    def get_gradient(self):
        with tf.GradientTape() as tape:
            tape.watch(self.Y.W_i)
            tape.watch(self.const)
            
            y_pred  = self.Y.out_g_x_0(self.x_train)       
            loss = tf.reduce_mean(tf.square(y_pred + self.const -self.y_train))
        g = tape.gradient(loss, [self.Y.W_i, self.const])
        return g, loss
    def train(self, epochs):        
        for i in range(epochs):    
            g, loss = self.get_gradient()
            self.opt.apply_gradients(zip(g,[self.Y.W_i, self.const]))
            if i%50 == 0:
                print('Epochs: %s, Loss: %s' % (i, loss.numpy()))
                if True:
                    drawnow(self.plot_iteration)
            else:
                pass
    def callback(self, xr=None):
        if self.lbfgs_step % 50 == 0:
            if True:
                drawnow(self.plot_iteration)
        self.lbfgs_step+=1
    def ScipyOptimizer(self,method='L-BFGS-B', **kwargs):    
        def get_weight_tensor():
            weight_list = []
            shape_list = []
            
            for v in [self.Y.W_i, self.const]:
                shape_list.append(v.shape)
                weight_list.extend(v.numpy().flatten())
            weight_list = tf.convert_to_tensor(weight_list)
            
            return weight_list, shape_list    
        x0, shape_list = get_weight_tensor()
        def set_weight_tensor(weight_list):        
            idx=0
            for v in [self.Y.W_i, self.const]:
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