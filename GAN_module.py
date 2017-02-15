import os
import numpy as np
from numpy.random import RandomState
from matplotlib import pyplot as plt
from matplotlib.pyplot import *
from time import time
import updates
import theano
import theano.tensor as T
from scipy.stats import gaussian_kde
from scipy.misc import imsave, imread


###################################### PT modification ######################################
# from theano_utils import floatX, sharedX

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)

#----------------------------------------------- Define activation functions---------------------------------------------------
class LeakyRectify(object):

    def __init__(self, leak=0.2):
        self.leak = leak

    def __call__(self, x):
        f1 = 0.5 * (1 + self.leak)
        f2 = 0.5 * (1 - self.leak)
        return f1 * x + f2 * abs(x)

class Rectify(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return (x + abs(x)) / 2.0

class Tanh(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return T.tanh(x)

class Sigmoid(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return T.nnet.sigmoid(x)

# ---------------------------------------- Define a Normal class that creates random gaussian variables ------------------------------------------
class Normal(object):
    def __init__(self, loc=0., scale=0.05):
        self.scale = scale
        self.loc = loc

    def __call__(self, shape, name=None):
        seed = 42
        return sharedX(RandomState(seed).normal(loc=self.loc, scale=self.scale, size=shape), name=name)

# ------------------------------------------------------ Define our optimizer (Adam) -------------------------------------------------------------

def clip_norm(g, c, n):
    if c > 0:
        g = T.switch(T.ge(n, c), g*c/n, g)
    return g

def clip_norms(gs, c):
    norm = T.sqrt(sum([T.sum(g**2) for g in gs]))
    return [clip_norm(g, c, norm) for g in gs]

class Regularizer(object):

    def __init__(self, l1=0., l2=0., maxnorm=0., l2norm=False, frobnorm=False):
        self.__dict__.update(locals())

    def max_norm(self, p, maxnorm):
        if maxnorm > 0:
            norms = T.sqrt(T.sum(T.sqr(p), axis=0))
            desired = T.clip(norms, 0, maxnorm)
            p = p * (desired/ (1e-7 + norms))
        return p

    def l2_norm(self, p, axis=0):
        return p/l2norm(p, axis=axis)

    def frob_norm(self, p, nrows):
        return (p/T.sqrt(T.sum(T.sqr(p))))*T.sqrt(nrows)

    def gradient_regularize(self, p, g):
        g += p * self.l2
        g += T.sgn(p) * self.l1
        return g

    def weight_regularize(self, p):
        p = self.max_norm(p, self.maxnorm)
        if self.l2norm:
            p = self.l2_norm(p, self.l2norm)
        if self.frobnorm > 0:
            p = self.frob_norm(p, self.frobnorm)
        return p

class Update(object):

    def __init__(self, regularizer=Regularizer(), clipnorm=0.):
        self.__dict__.update(locals())

    def __call__(self, params, grads):
        raise NotImplementedError

class Adam(Update):

    def __init__(self, lr=0.001, b1=0.9, b2=0.999, e=1e-8, l=1-1e-8, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())  

    def __call__(self, params, cost, consider_constant=None):
        updates = []
        # if self.clipnorm > 0:
            # print 'clipping grads', self.clipnorm
            # grads = T.grad(theano.gradient.grad_clip(cost, 0, self.clipnorm), params)
        grads = T.grad(cost, params, consider_constant=consider_constant)
        grads = clip_norms(grads, self.clipnorm)  
        t = theano.shared(floatX(1.))
        b1_t = self.b1*self.l**(t-1)
     
        for p, g in zip(params, grads):
            g = self.regularizer.gradient_regularize(p, g)
            m = theano.shared(p.get_value() * 0.)
            v = theano.shared(p.get_value() * 0.)
     
            m_t = b1_t*m + (1 - b1_t)*g
            v_t = self.b2*v + (1 - self.b2)*g**2
            m_c = m_t / (1-self.b1**t)
            v_c = v_t / (1-self.b2**t)
            p_t = p - (self.lr * m_c) / (T.sqrt(v_c) + self.e)
            p_t = self.regularizer.weight_regularize(p_t)
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t) )
        updates.append((t, t + 1.))
        return updates

# ----------------------------------------------------- Visualize Gaussian Curves ----------------------------------------------------------
def gaussian_likelihood(X, u=0., s=1.):
    return (1./(s*np.sqrt(2*np.pi)))*np.exp(-(((X - u)**2)/(2*s**2)))

