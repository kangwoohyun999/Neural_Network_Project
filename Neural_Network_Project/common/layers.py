import numpy as np
from common.functions import softmax,cross_entropy_error

class Affine:
    def __init__(self,W,b):
        self.W=W; self.b=b; self.x=None
    def forward(self,x):
        self.x=x; return x@self.W+self.b
    def backward(self,dout):
        dx=dout@self.W.T
        self.dW=self.x.T@dout
        self.db=dout.sum(axis=0)
        return dx

class Relu:
    def __init__(self): self.mask=None
    def forward(self,x):
        self.mask=(x<=0); out=x.copy(); out[self.mask]=0; return out
    def backward(self,dout):
        dout[self.mask]=0; return dout

class SoftmaxWithLoss:
    def __init__(self): self.y=None; self.t=None
    def forward(self,x,t):
        self.t=t; self.y=softmax(x)
        return cross_entropy_error(self.y,t)
    def backward(self,dout=1):
        N=self.t.shape[0]
        if self.t.size==self.y.size:
            return (self.y-self.t)/N
        dx=self.y.copy()
        dx[np.arange(N),self.t]-=1
        return dx/N

class BatchNorm:
    def __init__(self,gamma,beta,momentum=0.9):
        self.gamma=gamma; self.beta=beta; self.momentum=momentum
        self.running_mean=None; self.running_var=None
    def forward(self,x,train_flg=True):
        if self.running_mean is None:
            N,D=x.shape
            self.running_mean=np.zeros(D); self.running_var=np.zeros(D)
        if train_flg:
            mu=x.mean(axis=0); var=x.var(axis=0)
            std=np.sqrt(var+1e-7); xc=(x-mu)/std
            self.xc=xc; self.std=std
            self.running_mean=self.momentum*self.running_mean+(1-self.momentum)*mu
            self.running_var=self.momentum*self.running_var+(1-self.momentum)*var
            return self.gamma*xc+self.beta
        xc=(x-self.running_mean)/np.sqrt(self.running_var+1e-7)
        return self.gamma*xc+self.beta
    def backward(self,dout):
        N,D=dout.shape
        dxn=self.gamma*dout
        self.dgamma=np.sum(self.xc*dout,axis=0)
        self.dbeta=np.sum(dout,axis=0)
        dx=(1/N)*(1/self.std)*(N*dxn - dxn.sum(axis=0) - self.xc*(dxn*self.xc).sum(axis=0))
        return dx

class Dropout:
    def __init__(self,dropout_ratio=0.5):
        self.dropout_ratio=dropout_ratio
    def forward(self,x,train_flg=True):
        if train_flg:
            self.mask=np.random.rand(*x.shape)>self.dropout_ratio
            return x*self.mask
        return x*(1-self.dropout_ratio)
    def backward(self,dout):
        return dout*self.mask
