import numpy as np

class SGD:
    def __init__(self,lr=0.01): self.lr=lr
    def update(self,params,grads):
        for k in params: params[k]-=self.lr*grads[k]

class Adam:
    def __init__(self,lr=0.001,beta1=0.9,beta2=0.999):
        self.lr=lr; self.beta1=beta1; self.beta2=beta2
        self.iter=0; self.m={}; self.v={}
    def update(self,params,grads):
        if not self.m:
            for k in params:
                self.m[k]=np.zeros_like(params[k])
                self.v[k]=np.zeros_like(params[k])
        self.iter+=1
        lr_t=self.lr*(np.sqrt(1-self.beta2**self.iter)/(1-self.beta1**self.iter))
        for k in params:
            self.m[k]=self.beta1*self.m[k]+(1-self.beta1)*grads[k]
            self.v[k]=self.beta2*self.v[k]+(1-self.beta2)*(grads[k]**2)
            params[k]-=lr_t*self.m[k]/(np.sqrt(self.v[k])+1e-7)
