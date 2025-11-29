import numpy as np
def softmax(x):
    if x.ndim==2:
        x=x-x.max(axis=1,keepdims=True)
        y=np.exp(x)
        return y/np.sum(y,axis=1,keepdims=True)
    x=x-np.max(x)
    ex=np.exp(x)
    return ex/np.sum(ex)
def cross_entropy_error(y,t):
    if y.ndim==1:
        y=y.reshape(1,-1); t=t.reshape(1,-1)
    if t.size==y.size:
        t=t.argmax(axis=1)
    N=y.shape[0]
    return -np.sum(np.log(y[np.arange(N),t]+1e-7))/N
