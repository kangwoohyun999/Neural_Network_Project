import numpy as np
from collections import OrderedDict
from common.layers import Affine,Relu,SoftmaxWithLoss,BatchNorm,Dropout

class MultiLayerNet:
    def __init__(self,input_size,hidden_sizes,output_size,
                 weight_init_std='he',use_batchnorm=False,use_dropout=False,dropout_ratio=0.3):
        self.params={}; self.layers=OrderedDict()
        self.use_batchnorm=use_batchnorm; self.use_dropout=use_dropout
        layer_sizes=[input_size]+hidden_sizes+[output_size]
        L=len(layer_sizes)-1
        for i in range(1,L+1):
            in_size=layer_sizes[i-1]; out_size=layer_sizes[i]
            if weight_init_std=='he':
                self.params[f'W{i}']=np.random.randn(in_size,out_size)*np.sqrt(2/in_size)
            else:
                self.params[f'W{i}']=weight_init_std*np.random.randn(in_size,out_size)
            self.params[f'b{i}']=np.zeros(out_size)
            if use_batchnorm and i!=L:
                self.params[f'gamma{i}']=np.ones(out_size)
                self.params[f'beta{i}']=np.zeros(out_size)

        for i in range(1,L+1):
            self.layers[f'Affine{i}']=Affine(self.params[f'W{i}'],self.params[f'b{i}'])
            if i!=L:
                if use_batchnorm:
                    self.layers[f'BatchNorm{i}']=BatchNorm(self.params[f'gamma{i}'],self.params[f'beta{i}'])
                self.layers[f'Relu{i}']=Relu()
                if use_dropout:
                    self.layers[f'Dropout{i}']=Dropout(dropout_ratio)
        self.last_layer=SoftmaxWithLoss()

    def predict(self,x,train_flg=False):
        for layer in self.layers.values():
            if isinstance(layer,(BatchNorm,Dropout)):
                x=layer.forward(x,train_flg)
            else:
                x=layer.forward(x)
        return x

    def loss(self,x,t,train_flg=False):
        y=self.predict(x,train_flg)
        return self.last_layer.forward(y,t)

    def accuracy(self,x,t,batch_size=100):
        N=x.shape[0]; y=[] 
        for i in range(0,N,batch_size):
            yb=self.predict(x[i:i+batch_size],False)
            y+=list(np.argmax(yb,axis=1))
        y=np.array(y)
        if t.ndim!=1: t=t.argmax(axis=1)
        return np.mean(y==t)

    def gradient(self,x,t):
        self.loss(x,t,True)
        dout=self.last_layer.backward()
        for layer in reversed(list(self.layers.values())):
            dout=layer.backward(dout)
        grads={}
        affine_count=sum(k.startswith('W') for k in self.params)
        for i in range(1,affine_count+1):
            aff=self.layers[f'Affine{i}']
            grads[f'W{i}']=aff.dW; grads[f'b{i}']=aff.db
            if self.use_batchnorm and f'gamma{i}' in self.params:
                bn=self.layers.get(f'BatchNorm{i}')
                if bn:
                    grads[f'gamma{i}']=bn.dgamma
                    grads[f'beta{i}']=bn.dbeta
        return grads
