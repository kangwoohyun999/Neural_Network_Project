import pickle
import numpy as np
def load_data_from_pkl(path):
    with open(path,'rb') as f:
        data=pickle.load(f)
    if isinstance(data,tuple) and len(data)==4:
        return data
    if isinstance(data,tuple) and len(data)==2:
        x_test,t_test=data
        return None,None,x_test,t_test
    raise ValueError("Unknown pickle format")