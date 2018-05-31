import torch
import numpy as np

def saveParams( path, model ):
    i = 0
    params = np.empty(0)
    Dict = model.state_dict()
    for name in Dict:
        if "classifier" in name:
            continue
        param = Dict[name].numpy()
        param = param.reshape(param.size)
        params = np.concatenate((params, param));
    params.tofile(path+"/weights.dat")