import torch
import numpy as np
import os

def saveParams( path, model, fName="weights.dat", skipClassifier=False ):
    if not os.path.exists(path):
        os.makedirs(path)
    i = 0
    params = np.empty(0)
    Dict = model.state_dict()
    for name in Dict:
        if "classifier" in name and skipClassifier:
            print ("Classifier module skipped")
            continue
        param = Dict[name].numpy()
        param = param.reshape(param.size)
        params = np.concatenate((params, param));
    params.tofile(path+"/"+fName)