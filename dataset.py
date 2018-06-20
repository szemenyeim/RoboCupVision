import torch
import os.path as osp
from torch.utils import data
import glob
from PIL import Image
import numpy as np
import random
import re
import os
import pickle
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def default_loader(path):
    return Image.open(path)

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


class SSDataSet(data.Dataset):
    def __init__(self, root, split="train", camera = "both", img_transform=None, label_transform=None):
        self.root = root
        self.split = split
        self.images = []
        self.labels = []
        self.labels = []
        self.img_transform = img_transform
        self.label_transform = label_transform

        data_dir = osp.join(root, split)
        self.img_dir = osp.join(data_dir,"images")
        self.lab_dir = osp.join(data_dir,"labels")

        imgFiles = sorted(glob.glob1(self.img_dir, "*.png"),key=alphanum_key)
        txtFiles = sorted(glob.glob1(self.img_dir, "*.txt"),key=alphanum_key)
        labFiles = sorted(glob.glob1(self.lab_dir, "*.png"),key=alphanum_key)

        for img,lab,txt in zip(imgFiles,labFiles,txtFiles):
            char = open(osp.join( self.img_dir, txt )).read()
            condition = (camera== "both") or ((camera == "top") and (char == "u")) or ((camera == "bottom") and (char == "b"))
            if condition:
                self.images.append(img)
                self.labels.append(lab)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_file = osp.join( self.img_dir, self.images[index])
        lab_file = osp.join( self.lab_dir, self.labels[index])

        img = Image.open(img_file).convert('RGB')
        label = Image.open(lab_file).convert("I")

        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        random.seed(seed)  # apply this seed to img tranfsorms
        if self.img_transform is not None:
            imgs = self.img_transform(img)
        else:
            imgs = img

        random.seed(seed)  # apply this seed to target tranfsorms
        if self.label_transform is not None:
            labels = self.label_transform(label)
        else:
            labels = label

        return imgs, labels
