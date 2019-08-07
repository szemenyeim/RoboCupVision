import torch
import os.path as osp
from torch.utils import data
import glob
from PIL import Image
import numpy as np
import random
import re
import os
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from skimage.color import rgb2yuv

class ColorJitter(object):
    def __init__(self,b=0.3,c=0.3,s=0.3,h=3.1415/6):
        super(ColorJitter,self).__init__()
        self.b = b
        self.c = c
        self.s = s
        self.h = h

    def __call__(self, img):
        b_val = random.uniform(-self.b,self.b)
        c_val = random.uniform(1-self.c,1+self.c)
        s_val = random.uniform(1-self.s,1+self.s)
        h_val = random.uniform(-self.h,self.h)

        mtx = torch.FloatTensor([[s_val*np.cos(h_val),-np.sin(h_val)],[np.sin(h_val),s_val*np.cos(h_val)]])

        img[0] = (img[0]+b_val)*c_val
        if self.s > 0 and self.h > 0:
            img[1:] = torch.einsum('nm,mbc->nbc',mtx,img[1:])

        return img

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

class ToYUV(object):
    def __call__(self, img):
        return rgb2yuv(img)
        #return img.convert('YCbCr')

class SSYUVDataset(data.Dataset):
    def __init__(self, data_dir, img_size=(120,160), train=True, finetune = False,camera = "both"):
        self.img_shape = img_size
        self.train = train
        self.finetune = finetune
        self.img_size = img_size
        self.jitter = ColorJitter(0.3,0.3,0.3,3.1415/6)
        self.resize = transforms.Resize(img_size)
        self.labResize = transforms.Resize(img_size,Image.NEAREST)
        self.mean = [0.34190056, 0.4833289,  0.48565758] if finetune else [0.36269532, 0.41144562, 0.282713]
        self.std = [0.47421749, 0.13846053, 0.1714848] if finetune else [0.31111388, 0.21010718, 0.34060917]
        self.normalize = transforms.Normalize(mean=self.mean,std=self.std)
        self.images =[]
        self.labels =[]

        if finetune:
            data_dir = osp.join(data_dir,"FinetuneHorizon")

        data_dir = osp.join(data_dir,"train" if train else "val")
        self.img_dir = osp.join(data_dir, "images")
        self.lab_dir = osp.join(data_dir, "labels")

        imgFiles = sorted(glob.glob1(self.img_dir, "*.png"), key=alphanum_key)
        txtFiles = sorted(glob.glob1(self.img_dir, "*.txt"), key=alphanum_key)
        labFiles = sorted(glob.glob1(self.lab_dir, "*.png"), key=alphanum_key)

        if len(txtFiles) == len(imgFiles):
            for img, lab, txt in zip(imgFiles, labFiles, txtFiles):
                char = open(osp.join(self.img_dir, txt)).read()
                condition = (camera == "both") or ((camera == "top") and (char == "u")) or (
                            (camera == "bottom") and (char == "b"))
                if condition:
                    self.images.append(img)
                    self.labels.append(lab)
        else:
            for img, lab in zip(imgFiles, labFiles):
                self.images.append(img)
                self.labels.append(lab)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        #---------
        #  Image
        #---------
        img_file = osp.join(self.img_dir, self.images[index])
        lab_file = osp.join(self.lab_dir, self.labels[index])

        img = Image.open(img_file).convert('RGB')
        label = Image.open(lab_file).convert('I')

        if self.img_size[0] != img.size[1] and self.img_size[1] != img.size[0]:
            img = self.resize(img)
        if self.img_size[0] != label.size[1] and self.img_size[1] != label.size[0]:
            label = self.labResize(label)

        img = transforms.functional.to_tensor(img).float()
        label = transforms.functional.to_tensor(label)
        img = self.normalize(img)
        if self.train:
            p = torch.rand(1).item()
            if p > 0.5:
                img = img.flip(2)
                label = label.flip(2)
            img = self.jitter(img)

        return img, label.squeeze()

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

        if len(txtFiles) == len(imgFiles):
            for img,lab,txt in zip(imgFiles,labFiles,txtFiles):
                char = open(osp.join( self.img_dir, txt )).read()
                condition = (camera== "both") or ((camera == "top") and (char == "u")) or ((camera == "bottom") and (char == "b"))
                if condition:
                    self.images.append(img)
                    self.labels.append(lab)
        else:
            for img,lab in zip(imgFiles,labFiles):
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
