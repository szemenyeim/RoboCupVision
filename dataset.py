import torch
import os.path as osp
from torch.utils import data
import glob
from PIL import Image
import numpy as np
import random
import re
import os

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

class LPDataSet(data.Dataset):
    def __init__(self, root, split="train", img_transform=None, label_transform=None):
        self.root = root
        self.split = split
        self.images = []
        self.labels = []
        self.predictions = []
        self.img_transform = img_transform
        self.label_transform = label_transform

        data_dir = osp.join(root, split)

        for dir in get_immediate_subdirectories(data_dir):
            currDir = osp.join(data_dir,dir)
            img_dir = osp.join(currDir,"images")
            lab_dir = osp.join(currDir,"labels")
            images = []
            labels = []
            for file in sorted(glob.glob1(img_dir, "*.png"), key=alphanum_key):
                images.append(osp.join(img_dir, file))
            for file in sorted(glob.glob1(lab_dir, "*.png"), key=alphanum_key):
                labels.append(osp.join(lab_dir,file))
            self.images.append(images)
            self.labels.append(labels)

    def __len__(self):
        length = 0
        for imgs in self.images:
            length += len(imgs)-1
        return length

    def __getitem__(self, index):
        dirindex = 0
        itemindex = index

        #print index

        for imgs in self.images:
            #print dirindex, itemindex, len(imgs)
            if itemindex >= len(imgs) - 1:
                dirindex += 1
                itemindex -= (len(imgs))
            else:
                break

        img_file = self.images[dirindex][itemindex]
        img_file2 = self.images[dirindex][itemindex+1]
        lab_file = self.labels[dirindex][itemindex]
        lab_file2 = self.labels[dirindex][itemindex+1]

        img = Image.open(img_file).convert('RGB')
        img2 = Image.open(img_file2).convert('RGB')
        label = Image.open(lab_file).convert("I")
        label2 = Image.open(lab_file2).convert("I")

        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        if self.img_transform is not None:
            random.seed(seed)  # apply this seed to img tranfsorms
            img = torch.unsqueeze(self.img_transform(img),0)
            random.seed(seed)  # apply this seed to img tranfsorms
            img2 = torch.unsqueeze(self.img_transform(img2),0)
            imgs = torch.cat([img,img2])
        else:
            imgs = torch.cat([img,img2])

        random.seed(seed)  # apply this seed to target tranfsorms
        if self.label_transform is not None:
            random.seed(seed)  # apply this seed to img tranfsorms
            label = torch.unsqueeze(self.label_transform(label),0)
            random.seed(seed)  # apply this seed to img tranfsorms
            label2 = torch.unsqueeze(self.label_transform(label2),0)
            labels = torch.cat([label,label2])
        else:
            labels = torch.cat([label,label2])

        return imgs, labels


class SSDataSet(data.Dataset):
    def __init__(self, root, split="train", img_transform=None, label_transform=None):
        self.root = root
        self.split = split
        self.images = []
        self.labels = []
        self.img_transform = img_transform
        self.label_transform = label_transform

        data_dir = osp.join(root, split)
        self.img_dir = osp.join(data_dir,"images")
        self.lab_dir = osp.join(data_dir,"labels")

        for file in sorted(glob.glob1(self.img_dir, "*.png"),key=alphanum_key):
            self.images.append(file)
        for file in sorted(glob.glob1(self.lab_dir, "*.png"),key=alphanum_key):
            self.labels.append(file)


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
