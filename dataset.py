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


class ODDataSet(data.Dataset):
    def __init__(self, root, split="train", img_transform=None, label_transform=None, bbMean = None, bbStd = None, numBall = 1, numRobot = 5, numGoal = 2):

        self.numBBs = numBall+numGoal+numRobot
        self.root = root
        self.split = split
        self.images = []
        self.labels = np.empty([0,5*self.numBBs]).astype('float32')
        self.img_transform = img_transform
        self.label_transform = label_transform

        data_dir = osp.join(root, split)
        self.img_dir = osp.join(data_dir, "images")
        self.bMeans = np.load(osp.join(data_dir,'bMean.npy'))
        self.rMeans = np.load(osp.join(data_dir,'rMean.npy'))
        self.gMeans = np.load(osp.join(data_dir,'gMean.npy'))
        labs = pickle.load(open(osp.join(data_dir,'preds.pickle'),'rb'))

        for i, file in enumerate(sorted(glob.glob1(self.img_dir, "*.png"), key=alphanum_key)):
            self.images.append(file)
            currLab = labs[i]
            labArray = self.label2Array(currLab)
            self.labels = np.append(self.labels,labArray,0)


        self.means = bbMean if bbMean is not None else np.mean(self.labels,0)
        self.std = bbStd if bbStd is not None else (np.std(self.labels,0) +1e-5)

        self.labels = (self.labels - self.means) / self.std


    def label2Array(self,label):
            labArray = np.zeros((1, 5*self.numBBs))

            goals = np.empty((0,5))
            robots = np.empty((0,5))

            for BB in label:
                if BB[0] == 1:
                    labArray[0,0] = 1
                    labArray[0,1:5] = BB[1]
                elif BB[0] == 2:
                    robots = np.append(robots,BB[1])
                elif BB[0] == 3:
                    goals = np.append(goals,BB[1])

            if robots.shape[0] > 0:
                robots = np.reshape(robots,(-1,4))
                roboDist = cdist(robots,self.rMeans)
                row_ind, col_ind = linear_sum_assignment(roboDist)
                for i,ind in enumerate(row_ind):
                    arrOffs = (ind+1)*5
                    labArray[0,arrOffs] = 1
                    labArray[0,arrOffs+1:arrOffs+5] = robots[i]
            if goals.shape[0] > 0:
                goals = np.reshape(goals,(-1,4))
                goalDist = cdist(goals,self.gMeans)
                row_ind, col_ind = linear_sum_assignment(goalDist)
                for i,ind in enumerate(row_ind):
                    arrOffs = (ind+6)*5
                    labArray[0,arrOffs] = 1
                    labArray[0,arrOffs+1:arrOffs+5] = goals[i]

            return labArray.astype('float32')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_file = osp.join( self.img_dir, self.images[index])

        img = Image.open(img_file).convert('RGB')
        label = self.labels[index]

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
