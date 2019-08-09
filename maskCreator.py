import os
import glob
import cv2
import PIL.Image as Image
import torchvision.transforms as transforms
from skimage.color import rgb2yuv
import numpy as np

if __name__ == '__main__':

    root1 = "E:/RoboCup/YOLO/Train/"
    root2 = "E:/RoboCup/FinetuneHorizon/train/"

    imgSize = (256,192)
    imSize = (192,256)

    resize = transforms.Resize(imSize)
    labResize = transforms.Resize(imSize,Image.NEAREST)

    imgs = sorted(glob.glob1(root1,"*.png"))
    labels = sorted(glob.glob1(root2,"*.png"))

    if len(labels) != len(imgs):
        for i in imgs:
            img = cv2.resize(cv2.imread(root1 + i), imgSize)
            cv2.imwrite(root1+i,img)
    else:
        for i,l in zip(imgs,labels):
            img = cv2.cvtColor(cv2.cvtColor(cv2.resize(cv2.imread(root1+i),(160,120)),cv2.COLOR_BGR2YUV),cv2.COLOR_BGR2RGB)
            #img = cv2.resize(cv2.imread(root1+i),imgSize)
            label = Image.open(root2+l).convert('I')
            label = labResize(label)
            label.save(root2+l)
            cv2.imwrite(root1+i,img)





