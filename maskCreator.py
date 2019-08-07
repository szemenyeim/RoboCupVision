import os
import glob
import cv2
import PIL.Image as Image
import torchvision.transforms as transforms
from skimage.color import rgb2yuv
import numpy as np

if __name__ == '__main__':

    root1 = "E:/RoboCup/train/images/"
    root2 = "E:/RoboCup/train/labels/"

    resize = transforms.Resize((120,160))
    labResize = transforms.Resize((120,160),Image.NEAREST)

    imgs = sorted(glob.glob1(root1,"*.png"))
    labels = sorted(glob.glob1(root2,"*.png"))

    for i,l in zip(imgs,labels):
        img = cv2.cvtColor(cv2.cvtColor(cv2.resize(cv2.imread(root1+i),(160,120)),cv2.COLOR_BGR2YUV),cv2.COLOR_BGR2RGB)
        #label = Image.open(root2+l).convert('I')
        #label = labResize(label)
        #label.save(root2+l)
        cv2.imwrite(root1+i,img)





