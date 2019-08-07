import os
import glob
import cv2
import numpy as np

if __name__ == '__main__':

    root1 = "E:/RoboCup/val/images/"
    root2 = "E:/RoboCup/val/labels/"

    imgs = sorted(glob.glob1(root1,"*.png"))
    labels = sorted(glob.glob1(root2,"*.png"))

    for i,l in zip(imgs,labels):
        #img = cv2.cvtColor(cv2.cvtColor(cv2.resize(cv2.imread(root1+i),(320,240)),cv2.COLOR_BGR2YUV),cv2.COLOR_BGR2RGB)
        img = cv2.resize(cv2.imread(root1+i),(160,120))
        lab = cv2.resize(cv2.imread(root2+l),(160,120))
        cv2.imwrite(root1+i,img)
        cv2.imwrite(root2+l,lab)





