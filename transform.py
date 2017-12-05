import torch
from PIL import Image, ImageEnhance, ImageChops
import random
import numpy as np
from skimage.color import rgb2yuv
import cv2

class Scale(object):
    def __init__(self, factor, interpolation=Image.BILINEAR):
        self.factor = factor
        self.interpolation = interpolation

    def __call__(self, img):
        w,h = img.size
        if self.factor == 1:
            return img
        w = int(w/self.factor)
        h = int(h/self.factor)
        return img.resize((w,h), self.interpolation)

class ToYUV(object):
    def __call__(self, img):
        return rgb2yuv(img)*255
        #return img.convert('YCbCr')

class ToLabel(object):
    def __call__(self, tensor):
        return torch.squeeze(tensor.long())

class ToBinLabel(object):
    def __call__(self, tensor):
        tensor[tensor > 1] = 1
        return tensor

class HorizontalFlip(object):
    """Horizontally flips the given PIL.Image with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
        img (PIL.Image): Image to be flipped.
        Returns:
        PIL.Image: Randomly flipped image.
        """
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class VerticalFlip(object):
    def __call__(self, img):
        """
        Args:
        img (PIL.Image): Image to be flipped.
        Returns:
        PIL.Image: Randomly flipped image.
        """
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class RandomNoise(object):
    def __call__(self, img):
        if random.random() < 0.9:
            a = torch.normal(std=0.05*torch.ones(img.size()))
            return img + a
        return img

class RandomBrightness(object):
    def __call__(self, img):
        if random.random() < 0.9:
            a = 0.5 + random.random()
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(a)
        return img


class RandomContrast(object):
    def __call__(self, img):
        if random.random() < 0.9:
            a = 0.5 + random.random()
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(a)
        return img

class RandomColor(object):
    def __call__(self, img):
        if random.random() < 0.9:
            a = 0.5 + random.random()
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(a)
        return img

class RandomHue(object):
    def __init__(self, W, H):
        self.W = int(W)
        self.H = int(H)

    def __call__(self, img):
        if random.random() < 0.9:
            img = img.convert("HSV")
            zero = np.zeros((self.H,self.W,3))
            a = random.random()*30
            zero[:,:,0] += a
            zero = Image.fromarray(zero.astype('uint8'))
            if random.random() < 0.5:
                img = ImageChops.add(img,zero)
            else:
                img = ImageChops.subtract(img,zero)
            img = img.convert("RGB")
        return img

def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    cmap[0, 0] = 0
    cmap[0, 1] = 0
    cmap[0, 2] = 0
    cmap[1, 0] = 255
    cmap[1, 1] = 0
    cmap[1, 2] = 0
    cmap[2, 0] = 0
    cmap[2, 1] = 255
    cmap[2, 2] = 0
    cmap[3, 0] = 0
    cmap[3, 1] = 0
    cmap[3, 2] = 255
    cmap[4, 0] = 255
    cmap[4, 1] = 255
    cmap[4, 2] = 255
    return cmap

def Colorize(gray_image,n=5):
        cmap = labelcolormap(n)
        cmap = torch.from_numpy(cmap[:n])
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[0], size[1]).fill_(0)

        for label in range(0, len(cmap)):
            mask = (label == gray_image).cpu()
            color_image[0][mask] = cmap[label][0]
            color_image[1][mask] = cmap[label][1]
            color_image[2][mask] = cmap[label][2]

        return color_image

def labelToPred(label, numClass):
    B = label.size()[0]
    H = label.size()[1]
    W = label.size()[2]
    label = label.view(-1,1)
    if torch.cuda.is_available():
        out = torch.ones(B * H * W, numClass).cuda().scatter_(1,label.data,-1.0)*(-1)
    else:
        out = torch.ones(B * H * W, numClass).scatter_(1,label.data,-1.0)*(-1)
    out = out.view(B,H,W,numClass)
    out = out.permute(0,3,1,2)
    return out

def optFlow(imgp, imgn):
    imgp = ((imgp.data.cpu().numpy()+1.0)*127.5).astype('uint8')
    imgn = ((imgn.data.cpu().numpy()+1.0)*127.5).astype('uint8')
    flow = np.zeros(1)
    flow = cv2.calcOpticalFlowFarneback(imgp,imgn,flow,pyr_scale=0.5,levels=2,winsize=9,iterations=2,poly_n=7,poly_sigma=1.5,flags=0)
    return flow

def updateLabels(oldLab,flow):
    labels = torch.zeros(oldLab.size()).long()
    flow = np.rint(flow)
    for i in range(oldLab.size()[0]):
        for j in range(oldLab.size()[1]):
            otheri = i + flow[i,j,1]
            otherj = j + flow[i,j,0]
            if otherj >= 0 and otherj < oldLab.size()[1] and otheri >= 0 and otheri < oldLab.size()[0]:
                labels[otheri,otherj] = oldLab.data[i,j]
    return labels
