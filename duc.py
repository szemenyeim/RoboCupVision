import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models

import math


class DUC(nn.Module):
    def __init__(self, inplanes, planes, upscale_factor=2):
        super(DUC, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3,
                              padding=1)
        self.bn = nn.BatchNorm2d(planes)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x

class UpScale(nn.Module):
    def __init__(self):
        super(UpScale, self).__init__()

        self.duc1 = DUC(2048, 2048*2)
        self.duc2 = DUC(1024, 1024*2)
        self.duc3 = DUC(512, 512*2)
        self.duc4 = DUC(128, 128*2)
        self.duc5 = DUC(64, 64*2)

        self.transformer = nn.Conv2d(320, 128, kernel_size=1)

    def forward(self, x):

        dfm1 = x[4] + self.duc1(x[5])
        dfm2 = x[3] + self.duc2(dfm1)
        dfm3 = x[2] + self.duc3(dfm2)
        dfm3_t = self.transformer(torch.cat((dfm3, x[1]), 1))
        dfm4 = x[0] + self.duc4(dfm3_t)
        dfm5 = self.duc5(dfm4)

        return dfm1, dfm2, dfm3_t, dfm4, dfm5

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()

        resnet = models.resnet152(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn0 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        return conv_x, pool_x, fm1, fm2, fm3, fm4

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()

        self.num_classes = num_classes
        self.out = self._classifier(32)

    def _classifier(self, inplanes):
        return nn.Sequential(
            nn.Conv2d(inplanes, inplanes * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes * 2, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout2d(.5),
            nn.Conv2d(inplanes * 2, self.num_classes, 1),
        )

    def forward(self, x):

        out = self.out(x[4])

        return out

class SegFull(nn.Module):
    def __init__(self,num_classes):
        super(SegFull,self).__init__()
        self.pad = nn.ConstantPad2d((0,0,4,4),0)
        self.FCN = FCN()
        self.UpScale = UpScale()
        self.Classifier = Classifier(num_classes)

    def forward(self,x):
        x = self.pad(x)
        x = self.FCN(x)
        x = self.UpScale(x)
        x = self.Classifier(x)
        return x[:,:,4:124,:]
