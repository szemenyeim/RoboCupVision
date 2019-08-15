from __future__ import division

from model import *
from dataset import *

import os
import sys
import argparse

from torch import nn
from torch.utils import data
import lr_scheduler
from model import CrossEntropyLoss2d, PB_FCN, pruneModelNew, PB_FCN_2, DiceLoss
from dataset import SSDataSet
from transform import Colorize
from torchvision.transforms import Compose, Normalize, ToTensor, ColorJitter
import torch.optim as optim
torch.set_printoptions(precision=2,sci_mode=False)
import cv2


import progressbar


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", help="Finetuning", action="store_true", default=False)
    parser.add_argument("--noScale", help="Use VGA resolution", action="store_true", default=False)
    parser.add_argument("--UNet", help="Use Vanilla U-Net", action="store_true", default=False)
    parser.add_argument("--useDice", help="Use Dice Loss", action="store_true", default=False)
    parser.add_argument("--noBall", help="Treat Ball as Background", action="store_true")
    parser.add_argument("--noGoal", help="Treat Goal as Background", action="store_true")
    parser.add_argument("--noRobot", help="Treat Robot as Background", action="store_true")
    parser.add_argument("--noLine", help="Treat Lines as Background", action="store_true")
    parser.add_argument("--topCam", help="Use Top Camera images only", action="store_true")
    parser.add_argument("--bottomCam", help="Use Bottom Camera images only", action="store_true")
    opt = parser.parse_args()

    finetune = opt.finetune
    noScale = opt.noScale
    unet = opt.UNet
    nb = opt.noBall
    ng = opt.noGoal
    nr = opt.noRobot
    nl = opt.noLine
    tc = opt.topCam
    bc = opt.bottomCam

    fineTuneStr = "Finetune" if finetune else ""
    scaleStr = "VGA" if noScale else ""
    unetStr = "UNet" if unet else ""
    nbStr = "NoBall" if nb else ""
    ngStr = "NoGoal" if ng else ""
    nrStr = "NoRobot" if nr else ""
    nlStr = "NoLine" if nl else ""
    cameraString = "" if tc == bc else ("top" if tc else "bottom")
    cameraSaveStr = cameraString if finetune else ""
    scale = 2 if noScale else 4
    labSize = (480 // scale, 640 // scale)

    thresholds = [0.75, 0.5, 0.25, 0.1, 0.05]
    dThresholds = [1.25, 2.5, 5, 10, 20]
    if noScale:
        dThresholds = [d * 2 for d in dThresholds]

    name = "checkpoints/best%s%s%s%s%s%s%s%s" % (fineTuneStr, scaleStr, unetStr, nbStr, ngStr, nrStr, nlStr, cameraSaveStr)

    weights_path = name + ".weights"

    if nb and ng and nr and nl:
        print("You need to have at least one non-background class!")
        exit(-1)

    if cameraString != "both" and not finetune:
        print("You can only select camera images for the finetune dataset. Using both cameras by default")
        cameraString = "both"

    outSize = 1.0 / (labSize[0] * labSize[1])

    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    batchSize = 1

    root = "../data" if sys.platform != 'win32' else "E:/RoboCup"

    valloader = data.DataLoader(
        SSYUVDataset(root, img_size=labSize, train=False, finetune=finetune, camera=cameraString),
        batch_size=batchSize, shuffle=False, num_workers=5)

    numClass = 5 - nb - ng - nr - nl
    numPlanes = 8 if unet else 8
    levels = 3 if unet else 2
    depth = 4
    bellySize = 0 if unet else 5
    bellyPlanes = numPlanes * pow(2, depth)

    mapLoc = None if cuda else {'cuda:0': 'cpu'}

    print("######################################################")
    print("##################### Detection ######################")
    print("######################################################")

    # Initiate model
    model = ROBO_UNet(noScale, planes=numPlanes, depth=depth, levels=levels, bellySize=bellySize, bellyPlanes=bellyPlanes,pool=unet)
    model.load_state_dict(torch.load(weights_path))
    comp = model.get_computations(True)
    print(comp)
    print(sum(comp))

    if cuda:
        model = model.cuda()

    model.eval() # Set in evaluation mode

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


    print ('\nPerforming object detection:')
    bar = progressbar.ProgressBar(0, len(valloader), redirect_stdout=False)
    for batch_i, (imgs, targets) in enumerate(valloader):
        # Configure input
        input_imgs = imgs.type(Tensor)

        # Get detections
        with torch.no_grad():
            pred = model(input_imgs)
            _, predClass = torch.max(pred, 1)
            mask = Colorize(predClass.cpu().squeeze()).permute(1,2,0).numpy()

        cv2.imwrite('output/%d.png' % (batch_i),mask)

        # Log progress
        bar.update(batch_i)

    bar.finish()
