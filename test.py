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
from transform import Scale, ToLabel, HorizontalFlip, VerticalFlip, ToYUV, maskLabel
from torchvision.transforms import Compose, Normalize, ToTensor, ColorJitter
import torch.optim as optim
torch.set_printoptions(precision=2,sci_mode=False)
import cv2


import progressbar

def l1reg(model):
    regularization_loss = 0
    for param in model.parameters():
        regularization_loss += torch.sum(torch.abs(param))
    return regularization_loss

def getPrecRecall(maskPred,maskTarget):

    recall =0
    prec = 0
    nClass,bSize = maskPred.shape[0:2]

    for c in range(nClass):
        for b in range(bSize):
            imgPred = maskPred[c,b,:].cpu().numpy().astype('uint8')
            imgTar = maskTarget[c,b,:].cpu().numpy().astype('uint8')

            nPred,_ = cv2.connectedComponents(imgPred)
            nTrue,_ = cv2.connectedComponents(imgTar)

            if nPred > nTrue:
                recall += 1
                prec += nTrue/nPred
            else:
                prec += 1
                recall += nPred/nTrue if nTrue != 0 else 1

    prec /= nClass
    recall /= nClass
    return prec,recall

def valid():
    #############
    ####VALID####
    #############

    model.eval()

    losstotal = 0
    running_acc = 0
    imgCnt = 0
    conf = torch.zeros(numClass,numClass)
    IoU = torch.zeros(numClass)
    labCnts = torch.zeros(numClass)

    recall = 0
    precision = 0

    bar = progressbar.ProgressBar(0, len(valloader), redirect_stdout=False)

    for batch_i, (imgs, targets) in enumerate(valloader):
        imgs = imgs.type(Tensor)
        targets = targets.type(LongTensor)
        targets = maskLabel(targets, nb, nr, ng, nl)

        pred = model(imgs)
        loss = criterion(pred, targets)

        bar.update(batch_i)

        _, predClass = torch.max(pred, 1)
        running_acc += torch.sum(predClass == targets).item() * outSize * 100
        losstotal += loss.item()

        bSize = imgs.shape[0]
        imgCnt += bSize

        maskPred = torch.zeros(numClass, bSize, int(labSize[0]), int(labSize[1])).long()
        maskTarget = torch.zeros(numClass, bSize, int(labSize[0]), int(labSize[1])).long()
        for currClass in range(numClass):
            maskPred[currClass] = predClass == currClass
            maskTarget[currClass] = targets == currClass

        for imgInd in range(bSize):
            for labIdx in range(numClass):
                labCnts[labIdx] += torch.sum(maskTarget[labIdx, imgInd]).item()
                for predIdx in range(numClass):
                    inter = torch.sum(maskPred[predIdx, imgInd] & maskTarget[labIdx, imgInd]).item()
                    conf[(predIdx, labIdx)] += inter
                    if labIdx == predIdx:
                        union = torch.sum(maskPred[predIdx, imgInd] | maskTarget[labIdx, imgInd]).item()
                        if union == 0:
                            IoU[labIdx] += 1
                        else:
                            IoU[labIdx] += inter / union

        p,r = getPrecRecall(maskPred,maskTarget)
        precision += p
        recall += r

    bar.finish()
    prune = count_zero_weights(model)
    precision /= imgCnt
    recall /= imgCnt
    for labIdx in range(numClass):
        for predIdx in range(numClass):
            conf[(predIdx, labIdx)] /= (labCnts[labIdx] / 100.0)
    meanClassAcc = 0.0
    meanIoU = torch.sum(IoU / imgCnt).item() / numClass * 100
    for j in range(numClass):
        meanClassAcc += conf[(j, j)] / numClass
    currLoss = (meanClassAcc+meanIoU)/2
    print(
        "[Validate][Losses: pruned %f, total %f, avg: %f][Pixel Acc: %f, Mean Class Acc: %f, Mean IoU: %f, Precision: %f, Recall: %f]"
        % (
            prune,
            losstotal / float(len(valloader)),
            currLoss,
            running_acc / (imgCnt),
            meanClassAcc,
            meanIoU,
            precision,
            recall
        )
    )

    print(conf)

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", help="Finetuning", action="store_true", default=True)
    parser.add_argument("--noScale", help="Use VGA resolution", action="store_true", default=True)
    parser.add_argument("--v2", help="Use PB-FCNv2", action="store_true", default=False)
    parser.add_argument("--useDice", help="Use Dice Loss", action="store_true", default=False)
    parser.add_argument("--noBall", help="Treat Ball as Background", action="store_true")
    parser.add_argument("--noGoal", help="Treat Goal as Background", action="store_true")
    parser.add_argument("--noRobot", help="Treat Robot as Background", action="store_true")
    parser.add_argument("--noLine", help="Treat Lines as Background", action="store_true")
    parser.add_argument("--topCam", help="Use Top Camera images only", action="store_true")
    parser.add_argument("--bottomCam", help="Use Bottom Camera images only", action="store_true")
    parser.add_argument("--transfer", help="Layers to truly train", action="store_true", default=False)
    opt = parser.parse_args()

    finetune = opt.finetune
    noScale = opt.noScale
    v2 = opt.v2
    nb = opt.noBall
    ng = opt.noGoal
    nr = opt.noRobot
    nl = opt.noLine
    tc = opt.topCam
    bc = opt.bottomCam

    fineTuneStr = "Finetune" if finetune else ""
    scaleStr = "VGA" if noScale else ""
    v2Str = "v2" if v2 else ""
    nbStr = "NoBall" if nb else ""
    ngStr = "NoGoal" if ng else ""
    nrStr = "NoRobot" if nr else ""
    nlStr = "NoLine" if nl else ""
    cameraString = "" if tc == bc else ("top" if tc else "bottom")
    cameraSaveStr = cameraString if finetune else ""
    scale = 2 if noScale else 4
    labSize = (480.0/scale, 640.0/scale)

    name = "checkpoints/best%s%s%s%s%s%s%s%s" % (fineTuneStr,scaleStr,v2Str,nbStr,ngStr,nrStr,nlStr,cameraSaveStr)

    weights_path = []
    if opt.transfer:
        weights_path = sorted(glob.glob(name + "T*.weights"),reverse=True)
    elif opt.finetune:
        weights_path = sorted(glob.glob(name + "*_*.weights"),reverse=True)
    weights_path += [name + ".weights"]
    if not noScale:
        weights_path = [path for path in weights_path if "VGA" not in path]
    if not v2:
        weights_path = [path for path in weights_path if "v2" not in path]
    if not nb:
        weights_path = [path for path in weights_path if "NoBall" not in path]
    if not ng:
        weights_path = [path for path in weights_path if "NoGoal" not in path]
    if not nr:
        weights_path = [path for path in weights_path if "NoRobot" not in path]
    if not nl:
        weights_path = [path for path in weights_path if "NoLine" not in path]

    if nb and ng and nr and nl:
        print("You need to have at least one non-background class!")
        exit(-1)

    if cameraString != "both" and not finetune:
        print("You can only select camera images for the finetune dataset. Using both cameras by default")
        cameraString = "both"

    mean = [0.5, 0.0, 0.0] if not finetune else [0.5, 0.0, 0.0]
    std = [0.25, 0.25, 0.25] if not finetune  else [0.25, 0.25, 0.25]

    n_cpu = 4
    batch_size = 64
    channels = 3
    outSize = 1.0/(labSize[0] * labSize[1])

    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    input_transform = Compose([
        Scale(scale, Image.BILINEAR),
        ToYUV(),
        ToTensor(),
        Normalize(mean, std),

    ])
    target_transform = Compose([
        Scale(scale, Image.NEAREST),
        ToTensor(),
        ToLabel(),
    ])

    batchSize = 16 if (finetune or noScale) else 64

    root = "./data" if sys.platform != 'win32' else "E:/RoboCup"
    if finetune:
        root += "/FinetuneHorizon"

    valloader = data.DataLoader(SSDataSet(root, split="val", camera=cameraString, img_transform=input_transform,
                                          label_transform=target_transform),
                                batch_size=batchSize, shuffle=True, num_workers=4)

    numClass = 5 - nb - ng - nr - nl
    numPlanes = 32
    kernelSize = 1

    weights = Tensor([1, 2, 6, 3, 2]) if opt.useDice else Tensor([1, 10, 30, 5, 2])
    if finetune:
        weights = Tensor([1, 5, 2, 6, 4])
    classIndices = torch.LongTensor([1, (not nb), (not nr), (not ng), (not nl)])
    weights = weights[classIndices == 1]

    criterion = DiceLoss(weights) if opt.useDice else CrossEntropyLoss2d(weights)

    indices = None
    mapLoc = None if cuda else {'cuda:0': 'cpu'}

    for w_path in weights_path:

            print("######################################################")
            print("###### Testing %s ######" %w_path)
            print("######################################################")

            # Initiate model
            model = ROBO_Seg(v2,noScale)
            model.load_state_dict(torch.load(w_path))
            comp = model.get_computations(True)
            print(comp)
            print(sum(comp))

            if cuda:
                model = model.cuda()

            valid()
