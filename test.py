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

def getPrecRecall(maskPred,maskTarget, thresh, distanceThresh):

    recallI =0
    precI = 0
    recallD =0
    precD = 0
    nClass,bSize = maskPred.shape[0:2]

    for c in range(1,nClass):
        for b in range(bSize):
            imgPred = maskPred[c,b,:].cpu().numpy().astype('uint8')
            imgTar = maskTarget[c,b,:].cpu().numpy().astype('uint8')

            nPred,predLab = cv2.connectedComponents(imgPred)
            nTrue,tarLab = cv2.connectedComponents(imgTar)
            nPred -=1
            nTrue -=1

            usedTarI = np.zeros(nTrue)
            usedTarD = np.zeros(nTrue)

            nCorrI = 0
            nCorrD = 0

            for i in range(nPred):
                pred = (predLab == (i+1))
                predBox = cv2.boundingRect(pred.astype('uint8'))
                predCent = (predBox[0]+predBox[2]/2,predBox[1]+predBox[3]/2)

                foundI = False
                foundD = False

                for j in range(nTrue):
                    tar = (tarLab == (j+1))
                    tarBox = cv2.boundingRect(tar.astype('uint8'))
                    tarCent = (tarBox[0]+tarBox[2]/2,tarBox[1]+tarBox[3]/2)
                    dist = np.sqrt((predCent[0]-tarCent[0])**2+(predCent[1]-tarCent[1])**2)
                    Iou = (pred & tar).sum() / (pred | tar).sum()
                    if Iou > thresh and not foundI and usedTarI[j] == 0:
                        nCorrI += 1
                        foundI = True
                        usedTarI[j] = 1
                    if distanceThresh > dist and not foundD and usedTarD[j] == 0:
                        nCorrD += 1
                        foundD = True
                        usedTarD[j] = 1

            precI += nCorrI/nPred if nPred != 0 else 1
            recallI += nCorrI/nTrue if nTrue != 0 else 1
            precD += nCorrD/nPred if nPred != 0 else 1
            recallD += nCorrD/nTrue if nTrue != 0 else 1

    precI /= (nClass-1)
    recallI /= (nClass-1)
    precD /= (nClass-1)
    recallD /= (nClass-1)

    return (precI+recallI)/2,(precD+recallD)/2

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

    recPrec = np.zeros((2,5))

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

        for i,(thresh,dThresh) in enumerate(zip(thresholds,dThresholds)):
            valI,valD = getPrecRecall(maskPred,maskTarget,thresh,dThresh)
            recPrec[0,i] += valI
            recPrec[1,i] += valD

    bar.finish()
    prune = count_zero_weights(model)
    recPrec /= imgCnt
    for labIdx in range(numClass):
        for predIdx in range(numClass):
            conf[(predIdx, labIdx)] /= (labCnts[labIdx] / 100.0)
    meanClassAcc = 0.0
    meanIoU = torch.sum(IoU / imgCnt).item() / numClass * 100
    for j in range(numClass):
        meanClassAcc += conf[(j, j)] / numClass
    currLoss = (meanClassAcc+meanIoU)/2
    print(
        "[Validate][Losses: pruned %f, total %f, avg: %f][Pixel Acc: %f, Mean Class Acc: %f, Mean IoU: %f]"
        % (
            prune,
            losstotal / float(len(valloader)),
            currLoss,
            running_acc / (imgCnt),
            meanClassAcc,
            meanIoU,
        )
    )

    print("IoU:",recPrec[0])
    print("Dist:",recPrec[1])

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", help="Finetuning", action="store_true", default=False)
    parser.add_argument("--v2", help="Use v2 architecture", action="store_true", default=False)
    parser.add_argument("--noScale", help="Use VGA resolution", action="store_true", default=False)
    parser.add_argument("--UNet", help="Use Vanilla U-Net", action="store_true", default=False)
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
    unet = opt.UNet
    nb = opt.noBall
    ng = opt.noGoal
    nr = opt.noRobot
    nl = opt.noLine
    tc = opt.topCam
    bc = opt.bottomCam

    fineTuneStr = "Finetune" if finetune else ""
    scaleStr = "VGA" if noScale else ""
    v2Str = "v2" if v2 else ""
    unetStr = "UNet" if unet else ""
    nbStr = "NoBall" if nb else ""
    ngStr = "NoGoal" if ng else ""
    nrStr = "NoRobot" if nr else ""
    nlStr = "NoLine" if nl else ""
    cameraString = "" if tc == bc else ("top" if tc else "bottom")
    cameraSaveStr = cameraString if finetune else ""
    scale = 2 if noScale else 4
    labSize = (480//scale, 640//scale)

    thresholds = [0.75, 0.5, 0.25, 0.1, 0.05]
    dThresholds = [1.25, 2.5, 5, 10, 20]
    if noScale:
        dThresholds = [d*2 for d in dThresholds]

    name = "checkpoints/best%s%s%s%s%s%s%s%s%s" % (fineTuneStr,v2Str,scaleStr,unetStr,nbStr,ngStr,nrStr,nlStr,cameraSaveStr)

    weights_path = []
    if opt.transfer:
        weights_path = sorted(glob.glob(name + "T*.weights"),reverse=True)
    elif opt.finetune:
        weights_path = sorted(glob.glob(name + "*_*.weights"),reverse=True)
    weights_path += [name + ".weights"]
    if not noScale:
        weights_path = [path for path in weights_path if "VGA" not in path]
    if not unet:
        weights_path = [path for path in weights_path if "UNet" not in path]
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

    n_cpu = 4
    batch_size = 64
    channels = 3
    outSize = 1.0/(labSize[0] * labSize[1])

    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    batchSize = 16 if (finetune or noScale) else 64

    root = "../data" if sys.platform != 'win32' else "D:/Datasets/RoboCup"

    valloader = data.DataLoader(SSYUVDataset(root, img_size=labSize, train=False, finetune=finetune, camera=cameraString),
        batch_size=batchSize, shuffle=True, num_workers=5)

    numClass = 5 - nb - ng - nr - nl
    numPlanes = 8 if unet else 8
    levels = 3 if unet else 2
    depth = 4
    bellySize = 0 if unet else 5
    bellyPlanes = numPlanes * pow(2, depth)

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
            model = ROBO_UNet(noScale,planes=numPlanes,depth=depth,levels=levels,bellySize=bellySize,bellyPlanes=bellyPlanes,pool=unet)
            model.load_state_dict(torch.load(w_path))
            comp = model.get_computations(True)
            print(comp)
            print(sum(comp))

            if cuda:
                model = model.cuda()

            valid()
