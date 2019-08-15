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
from dataset import SSYUVDataset
from transform import Scale, ToLabel, HorizontalFlip, VerticalFlip, ToYUV, maskLabel
from torchvision.transforms import Compose, Normalize, ToTensor, ColorJitter
import torch.optim as optim
torch.set_printoptions(precision=2,sci_mode=False)


import progressbar

def l1reg(model):
    regularization_loss = 0
    for param in model.parameters():
        regularization_loss += torch.sum(torch.abs(param))
    return regularization_loss

def train(epoch,epochs,bestLoss,indices = None):
    #############
    ####TRAIN####
    #############

    lossreg = 0
    losstotal = 0
    running_acc = 0
    imgCnt = 0

    model.train()

    bar = progressbar.ProgressBar(0, len(trainloader), redirect_stdout=False)

    for batch_i, (imgs, targets) in enumerate(trainloader):
        imgs = imgs.type(Tensor)
        targets = targets.type(LongTensor).long()
        targets = maskLabel(targets, nb, nr, ng, nl)

        optimizer.zero_grad()

        pred = model(imgs)
        loss = criterion(pred,targets)
        reg = Tensor([0.0])
        if indices is None:
            reg = decay * l1reg(model)
            loss += reg

        loss.backward()

        if indices is not None:
            pIdx = 0
            for param in model.parameters():
                if param.dim() > 1:
                    if param.grad is not None:
                        param.grad[indices[pIdx]] = 0
                    pIdx += 1

        optimizer.step()
        bar.update(batch_i)

        _, predClass = torch.max(pred, 1)
        running_acc += torch.sum(predClass == targets).item() * outSize * 100
        lossreg += reg.item()
        losstotal += loss.item()
        imgCnt += imgs.shape[0]

    bar.finish()
    prune = count_zero_weights(model)
    print(
        "[Epoch Train %d/%d lr: %.4f][Losses: reg %f, pruned %f, total %f][Pixel Acc: %f]"
        % (
            epoch + 1,
            epochs,
            scheduler.get_lr()[-1]/learning_rate,
            lossreg / float(len(trainloader)),
            prune,
            losstotal / float(len(trainloader)),
            running_acc/(imgCnt),
        )
    )

    if indices is None:
        scheduler.step()

    return bestLoss


def valid(epoch,epochs,bestLoss,pruned):
    #############
    ####VALID####
    #############

    model.eval()

    lossreg = 0
    losstotal = 0
    running_acc = 0
    imgCnt = 0
    conf = torch.zeros(numClass,numClass)
    IoU = torch.zeros(numClass)
    labCnts = torch.zeros(numClass)

    bar = progressbar.ProgressBar(0, len(valloader), redirect_stdout=False)

    for batch_i, (imgs, targets) in enumerate(valloader):
        imgs = imgs.type(Tensor)
        targets = targets.type(LongTensor)
        targets = maskLabel(targets, nb, nr, ng, nl)

        pred = model(imgs)
        loss = criterion(pred, targets)
        reg = Tensor([0.0])
        if indices is None:
            reg = decay * l1reg(model)
            loss += reg

        bar.update(batch_i)

        _, predClass = torch.max(pred, 1)
        running_acc += torch.sum(predClass == targets).item() * outSize * 100
        lossreg += reg.item()
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

    bar.finish()
    prune = count_zero_weights(model)
    for labIdx in range(numClass):
        for predIdx in range(numClass):
            conf[(predIdx, labIdx)] /= (labCnts[labIdx] / 100.0)
    meanClassAcc = 0.0
    meanIoU = torch.sum(IoU / imgCnt).item() / numClass * 100
    for j in range(numClass):
        meanClassAcc += conf[(j, j)] / numClass
    currLoss = (meanClassAcc+meanIoU)/2
    print(
        "[Epoch Val %d/%d lr: %.4f][Losses: reg %f, pruned %f, total %f][Pixel Acc: %f, Mean Class Acc: %f, Mean IoU: %f]"
        % (
            epoch + 1,
            epochs,
            scheduler.get_lr()[-1] / learning_rate,
            lossreg / float(len(valloader)),
            prune,
            losstotal / float(len(valloader)),
            running_acc / (imgCnt),
            meanClassAcc,
            meanIoU,
        )
    )

    name = "bestFinetune" if finetune else "best"
    name += scaleStr

    name += unetStr
    name += nbStr
    name += ngStr
    name += nrStr
    name += nlStr
    name += cameraSaveStr
    if transfer != 0:
        name += "T%d" % transfer
    if pruned:
        pruneP = round(prune * 100)
        comp = round(sum(model.get_computations(True))/1000000)
        name = name + ("%d_%d" %(pruneP,comp))

    if bestLoss < (currLoss):
        print("Saving best model")
        print(conf)
        bestLoss = (currLoss)
        torch.save(model.state_dict(), "checkpoints/%s.weights" % name)

    return bestLoss

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
    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-1)
    parser.add_argument("--decay", help="Weight decay", type=float, default=5e-5)
    parser.add_argument("--transfer", help="Layers to truly train", action="store_true")
    opt = parser.parse_args()

    finetune = opt.finetune
    learning_rate = opt.lr#*2 if finetune and not opt.transfer else opt.lr
    dec = opt.decay if finetune else opt.decay/10
    transfers = [1, 2, 3, 4] if opt.transfer else [0]
    decays = [10*dec, 5*dec, 2*dec, dec] if (finetune and not opt.transfer) else [dec]
    '''if opt.UNet:
        decays = [d*2 for d in decays]'''
    noScale = opt.noScale
    unet = opt.UNet
    nb = opt.noBall
    ng = opt.noGoal
    nr = opt.noRobot
    nl = opt.noLine
    tc = opt.topCam
    bc = opt.bottomCam

    fineTuneStr = "Finetuned" if finetune else ""
    scaleStr = "VGA" if noScale else ""
    unetStr = "UNet" if unet else ""
    nbStr = "NoBall" if nb else ""
    ngStr = "NoGoal" if ng else ""
    nrStr = "NoRobot" if nr else ""
    nlStr = "NoLine" if nl else ""
    cameraString = "" if tc == bc else ("top" if tc else "bottom")
    cameraSaveStr = cameraString if finetune else ""
    if tc == bc:
        cameraString = "both"
    scale = 2 if noScale else 4
    labSize = (480//scale, 640//scale)

    weights_path = "checkpoints/best%s%s%s%s%s%s%s.weights" % (scaleStr,unetStr,nbStr,ngStr,nrStr,nlStr,cameraSaveStr)

    if nb and ng and nr and nl:
        print("You need to have at least one non-background class!")
        exit(-1)

    if cameraString != "both" and not finetune:
        print("You can only select camera images for the finetune dataset. Using both cameras by default")
        cameraString = "both"

    n_cpu = 4
    channels = 3
    epochs = 100 if noScale or not finetune else 200
    momentum = 0.5

    if finetune:
        learning_rate *= 0.1
        momentum = 0.5
        epochs = 200 if noScale else 200
    outSize = 1.0/(labSize[0] * labSize[1])

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    seed = 12345678
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    batchSize = 16 if finetune else (32 if noScale else 64)

    root = "../data" if sys.platform != 'win32' else "E:/RoboCup"

    trainloader = data.DataLoader(SSYUVDataset(root,img_size=labSize,train=True,finetune=finetune,camera=cameraString),
                                  batch_size=batchSize, shuffle=True, num_workers=5)

    valloader = data.DataLoader(SSYUVDataset(root,img_size=labSize,train=False,finetune=finetune,camera=cameraString),
                                batch_size=batchSize, shuffle=True, num_workers=5)

    numClass = 5 - nb - ng - nr - nl
    numPlanes = 8 if unet else 8
    levels = 3 if unet else 2
    depth = 4 if unet else 4
    bellySize = 0 if unet else 5
    bellyPlanes = numPlanes*pow(2,depth)

    weights = Tensor([1, 2, 6, 3, 2]) if opt.useDice else Tensor([1, 10, 30, 10, 2])
    if finetune:
        weights = Tensor([1, 6, 2, 10, 4])
    classIndices = torch.LongTensor([1, (not nb), (not nr), (not ng), (not nl)])
    weights = weights[classIndices == 1]

    criterion = DiceLoss(weights) if opt.useDice else CrossEntropyLoss2d(weights)

    indices = None
    mapLoc = None if cuda else {'cuda:0': 'cpu'}

    for transfer in transfers:
        if len(transfers) > 1:
            print("######################################################")
            print("############# Finetune with transfer: %d #############" % transfer)
            print("######################################################")
        for decay in decays:

            if len(decays) > 1:
                print("######################################################")
                print("############ Finetune with decay: %.1E ############" % decay)
                print("######################################################")

            torch.random.manual_seed(12345678)
            if cuda:
                torch.cuda.manual_seed(12345678)

            # Initiate model
            model = ROBO_UNet(noScale,planes=numPlanes,depth=depth,levels=levels,bellySize=bellySize,bellyPlanes=bellyPlanes,pool=unet)
            comp = model.get_computations()
            print(comp)
            print(sum(comp))

            if finetune:
                model.load_state_dict(torch.load(weights_path))

            if cuda:
                model = model.cuda()

            bestLoss = 0

            optimizer = torch.optim.SGD([
                        {'params': model.downPart[0:transfer].parameters(), 'lr': learning_rate*10},
                        {'params': model.downPart[transfer:].parameters()},
                        {'params': model.PB.parameters()},
                        {'params': model.upPart.parameters()},
                        {'params': model.segmenter.parameters()}
                    ],lr=learning_rate,momentum=momentum)
            #optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=momentum)
            eta_min = learning_rate/25 if opt.transfer else learning_rate/10
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=eta_min)

            for epoch in range(epochs):
                #if finetune:
                train(epoch,epochs,100)
                bestLoss = valid(epoch,epochs,bestLoss,False)
                #else:
                    #bestLoss = train(epoch,epochs,bestLoss)

            if finetune and (transfer == 0):
                model.load_state_dict(torch.load("checkpoints/bestFinetune%s%s%s%s%s%s%s.weights" % (scaleStr,unetStr,nbStr,ngStr,nrStr,nlStr,cameraSaveStr)))
                with torch.no_grad():
                    indices = pruneModelNew(model.parameters())

                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate/20)
                print("Finetuning")

                bestLoss = 0

                for epoch in range(25):
                    train(epoch, 25, 100, indices=indices)
                    bestLoss = valid(epoch,25,bestLoss,True)
