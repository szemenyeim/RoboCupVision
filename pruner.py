import torch
from torch.utils import data
import lr_scheduler
from model import  CrossEntropyLoss2d, PB_FCN, loadModel, pruneModel2, PB_FCN_2, getParamSize
from duc import SegFull
from dataset import SSDataSet
from transform import Scale, ToLabel, HorizontalFlip, VerticalFlip, ToYUV
from visualize import LinePlotter
from torchvision.transforms import Compose, Normalize, ToTensor, ColorJitter
from PIL import Image
import numpy as np
import random
import progressbar
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--deep", help="Use Very deep model for reference",
                        action="store_true")
    parser.add_argument("--noScale", help="Use VGA resolution",
                        action="store_true")
    parser.add_argument("--v2", help="Use PB-FCNv2",
                        action="store_true")
    parser.add_argument("--ballOnly", help="Train Binary segmenter for ball",
                        action="store_true")
    args = parser.parse_args()

    deep = args.deep
    noScale = args.noScale
    v2 = args.v2
    bo = args.ballOnly
    haveCuda = torch.cuda.is_available()

    fineTuneStr = "Finetuned"
    pruneStr = "Pruned2"
    deepStr = "Deep" if deep else ""
    scaleStr = "VGA" if noScale else ""
    v2Str = "v2" if v2 else ""
    boStr = "bo" if bo else ""
    scale = 1 if noScale else 4

    labSize = (480.0/scale, 640.0/scale)

    input_transform = Compose([
        Scale(scale, Image.BILINEAR),
        ToYUV(),
        ToTensor(),
        Normalize([.5, 0, 0], [.5, .5, .5]),

    ])
    target_transform = Compose([
        Scale(scale, Image.NEAREST),
        ToTensor(),
        ToLabel(),
    ])

    input_transform_tr = Compose([
        Scale(scale, Image.BILINEAR),
        HorizontalFlip(),
        VerticalFlip(),
        ColorJitter(brightness=0.5,contrast=0.5,saturation=0.4,hue=0.3),
        ToYUV(),
        ToTensor(),
        Normalize([.5, 0, 0], [.5, .5, .5]),

    ])
    target_transform_tr = Compose([
        Scale(scale, Image.NEAREST),
        HorizontalFlip(),
        VerticalFlip(),
        ToTensor(),
        ToLabel(),
    ])

    seed = 12345678
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if haveCuda:
        torch.cuda.manual_seed(seed)

    batchSize = 8

    root = "./data/FinetuneHorizon"

    camera = "bottom" if bo else "top"

    trainloader = data.DataLoader(SSDataSet(root, split="train", camera=camera, img_transform=input_transform_tr,
                                             label_transform=target_transform_tr),
                                  batch_size=batchSize, shuffle=True, num_workers=6)

    valloader = data.DataLoader(SSDataSet(root, split="val", camera=camera, img_transform=input_transform,
                                             label_transform=target_transform),
                                  batch_size=batchSize, shuffle=True, num_workers=6)


    numClass = 2 if bo else 5
    numPlanes = 32
    kernelSize = 1
    if deep:
        model = SegFull(numClass)
    elif v2:
        model = PB_FCN_2(False, nClass=numClass)
    else:
        model = PB_FCN(numPlanes, numClass, kernelSize, noScale, 0)

    weights = torch.FloatTensor([1,4,1.5,5,1.5])
    if bo:
        weights = weights[0:2]

    indices = []
    mapLoc = None if haveCuda else {'cuda:0': 'cpu'}
    if haveCuda:
        model = model.cuda()
        weights = weights.cuda()

    loadModel(model,noScale,v2,bo,deep,True,True,mapLoc)

    criterion = CrossEntropyLoss2d(weights)

    lr = 1e-2
    weight_decay = 1e-3
    momentum = 0.1
    epochs = 10
    iters = 10

    outSize = 1.0/(labSize[0] * labSize[1])



    def cb():
        print("Best Model reloaded")
        stateDict = torch.load("./pth/bestModelSeg" + scaleStr  + v2Str + boStr + deepStr + fineTuneStr + pruneStr + ".pth",
                               map_location=mapLoc)
        model.load_state_dict(stateDict)

    ploter = LinePlotter("RoboCup")

    for iter in range(iters):

        limit = (iter + 1) * epochs
        optimizer = torch.optim.SGD([{'params': model.parameters()}, ],
                                    lr=lr, momentum=momentum,
                                    weight_decay=weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,limit,1e-3)

        bestLoss = 100
        bestAcc = 0
        bestIoU = 0
        bestTAcc = 0
        bestConf = torch.zeros(numClass, numClass)
        
        pruneAm = 0.085 if bo else 0.08
        pruneThreshLow = 500 if v2 else 1000
        pruneThreshHigh = 15000 if v2 else 50000

        if iter > 0:
            cb()
        with torch.no_grad():
            indices = pruneModel2(model.parameters(),(iter+1)*pruneAm,pruneThreshLow,pruneThreshHigh)

        for epoch in range(limit):

            scheduler.step()
            model.train()

            running_loss = 0.0
            running_acc = 0.0
            imgCnt = 0
            bar = progressbar.ProgressBar(0,len(trainloader),redirect_stdout=False)
            for i, (images, labels) in enumerate(trainloader):
                if torch.cuda.is_available():
                    images = images.float().cuda()
                    labels = labels.cuda()
                if bo:
                    labels[labels>1] = 0

                optimizer.zero_grad()

                pred = model(images)
                loss = criterion(pred,labels)

                loss.backward()
                pIdx = 0
                for param in model.parameters():
                    if param.dim() > 1:
                        if param.grad is not None:
                            param.grad[indices[pIdx]] = 0.0
                        pIdx += 1

                optimizer.step()

                running_loss += loss.item()
                _, predClass = torch.max(pred, 1)
                running_acc += torch.sum( predClass.data == labels.data ).item()*outSize*100

                bSize = images.size()[0]
                imgCnt += bSize

                bar.update(i)

            bar.finish()
            print("Epoch [%d] Training Loss: %.4f Training Pixel Acc: %.2f" % (epoch+1, running_loss/(i+1), running_acc/(imgCnt)))
            ploter.plot("loss", "train", epoch+1, running_loss/(i+1))

            running_loss = 0.0
            running_acc = 0.0
            imgCnt = 0
            conf = torch.zeros(numClass,numClass)
            IoU = torch.zeros(numClass)
            labCnts = torch.zeros(numClass)
            model.eval()
            bar = progressbar.ProgressBar(0, len(valloader), redirect_stdout=False)
            for i, (images, labels) in enumerate(valloader):
                if torch.cuda.is_available():
                    images = images.float().cuda()
                    labels = labels.cuda()
                if bo:
                    labels[labels>1] = 0

                pred = model(images)
                loss = criterion(pred,labels)

                running_loss += loss.item()
                _, predClass = torch.max(pred, 1)
                running_acc += torch.sum(predClass.data == labels.data).item()*outSize*100

                bSize = images.size()[0]
                imgCnt += bSize

                maskPred = torch.zeros(numClass,bSize,int(labSize[0]), int(labSize[1])).long()
                maskLabel = torch.zeros(numClass,bSize,int(labSize[0]), int(labSize[1])).long()
                for currClass in range(numClass):
                    maskPred[currClass] = predClass == currClass
                    maskLabel[currClass] = labels == currClass

                for imgInd in range(bSize):
                    for labIdx in range(numClass):
                        labCnts[labIdx] += torch.sum(maskLabel[labIdx,imgInd]).item()
                        for predIdx in range(numClass):
                            inter = torch.sum(maskPred[predIdx,imgInd] & maskLabel[labIdx,imgInd]).item()
                            conf[(predIdx, labIdx)] += inter
                            if labIdx == predIdx:
                                union = torch.sum(maskPred[predIdx,imgInd] | maskLabel[labIdx,imgInd]).item()
                                if union == 0:
                                    IoU[labIdx] += 1
                                else:
                                    IoU[labIdx] += inter/union

                bar.update(i)

            bar.finish()
            for labIdx in range(numClass):
                for predIdx in range(numClass):
                    conf[(predIdx, labIdx)] /= (labCnts[labIdx] / 100.0)
            meanClassAcc = 0.0
            meanIoU = torch.sum(IoU/imgCnt).item() / numClass * 100
            currLoss = running_loss/(i+1)
            for j in range(numClass):
                meanClassAcc += conf[(j,j)]/numClass
            print("Epoch [%d] Validation Loss: %.4f Validation Pixel Acc: %.2f Mean Class Acc: %.2f IoU: %.2f" %
                  (epoch+1, running_loss/(i+1), running_acc/(imgCnt), meanClassAcc, meanIoU))
            ploter.plot("loss", "val", epoch+1, running_loss/(i+1))

            if bestLoss > currLoss:
                conf[conf<0.001] = 0
                print(conf)
                bestConf = conf
                bestLoss = currLoss
                bestIoU = meanIoU
                bestAcc = meanClassAcc
                bestTAcc = running_acc/(imgCnt)

                torch.save(model.state_dict(), "./pth/bestModelSeg" + scaleStr + v2Str + boStr + deepStr + fineTuneStr + pruneStr + ".pth")

    print("Optimization finished Validation Loss: %.4f Pixel Acc: %.2f Mean Class Acc: %.2f IoU: %.2f" % (bestLoss, bestTAcc, bestAcc, bestIoU))
    print(bestConf)

