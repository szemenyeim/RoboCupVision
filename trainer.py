import torch
from torch.autograd import Variable
from torch.utils import data
import lr_scheduler
from model import  CrossEntropyLoss2d, PB_FCN, loadModel, pruneModel
from duc import SegFull
from dataset import SSDataSet
from transform import Scale, ToLabel, HorizontalFlip, VerticalFlip, ToYUV, RandomNoise, RandomBrightness, RandomColor, RandomContrast, RandomHue
from visualize import LinePlotter
from torchvision.transforms import Compose, Normalize, ToTensor
from PIL import Image
import numpy as np
import random
import progressbar
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--finetune", help="Finetune the network on the real dataset",
                    action="store_true")
parser.add_argument("--prune", help="Prune network weights",
                    action="store_true")
parser.add_argument("--deep", help="Use Very deep model for reference",
                    action="store_true")
parser.add_argument("--noScale", help="Use VGA resolution",
                    action="store_true")
args = parser.parse_args()

fineTune = args.finetune
pruning = args.prune
deep = args.deep
noScale = args.noScale
haveCuda = torch.cuda.is_available()

fineTuneStr = "Finetuned" if fineTune else ""
pruneStr = "Pruned" if pruning else ""
deepStr = "Deep" if deep else ""
scaleStr = "VGA" if noScale else ""
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
    RandomColor(),
    RandomContrast(),
    RandomBrightness(),
    RandomHue(labSize[1],labSize[0]),
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

batchSize = 8 if (fineTune or noScale) else 32

root = "./data/FinetuneHorizon" if fineTune else "./data"

trainloader = data.DataLoader(SSDataSet(root, split="train", img_transform=input_transform_tr,
                                         label_transform=target_transform_tr),
                              batch_size=batchSize, shuffle=True)

valloader = data.DataLoader(SSDataSet(root, split="val", img_transform=input_transform,
                                         label_transform=target_transform),
                              batch_size=1, shuffle=True)


numClass = 5
numPlanes = 32
kernelSize = 1
if deep:
    model = SegFull(numClass)
else:
    model = PB_FCN(numPlanes, numClass, kernelSize, noScale)

weights = torch.FloatTensor([1,6,1.5,3,3])
if fineTune:
    weights = torch.FloatTensor([1,4,1,4,1.5])

indices = []
mapLoc = None if haveCuda else {'cuda:0': 'cpu'}
if haveCuda:
    model = model.cuda()
    weights = weights.cuda()

loadModel(model,noScale,deep,fineTune,pruning,mapLoc)

if fineTune & pruning:
    indices = pruneModel(model.parameters())

criterion = CrossEntropyLoss2d(weights)

epochs = 100 if noScale else 200
lr = 1e-1
weight_decay = 1e-3
momentum = 0.5
patience = 10 if noScale else 20

if fineTune:
    lr *= 0.1
    momentum = 0.1
    epochs = 250 if noScale else 500
    patience = 25 if noScale else 50

outSize = 1.0/(labSize[0] * labSize[1])


optimizer = torch.optim.SGD( [ { 'params': model.parameters()}, ],
                            lr=lr, momentum=momentum,
                            weight_decay=weight_decay)

def cb():
    print "Best Model reloaded"
    stateDict = torch.load("./pth/bestModelSeg" + scaleStr + deepStr + fineTuneStr + pruneStr + ".pth",
                           map_location=mapLoc)
    model.load_state_dict(stateDict)

scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5,patience=patience,verbose=True,cb=cb)
ploter = LinePlotter()

bestLoss = 100
bestAcc = 0
bestIoU = 0
bestTAcc = 0
bestConf = torch.zeros(numClass,numClass)

for epoch in range(epochs):

    model.train()
    running_loss = 0.0
    running_acc = 0.0
    imgCnt = 0
    bar = progressbar.ProgressBar(0,len(trainloader),redirect_stdout=False)
    for i, (images, labels) in enumerate(trainloader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images)
            labels = Variable(labels)

        optimizer.zero_grad()

        pred = model(images)
        loss = criterion(pred,labels)

        loss.backward()
        if pruning:
            pIdx = 0
            for param in model.parameters():
                if param.dim() > 1:
                    param.grad.data[indices[pIdx]] = 0
                    pIdx += 1

        optimizer.step()

        running_loss += loss.data[0]
        _, predClass = torch.max(pred, 1)
        running_acc += torch.sum( predClass.data == labels.data )*outSize*100

        bSize = images.data.size()[0]
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
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images)
            labels = Variable(labels)

        pred = model(images)
        loss = criterion(pred,labels)

        running_loss += loss.data[0]
        _, predClass = torch.max(pred, 1)
        running_acc += torch.sum(predClass.data == labels.data)*outSize*100

        bSize = images.data.size()[0]
        imgCnt += bSize

        maskPred = torch.zeros(numClass,bSize,int(labSize[0]), int(labSize[1])).long()
        maskLabel = torch.zeros(numClass,bSize,int(labSize[0]), int(labSize[1])).long()
        for currClass in range(numClass):
            maskPred[currClass] = predClass.data == currClass
            maskLabel[currClass] = labels.data == currClass

        for labIdx in range(numClass):
            labCnts[labIdx] += torch.sum(maskLabel[labIdx])
            for predIdx in range(numClass):
                inter = torch.sum(maskPred[predIdx] & maskLabel[labIdx])
                conf[(predIdx, labIdx)] += inter
                if labIdx == predIdx:
                    union = torch.sum(maskPred[predIdx] | maskLabel[labIdx])
                    if union == 0:
                        IoU[labIdx] += 1
                    else:
                        IoU[labIdx] += float(inter)/(float(union))

        bar.update(i)

    bar.finish()
    for labIdx in range(numClass):
        for predIdx in range(numClass):
            conf[(predIdx, labIdx)] /= (labCnts[labIdx] / 100.0)
    meanClassAcc = 0.0
    meanIoU = torch.sum(IoU/imgCnt) / numClass * 100
    currLoss = running_loss/(i+1)
    for j in range(numClass):
        meanClassAcc += conf[(j,j)]/numClass
    print("Epoch [%d] Validation Loss: %.4f Validation Pixel Acc: %.2f Mean Class Acc: %.2f IoU: %.2f" %
          (epoch+1, running_loss/(i+1), running_acc/(imgCnt), meanClassAcc, meanIoU))
    ploter.plot("loss", "val", epoch+1, running_loss/(i+1))

    if bestLoss > currLoss:
        conf[conf<0.001] = 0
        print conf
        bestConf = conf
        bestLoss = currLoss
        bestIoU = meanIoU
        bestAcc = meanClassAcc
        bestTAcc = running_acc/(imgCnt)

        torch.save(model.state_dict(), "./pth/bestModelSeg" + scaleStr + deepStr + fineTuneStr + pruneStr + ".pth")

    scheduler.step(currLoss)

print("Optimization finished Validation Loss: %.4f Pixel Acc: %.2f Mean Class Acc: %.2f IoU: %.2f" % (bestLoss, bestTAcc, bestAcc, bestIoU))
print bestConf

