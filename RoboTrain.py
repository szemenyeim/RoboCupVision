import torch
from torch.utils import data
from torch.nn import CrossEntropyLoss
import lr_scheduler
from model import  ROBOLoss, ROBO, pruneModel, ErrorMeasures
from dataset import ODDataSet
from transform import ToYUV
from visualize import LinePlotter
from torchvision.transforms import Compose, Normalize, ToTensor, ColorJitter, Resize
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import random
import progressbar
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", help="Pretrain on the classification dataset",
                        action="store_true")
    parser.add_argument("--finetune", help="Finetune the network on the real dataset",
                        action="store_true")
    parser.add_argument("--prune", help="Prune network weights",
                        action="store_true")
    parser.add_argument("--deep", help="Use Very deep model for reference",
                        action="store_true")
    parser.add_argument("--noScale", help="Use VGA resolution",
                        action="store_true")
    args = parser.parse_args()

    preTrain = args.pretrain
    fineTune = args.finetune
    pruning = args.prune
    deep = args.deep
    noScale = args.noScale
    haveCuda = torch.cuda.is_available()

    preTrainStr = "Pretrained" if preTrain else ""
    fineTuneStr = "Finetuned" if fineTune else ""
    pruneStr = "Pruned" if pruning else ""
    deepStr = "Deep" if deep else ""
    scaleStr = "VGA" if noScale else ""

    labSize = (32,32) if preTrain else (640,480) if noScale else (160,120)

    input_transform = Compose([
        Resize(labSize, Image.BILINEAR),
        ToYUV(),
        ToTensor(),
        Normalize([.5, 0, 0], [.5, .5, .5]),

    ])

    input_transform_tr = Compose([
        Resize(labSize, Image.BILINEAR),
        ColorJitter(brightness=0.5,contrast=0.5,saturation=0.4,hue=0.3),
        ToYUV(),
        ToTensor(),
        Normalize([.5, 0, 0], [.5, .5, .5]),

    ])

    seed = 12345678
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if haveCuda:
        torch.cuda.manual_seed(seed)

    batchSize = 8 if (fineTune or noScale) else 64 if preTrain else 32

    root = "./data/FinetuneHorizon" if fineTune else "./data"

    if preTrain:
        trainSet = ImageFolder("./data/Classification/train",input_transform_tr)
        valSet = ImageFolder("./data/Classification/val",input_transform)

    else:
        trainSet = ODDataSet(root, split="train", img_transform=input_transform_tr,label_transform=None)
        valSet = ODDataSet(root, split="val", img_transform=input_transform, label_transform=None, bbMean=trainSet.means, bbStd=trainSet.std)

    Tensor = torch.cuda.FloatTensor if haveCuda else torch.FloatTensor
    BBMeans = None if preTrain else Tensor(trainSet.means)
    BBSTD = None if preTrain else Tensor(trainSet.std)

    sampler = None #data.sampler.SubsetRandomSampler(range(64))

    trainloader = data.DataLoader(trainSet, batch_size=batchSize, sampler=sampler, shuffle=True, num_workers=6)
    valloader = data.DataLoader(valSet, batch_size=batchSize, sampler=sampler, shuffle=False, num_workers=6)

    numClass = 5
    numPlanes = 32 if deep else 8
    levels = 3 if deep else 1
    depth = 10 if deep or noScale else 7

    model = ROBO(numPlanes,depth,levels,numClass=numClass,classify=preTrain)

    indices = []
    mapLoc = None if haveCuda else {'cuda:0': 'cpu'}
    if haveCuda:
        model = model.cuda()

    if not preTrain:
        stateDict = torch.load("./pth/bestROBO" + scaleStr + deepStr + ("" if fineTune else "Pretrained") + ("Finetuned" if pruning else "") + ".pth",
                               map_location=mapLoc)
        model.load_state_dict(stateDict)


    if fineTune & pruning:
        indices = pruneModel(model.parameters())

    criterion = CrossEntropyLoss() if preTrain else ROBOLoss()
    measure = ErrorMeasures(preTrain,BBMeans,BBSTD)

    epochs = 100 if noScale else 200
    lr = 1e-1
    weight_decay = 1e-5
    momentum = 0.1
    patience = 10 if noScale else 20

    if fineTune:
        lr *= 0.1
        momentum = 0.1
        epochs = 250 if noScale else 500
        patience = 25 if noScale else 50


    optimizer = torch.optim.SGD( [ { 'params': model.parameters()}, ],
                                lr=lr, momentum=momentum,
                                weight_decay=weight_decay)

    def cb():
        print("Best Model reloaded")
        stateDict = torch.load("./pth/bestROBO" + preTrainStr + scaleStr + deepStr + fineTuneStr + pruneStr + ".pth",
                               map_location=mapLoc)
        model.load_state_dict(stateDict)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5,patience=patience,verbose=True,cb=cb)
    ploter = LinePlotter()

    bestLoss = 100
    bestAcc = 0
    bestIoU = 0
    bestConf = torch.zeros(numClass,numClass)

    for epoch in range(epochs):

        torch.set_grad_enabled(True)
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        IoU = 0.0
        bbCnt = 0
        imgCnt = 0
        bar = progressbar.ProgressBar(0,len(trainloader),redirect_stdout=False)
        for i, (images, labels) in enumerate(trainloader):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

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

            running_loss += loss.item()

            correct, conf, currIoU, cnt = measure.forward(pred,labels)
            running_acc += correct
            IoU += currIoU
            bbCnt += cnt

            bSize = images.size()[0]
            imgCnt += bSize

            bar.update(i)

        bar.finish()
        print("Epoch [%d] Training Loss: %.4f Training Acc: %.2f IoU: %.2f" % (epoch+1, running_loss/(i+1), running_acc/(imgCnt)*100, IoU/bbCnt*100))
        #ploter.plot("loss", "train", epoch+1, running_loss/(i+1))

        torch.set_grad_enabled(False)
        running_loss = 0.0
        running_acc = 0.0
        imgCnt = 0
        bbCnt = 0
        conf = torch.zeros(numClass,numClass).long() if preTrain else torch.zeros(4,3).long()
        IoU = 0.0

        model.eval()
        bar = progressbar.ProgressBar(0, len(valloader), redirect_stdout=False)
        for i, (images, labels) in enumerate(valloader):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            pred = model(images)
            loss = criterion(pred,labels)

            running_loss += loss.item()

            correct, confusion, currIoU, cnt = measure.forward(pred,labels)
            running_acc += correct
            conf += confusion
            IoU += currIoU
            bbCnt += cnt

            bSize = images.size()[0]
            imgCnt += bSize

            bar.update(i)

        bar.finish()
        currLoss = running_loss/(i+1)
        print("Epoch [%d] Validation Loss: %.4f Validation Acc: %.2f IoU: %.2f" %
              (epoch+1, running_loss/(i+1), running_acc/(imgCnt)*100,IoU/bbCnt*100))
        #ploter.plot("loss", "val", epoch+1, running_loss/(i+1))

        if bestLoss > currLoss:
            conf[conf<0.001] = 0
            print(conf)
            bestConf = conf
            bestLoss = currLoss
            bestIoU = IoU/bbCnt*100
            bestAcc = running_acc/(imgCnt)

            torch.save(model.state_dict(), "./pth/bestROBO" + preTrainStr + scaleStr + deepStr + fineTuneStr + pruneStr + ".pth")

        scheduler.step(currLoss)

    print("Optimization finished Validation Loss: %.4f Pixel Acc: %.2f Mean Class Acc: %.2f IoU: %.2f" % (bestLoss, bestTAcc, bestAcc, bestIoU))
    print(bestConf)

