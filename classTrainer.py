import torch
import torch.nn as nn
from torch.utils import data
from model import DownSampler, Classifier, PB_FCN_2
import lr_scheduler
from visualize import LinePlotter
from torchvision.transforms import Compose, Normalize, ToTensor, RandomHorizontalFlip, ColorJitter
from transform import ToYUV
import torchvision.datasets as datasets
import progressbar
import numpy as np
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--noScale", help="Use VGA resolution",
                        action="store_true")
    parser.add_argument("--v2", help="Use PB-FCNv2",
                        action="store_true")
    parser.add_argument("--ballOnly", help="Train Binary segmenter for ball",
                        action="store_true")
    args = parser.parse_args()
    noScale = args.noScale
    v2 = args.v2
    bo = args.ballOnly
    VGAStr = "VGA" if noScale else ""
    v2Str = "v2" if v2 else ""
    boStr = "bo" if bo else ""


    input_transform = Compose([
        ToYUV(),
        ToTensor(),
        Normalize([.5, 0, 0], [.5, .5, .5]),

    ])

    input_transform_tr = Compose([
        RandomHorizontalFlip(),
        ColorJitter(brightness=0.5,contrast=0.5,saturation=0.4,hue=0.3),
        ToYUV(),
        ToTensor(),
        Normalize([.5, 0, 0], [.5, .5, .5]),

    ])

    seed = 12345678
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    batchSize = 64 if v2 else 32

    trainloader = data.DataLoader(datasets.ImageFolder("./data/Classification/train/", transform=input_transform_tr),
                                  batch_size=batchSize, shuffle=True, num_workers=6)

    valloader = data.DataLoader(datasets.ImageFolder("./data/Classification/val", transform=input_transform),
                                  batch_size=batchSize, shuffle=True,num_workers=6)

    numClass = 2 if bo else 5
    numFeat = 32
    dropout = 0.1
    poolFact = 2 if noScale else 4

    class PB_FCN(nn.Module):
        def __init__(self):
            super(PB_FCN,self).__init__()
            self.conv = DownSampler(numFeat, noScale, dropout)
            self.classif = Classifier(numFeat*2,numClass,poolFact)
        def forward(self, x):
            ind = 0 if noScale else 1
            return self.classif(self.conv(x)[ind])

    model = PB_FCN_2(True,nClass=numClass) if v2 else PB_FCN()

    weights = torch.ones(numClass)
    if torch.cuda.is_available():
        model = model.cuda()
        weights = weights.cuda()

    criterion = torch.nn.CrossEntropyLoss(weights)

    mapLoc = None if torch.cuda.is_available() else {'cuda:0': 'cpu'}

    epochs = 200
    lr = 1e-2
    weight_decay = 1e-5
    momentum = 0.9

    def cb():
        print("Best Model reloaded")
        stateDict = torch.load("./pth/bestModel" + VGAStr + v2Str + boStr + ".pth",
                               map_location=mapLoc)
        model.load_state_dict(stateDict)

    optimizer = torch.optim.SGD( [{ 'params': model.parameters()}, ],
                                 lr=lr, momentum=momentum, weight_decay=weight_decay )
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5,patience=20,verbose=True,threshold=1e-3, cb=cb)

    ploter = LinePlotter("RoboCup")

    bestLoss = 100
    bestAcc = 0
    bestTest = 0

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0
        running_acc = 0.0
        imgCnt = 0
        conf = torch.zeros(numClass,numClass).long()
        bar = progressbar.ProgressBar(0,len(trainloader),redirect_stdout=False)
        for i, (images, labels) in enumerate(trainloader):
            if torch.cuda.is_available():
                images = images.float().cuda()
                labels = labels.cuda()
            if bo:
                labels[labels>1] = 0

            optimizer.zero_grad()

            pred = torch.squeeze(model(images))
            loss = criterion(pred,labels)

            loss.backward()
            optimizer.step()

            bSize = images.size()[0]
            imgCnt += bSize

            running_loss += loss.item()
            _, predClass = torch.max(pred, 1)
            running_acc += torch.sum( predClass == labels ).item()*100

            for j in range(bSize):
                conf[(predClass[j],labels[j])] += 1

            bar.update(i)

        bar.finish()
        print("Epoch [%d] Training Loss: %.4f Training Acc: %.2f" % (epoch+1, running_loss/(i+1), running_acc/(imgCnt)))
        #ploter.plot("loss", "train", epoch+1, running_loss/(i+1))

        running_loss = 0.0
        running_acc = 0.0
        imgCnt = 0
        conf = torch.zeros(numClass,numClass).long()
        model.eval()
        bar = progressbar.ProgressBar(0, len(valloader), redirect_stdout=False)
        for i, (images, labels) in enumerate(valloader):
            if torch.cuda.is_available():
                images = images.float().cuda()
                labels = labels.cuda()

            pred = torch.squeeze(model(images))
            loss = criterion(pred, labels)

            bSize = images.size()[0]
            imgCnt += bSize

            running_loss += loss.item()
            _, predClass = torch.max(pred, 1)
            running_acc += torch.sum( predClass == labels ).item()*100

            for j in range(bSize):
                conf[(predClass[j],labels[j])] += 1

            bar.update(i)

        bar.finish()
        print("Epoch [%d] Validation Loss: %.4f Validation Acc: %.2f" % (epoch+1, running_loss/(i+1), running_acc/(imgCnt)))
        #ploter.plot("loss", "val", epoch+1, running_loss/(i+1))

        if bestLoss > running_loss/(i+1):
            bestLoss = running_loss/(i+1)
            bestAcc = running_acc/(imgCnt)
            print(conf)
            torch.save(model.state_dict(), "./pth/bestModel" + VGAStr + v2Str + boStr + ".pth")

        scheduler.step(running_loss/(i+1))

    if not v2:
        cb()
        torch.save(model.conv.state_dict(), "./pth/bestModel" + VGAStr + v2Str + boStr + ".pth")
    print("Finished: Best Validation Loss: %.4f Best Validation Acc: %.2f"  % (bestLoss, bestAcc))

