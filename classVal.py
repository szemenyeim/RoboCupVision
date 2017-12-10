import torch
from torch.autograd import Variable
from torch.utils import data
from model import DownSampler, Classifier, BNNL, BNNMC
import lr_scheduler
from visualize import LinePlotter
from torchvision.transforms import Compose, Normalize, ToTensor, RandomHorizontalFlip
from transform import ToYUV, RandomBrightness, RandomColor, RandomContrast, RandomHue
import torchvision.datasets as datasets
import progressbar
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--hessL", help="Use BNN-L from Hess et. al.",
                    action="store_true")
parser.add_argument("--hessMC", help="Use BNN-M-C from Hess et. al.",
                    action="store_true")
args = parser.parse_args()
hessL = args.hessL
hessMC = args.hessMC

input_transform = Compose([
    ToYUV(),
    ToTensor(),
    Normalize([.5, 0, 0], [.5, .5, .5]),

])

input_transform_tr = Compose([
    RandomHorizontalFlip(),
    RandomColor(),
    RandomContrast(),
    RandomBrightness(),
    RandomHue(32,32),
    ToYUV(),
    ToTensor(),
    Normalize([.5, 0, 0], [.5, .5, .5]),

])

seed = 12345678
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

batchSize = 64

trainDataRoot = "./data/Classification/trainBig/"

trainloader = data.DataLoader(datasets.ImageFolder(trainDataRoot, transform=input_transform_tr),
                              batch_size=batchSize, shuffle=True)

valloader = data.DataLoader(datasets.ImageFolder("./data/Classification/test", transform=input_transform),
                              batch_size=batchSize, shuffle=True)

numClass = 4
numFeat = 32
dropout = 0.25
modelConv = DownSampler(numFeat, False, dropout)
modelClass = Classifier(numFeat*2,numClass,4)
modelHess = BNNL()
if hessMC:
    modelHess = BNNMC()
weights = torch.ones(numClass)
if torch.cuda.is_available():
    modelConv = modelConv.cuda()
    modelClass = modelClass.cuda()
    modelHess = modelHess.cuda()
    weights = weights.cuda()

criterion = torch.nn.CrossEntropyLoss(weights)

mapLoc = None if torch.cuda.is_available() else {'cuda:0': 'cpu'}

epochs = 80
lr = 1e-2
weight_decay = 5e-4
momentum = 0.9


def cb():
    print "Best Model reloaded"
    if hessMC:
        stateDict = torch.load("./pth/bestModelHessMC" + ".pth",
                               map_location=mapLoc)
        modelHess.load_state_dict(stateDict)
    elif hessL:
        stateDict = torch.load("./pth/bestModelHessL" + ".pth",
                               map_location=mapLoc)
        modelHess.load_state_dict(stateDict)
    else:
        stateDict = torch.load("./pth/bestModelB" + ".pth",
                               map_location=mapLoc)
        modelConv.load_state_dict(stateDict)
        stateDict = torch.load("./pth/bestClassB" + ".pth",
                               map_location=mapLoc)
        modelClass.load_state_dict(stateDict)

optimizer = torch.optim.SGD( [
                                { 'params': modelConv.parameters()},
                                { 'params': modelClass.parameters()},
                                { 'params': modelHess.parameters()}, ],
                             lr=lr, momentum=momentum, weight_decay=weight_decay )
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.2,patience=10,verbose=True,threshold=1e-3,cb=cb)

ploter = LinePlotter()

bestLoss = 100
bestAcc = 0
bestTest = 0

for epoch in range(epochs):

    modelConv.train()
    modelClass.train()
    modelHess.train()
    running_loss = 0.0
    running_acc = 0.0
    imgCnt = 0
    conf = torch.zeros(numClass,numClass)
    bar = progressbar.ProgressBar(0,len(trainloader),redirect_stdout=False)
    for i, (images, labels) in enumerate(trainloader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images)
            labels = Variable(labels)

        optimizer.zero_grad()

        if hessL or hessMC:
            pred = torch.squeeze(modelHess(images))
        else:
            final = modelConv(images)[1]
            pred = torch.squeeze(modelClass(final))
        loss = criterion(pred,labels)

        loss.backward()
        optimizer.step()

        bSize = images.data.size()[0]
        imgCnt += bSize

        running_loss += loss.data[0]
        _, predClass = torch.max(pred, 1)
        running_acc += torch.sum( predClass.data == labels.data )*100

        for j in range(bSize):
            conf[(predClass.data[j],labels.data[j])] += 1

        bar.update(i)

    bar.finish()
    print("Epoch [%d] Training Loss: %.4f Training Acc: %.2f" % (epoch+1, running_loss/(i+1), running_acc/(imgCnt)))
    #ploter.plot("loss", "train", epoch+1, running_loss/(i+1))

    running_loss = 0.0
    running_acc = 0.0
    imgCnt = 0
    conf = torch.zeros(numClass,numClass)
    modelConv.eval()
    modelClass.eval()
    modelHess.eval()
    bar = progressbar.ProgressBar(0, len(valloader), redirect_stdout=False)
    for i, (images, labels) in enumerate(valloader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images)
            labels = Variable(labels)

        optimizer.zero_grad()

        if hessL or hessMC:
            pred = torch.squeeze(modelHess(images))
        else:
            final = modelConv(images)[1]
            pred = torch.squeeze(modelClass(final))
        loss = criterion(pred, labels)

        bSize = images.data.size()[0]
        imgCnt += bSize

        running_loss += loss.data[0]
        _, predClass = torch.max(pred, 1)
        running_acc += torch.sum(predClass.data == labels.data)*100

        for j in range(bSize):
            conf[(predClass.data[j],labels.data[j])] += 1

        bar.update(i)

    bar.finish()
    print("Epoch [%d] Validation Loss: %.4f Validation Acc: %.2f" % (epoch+1, running_loss/(i+1), running_acc/(imgCnt)))
    #ploter.plot("loss", "val", epoch+1, running_loss/(i+1))

    if bestAcc < running_acc/(imgCnt):
        bestLoss = running_loss/(i+1)
        bestAcc = running_acc/(imgCnt)
        print conf
        if hessL:
            torch.save(modelConv.state_dict(), "./pth/bestModelHessL" + ".pth")
        elif hessMC:
            torch.save(modelConv.state_dict(), "./pth/bestModelHessMC" + ".pth")
        else:
            torch.save(modelConv.state_dict(), "./pth/bestModelB" + ".pth")
            torch.save(modelClass.state_dict(), "./pth/bestClassB" + ".pth")

    scheduler.step(running_loss/(i+1))

print("Finished: Best Validation Loss: %.4f Best Validation Acc: %.2f" % (bestLoss, bestAcc))

