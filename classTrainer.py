import torch
from torch.autograd import Variable
from torch.utils import data
from model import DownSampler, Classifier
import lr_scheduler
from visualize import LinePlotter
from torchvision.transforms import Compose, Normalize, ToTensor, RandomHorizontalFlip
from transform import ToYUV, RandomBrightness, RandomColor, RandomContrast, RandomHue
import torchvision.datasets as datasets
import progressbar
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--noScale", help="Use VGA resolution",
                    action="store_true")
args = parser.parse_args()
noScale = args.noScale
VGAStr = "VGA" if noScale else ""

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

batchSize = 32

trainloader = data.DataLoader(datasets.ImageFolder("./data/Classification/train/", transform=input_transform_tr),
                              batch_size=batchSize, shuffle=True)

valloader = data.DataLoader(datasets.ImageFolder("./data/Classification/val", transform=input_transform),
                              batch_size=batchSize, shuffle=True)

numClass = 5
numFeat = 32
modelConv = DownSampler(numFeat, noScale)
poolFact = 2 if noScale else 4
modelClass = Classifier(numFeat*2,numClass,poolFact)
weights = torch.ones(numClass)
if torch.cuda.is_available():
    modelConv = modelConv.cuda()
    modelClass = modelClass.cuda()
    weights = weights.cuda()

criterion = torch.nn.CrossEntropyLoss(weights)

mapLoc = None if torch.cuda.is_available() else {'cuda:0': 'cpu'}

epochs = 200
lr = 1e-2
weight_decay = 1e-3
momentum = 0.5

def cb():
    print "Best Model reloaded"
    stateDict = torch.load("./pth/bestModel" + VGAStr + ".pth",
                           map_location=mapLoc)
    modelConv.load_state_dict(stateDict)
    stateDict = torch.load("./pth/bestClass" + VGAStr + ".pth",
                           map_location=mapLoc)
    modelClass.load_state_dict(stateDict)

optimizer = torch.optim.SGD( [
                                { 'params': modelConv.parameters()},
                                { 'params': modelClass.parameters()}, ],
                             lr=lr, momentum=momentum, weight_decay=weight_decay )
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5,patience=20,verbose=True,threshold=1e-3, cb=cb)

ploter = LinePlotter()

bestLoss = 100
bestAcc = 0
bestTest = 0

for epoch in range(epochs):

    modelConv.train()
    modelClass.train()
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

        final = modelConv(images)[0] if noScale else modelConv(images)[1]
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
    bar = progressbar.ProgressBar(0, len(valloader), redirect_stdout=False)
    for i, (images, labels) in enumerate(valloader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images)
            labels = Variable(labels)

        final = modelConv(images)[0] if noScale else modelConv(images)[1]
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
        torch.save(modelConv.state_dict(), "./pth/bestModel" + VGAStr + ".pth")

    scheduler.step(running_loss/(i+1))

print("Finished: Best Validation Loss: %.4f Best Validation Acc: %.2f"  % (bestLoss, bestAcc))

