import torch
from torch.autograd import Variable
from torch.utils import data
from model import PB_FCN, LabelProp
from dataset import SSDataSet
from transform import Scale, ToLabel, Colorize, ToYUV, labelToPred
from torchvision.transforms import Compose, Normalize, ToTensor
from PIL import Image
import progressbar
from paramSave import saveParams
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--finetuned", help="Use finetuned net and dataset",
                    action="store_true")
parser.add_argument("--pruned", help="Use pruned net",
                    action="store_true")
args = parser.parse_args()

fineTune = args.finetuned
pruned = args.pruned


fineTuneStr = "Finetuned" if fineTune else ""
pruneStr = "Pruned" if pruned else ""
scale =  4

input_transform = Compose([
    Scale(scale, Image.BILINEAR),
    ToYUV(),
    ToTensor(),
    Normalize([.5, .0, .0], [.5, .5, .5]),

])
target_transform = Compose([
    Scale(scale, Image.NEAREST),
    ToTensor(),
    ToLabel(),
])

labSize = (480.0/scale, 640.0/scale)
outSize = 1.0/(labSize[0] * labSize[1])

batchSize = 1

root = "./data/"
outDir = "./output/"
if fineTune:
    outDir = "./output/FinetuneHorizon/"
    root = "./data/FinetuneHorizon"

valloader = data.DataLoader(SSDataSet(root, split="val", img_transform=input_transform,
                                         label_transform=target_transform),
                              batch_size=batchSize, shuffle=False)

numClass = 5
kernelSize = 1
numPlanes = 32
model = PB_FCN(numPlanes, numClass, kernelSize, False)
modelLP = LabelProp(numClass,numPlanes)
mapLoc = {'cuda:0': 'cpu'}
if torch.cuda.is_available():
    model = model.cuda()
    mapLoc = None

stateDict = torch.load("./pth/bestModelSeg" + fineTuneStr + pruneStr + ".pth", map_location=mapLoc)
model.load_state_dict(stateDict)

stateDict = torch.load("./pth/bestModelLP" + fineTuneStr + pruneStr + ".pth", map_location=mapLoc)
modelLP.load_state_dict(stateDict)

saveParams("./weightsLP", modelLP.cpu())

running_acc = 0.0
imgCnt = 0
conf = torch.zeros(numClass, numClass)
IoU = torch.zeros(numClass)
labCnts = torch.zeros(numClass)
model.eval()
#print model
t = 0
bar = progressbar.ProgressBar(0, len(valloader), redirect_stdout=False)
for i, (images, labels) in enumerate(valloader):
    if torch.cuda.is_available():
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        inputs = Variable(torch.cuda.FloatTensor(batchSize, 8, 120, 160))
    else:
        images = Variable(images)
        labels = Variable(labels)
        inputs = Variable(torch.FloatTensor(batchSize, 8, 120, 160))

    beg = time.clock()
    pred = model(images)
    _, predClass = torch.max(pred, 1)
    for j in range(1):
        lab = labelToPred(labels, numClass)
        chY = torch.unsqueeze(torch.unsqueeze(images[0][0], 0), 0)
        inputs = torch.cat( [chY, chY, chY-chY,lab], 1 )
        pred = modelLP(inputs)
        _, predClass = torch.max(pred, 1)

    t += time.clock() - beg

    running_acc += torch.sum(predClass.data == labels.data) * outSize * 100

    bSize = images.data.size()[0]
    for j in range(bSize):
        img = Image.fromarray(Colorize(predClass.data[j]).permute(1, 2, 0).numpy().astype('uint8'))
        img.save(outDir + "%d.png" % (imgCnt + j))
    imgCnt += bSize


    maskPred = torch.zeros(numClass, bSize, int(labSize[0]), int(labSize[1])).long()
    maskLabel = torch.zeros(numClass, bSize, int(labSize[0]), int(labSize[1])).long()
    for currClass in range(numClass):
        maskPred[currClass] = predClass.data == currClass
        maskLabel[currClass] = labels.data == currClass

    for labIdx in range(numClass):
        labCnts[labIdx] += torch.sum(maskLabel[labIdx])
        for predIdx in range(numClass):
            inter = torch.sum(maskPred[predIdx] & maskLabel[labIdx])
            conf[(predIdx, labIdx)] += inter
            if labIdx == predIdx:
                if labIdx == predIdx:
                    union = torch.sum(maskPred[predIdx] | maskLabel[labIdx])
                    if union == 0:
                        IoU[labIdx] += 1
                    else:
                        IoU[labIdx] += float(inter)/(float(union))

    bar.update(i)

bar.finish()
t = t/imgCnt*1000
for labIdx in range(numClass):
    for predIdx in range(numClass):
        conf[(predIdx, labIdx)] /= (labCnts[labIdx] / 100.0)
meanClassAcc = 0.0
for j in range(numClass):
    meanClassAcc += conf[(j, j)] / numClass

meanIoU = torch.sum(IoU/imgCnt)/numClass*100
print("Validation Pixel Acc: %.2f Mean Class Acc: %.2f Mean IoU: %.2f" % (running_acc / (imgCnt), meanClassAcc, meanIoU))
print conf
print t