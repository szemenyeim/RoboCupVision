import torch
from torch.autograd import Variable
from torch.utils import data
from model import PB_FCN, FCN, PB_FCN_2
from duc import SegFull
from dataset import SSDataSet
from transform import Scale, ToLabel, Colorize, ToYUV
from torchvision.transforms import Compose, Normalize, ToTensor
from PIL import Image
import progressbar
from paramSave import saveParams
import argparse
import time

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned", help="Use finetuned net and dataset",
                        action="store_true")
    parser.add_argument("--pruned", help="Use pruned net",
                        action="store_true")
    parser.add_argument("--deep", help="Use Very deep model for reference",
                        action="store_true")
    parser.add_argument("--noScale", help="Use VGA resolution",
                        action="store_true")
    parser.add_argument("--FCN", help="Use Normal FCN Network",
                        action="store_true")
    parser.add_argument("--v2", help="Use PB-FCNv2",
                        action="store_true")
    parser.add_argument("--dump", help="Dump model parameters",
                        action="store_true")
    parser.add_argument("--useCuda", help="Test on GPU",
                        action="store_true")
    args = parser.parse_args()

    fineTune = args.finetuned
    pruned = args.pruned
    deep = args.deep
    noScale = args.noScale
    useFCN = args.FCN
    v2 = args.v2
    dump = args.dump
    useCuda = torch.cuda.is_available() if args.useCuda else False
    if useFCN: noScale = False


    fineTuneStr = "Finetuned" if fineTune else ""
    pruneStr = "Pruned" if pruned else ""
    deepStr = "Deep" if deep else ""
    FCNStr = "1" if useFCN else ""
    scaleStr = "VGA" if noScale else ""
    v2Str = "v2" if v2 else ""
    scale = 1 if noScale else 4

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
    if deep:
        model = SegFull(numClass)
    elif v2:
        model = PB_FCN_2(False)
    elif useFCN:
        model = FCN()
    else:
        model = PB_FCN(numPlanes, numClass, kernelSize, noScale, 0)

    mapLoc = {'cuda:0': 'cpu'}
    if useCuda:
        model = model.cuda()
        mapLoc = None

    stateDict = torch.load("./pth/bestModelSeg" + FCNStr + scaleStr + v2Str + deepStr + fineTuneStr + pruneStr + ".pth", map_location=mapLoc)
    model.load_state_dict(stateDict)

    if dump:
        saveParams("./weights" + ("VGA" if noScale else ""), model.cpu())
        if useCuda:
            model = model.cuda()

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
        if useCuda:
            images = images.cuda()
            labels = labels.cuda()

        beg = time.clock()
        pred = model(images)
        t += time.clock() - beg

        _, predClass = torch.max(pred, 1)
        running_acc += torch.sum(predClass == labels).item() * outSize * 100

        bSize = images.data.size()[0]
        for j in range(bSize):
            img = Image.fromarray(Colorize(predClass.data[j]).permute(1, 2, 0).numpy().astype('uint8'))
            img.save(outDir + "%d.png" % (imgCnt + j))
        imgCnt += bSize


        maskPred = torch.zeros(numClass, bSize, int(labSize[0]), int(labSize[1])).long()
        maskLabel = torch.zeros(numClass, bSize, int(labSize[0]), int(labSize[1])).long()
        for currClass in range(numClass):
            maskPred[currClass] = predClass == currClass
            maskLabel[currClass] = labels == currClass

        for imgInd in range(bSize):
            for labIdx in range(numClass):
                labCnts[labIdx] += torch.sum(maskLabel[labIdx, imgInd]).item()
                for predIdx in range(numClass):
                    inter = torch.sum(maskPred[predIdx, imgInd] & maskLabel[labIdx, imgInd]).item()
                    conf[(predIdx, labIdx)] += inter
                    if labIdx == predIdx:
                        union = torch.sum(maskPred[predIdx, imgInd] | maskLabel[labIdx, imgInd]).item()
                        if union == 0:
                            IoU[labIdx] += 1
                        else:
                            IoU[labIdx] += inter / union

        bar.update(i)

    bar.finish()
    t = t/imgCnt*1000
    for labIdx in range(numClass):
        for predIdx in range(numClass):
            conf[(predIdx, labIdx)] /= (labCnts[labIdx] / 100.0)
    meanClassAcc = 0.0
    for j in range(numClass):
        meanClassAcc += conf[(j, j)] / numClass

    meanIoU = torch.sum(IoU/imgCnt).item()/numClass*100
    print("Validation Pixel Acc: %.2f Mean Class Acc: %.2f Mean IoU: %.2f" % (running_acc / (imgCnt), meanClassAcc, meanIoU))
    print(conf)
    print(t)