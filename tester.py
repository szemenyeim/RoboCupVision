import torch
from torch.autograd import Variable
from torch.utils import data
from model import PB_FCN, FCN, PB_FCN_2
from duc import SegFull
from dataset import SSDataSet
from transform import Scale, ToLabel, Colorize, ToYUV, maskLabel
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
    parser.add_argument("--pruned2", help="Use pruned2 net",
                        action="store_true")
    parser.add_argument("--noScale", help="Use VGA resolution",
                        action="store_true")
    parser.add_argument("--FCN", help="Use Normal FCN Network",
                        action="store_true")
    parser.add_argument("--v2", help="Use PB-FCNv2",
                        action="store_true")
    parser.add_argument("--noBall", help="Treat Ball as Background",
                        action="store_true")
    parser.add_argument("--noGoal", help="Treat Goal as Background",
                        action="store_true")
    parser.add_argument("--noRobot", help="Treat Robot as Background",
                        action="store_true")
    parser.add_argument("--noLine", help="Treat Lines as Background",
                        action="store_true")
    parser.add_argument("--topCam", help="Use Top Camera images only",
                        action="store_true")
    parser.add_argument("--bottomCam", help="Use Bottom Camera images only",
                        action="store_true")
    parser.add_argument("--dump", help="Dump model parameters",
                        action="store_true")
    parser.add_argument("--useCuda", help="Test on GPU",
                        action="store_true")
    args = parser.parse_args()

    fineTune = args.finetuned
    pruned = args.pruned
    pruned2 = args.pruned2
    noScale = args.noScale
    useFCN = args.FCN
    v2 = args.v2
    nb = args.noBall
    ng = args.noGoal
    nr = args.noRobot
    nl = args.noLine
    tc = args.topCam
    bc = args.bottomCam
    dump = args.dump
    useCuda = torch.cuda.is_available() if args.useCuda else False
    if useFCN: noScale = False


    fineTuneStr = "Finetuned" if fineTune else ""
    pruneStr = "Pruned" if pruned else "Pruned2" if pruned2 else ""
    FCNStr = "1" if useFCN else ""
    scaleStr = "VGA" if noScale else ""
    v2Str = "v2" if v2 else ""
    nbStr = "NoBall" if nb else ""
    ngStr = "NoGoal" if ng else ""
    nrStr = "NoRobot" if nr else ""
    nlStr = "NoLine" if nl else ""
    cameraString = "both" if tc == bc else( "top" if tc else "bottom")
    cameraLoadStr = cameraString if fineTune else ""
    scale = 1 if noScale else 4

    if nb and ng and nr and nl:
        print("You need to have at least one non-background class!")
        exit(-1)

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

    valloader = data.DataLoader(SSDataSet(root, split="val", camera=cameraString, img_transform=input_transform,
                                             label_transform=target_transform),
                                  batch_size=batchSize, shuffle=False)

    numClass = 5 - nb - ng - nr - nl
    kernelSize = 1
    numPlanes = 32
    if v2:
        model = PB_FCN_2(False, nClass=numClass)
    elif useFCN:
        model = FCN()
    else:
        model = PB_FCN(numPlanes, numClass, kernelSize, noScale, 0)

    mapLoc = {'cuda:0': 'cpu'}
    if useCuda:
        model = model.cuda()
        mapLoc = None

    stateDict = torch.load("./pth/bestModelSeg" + FCNStr + scaleStr + v2Str + nbStr + ngStr + nrStr + nlStr + cameraLoadStr + fineTuneStr + pruneStr + ".pth", map_location=mapLoc)
    model.load_state_dict(stateDict)

    if dump:
        saveParams("./weights/" + scaleStr + v2Str + nbStr + ngStr + nrStr + nlStr + cameraLoadStr, model.cpu(), "weights.dat" if pruned else "weights2.dat", v2)
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
        images = images.float()
        if useCuda:
            images = images.float().cuda()
            labels = labels.cuda()
        labels = maskLabel(labels, nb, nr, ng, nl)

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
        maskTarget = torch.zeros(numClass, bSize, int(labSize[0]), int(labSize[1])).long()
        for currClass in range(numClass):
            maskPred[currClass] = predClass == currClass
            maskTarget[currClass] = labels == currClass

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