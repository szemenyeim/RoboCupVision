import torch
from torch.autograd import Variable
from torch.utils import data
from model import PB_FCN, LabelProp
from dataset import LPDataSet
from transform import Scale, ToLabel, Colorize, ToYUV, labelToPred, optFlow, updateLabels
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
    parser.add_argument("--optFlow", help="Use optical flow",
                        action="store_true")
    args = parser.parse_args()

    fineTune = args.finetuned
    pruned = args.pruned
    optflow = args.optFlow


    fineTuneStr = "Finetuned" if fineTune else ""
    pruneStr = "Pruned" if pruned else ""
    scale = 4

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

    root = "./data/LabelProp/Synthetic/"
    outDir = "./output/LabelProp/Synthetic/"
    if fineTune:
        outDir = "./output/LabelProp/Real/"
        root = "./data/LabelProp/Real/"

    valloader = data.DataLoader(LPDataSet(root, split="val", img_transform=input_transform,
                                             label_transform=target_transform),
                                  batch_size=batchSize, shuffle=False)

    numClass = 5
    kernelSize = 5
    numPlanes = 32

    modelSeg = PB_FCN(numPlanes, numClass, kernelSize, False, 0)
    model = LabelProp(numClass,numPlanes, 0)
    mapLoc = {'cuda:0': 'cpu'}
    if torch.cuda.is_available():
        modelSeg = modelSeg.cuda()
        model = model.cuda()
        mapLoc = None

    if not optflow:
        stateDict = torch.load("./pth/bestModelSeg" + fineTuneStr + pruneStr + ".pth", map_location=mapLoc)
        modelSeg.load_state_dict(stateDict)
        stateDict = torch.load("./pth/bestModelLP" + fineTuneStr + pruneStr + ".pth", map_location=mapLoc)
        model.load_state_dict(stateDict)

        saveParams("./weightsLP", model.cpu())


    running_acc = 0.0
    imgCnt = 0
    conf = torch.zeros(numClass, numClass)
    IoU = torch.zeros(numClass)
    labCnts = torch.zeros(numClass)
    model.eval()
    t=0
    bar = progressbar.ProgressBar(0, len(valloader), redirect_stdout=False)
    for i, (images, labels) in enumerate(valloader):

        currBSize = images.size()[0] * 2
        chCnt = images.size()[2] + numClass
        H = images.size()[3]
        W = images.size()[4]

        inputs = torch.FloatTensor(currBSize, chCnt, H, W)
        outputs = torch.LongTensor(currBSize, H, W)
        predClass = torch.LongTensor(currBSize, H, W)

        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
            inputs = inputs.cuda()
            outputs = outputs.cuda()
            predClass = predClass.cuda()

        if optflow:
            cnt = 0
            for img, lab in zip(images, labels):
                outputs[cnt] = lab[0]
                outputs[cnt + 1] = lab[1]
                predClass[cnt] = updateLabels( lab[1], optFlow(img[1][0], img[0][0]))
                predClass[cnt + 1] = updateLabels( lab[0], optFlow(img[0][0], img[1][0]))
        else:
            cnt = 0
            for img, lab in zip(images, labels):
                # preds = F.softmax(modelSeg(img))*2 - 1.0
                preds = labelToPred(lab, numClass)
                inputs[cnt] = torch.cat(
                    [torch.unsqueeze(img[0][0], 0), torch.unsqueeze(img[1][0], 0), torch.unsqueeze(img[0][0] - img[1][0], 0),
                     preds[1]])
                inputs[cnt + 1] = torch.cat(
                    [torch.unsqueeze(img[1][0], 0), torch.unsqueeze(img[0][0], 0), torch.unsqueeze(img[1][0] - img[0][0], 0),
                     preds[0]])
                # inputs[cnt] = torch.cat([img[0], preds[0]])
                # inputs[cnt + 1] = torch.cat([img[1], preds[1]])
                outputs[cnt] = lab[0]
                outputs[cnt + 1] = lab[1]
                cnt += 2

            beg = time.clock()
            pred = model(inputs)
            t += time.clock() - beg
            _, predClass = torch.max(pred, 1)

        bSize = inputs.data.size()[0]

        running_acc += torch.sum(predClass == outputs).item() * outSize * 100

        for j in range(bSize):
            img = Image.fromarray(Colorize(predClass.data[j]).permute(1, 2, 0).numpy().astype('uint8'))
            img.save(outDir + "%d.png" % (imgCnt + j))
        imgCnt += bSize


        maskPred = torch.zeros(numClass, bSize, int(labSize[0]), int(labSize[1])).long()
        maskLabel = torch.zeros(numClass, bSize, int(labSize[0]), int(labSize[1])).long()
        for currClass in range(numClass):
            maskPred[currClass] = predClass == currClass
            maskLabel[currClass] = outputs == currClass

        for labIdx in range(numClass):
            labCnts[labIdx] += torch.sum(maskLabel[labIdx]).item()
            for predIdx in range(numClass):
                inter = torch.sum(maskPred[predIdx] & maskLabel[labIdx]).item()
                conf[(predIdx, labIdx)] += inter
                if labIdx == predIdx:
                    if labIdx == predIdx:
                        union = torch.sum(maskPred[predIdx] | maskLabel[labIdx]).item()
                        if union == 0:
                            IoU[labIdx] += 1
                        else:
                            IoU[labIdx] += inter/union

        bar.update(i)

    bar.finish()
    t = t/imgCnt*1000
    for labIdx in range(numClass):
        for predIdx in range(numClass):
            conf[(predIdx, labIdx)] /= (labCnts[labIdx] / 100.0)
    meanClassAcc = 0.0
    for j in range(numClass):
        meanClassAcc += conf[(j, j)] / numClass

    meanIoU = torch.sum(IoU/imgCnt).item()/numClass*100*2
    print("Validation Pixel Acc: %.2f Mean Class Acc: %.2f Mean IoU: %.2f" % (running_acc / (imgCnt), meanClassAcc, meanIoU))
    print(conf)
    print(t)