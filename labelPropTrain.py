import torch
from torch.autograd import Variable
from torch.utils import data
import torch.nn.functional as F
import lr_scheduler
from model import  CrossEntropyLoss2d, LabelProp, loadModel, PB_FCN, pruneModel
from duc import SegFull
from dataset import LPDataSet
from transform import Scale, ToLabel, HorizontalFlip, VerticalFlip, ToYUV, labelToPred
from visualize import LinePlotter
from torchvision.transforms import Compose, Normalize, ToTensor, ColorJitter
from PIL import Image
import numpy as np
import random
import progressbar
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", help="Finetune the network on the real dataset",
                        action="store_true")
    parser.add_argument("--prune", help="Prune network weights",
                        action="store_true")
    args = parser.parse_args()

    fineTune = args.finetune
    pruning = args.prune
    haveCuda = torch.cuda.is_available()

    fineTuneStr = "Finetuned" if fineTune else ""
    pruneStr = "Pruned" if pruning else ""
    scale = 4

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
        #RandomNoise(),

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

    batchSize = 8 if fineTune else 8

    root = "./data/LabelProp/Real" if fineTune else "./data/LabelProp/Synthetic"

    trainloader = data.DataLoader(LPDataSet(root, split="train", img_transform=input_transform_tr,
                                             label_transform=target_transform_tr),
                                  batch_size=batchSize, shuffle=True)

    valloader = data.DataLoader(LPDataSet(root, split="val", img_transform=input_transform,
                                             label_transform=target_transform),
                                  batch_size=1, shuffle=True)


    numClass = 5
    numPlanes = 32
    kernelSize = 5
    model = LabelProp(numClass, numPlanes, 0)
    modelSeg = PB_FCN(numPlanes, numClass, kernelSize, False, 0)

    weights = torch.FloatTensor([1,6,1,3,2])
    if fineTune:
        weights = torch.FloatTensor([1,3,0.5,2,1])

    mapLoc = None if haveCuda else {'cuda:0': 'cpu'}
    if haveCuda:
        model = model.cuda()
        modelSeg = modelSeg.cuda()
        weights = weights.cuda()

    stateDict = torch.load("./pth/bestModelSeg" + fineTuneStr + pruneStr + ".pth", map_location=mapLoc)
    modelSeg.load_state_dict(stateDict)

    if fineTune:
        stateDict = torch.load("./pth/bestModelLP" + (fineTuneStr if pruning  else "") + ".pth", map_location=mapLoc)
        model.load_state_dict(stateDict)

    indices = []
    if fineTune & pruning:
        indices = pruneModel(model.parameters())

    criterion = CrossEntropyLoss2d(weights)

    epochs = 200
    lr = 2e-1
    weight_decay = 1e-3
    momentum = 0.5
    patience = 20

    if fineTune:
        lr *= 0.25
        momentum = 0.1
        epochs = 500
        patience = 50

    outSize = 1.0/(labSize[0] * labSize[1])


    optimizer = torch.optim.SGD( [ { 'params': model.parameters()}, ],
                                lr=lr, momentum=momentum,
                                weight_decay=weight_decay)

    def cb():
        print("Best Model reloaded")
        stateDict = torch.load("./pth/bestModelLP" +  fineTuneStr + pruneStr + ".pth",
                               map_location=mapLoc)
        model.load_state_dict(stateDict)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5,patience=patience,verbose=True,cb=cb)
    ploter = LinePlotter()

    bestLoss = 100
    bestAcc = 0
    bestIoU = 0
    bestTPA = 0
    bestConf = torch.zeros(numClass,numClass)

    modelSeg.eval()

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0
        running_acc = 0.0
        imgCnt = 0
        bar = progressbar.ProgressBar(0,len(trainloader),redirect_stdout=False)
        for i, (images, labels) in enumerate(trainloader):

            currBSize = images.size()[0]*2
            chCnt = images.size()[2]+numClass
            H = images.size()[3]
            W = images.size()[4]

            inputs = torch.FloatTensor(currBSize, chCnt, H, W)
            outputs = torch.LongTensor(currBSize, H, W)

            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
                inputs = inputs.cuda()
                outputs = outputs.cuda()


            cnt = 0
            for img,lab in zip(images,labels):
                #preds = F.softmax(modelSeg(img))*2 - 1.0
                preds = labelToPred(lab,numClass)
                inputs[cnt] = torch.cat([torch.unsqueeze(img[0][0],0), torch.unsqueeze(img[1][0],0), torch.unsqueeze(img[0][0] - img[1][0],0), preds[1]])
                inputs[cnt + 1] = torch.cat([torch.unsqueeze(img[1][0],0), torch.unsqueeze(img[0][0],0), torch.unsqueeze(img[1][0] - img[0][0],0), preds[0]])

                '''trans = ToPILImage()
                trans(img[0].data).show()
                trans(img[1].data).show()
                raw_input()'''

                #inputs[cnt] = torch.cat([img[0], preds[0]])
                #inputs[cnt + 1] = torch.cat([img[1], preds[1]])
                outputs[cnt] = lab[0]
                outputs[cnt+1] = lab[1]
                cnt += 2

            optimizer.zero_grad()

            pred = model(inputs)
            loss = criterion(pred,outputs)

            loss.backward()
            if pruning:
                pIdx = 0
                for param in model.parameters():
                    if param.dim() > 1:
                        param.grad[indices[pIdx]] = 0
                        pIdx += 1

            optimizer.step()

            running_loss += loss.item()
            _, predClass = torch.max(pred, 1)
            running_acc += torch.sum( predClass == outputs ).item()*outSize*100

            bSize = inputs.data.size()[0]
            imgCnt += bSize

            bar.update(i)

        bar.finish()
        print("Epoch [%d] Training Loss: %.4f Training Pixel Acc: %.2f" % (epoch+1, running_loss/(i+1), running_acc/(imgCnt)))
        ploter.plot("loss", "train", epoch+1, running_loss/(i+1))
        #print conf/float(i+1)

        running_loss = 0.0
        running_acc = 0.0
        imgCnt = 0
        conf = torch.zeros(numClass,numClass)
        IoU = torch.zeros(numClass)
        labCnts = torch.zeros(numClass)
        model.eval()
        bar = progressbar.ProgressBar(0, len(valloader), redirect_stdout=False)
        for i, (images, labels) in enumerate(valloader):

            currBSize = images.size()[0] * 2
            chCnt = images.size()[2] + numClass
            H = images.size()[3]
            W = images.size()[4]

            inputs = torch.FloatTensor(currBSize, chCnt, H, W)
            outputs = torch.LongTensor(currBSize, H, W)

            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
                inputs = inputs.cuda()
                outputs = outputs.cuda()

            cnt = 0
            for img, lab in zip(images, labels):
                #preds = F.softmax(modelSeg(img))*2 - 1.0
                preds = labelToPred(lab,numClass)
                inputs[cnt] = torch.cat([torch.unsqueeze(img[0][0],0), torch.unsqueeze(img[1][0],0), torch.unsqueeze(img[0][0] - img[1][0],0), preds[1]])
                inputs[cnt + 1] = torch.cat([torch.unsqueeze(img[1][0],0), torch.unsqueeze(img[0][0],0), torch.unsqueeze(img[1][0] - img[0][0],0), preds[0]])
                #inputs[cnt] = torch.cat([img[0], preds[0]])
                #inputs[cnt + 1] = torch.cat([img[1], preds[1]])
                outputs[cnt] = lab[0]
                outputs[cnt + 1] = lab[1]
                cnt += 2

            optimizer.zero_grad()

            pred = model(inputs)
            loss = criterion(pred, outputs)

            running_loss += loss.item()
            _, predClass = torch.max(pred, 1)
            running_acc += torch.sum(predClass == outputs).item()*outSize*100

            bSize = inputs.size()[0]
            imgCnt += bSize

            maskPred = torch.zeros(numClass,bSize,int(labSize[0]), int(labSize[1])).long()
            maskLabel = torch.zeros(numClass,bSize,int(labSize[0]), int(labSize[1])).long()
            for currClass in range(numClass):
                maskPred[currClass] = predClass.data == currClass
                maskLabel[currClass] = outputs.data == currClass

            for labIdx in range(numClass):
                labCnts[labIdx] += torch.sum(maskLabel[labIdx]).item()
                for predIdx in range(numClass):
                    inter = torch.sum(maskPred[predIdx] & maskLabel[labIdx]).item()
                    conf[(predIdx, labIdx)] += inter
                    if labIdx == predIdx:
                        union = torch.sum(maskPred[predIdx] | maskLabel[labIdx]).item()
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
        meanIoU = torch.sum(IoU/imgCnt).item() / numClass * 200
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
            bestTPA = running_acc/(imgCnt)
            bestAcc = meanClassAcc

            torch.save(model.state_dict(), "./pth/bestModelLP" + fineTuneStr + pruneStr + ".pth")

        scheduler.step(currLoss)

    print("Optimization finished Validation Loss: %.4f Total Acc: %.2f Mean Class Acc: %.2f IoU: %.2f" % (bestLoss, bestTPA, bestAcc, bestIoU))
    print(bestConf)