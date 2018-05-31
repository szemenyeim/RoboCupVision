import torch
from torch import nn


class Concat(nn.Module):
    def __init__(self):
        super(Concat,self).__init__()

    def forward(self, in1,in2):
        return torch.cat([in1,in2],1)

class Shortcut(nn.Module):
    def __init__(self):
        super(Shortcut,self).__init__()

    def forward(self, in1,in2):
        return in1+in2

if __name__ == "__main__":


    dataC1 = torch.randn((1,4,32,32))
    dataC2 = torch.randn((1,4,32,32))
    dataF = torch.randn(32)

    layers = nn.ModuleList()

    layers.add_module("FC",nn.Linear(32,16))
    layers.add_module("BN",nn.BatchNorm2d(4,track_running_stats=False))
    layers.add_module("Cat",Concat())
    layers.add_module("Short",Shortcut())
    layers.add_module("Reorg",nn.PixelShuffle(2))
    layers.add_module("SM",nn.Softmax2d())
    layers.add_module("MP",nn.MaxPool2d(2,2))
    layers.add_module("AP",nn.AvgPool2d(2,2))

    layers.add_module("C1",nn.Conv2d(4,8,kernel_size=3,stride=1,padding=1,dilation=1))
    layers.add_module("C2",nn.Conv2d(4,8,kernel_size=3,stride=2,padding=1,dilation=1))
    layers.add_module("C3",nn.Conv2d(4,8,kernel_size=3,stride=1,padding=2,dilation=2))
    layers.add_module("C4",nn.Conv2d(4,8,kernel_size=3,stride=2,padding=2,dilation=2))
    layers.add_module("C5",nn.Conv2d(4,8,kernel_size=(3,1),stride=1,padding=(1,0),dilation=1))
    layers.add_module("C6",nn.Conv2d(4,8,kernel_size=(3,1),stride=2,padding=(1,0),dilation=1))
    layers.add_module("C7",nn.Conv2d(4,8,kernel_size=(3,1),stride=1,padding=(2,0),dilation=(2,1)))
    layers.add_module("C8",nn.Conv2d(4,8,kernel_size=(3,1),stride=2,padding=(2,0),dilation=(2,1)))
    layers.add_module("C9",nn.Conv2d(4,8,kernel_size=(1,3),stride=1,padding=(0,1),dilation=1))
    layers.add_module("C10",nn.Conv2d(4,8,kernel_size=(1,3),stride=2,padding=(0,1),dilation=1))
    layers.add_module("C11",nn.Conv2d(4,8,kernel_size=(1,3),stride=1,padding=(0,2),dilation=(1,2)))
    layers.add_module("C12",nn.Conv2d(4,8,kernel_size=(1,3),stride=2,padding=(0,2),dilation=(1,2)))
    layers.add_module("C13",nn.Conv2d(4,8,kernel_size=1,stride=1,padding=0,dilation=1))

    layers.add_module("TrC",nn.ConvTranspose2d(4,8,kernel_size=3,stride=3,padding=1,dilation=1,output_padding=1))

    torch.save(layers,"./tests/testLayers.pth")
    torch.save(dataC1, "./tests/dataC1.pth")
    torch.save(dataC2, "./tests/dataC2.pth")
    torch.save(dataF, "./tests/dataF.pth")

    for name,layer in layers.named_modules():
        if name == "":
            continue
        print(name)
        if name == "FC":
            out = layer(dataF)
        elif name == "Cat" or name == "Short":
            out = layer(dataC1,dataC2)
        else:
            out = layer(dataC1)
        torch.save(out,"./tests/out" + name + ".pth")