import torch
from torch import nn, optim
import cv2
import numpy
import torch.fft
from complexPyTorch.complexLayers import ComplexConvTranspose2d,ComplexConv2d
from complexPyTorch.complexFunctions import complex_relu

class CDown(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.COV1 = nn.Sequential(ComplexConv2d(in_channels, out_channels, 3, stride=2, padding=1))
    def forward(self, x):
        out1 = complex_relu((self.COV1(x)))
        return out1

class CUp(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.COV1=nn.Sequential(ComplexConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1))
    def forward(self, x):
        out1 = complex_relu((self.COV1(x)))
        return out1

class CUp2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.COV1=nn.Sequential(ComplexConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1))
    def forward(self, x):
        out1 =self.COV1(x)
        return out1


class CCNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.netdown1 = CDown(1, 4)
        self.netdown2 = CDown(4, 8)
        self.netdown3 = CDown(8, 16)
        self.netdown4 = CDown(16, 32)

        self.netup4 = CUp(32, 16)
        self.netup3 = CUp(16, 8)
        self.netup2 = CUp(8, 4)
        self.netup1 = CUp2(4, 1)

    def forward(self, x):
        out1 = self.netdown1(x)
        out2 = self.netdown2(out1)
        out3 = self.netdown3(out2)
        out4 = self.netdown4(out3)

        out17 = self.netup4(out4)
        out18 = self.netup3(out17+out3)
        out19 = self.netup2(out18 + out2)
        out20 = self.netup1(out19 + out1)
        out20 = torch.squeeze(out20)
        predictphase = torch.atan2(out20.imag, out20.real)


        return predictphase

class CCNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.netdown1 = CDown(1, 4)
        self.netdown2 = CDown(4, 8)
        self.netdown3 = CDown(8, 16)

        self.netup3 = CUp(16, 8)
        self.netup2 = CUp(8, 4)
        self.netup1 = CUp2(4, 1)

    def forward(self, x):
        out1 = self.netdown1(x)
        out2 = self.netdown2(out1)
        out3 = self.netdown3(out2)

        out18 = self.netup3(out3)
        out19 = self.netup2(out18 + out2)
        out20 = self.netup1(out19 + out1)
        out20 = torch.squeeze(out20)
        holophase = torch.atan2(out20.imag, out20.real)
        return holophase

