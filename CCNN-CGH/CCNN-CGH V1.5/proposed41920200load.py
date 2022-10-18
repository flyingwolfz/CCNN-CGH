import torch
from torch import nn, optim
import cv2
import numpy
import torch.fft
from complexPyTorch.complexLayers import ComplexConvTranspose2d,ComplexConv2d
from complexPyTorch.complexFunctions import complex_relu
import time
import math
from torchsummary import complexsummary
from skimage.metrics import structural_similarity as ssim
z=200 #juli

pitch=0.008
wavelength=0.000639
n = 1072
m = 1920


x = numpy.linspace(-n//2, n//2-1, n)
y = numpy.linspace(-m//2, m//2-1, m)
x=torch.from_numpy(x)
y=torch.from_numpy(y)
n=numpy.array(n)
m=numpy.array(m)
n = torch.from_numpy(n)
m = torch.from_numpy(m)

v = 1 / (n * pitch)
u = 1 / (m * pitch)
fx = x * v
fy = y * u

fX, fY = torch.meshgrid(fx, fy)

H = (-1)*(2*numpy.pi/wavelength) * z * torch.sqrt(1 - (wavelength*fX)**2 - (wavelength*fY)**2)
H2 = (2*numpy.pi/wavelength) * z * torch.sqrt(1 - (wavelength*fX)**2 - (wavelength*fY)**2)
Hreal=torch.cos(H)
Himage=torch.sin(H)
H2real=torch.cos(H2)
H2image=torch.sin(H2)
xlimit=1/torch.sqrt((2*1/m/pitch*z)**2+1)/wavelength
ylimit=1/torch.sqrt((2*1/n/pitch*z)**2+1)/wavelength
a = (abs(fX) < xlimit) & (abs(fY) < ylimit)
a=a.numpy()
a=a+0
Hreal = Hreal * a
Himage = Himage * a
Hreal=Hreal.float()
Himage=Himage.float()
H = torch.complex(Hreal, Himage)
H2real = H2real * a
H2image = H2image * a
H2real=H2real.float()
H2image=H2image.float()
H2 = torch.complex(H2real, H2image)
H = H.cuda()
H2 = H2.cuda()


def psnr(img1, img2):
    mse = numpy.mean((img1 - img2) ** 2 )
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

class Down4(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.COV1 = nn.Sequential(ComplexConv2d(in_channels, out_channels, 3, stride=2, padding=1))
    def forward(self, x):
        out1 = complex_relu((self.COV1(x)))
        return out1

class Up4(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.COV1=nn.Sequential(ComplexConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1))

    def forward(self, x):
        out1 = complex_relu((self.COV1(x)))
        return out1

class Up5(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.COV1=nn.Sequential(ComplexConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1))
    def forward(self, x):
        out1 =self.COV1(x)
        return out1

class net5(nn.Module):
    def __init__(self):
        super().__init__()
        self.netdown1 = Down4(1, 4)
        self.netdown2 = Down4(4, 8)
        self.netdown3 = Down4(8, 16)
        self.netdown4 = Down4(16, 32)

        self.netup4 = Up4(32, 16)
        self.netup3 = Up4(16, 8)
        self.netup2 = Up4(8, 4)
        self.netup1 = Up5(4, 1)


    def forward(self, x):
        out1=self.netdown1(x)
        out2=self.netdown2(out1)
        out3=self.netdown3(out2)
        out4=self.netdown4(out3)

        out17 = self.netup4(out4)
        out18 = self.netup3(out17+out3)
        out19 = self.netup2(out18+out2)
        out20 = self.netup1(out19+out1)
        phase = torch.atan2(out20.real, out20.imag)
        init_complex = torch.complex(x.real * torch.cos(phase), x.real * torch.sin(phase))

        return init_complex
class net6(nn.Module):
    def __init__(self):
        super().__init__()
        self.netdown1 = Down4(1, 4)
        self.netdown2 = Down4(4, 8)
        self.netdown3 = Down4(8, 16)

        self.netup3 = Up4(16, 8)
        self.netup2 = Up4(8, 4)
        self.netup1 = Up5(4, 1)


    def forward(self, x):
        out1=self.netdown1(x)
        out2=self.netdown2(out1)
        out3=self.netdown3(out2)

        out18 = self.netup3(out3)
        out19 = self.netup2(out18 + out2)
        out20 = self.netup1(out19 + out1)
        out20 = torch.squeeze(out20)
        holophase = torch.atan2(out20.real, out20.imag)

        return holophase
class proposednet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = net5()
        self.net2 = net6()

    def forward(self, x):
        init_complex = self.net1(x)
        quan = torch.fft.fftn(init_complex)
        quan = quan * H2
        slmfield = torch.fft.ifftn(quan)

        out=self.net2(slmfield)

        return out
lr=0.0001
model = proposednet()
#path='E:\\DIV2K\\DIV2K_train_HR'
path='E:\\DIV2K\\DIV2K_valid_HR'

model.load_state_dict(torch.load('proposedstate.pth'))

if torch.cuda.is_available():
    model.cuda()
c = 839
b = '\\0' + str(c)
imgpath = path + b + '.png'
#imgpath ='long.png'
#imgpath='D:\\zcl\\python\\pytorchsci\\dog\\2.jpg'
#imgpath ='triss.png'
img = cv2.imread(imgpath)
img2 = cv2.resize(img, (1920,1072))
gray = cv2.split(img2)[2]
#gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
#gray = cv2.split(img2)[1]
cv2.imwrite('2.png', gray)
gray = numpy.reshape(gray, (1, 1, 1072,1920))

target_amp = torch.from_numpy(gray)
target_amp = target_amp / 255.0
target_amp = torch.sqrt(target_amp)
target_amp = target_amp.cuda()
target_ampc = torch.complex(target_amp, torch.zeros_like(target_amp))
#complexsummary(model.cuda(), input_size=(1, 1072,1920), batch_size=1)
output = model(target_ampc)
print('success')
time_start=time.time()
with torch.no_grad():
 for k in range(100):
  output = model(target_ampc)

time_end=time.time()
print('totally cost',(time_end-time_start)/100.0)
holo=output/2.0/3.14159+0.5
#holo= numpy.uint8(ab.cpu().data.numpy()*255)



holo= numpy.uint8(holo.cpu().data.numpy()*255)
cv2.imwrite('h.png', holo)
grayreal = torch.cos(output)
grayimage = torch.sin(output)
gray = torch.complex(grayreal, grayimage)

quan = torch.fft.fftn(gray)
quan2 = quan * H
final = torch.fft.ifftn(quan2)
final = torch.abs(final)
finalpic=final
ffinal = final*final

final = final.cpu().numpy()
target_amp=torch.squeeze(target_amp)
target_amp = target_amp.cpu().numpy()
psnrr = psnr(final, target_amp)
print('psnr:',psnrr)
ssimm = ssim(target_amp, final)
print('ssim:',ssimm)
a = (ffinal>1.0)
#ffinal[a]=1.0
#a=a.numpy()
#a=a+0
#Hreal=Hreal*a


ffinal=ffinal/torch.max(ffinal)
amp= numpy.uint8(ffinal.cpu().data.numpy()*255)
cv2.imwrite('1.png', amp)
cv2.imshow('amp', amp)

#cv2.waitKey(0)