import torch
from torch import nn, optim
import cv2
import numpy
import torch.fft
from complexPyTorch.complexLayers import ComplexConvTranspose2d,ComplexConv2d
from complexPyTorch.complexFunctions import complex_relu
from tqdm import trange
import time
from scipy import io
z=200
num=30
rangege=700
rangegege=100
pitch=0.008
wavelength=0.000639
n = 1072
m = 1920
slm_res = (n, m)



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

H = (-1) * (2 * numpy.pi / wavelength) * z * torch.sqrt(1 - (wavelength * fX) ** 2 - (wavelength * fY) ** 2)
H2 = (2 * numpy.pi / wavelength) * z * torch.sqrt(1 - (wavelength * fX) ** 2 - (wavelength * fY) ** 2)
Hreal = torch.cos(H)
Himage = torch.sin(H)
H2real = torch.cos(H2)
H2image = torch.sin(H2)
xlimit = 1 / torch.sqrt((2 * 1 / m / pitch * z) ** 2 + 1) / wavelength
ylimit = 1 / torch.sqrt((2 * 1 / n / pitch * z) ** 2 + 1) / wavelength
a = (abs(fX) < xlimit) & (abs(fY) < ylimit)
a = a.numpy()
a = a + 0
Hreal = Hreal * a
Himage = Himage * a
H = torch.complex(Hreal, Himage)
H2real = H2real * a
H2image = H2image * a
H2 = torch.complex(H2real, H2image)
H = H.cuda()
H2 = H2.cuda()


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
        slmfield=torch.complex(slmfield.real.float(),slmfield.imag.float())
        out=self.net2(slmfield)

        return out
lr=0.001
model = proposednet()
criterion = nn.MSELoss()

if torch.cuda.is_available():
    model=model.cuda()
    criterion=criterion.cuda()

optvars = [{'params': model.parameters()}]
optimizier = torch.optim.Adam(optvars, lr=lr)

trainpath='E:\\DIV2K\\DIV2K_train_HR'
validpath='E:\\DIV2K\\DIV2K_valid_HR'
l=[]
tl=[]

for k in trange(num):
    currenttloss = 0
    currentloss = 0
    for kk in range(rangege):
        c = 100 + kk
        b = '\\0' + str(c)
        imgpath = trainpath + b + '.png'
        img = cv2.imread(imgpath)
        img2 = cv2.resize(img, (1920,1072))
        gray = cv2.split(img2)[2]
        gray = numpy.reshape(gray, (1, 1, 1072,1920))
        target_amp = torch.from_numpy(gray)
        target_amp = target_amp.cuda()
        target_amp = target_amp / 255.0
        target_amp_complex = torch.complex(target_amp, torch.zeros_like(target_amp))
        target_amp=target_amp.double()
        target_amp=torch.squeeze(target_amp)

        target_amp = target_amp.cuda()
        target_amp_complex = target_amp_complex.cuda()

        output = model(target_amp_complex)

        grayreal = torch.cos(output)
        grayimage = torch.sin(output)

        gray = torch.complex(grayreal, grayimage)
        quan = torch.fft.fftn(gray)
        quan2 = quan * H
        final = torch.fft.ifftn(quan2)
        final = torch.abs(final)
        loss = criterion(final,target_amp)
        currenttloss = currenttloss + loss.cpu().data.numpy()
        optimizier.zero_grad()
        loss.backward()
        optimizier.step()
        #time.sleep(0.001)

    tl.append(currenttloss / rangege)
    print('trainloss:', currenttloss / rangege)
    c = k
    b = '1\\' + str(c)
    imgpath = b + '.png'
    finalpic = final
    finalpic = finalpic / torch.max(finalpic)
    amp = numpy.uint8(finalpic.cpu().data.numpy() * 255)
    cv2.imwrite(imgpath, amp)
    with torch.no_grad():
        for kk in range(rangegege):
            c = 801 + kk
            b = '\\0' + str(c)
            imgpath = validpath + b + '.png'
            img = cv2.imread(imgpath)
            img2 = cv2.resize(img, (1920,1072))
            gray = cv2.split(img2)[2]
            gray = numpy.reshape(gray, (1, 1, 1072,1920))
            target_amp = torch.from_numpy(gray)
            target_amp = target_amp.cuda()
            target_amp = target_amp / 255.0
            target_amp_complex = torch.complex(target_amp, torch.zeros_like(target_amp))
            target_amp = target_amp.double()
            target_amp = torch.squeeze(target_amp)

            target_amp = target_amp.cuda()
            target_amp_complex = target_amp_complex.cuda()

            output = model(target_amp_complex)

            grayreal = torch.cos(output)
            grayimage = torch.sin(output)

            gray = torch.complex(grayreal, grayimage)
            quan = torch.fft.fftn(gray)
            quan2 = quan * H
            final = torch.fft.ifftn(quan2)
            final = torch.abs(final)
            loss = criterion(final, target_amp)
            currentloss = currentloss + loss.cpu().data.numpy()
            if kk == 38:
                finalpic = final
                finalpic = finalpic / torch.max(finalpic)
                c = k
                b = '2\\' + str(c)
                imgpath = b + '.png'
                amp = numpy.uint8(finalpic.cpu().data.numpy() * 255)
                cv2.imwrite(imgpath, amp)
        l.append(currentloss / rangegege)
        print('validloss:', currentloss / rangegege)
    time.sleep(1)
pass

torch.save(model.state_dict(), 'proposedstate.pth')
l=numpy.mat(l)
io.savemat('avgloss.mat',{'avgloss': l})
tl=numpy.mat(tl)
io.savemat('avgtloss.mat',{'avgtloss': tl})