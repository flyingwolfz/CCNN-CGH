import torch
from torch import nn, optim
import cv2
import numpy
import torch.fft
from utils.pytorch_prototyping.pytorch_prototyping import Conv2dSame, Unet
import math
from tqdm import trange
from scipy import io
import time
z=200
num=30
rangege=700
rangegege=100
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
Hreal=Hreal*a
Himage=Himage*a
H=torch.complex(Hreal,Himage)
H2real=H2real*a
H2image=H2image*a
H2=torch.complex(H2real,H2image)
H=H.cuda()
H2=H2.cuda()

def rect_to_polar(real, imag):
    """Converts the rectangular complex representation to polar"""
    mag = torch.pow(real**2 + imag**2, 0.5)
    ang = torch.atan2(imag, real)
    return mag, ang


def polar_to_rect(mag, ang):
    """Converts the polar complex representation to rectangular"""
    real = mag * torch.cos(ang)
    imag = mag * torch.sin(ang)
    return real, imag

class InitialPhaseUnet(nn.Module):
    """computes the initial input phase given a target amplitude"""
    def __init__(self, num_down=8, num_features_init=32, max_features=256,
                 norm=nn.BatchNorm2d):
        super(InitialPhaseUnet, self).__init__()

        net = [Unet(1, 1, num_features_init, num_down, max_features,
                    use_dropout=False,
                    upsampling_mode='transpose',
                    norm=norm,
                    outermost_linear=True),
               nn.Hardtanh(-math.pi, math.pi)]

        self.net = nn.Sequential(*net)

    def forward(self, amp):
        out_phase = self.net(amp)
        return out_phase


class FinalPhaseOnlyUnet(nn.Module):
    """computes the final SLM phase given a naive SLM amplitude and phase"""
    def __init__(self, num_down=8, num_features_init=32, max_features=256,
                 norm=nn.BatchNorm2d, num_in=2):
        super(FinalPhaseOnlyUnet, self).__init__()

        net = [Unet(num_in, 1, num_features_init, num_down, max_features,
                    use_dropout=False,
                    upsampling_mode='transpose',
                    norm=norm,
                    outermost_linear=True),
               nn.Hardtanh(-math.pi, math.pi)]

        self.net = nn.Sequential(*net)

    def forward(self, amp_phase):
        out_phase = self.net(amp_phase)
        return out_phase


class holonet(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_phase = InitialPhaseUnet(4, 16)
        self.final_phase_only = FinalPhaseOnlyUnet(4, 16)

    def forward(self, x):
        init_phase = self.initial_phase(x)
        real, imag = polar_to_rect(x, init_phase)
        target_complex = torch.complex(real, imag)
        quan = torch.fft.fftn(target_complex)
        quan = quan * H2
        slmfield = torch.fft.ifftn(quan)
        amp, ang = rect_to_polar(slmfield.real, slmfield.imag)
        slm_amp_phase = torch.cat((amp, ang), -3)
        slm_amp_phase = slm_amp_phase.float()
        out=self.final_phase_only(slm_amp_phase)

        return out
#lr=0.0001
lr=0.001
model = holonet()
criterion = nn.MSELoss()

if torch.cuda.is_available():
    model.cuda()


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
        target_amp = target_amp / 255.0

        target_amp = target_amp.cuda()

        output = model(target_amp)

        output = torch.squeeze(output)

        grayreal = torch.cos(output)
        grayimage = torch.sin(output)
        gray = torch.complex(grayreal, grayimage)
        quan = torch.fft.fftn(gray)
        quan2 = quan * H
        final = torch.fft.ifftn(quan2)
        final = torch.abs(final)
        target_amp = target_amp.double()
        target_amp = torch.squeeze(target_amp)
        loss = criterion(final,target_amp)
        currenttloss = currenttloss + loss.cpu().data.numpy()
        optimizier.zero_grad()
        loss.backward()
        optimizier.step()
    tl.append(currenttloss / rangege)
    print('trainloss:',currenttloss / rangege)
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
        target_amp = target_amp / 255.0

        target_amp = target_amp.cuda()

        output = model(target_amp)

        output = torch.squeeze(output)

        grayreal = torch.cos(output)
        grayimage = torch.sin(output)
        gray = torch.complex(grayreal, grayimage)
        quan = torch.fft.fftn(gray)
        quan2 = quan * H
        final = torch.fft.ifftn(quan2)
        final = torch.abs(final)
        target_amp = target_amp.double()
        target_amp = torch.squeeze(target_amp)
        loss = criterion(final, target_amp)
        currentloss = currentloss + loss.cpu().data.numpy()
        if kk==38:
         finalpic = final
         finalpic = finalpic / torch.max(finalpic)
         c = k
         b = '2\\' + str(c)
         imgpath = b + '.png'
         amp = numpy.uint8(finalpic.cpu().data.numpy() * 255)
         cv2.imwrite(imgpath, amp)
      l.append(currentloss / rangegege)
      print('validloss:',currentloss / rangegege)
    time.sleep(1)
pass


torch.save(model.state_dict(), 'holonetstate.pth')
l=numpy.mat(l)
io.savemat('avgloss.mat',{'avgloss': l})
tl=numpy.mat(tl)
io.savemat('avgtloss.mat',{'avgtloss': tl})

