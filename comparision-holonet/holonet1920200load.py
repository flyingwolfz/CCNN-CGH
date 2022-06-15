import torch
from torch import nn, optim
import cv2
import numpy
import torch.fft
import time
from utils.pytorch_prototyping.pytorch_prototyping import Conv2dSame, Unet
import math
from skimage.metrics import structural_similarity as ssim
z=200

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


def psnr(img1, img2):
    mse = numpy.mean((img1 - img2) ** 2 )
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

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


#path='E:\\DIV2K\\DIV2K_train_HR'
path='E:\\DIV2K\\DIV2K_valid_HR'
model = holonet()
model.load_state_dict(torch.load('holonetstate.pth'))

if torch.cuda.is_available():
    model.cuda()
c = 879
b = '\\0' + str(c)
imgpath = path + b + '.png'
imgpath = 'triss.png'
img = cv2.imread(imgpath)
img2 = cv2.resize(img, (1920,1072))
gray = cv2.split(img2)[2]

cv2.imwrite('2.png', gray)
gray = numpy.reshape(gray, (1, 1, 1072,1920))

target_amp = torch.from_numpy(gray)
target_amp = target_amp / 255.0
target_amp = target_amp.cuda()


output = model(target_amp)
time_start=time.time()
with torch.no_grad():
  for k in range(10):
    output = model(target_amp)
time_end=time.time()
print('totally cost',(time_end-time_start)/10.0)


output = torch.squeeze(output)
target_amp = torch.squeeze(target_amp)
phase=output.cpu().data.numpy()
phase=phase/math.pi/2.0+0.5
phase=phase*255.0
phase=numpy.uint8(phase)
cv2.imwrite('3.png', phase)
cv2.imshow('phase', phase)

grayreal = torch.cos(output)
grayimage = torch.sin(output)
gray = torch.complex(grayreal, grayimage)
quan = torch.fft.fftn(gray)
quan2 = quan * H
final = torch.fft.ifftn(quan2)
final = torch.abs(final)
finalpic=final

final = final.cpu().numpy()
target_amp = target_amp.cpu().numpy()
psnrr = psnr(final, target_amp)
print('psnr:',psnrr)
ssimm = ssim(target_amp, final)
print('ssim:',ssimm)

finalpic=finalpic/torch.max(finalpic)
amp= numpy.uint8(finalpic.cpu().data.numpy()*255)
cv2.imwrite('1.png', amp)
cv2.imshow('amp', amp)

#cv2.waitKey(0)