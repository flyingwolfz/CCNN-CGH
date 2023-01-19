
import cv2
import numpy
import torch.fft
import time
import math
from skimage.metrics import structural_similarity as ssim
from scipy import io
import asm
import model

testpic=829
pitch=0.008
wavelength=0.000638
n = 1072
m = 1920
z=200

slm_res = (n, m)
pad=False
band_limit=True

Hbackward = asm.calh(cuda=True, pad=pad, band_limit=band_limit, slm_res=slm_res,
                     pitch=pitch, z=-z, wavelength=wavelength)
Hforward= asm.calh(cuda=True, pad=pad, band_limit=band_limit, slm_res=slm_res,
                     pitch=pitch, z=z, wavelength=wavelength)
Hbackward = Hbackward.cuda()
Hforward = Hforward.cuda()
def psnr(img1, img2):
    mse = numpy.mean((img1 - img2) ** 2 )
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

ccnn1 = model.CCNN1()
ccnn2 = model.CCNN2()
ccnn1.load_state_dict(torch.load('ccnn1.pth'))
ccnn2.load_state_dict(torch.load('ccnn2.pth'))
ccnn1=ccnn1.cuda()
ccnn2=ccnn2.cuda()

path='E:\\DIV2K\\DIV2K_valid_HR'
b = '\\0' + str(testpic)
imgpath = path + b + '.png'

img = cv2.imread(imgpath)
img2 = cv2.resize(img, (m,n))
gray = cv2.split(img2)[2]
cv2.imwrite('2.png', gray)

target_amp = torch.from_numpy(gray)
target_amp = target_amp / 255.0
target_amp = target_amp.cuda()
target_amp=torch.sqrt(target_amp)

init_phase=torch.zeros(n,m)
init_phase=init_phase.cuda()
real = torch.cos(init_phase * 2 * torch.pi)
imag = torch.sin(init_phase * 2 * torch.pi)
target_amp_complex = torch.complex(target_amp * real, target_amp * imag)
target_amp_complex = target_amp_complex.view(1, 1, n, m)

predictphase = ccnn1(target_amp_complex)
target_amp_complex2 = torch.complex(target_amp * torch.cos(predictphase), target_amp * torch.sin(predictphase))
slmfield = asm.asmprop2(H=Hforward, pad=pad, input=target_amp_complex2)
slmfield = slmfield.view(1, 1, n, m)
output = ccnn2(slmfield)

print('pass, start testing')

time_start=time.time()
with torch.no_grad():
    for k in range(100):
        predictphase = ccnn1(target_amp_complex)
        target_amp_complex2 = torch.complex(target_amp * torch.cos(predictphase), target_amp * torch.sin(predictphase))
        slmfield = asm.asmprop2(H=Hforward, pad=pad, input=target_amp_complex2)
        slmfield = slmfield.view(1, 1, n, m)
        output = ccnn2(slmfield)

time_end=time.time()
print('time',(time_end-time_start)/100.0)

max_phs = 2*torch.pi
holophase = output - output.mean()
holophase = ((holophase + max_phs / 2) % max_phs) / max_phs
holo = numpy.uint8(holophase.cpu().data.numpy() * 255)

cv2.imwrite('h.png', holo)
grayreal = torch.cos(output)
grayimage = torch.sin(output)
gray = torch.complex(grayreal, grayimage)

final = asm.asmprop2(H=Hbackward, pad=pad, input=gray)

final = torch.abs(final)
ffinal = final*final

final = final.cpu().numpy()
target_amp = target_amp.cpu().numpy()
psnrr = psnr(final, target_amp)
print('psnr:',psnrr)
ssimm = ssim(target_amp, final)
print('ssim:',ssimm)

ffinal=ffinal/torch.max(ffinal)
amp= numpy.uint8(ffinal.cpu().data.numpy()*255)
cv2.imwrite('1.png', amp)