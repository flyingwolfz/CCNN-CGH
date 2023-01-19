import math
import numpy
import torch
import torch.fft
import torch.nn as nn
import cv2

def calh(cuda=True, pad=True, band_limit=True, slm_res=(1080, 1920),
             pitch=0.0036, z=-150, wavelength=0.000638):
    padsize=1
    if pad:
        padsize=2
    n, m = slm_res[0] * padsize, slm_res[1] * padsize
    x = numpy.linspace(-n // 2+0.5, n // 2 - 0.5, n)
    y = numpy.linspace(-m // 2+0.5, m // 2 - 0.5, m)
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    n = numpy.array(n)
    m = numpy.array(m)
    n = torch.from_numpy(n)
    m = torch.from_numpy(m)
    v = 1 / (n * pitch)
    u = 1 / (m * pitch)
    fx = x * v
    fy = y * u
    fX, fY = torch.meshgrid(fx, fy)
    H =(2 * numpy.pi / wavelength) * z * torch.sqrt(1 - (wavelength * fX) ** 2 - (wavelength * fY) ** 2)
    Hreal = torch.cos(H)
    Himag = torch.sin(H)
    if band_limit:
       xlimit = 1 / torch.sqrt((2 * 1 / n / pitch * z) ** 2 + 1) / wavelength
       ylimit = 1 / torch.sqrt((2 * 1 / m / pitch * z) ** 2 + 1) / wavelength
       a = (abs(fX) < xlimit) & (abs(fY) < ylimit)
       a = a.numpy()
       a = a + 0
       Hreal=Hreal*a
       Himag = Himag * a
       #writea = numpy.uint8(a * 255)
       #cv2.imwrite('bandlimit.png', writea)
    H=torch.complex(Hreal.float(),Himag.float())
    if cuda:
        H = H.cuda()
    return H


def asmprop(H=None, pad=True, input=None):
    y, x = input.size()[0], input.size()[1]
    if pad:
       padx = int(x / 2)
       pady = int(y / 2)
       ZeroPad = nn.ZeroPad2d(padding=(padx, padx, pady, pady))
       input = ZeroPad(input)
    quan = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(input)))
    quan2 = quan * H
    holo = torch.fft.fftshift(torch.fft.ifftn(torch.fft.fftshift(quan2)))
    if pad:
        padx = int(x / 2)
        pady = int(y / 2)
        holo = holo[pady:pady + y, padx:padx + x]
    return holo

def asmprop2(H=None, pad=True, input=None):
    y, x = input.size()[0], input.size()[1]
    if pad:
       padx = int(x / 2)
       pady = int(y / 2)
       ZeroPad = nn.ZeroPad2d(padding=(padx, padx, pady, pady))
       input = ZeroPad(input)
    quan = torch.fft.fftn(input)
    quan2 = quan * H
    holo = torch.fft.ifftn((quan2))
    if pad:
        padx = int(x / 2)
        pady = int(y / 2)
        holo = holo[pady:pady + y, padx:padx + x]
    return holo