
import torch
import numpy
import cv2

def srgb_to_lin(image):

    thresh = 0.04045

    im_out = numpy.where(image <= thresh, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)

    return im_out

def lin_to_srgb(image):

    thresh = 0.0031308

    im_out = numpy.where(image <= thresh, 12.92 * image, 1.055 * (image**(1 / 2.4)) - 0.055)


    return im_out


def loadimage(path,image_index,channel=2,flip=0,m=1920,n=1080,convert=True):
    #DIV2K loader

    b = '\\0' + str(image_index)
    imgpath = path + b + '.png'
    img = cv2.imread(imgpath)
    img2 = cv2.resize(img, (m, n))
    gray = cv2.split(img2)[channel]

    flip = flip % 4
    if flip == 1:
        gray = cv2.flip(gray, flipCode=1)
    if flip == 2:
        gray = cv2.flip(gray, flipCode=0)
    if flip == 3:
        gray = cv2.flip(gray, flipCode=-1)

    gray=gray/255.0

    if convert==True:
        gray=srgb_to_lin(gray)

    return gray

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