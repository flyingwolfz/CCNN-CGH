
import torch
import numpy
import cv2

def srgb_to_lin(image):

    thresh = 0.04045

    if torch.is_tensor(image):
        low_val = image <= thresh
        im_out = torch.zeros_like(image)
        im_out[low_val] = 25 / 323 * image[low_val]
        im_out[torch.logical_not(low_val)] = ((200 * image[torch.logical_not(low_val)] + 11)
                                              / 211) ** (12 / 5)
    else:
        im_out = numpy.where(image <= thresh, image / 12.92, ((image + 0.055) / 1.055) ** (12 / 5))

    return im_out


def lin_to_srgb(image):

    thresh = 0.0031308

    im_out = numpy.where(image <= thresh, 12.92 * image, 1.055 * (image**(1 / 2.4)) - 0.055)


    return im_out

def Loadimage(path,channel=2,flip=0,m=1920,n=1080,convert=True):
    #any image loader

    img = cv2.imread(path)
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

def loadimage(path,image_index,channel=2,flip=0,m=1920,n=1080,convert=True,cuda=False):
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

    if cuda==True:
        gray = gray / 255.0
        gray = torch.from_numpy(gray)
        gray = gray.cuda()

    if convert==True:
        gray=srgb_to_lin(gray)

    return gray

