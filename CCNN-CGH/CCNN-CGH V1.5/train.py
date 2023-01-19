from torch import nn
import cv2
import numpy
import torch.fft
from tqdm import trange
import time
from scipy import io
import asm
import model

#you can change asmprop2 to asmprop, see asm.py
num=30
trainnum=700
validnum=100
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
Hforward = asm.calh(cuda=True, pad=pad, band_limit=band_limit, slm_res=slm_res,
                    pitch=pitch, z=z, wavelength=wavelength)

lr=0.001
ccnn1 = model.CCNN1()
ccnn2 = model.CCNN2()
criterion = nn.MSELoss()
ccnn1=ccnn1.cuda()
ccnn2=ccnn2.cuda()
criterion=criterion.cuda()

init_phase=torch.zeros(n,m)
init_phase=init_phase.cuda()

optvars = [{'params': ccnn1.parameters()}]
optvars += [{'params': ccnn2.parameters()}]
optimizier = torch.optim.Adam(optvars, lr=lr)

trainpath='E:\\DIV2K\\DIV2K_train_HR'
validpath='E:\\DIV2K\\DIV2K_valid_HR'
l=[]
tl=[]

for k in trange(num):
    currenttloss = 0
    currentloss = 0
    for kk in range(trainnum):
        c = 100 + kk
        b = '\\0' + str(c)
        imgpath = trainpath + b + '.png'
        img = cv2.imread(imgpath)
        img2 = cv2.resize(img, (m,n))
        #gray = cv2.split(img2)[2]

        enhance1 = numpy.random.randint(low=0, high=100)
        enhance2 = numpy.random.randint(low=0, high=100)
        channel = enhance1 % 3
        gray = cv2.split(img2)[channel]
        fliper = enhance2 % 4
        if fliper == 0:
            gray = cv2.flip(gray, flipCode=1)
        if fliper == 1:
            gray = cv2.flip(gray, flipCode=0)
        if fliper == 2:
            gray = cv2.flip(gray, flipCode=-1)

        target_amp = torch.from_numpy(gray)
        target_amp = target_amp.cuda()
        target_amp = target_amp / 255.0

        target_amp = torch.sqrt(target_amp)

        real = torch.cos(init_phase* 2 * torch.pi)
        imag = torch.sin(init_phase* 2 * torch.pi)
        target_amp_complex = torch.complex(target_amp*real, target_amp*imag)
        target_amp_complex=target_amp_complex.view(1, 1, n, m)

        predictphase = ccnn1(target_amp_complex)
        target_amp_complex = torch.complex(target_amp * torch.cos(predictphase), target_amp * torch.sin(predictphase))

        slmfield = asm.asmprop2(H=Hforward, pad=pad, input=target_amp_complex)

        slmfield = slmfield.view(1, 1, n, m)
        output = ccnn2(slmfield)

        slmcomplex = torch.complex(torch.cos(output), torch.sin(output))
        final = asm.asmprop2(H=Hbackward, pad=pad, input=slmcomplex)

        final = torch.abs(final)
        loss = criterion(final,target_amp)

        currenttloss = currenttloss + loss.cpu().data.numpy()
        optimizier.zero_grad()
        loss.backward()
        optimizier.step()
    tl.append(currenttloss / trainnum)
    print('trainloss:', currenttloss / trainnum)
    if k%1==0:
        c = k
        b = '1\\' + str(c)
        imgpath = b + '.png'
        finalpic = final*final
        finalpic = finalpic / torch.max(finalpic)
        pic = numpy.uint8(finalpic.cpu().data.numpy() * 255)
        cv2.imwrite(imgpath, pic)

        b = '3\\' + str(c)
        imgpath = b + '.png'
        phasepic = predictphase
        max_phs = 1
        phasepic = phasepic - phasepic.mean()
        phasepic = ((phasepic + max_phs / 2) % max_phs) / max_phs
        pic = numpy.uint8(phasepic.cpu().data.numpy() * 255)
        cv2.imwrite(imgpath, pic)

    with torch.no_grad():
        for kk in range(validnum):
            c = 801 + kk
            b = '\\0' + str(c)
            imgpath = validpath + b + '.png'
            img = cv2.imread(imgpath)
            img2 = cv2.resize(img, (m,n))
            gray = cv2.split(img2)[2]
            target_amp = torch.from_numpy(gray)
            target_amp = target_amp.cuda()
            target_amp = target_amp / 255.0

            target_amp = torch.sqrt(target_amp)

            real = torch.cos(init_phase * 2 * torch.pi)
            imag = torch.sin(init_phase * 2 * torch.pi)
            target_amp_complex = torch.complex(target_amp * real, target_amp * imag)
            target_amp_complex = target_amp_complex.view(1, 1, n, m)

            predictphase = ccnn1(target_amp_complex)
            target_amp_complex = torch.complex(target_amp * torch.cos(predictphase), target_amp * torch.sin(predictphase))

            slmfield = asm.asmprop2(H=Hforward, pad=pad, input=target_amp_complex)

            slmfield = slmfield.view(1, 1, n, m)
            output = ccnn2(slmfield)

            slmcomplex = torch.complex(torch.cos(output), torch.sin(output))
            final = asm.asmprop2(H=Hbackward, pad=pad, input=slmcomplex)

            final = torch.abs(final)
            loss = criterion(final, target_amp)
            currentloss = currentloss + loss.cpu().data.numpy()
            if  k%1==0 and kk == 38:
                finalpic = final*final
                finalpic = finalpic / torch.max(finalpic)
                c = k
                b = '2\\' + str(c)
                imgpath = b + '.png'
                pic = numpy.uint8(finalpic.cpu().data.numpy() * 255)
                cv2.imwrite(imgpath, pic)

                max_phs = 2 * torch.pi
                holophase = output - output.mean()
                holophase = ((holophase + max_phs / 2) % max_phs) / max_phs
                holo = numpy.uint8(holophase.cpu().data.numpy() * 255)
                b = '4\\' + str(c)
                imgpath = b + '.png'
                cv2.imwrite(imgpath, holo)

        l.append(currentloss / validnum)
        print('validloss:', currentloss / validnum)
    time.sleep(1)
pass

torch.save(ccnn1.state_dict(), 'ccnn1.pth')
torch.save(ccnn2.state_dict(), 'ccnn2.pth')
l=numpy.mat(l)
io.savemat('avgloss.mat',{'avgloss': l})
tl=numpy.mat(tl)
io.savemat('avgtloss.mat',{'avgtloss': tl})

