from torch import nn
import cv2
import numpy
from tqdm import trange
import time
from scipy import io
import oldmodel
from propagation_ASM import *
import tools
num=30
trainnum=700
validnum=100
pitch=0.0036
wavelength=0.000638
n = 2160
m = 3840
z=150
layernum=1
interval=10
slm_res = (n, m)
pad=False
method='ccnncgh'
#method='holonet'
#method='holoencoder'

Hbackward= propagation_ASM(torch.empty(1, 1, n, m), feature_size=[pitch, pitch],
                                wavelength=wavelength, z=-z, linear_conv=pad,return_H=True)
Hforward= propagation_ASM(torch.empty(1, 1, n, m), feature_size=[pitch, pitch],
                                wavelength=wavelength, z=z, linear_conv=pad,return_H=True)
Hbackward = Hbackward.cuda()
Hforward = Hforward.cuda()
lr=0.001
if method=='ccnncgh':
    net = oldmodel.ccnncgh()
if method == 'holonet':
    net = oldmodel.holonet()
if method=='holoencoder':
    net = oldmodel.holoencoder()

criterion = nn.MSELoss()
net=net.cuda()
criterion=criterion.cuda()

init_phase=torch.zeros(1,1,n,m)
init_phase=init_phase.cuda()
optvars = [{'params': net.parameters()}]
optimizier = torch.optim.Adam(optvars, lr=lr)

trainpath='D:\\DIV2K_train_HR'
validpath='D:\\DIV2K_valid_HR'
l=[]
tl=[]

for k in trange(num):
    currenttloss = 0
    currentloss = 0
    for kk in range(trainnum):

        optimizier.zero_grad()

        image_index = 100 + kk
        flip = numpy.random.randint(low=0, high=100)

        input_image=tools.loadimage(path=trainpath,image_index=image_index,channel=2,flip=flip,m=m,n=n,convert=True,cuda=True)


        target_amp = torch.sqrt(input_image)
        target_amp = target_amp.view(1, 1, n, m)
        target_amp=target_amp.float()

        holo_phase,predict_phase =net(target_amp,init_phase,z,pad,pitch,wavelength,Hforward)

        slm_complex = torch.complex(torch.cos(holo_phase), torch.sin(holo_phase))
        recon_complex = propagation_ASM(u_in=slm_complex, z=-z, linear_conv=pad, feature_size=[pitch, pitch],
                                        wavelength=wavelength,
                                        precomped_H=Hbackward)

        recon_amp = torch.abs(recon_complex)
        loss = criterion(recon_amp,target_amp)

        currenttloss = currenttloss + loss.cpu().data.numpy()
        loss.backward()
        optimizier.step()

    tl.append(currenttloss / trainnum)
    print('trainloss:', currenttloss / trainnum)
    if k%10==0:
        c = k
        b = '1\\' + method+str(c)
        imgpath = b + '.png'
        recon_amp = torch.squeeze(recon_amp)
        recon_amp=recon_amp.cpu().data.numpy()
        recon=recon_amp*recon_amp

        recon=tools.lin_to_srgb(recon)

        recon = recon / recon.max()
        pic = numpy.uint8(recon * 255)
        cv2.imwrite(imgpath, pic)

        b = '3\\' + method+str(c)
        imgpath = b + '.png'
        phasepic = torch.squeeze(predict_phase)
        max_phs = 2 * torch.pi
        phasepic = ((phasepic + max_phs / 2) % max_phs) / max_phs
        pic = numpy.uint8(phasepic.cpu().data.numpy() * 255)
        cv2.imwrite(imgpath, pic)

    with torch.no_grad():
        for kk in range(validnum):
            image_index = 801 + kk
            flip = numpy.random.randint(low=0, high=100)

            input_image = tools.loadimage(path=validpath, image_index=image_index, channel=2, flip=flip, m=m, n=n,
                                          convert=True,cuda=True)

            target_amp = torch.sqrt(input_image)
            target_amp = target_amp.view(1, 1, n, m)
            target_amp = target_amp.float()

            holo_phase, precict_phase = net(target_amp, init_phase, z, pad, pitch, wavelength, Hforward)

            slm_complex = torch.complex(torch.cos(holo_phase), torch.sin(holo_phase))
            recon_complex = propagation_ASM(u_in=slm_complex, z=-z, linear_conv=pad, feature_size=[pitch, pitch],
                                             wavelength=wavelength,
                                             precomped_H=Hbackward)

            recon_amp = torch.abs(recon_complex)
            loss = criterion(recon_amp, target_amp)
            currentloss = currentloss + loss.cpu().data.numpy()

            if  k%10==0 and kk == 38:
                c = k
                b = '2\\' + method+str(c)
                imgpath = b + '.png'
                recon_amp = torch.squeeze(recon_amp)
                recon_amp = recon_amp.cpu().data.numpy()
                recon = recon_amp * recon_amp

                recon = tools.lin_to_srgb(recon)

                recon = recon / recon.max()
                pic = numpy.uint8(recon * 255)
                cv2.imwrite(imgpath, pic)

                max_phs = 2 * torch.pi
                holo_phase = torch.squeeze(holo_phase)
                #holophase = output - output.mean()
                holophase = ((holo_phase + max_phs / 2) % max_phs) / max_phs
                holo = numpy.uint8(holophase.cpu().data.numpy() * 255)
                b = '4\\' + method+str(c)
                imgpath = b + '.png'
                cv2.imwrite(imgpath, holo)

        l.append(currentloss / validnum)
        print('validloss:', currentloss / validnum)
    time.sleep(1)
pass

pthname=method+'.pth'
lossname=method+'avgloss.mat'
tlossname=method+'avgtloss.mat'

torch.save(net.state_dict(), pthname)

l=numpy.mat(l)
io.savemat(lossname,{'avgloss': l})
tl=numpy.mat(tl)
io.savemat(tlossname,{'avgtloss': tl})