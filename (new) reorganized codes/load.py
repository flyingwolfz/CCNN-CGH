import cv2
import numpy
import time
from skimage.metrics import structural_similarity as ssim
from propagation_ASM import *
import model
import tools
testpic=829
pitch=0.0036
wavelength=0.000638
n = 2160
m = 3840
z=150

slm_res = (n, m)
pad=False
method='ccnncgh'
#method='holonet'
#method='holoencoder'

Hbackward = propagation_ASM(torch.empty(1, 1, n, m), feature_size=[pitch, pitch],
                                wavelength=wavelength, z=-z, linear_conv=pad,return_H=True)
Hbackward = Hbackward.cuda()

Hforward = propagation_ASM(torch.empty(1, 1, n, m), feature_size=[pitch, pitch],
                                wavelength=wavelength, z=z, linear_conv=pad,return_H=True)
Hforward = Hforward.cuda()


def psnr(img1, img2):
    mse = numpy.mean((img1 - img2) ** 2 )
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

if method=='ccnncgh':
    net = model.ccnncgh()
if method == 'holonet':
    net = model.holonet()
if method=='holoencoder':
    net = model.holoencoder()
pthname=method+'.pth'
validpath='D:\\DIV2K_valid_HR'

net.load_state_dict(torch.load(pthname))
net=net.cuda()
input_image = tools.loadimage(path=validpath, image_index=testpic, channel=2, flip=0, m=m, n=n, convert=True, cuda=True)
target_amp = torch.sqrt(input_image)
target_amp = target_amp.view(1, 1, n, m)
target_amp = target_amp.float()


init_phase=torch.zeros(1,1,n,m)
init_phase=init_phase.cuda()

holo_phase, predict_phase = net(target_amp, init_phase, z, pad, pitch, wavelength, Hforward)

print('pass, start testing')

time_start=time.time()
with torch.no_grad():
    for k in range(100):
        holo_phase, predict_phase = net(target_amp, init_phase, z, pad, pitch, wavelength, Hforward)
time_end=time.time()
print('time',(time_end-time_start)/100.0)

slm_complex = torch.complex(torch.cos(holo_phase), torch.sin(holo_phase))
recon_complex = propagation_ASM(u_in=slm_complex, z=-z, linear_conv=pad, feature_size=[pitch, pitch],
                                wavelength=wavelength,
                                precomped_H=Hbackward)

recon_amp = torch.abs(recon_complex)
recon_amp = torch.squeeze(recon_amp)
recon_amp=recon_amp.cpu().data.numpy()
target_amp=torch.squeeze(target_amp)
target_amp = target_amp.cpu().numpy()
psnrr = psnr(recon_amp, target_amp)
print('psnr:',psnrr)
ssimm = ssim(recon_amp, target_amp)
print('ssim:',ssimm)

recon=recon_amp*recon_amp
recon=tools.lin_to_srgb(recon)
recon = recon / recon.max()
pic = numpy.uint8(recon * 255)
cv2.imwrite('recon.png', pic)

max_phs = 2 * torch.pi
holo_phase = torch.squeeze(holo_phase)
# holophase = output - output.mean()
holophase = ((holo_phase + max_phs / 2) % max_phs) / max_phs
holo = numpy.uint8(holophase.cpu().data.numpy() * 255)
cv2.imwrite('h.png', holo)


phasepic = torch.squeeze(predict_phase)
phasepic = ((phasepic + max_phs / 2) % max_phs) / max_phs
pic = numpy.uint8(phasepic.cpu().data.numpy() * 255)
cv2.imwrite('predict_phase.png', pic)
