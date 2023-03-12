import numpy
from skimage.metrics import structural_similarity as ssim
import oldmodel
from propagation_ASM import *
import tools
def psnr(img1, img2):
    mse = numpy.mean((img1 - img2) ** 2 )
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

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

if method=='ccnncgh':
    net = oldmodel.ccnncgh()
if method == 'holonet':
    net = oldmodel.holonet()
if method=='holoencoder':
    net = oldmodel.holoencoder()
pthname=method+'.pth'
validpath='D:\\DIV2K_valid_HR'

net.load_state_dict(torch.load(pthname))
net.cuda()

init_phase=torch.zeros(1,1,n,m)
init_phase=init_phase.cuda()

rangege=100
currentssim=0
currentpsnr=0
with torch.no_grad():
    for kk in range(rangege):
        image_index = 801 + kk
        flip = numpy.random.randint(low=0, high=100)

        input_image = tools.loadimage(path=validpath, image_index=image_index, channel=2, flip=flip, m=m, n=n,
                                      convert=True, cuda=True)

        target_amp = torch.sqrt(input_image)
        target_amp = target_amp.view(1, 1, n, m)
        target_amp = target_amp.float()

        holo_phase, precict_phase = net(target_amp, init_phase, z, pad, pitch, wavelength, Hforward)

        slm_complex = torch.complex(torch.cos(holo_phase), torch.sin(holo_phase))
        recon_complex = propagation_ASM(u_in=slm_complex, z=-z, linear_conv=pad, feature_size=[pitch, pitch],
                                        wavelength=wavelength,
                                        precomped_H=Hbackward)



        recon_amp = torch.abs(recon_complex)
        recon_amp = torch.squeeze(recon_amp)
        recon_amp = recon_amp.cpu().data.numpy()
        target_amp = torch.squeeze(target_amp)
        target_amp = target_amp.cpu().numpy()

        psnrr = psnr(recon_amp, target_amp)
        currentpsnr = currentpsnr + psnrr
        ssimm = ssim(recon_amp, target_amp)

        currentssim = currentssim + ssimm
print('avgpsnr:',currentpsnr/rangege)
print('avgssim:',currentssim/rangege)