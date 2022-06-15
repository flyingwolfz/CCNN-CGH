clc
clear

%tf=im2double((imread('red.png')));
%ccnn=im2double((imread('h.png')));
th=im2double((imread('red2.png')));
ccnn=im2double((imread('h2.png')));
th=exp(1i*2*pi*th);
ccnn=exp(1i*2*pi*ccnn);

pitch=8*10^(-3);
z=200;
lambdaccnn=639*10^(-6);
lambdath=638*10^(-6);

finalth=ASM('ncut','backward','limit',th,1,z,pitch,lambdath);
finalccnn=ASM('ncut','backward','limit',ccnn,1,z,pitch,lambdaccnn);
finalth=abs(finalth);
finalccnn=abs(finalccnn);

%img=im2double((imread('0889.png')));
img=im2double((imread('0879.png')));
img=imresize(img,size(finalth));
img=img(:,:,1);

partth=finalth(100:700,600:1400);
partimg=img(100:700,600:1400);

%SSIM=ssim(parttf,partimg)
Diff=255*double(partimg)-255*double(partth);
MSE=sum(Diff(:).^2)/numel(partimg);
PSNR=10*log10(255^2/MSE)

%img=im2double((imread('0889.png')));
img=im2double((imread('0879.png')));
img=imresize(img,size(finalccnn));
img=img(:,:,1);

partccnn=finalccnn(100:700,600:1400);
partimg=img(100:700,600:1400);

%SSIM=ssim(partccnn,partimg)
Diff=255*double(partimg)-255*double(partccnn);
MSE=sum(Diff(:).^2)/numel(partimg);
PSNR=10*log10(255^2/MSE)
imwrite(partccnn,'partccnn.png');
imwrite(partth,'partth.png');
imwrite(partimg,'partimg.png');



