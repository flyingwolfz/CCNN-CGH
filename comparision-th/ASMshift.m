%angular spectrum method of fast calculation of diffraction 角谱衍射计算
%shift:fftshift 要不要加fftshift
%cut:output is set as the same size of input 要不要剪裁为原图大小
%direction:direction of the propagation 传播方向
%bandlimit:band-limit ASM 是否用带限角谱
%quan:input of the complex wave field 输入复振幅
%mu:zero padding size(How many times the size of the original complex wave
%   field) 填0为原图大小的几倍
%z:distance 距离
%pitch: pixel size 像素大小
%lambda: wavelength 波长
function[final] = ASM(shift,cut,direction,bandlimit,quan,mu,z,pitch,lambda)

[kuan,chang] = size(quan);
% pitch=8*10^(-3);
% lambda=638*10^(-6);
a=mu*kuan;
b=mu*chang;
u0=1/b/pitch;
v0=1/a/pitch;
quan2=zeros(a,b);
yy=-a/2+0.5:(a/2-1)+0.5;
xx=-b/2+0.5:(b/2-1)+0.5;
[x,y]=meshgrid(xx*u0,yy*v0);
quan2(1+a/2-kuan/2:a/2+kuan/2,1+b/2-chang/2:b/2+chang/2)=quan;
if(strcmp(direction,'forward'))
    trans=exp(1i*2*pi/lambda*z*sqrt(1-(lambda*x).^2-(lambda*y).^2));
else
    trans=exp(-1i*2*pi/lambda*z*sqrt(1-(lambda*x).^2-(lambda*y).^2));
end
if(strcmp(bandlimit,'limit'))
    xlimit=1/sqrt((2*1/b/pitch*z)^2+1)/lambda;
    ylimit=1/sqrt((2*1/a/pitch*z)^2+1)/lambda;
    trans(abs(x)>xlimit)=0;
    trans(abs(y)>ylimit)=0;
end
if(strcmp(shift,'shift'))
    final=ifft2(trans.*fftshift(fft2(quan2)));
else
    final=ifft2(trans.*(fft2(quan2)));
end
if(strcmp(cut,'cut'))
 cutfinal=final(1+a/2-kuan/2:a/2+kuan/2,1+b/2-chang/2:b/2+chang/2);
 final=cutfinal;
end

end

