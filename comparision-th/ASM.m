
function[final] = ASM(cut,direction,bandlimit,quan,mu,z,pitch,lambda)

[kuan,chang] = size(quan);
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

final=ifftshift(ifft2(ifftshift(trans.*fftshift(fft2(fftshift(quan2))))));

if(strcmp(cut,'cut'))
 cutfinal=final(1+a/2-kuan/2:a/2+kuan/2,1+b/2-chang/2:b/2+chang/2);
 final=cutfinal;
end

end

