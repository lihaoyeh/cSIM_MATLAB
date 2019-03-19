% speckle coherent structured illumination microscopy
% Coded by Li-Hao Yeh 2017.02.22
% Last update 2018.08.02

clear all;
set(0,'DefaultFigureWindowStyle','docked');
gpuDevice(4);
%% Load data


% F = @(x) fftshift(fft2(ifftshift(x)));
% iF = @(x) fftshift(ifft2(ifftshift(x)));
% % 
% Nimg = 1296;
% Ns = 300;
% Ms = 300;
% % 
% 
% nstart = [480,600];
% nstart_large = [34,16]+nstart;
% nstart_large2 = [1,74]+nstart;
% 
% Ic_image = zeros(Ns,Ms,Nimg,2);
% 
% data_dir = ['D:\Li-Hao\cSIM\data\data_035\data_image2\'];
% 
% for i = 1:Nimg
% 
%     I1 = max(0,double(imread([data_dir,'ptgrey1_acq',num2str(i),'.tif'])) );                
%     Ic_image(:,:,i,1) = I1((1:Ns)+nstart_large(1),(1:Ms)+nstart_large(2));
% 
%     I2 = gpuArray(imresize(fliplr(max(0,double(imread([data_dir,'ptgrey2_acq',num2str(i),'.tif'])) )),[989,1302]));
%     I2 = gather(abs(rotate_re(I2(1:988,1:1300),-pi*0.43/180)));
% 
%     Ic_image(:,:,i,2) = I2((1:Ns)+nstart_large2(1),(1:Ms)+nstart_large2(2));
%     if mod(i,100) == 0
%         fprintf('Data loading process (%2d / %2d)\n',i,Nimg);
%     end
% 
% end
% 
% % % % % 
% % % % % % % % 
% save('registered_data','-v7.3');
load registered_data.mat;
% Ic_image = Ic_image/2^4;

%% Experimental parameters

lambda = 0.532; k=2*pi/lambda;  % wavelength (micons) and wave number
mag = 5; % magnification
pscrop = 5.5/mag; % Pixels size (microns)
NA_obj = 0.1; % detection NA
NAs = 0.3;
z_camera = [0;-32];

%% data pre-processing

intensity = squeeze(mean(mean(Ic_image,2),1));

[N_large,M_large,~,N_defocus] = size(Ic_image);

fx_large = (-M_large/2:(M_large/2-1))./(pscrop*M_large); 
fy_large = (-N_large/2:(N_large/2-1))./(pscrop*N_large);
[fxx_large,fyy_large] = meshgrid(fx_large,fy_large);
fxx_large = ifftshift(fxx_large);
fyy_large = ifftshift(fyy_large);

cor_shiftx = zeros(Nimg,1);
cor_shifty = zeros(Nimg,1);

for i = 1:Nimg
    [output, ~] = dftregistration(fft2(Ic_image(:,:,i,2)),fft2(Ic_image(:,:,i,1)),100);
    cor_shifty(i) = output(3);
    cor_shiftx(i) = output(4);
%     Ic_image(:,:,i,1) = max(0,real(ifft2(fft2(Ic_image(:,:,i,1)).*exp(-1j*2*pi*pscrop*(fxx_large*cor_shiftx(i) + fyy_large*cor_shifty(i))))));

end

global_shiftx = mean(cor_shiftx);
global_shifty = mean(cor_shifty);
% 
Ic_image(:,:,:,1) = max(0,real(ifft2(bsxfun(@times,fft2(Ic_image(:,:,:,1)),exp(-1j*2*pi*pscrop*(fxx_large*global_shiftx + fyy_large*global_shifty))))));


Ic_image(:,:,:,2) = Ic_image(:,:,:,2)*mean(intensity(:,1))/mean(intensity(:,2));
%%

Ic_image = Ic_image(71:170,11:110,:,:); %3t3_2

%% Coordinate assignment

[Ncrop,Mcrop,Nimg,N_defocus] = size(Ic_image);

upsampling_factor = 4; 

N = Ncrop*upsampling_factor; M = Mcrop*upsampling_factor; ps = pscrop/upsampling_factor;

xh = (-M/2:(M/2-1)).*ps; yh = (-N/2:(N/2-1)).*ps;
fx = (-M/2:(M/2-1))./(ps*M); fy = (-N/2:(N/2-1))./(ps*N);
NAx = fx*lambda; NAy = fy*lambda;
[xhh,yhh] = meshgrid(xh,yh);
[fxx,fyy] = meshgrid(fx,fy);

Fourier_crop = zeros(N,M);
Fourier_crop(sqrt(fxx.^2+fyy.^2)<2*NA_obj/lambda) =1;
% 


%% Upsampling the data


I_image_up = zeros(N,M,Nimg,N_defocus);
for i = 1:Nimg
    for j = 1:N_defocus
        I_image_up(:,:,i,j) = abs(iF(padarray(F(max(0,Ic_image(:,:,i,j)-96)),[(N-Ncrop)/2,(M-Mcrop)/2]).*Fourier_crop));
    end
end

% I_image_up = I_image_up/max(I_image_up(:));
% I_image_up = max(0,I_image_up - 0.08);

fxx = ifftshift(fxx);
fyy = ifftshift(fyy);

%% shift estimation


% N_shift = Nimg;
% 
% yshift = zeros(N_shift,N_defocus);
% xshift = zeros(N_shift,N_defocus);
% 
% for j = 1:Nimg
%     if j == 1
%         yshift(j,:) = 0;
%         xshift(j,:) = 0;
%     else
%         for i = 1:N_defocus
%             [output, ~] = dftregistration(fft2(I_image_up(:,:,1,i)),fft2(I_image_up(:,:,j,i)),100);
%             yshift(j,i) = (output(3)); 
%             xshift(j,i) = (output(4));
%         end
%     end
% end
% 
% save('xyshift','xshift','yshift');
load('xyshift_exp');

%% Set iterative parameters

F = @(x) fft2(x);
iF = @(x) ifft2(x);


N_bound_pad = 80;
Nc = N + 2*N_bound_pad;
Mc = M + 2*N_bound_pad;


fx_c = (-Mc/2:(Mc/2-1))./(ps*Mc); fy_c = (-Nc/2:(Nc/2-1))./(ps*Nc);
[fxx_c,fyy_c] = meshgrid(fx_c,fy_c);

fxx_c = ifftshift(fxx_c);
fyy_c = ifftshift(fyy_c);



% initialization

rng(50);

obj = gpuArray(ones(Nc,Mc));
% bg = zeros(N_defocus,1);
% bg(1) = 3;
% bg(2) = 3;
% scaling = ones(N_defocus,1);

I_forward = zeros(N,M,Nimg,N_defocus);



yshift_max = round(max(abs(yshift(:))));
xshift_max = round(max(abs(xshift(:))));

xshift_obj = zeros(Nimg,1);
yshift_obj = zeros(Nimg,1);

field_p_whole = ones(Nc+2*yshift_max,Mc+2*xshift_max,'gpuArray');

% field_p = padarray(sqrt(I_image_up),[N_bound_pad,N_bound_pad]);
% field_p = zeros(N,M,Nimg);

% convolution with larger field of view

r_obj=(fxx_c.^2+fyy_c.^2).^(1/2);
Pupil_support = zeros(Nc,Mc);
Pupil_support(r_obj<NA_obj/lambda) = 1;
Pupil_support = gpuArray(Pupil_support);

H_coherent = zeros(Nc,Mc,N_defocus,'gpuArray');
Pupil_obj = Pupil_support;

for i = 1:N_defocus
    H_coherent(:,:,i) = Pupil_support.*gpuArray(exp(-1j*pi*lambda*z_camera(i)*(fxx_c.^2+fyy_c.^2)));
end

Np = Nc + 2*yshift_max;
Mp = Mc + 2*xshift_max;

fxp = (-Mp/2:(Mp/2-1))./(ps*Mp); fyp = (-Np/2:(Np/2-1))./(ps*Np);
[fxxp,fyyp] = meshgrid(fxp,fyp);
% 
fxxp = gpuArray(ifftshift(fxxp));
fyyp = gpuArray(ifftshift(fyyp));

for j = 1:Nimg    
    fieldp_shift_back = iF(F(padarray(sqrt(padarray(gpuArray(I_image_up(:,:,j,1)),[N_bound_pad,N_bound_pad])),[yshift_max,xshift_max],0)).*exp(-1j*2*pi*ps*(fxxp.*xshift(j,1) + fyyp.*yshift(j,1))));
    field_p_whole = field_p_whole + fieldp_shift_back/Nimg;
end

Pupil_NAeff = zeros(Nc,Mc);
Pupil_NAeff(find(r_obj<(NA_obj+NAs)/lambda))=1;
Pupil_NAeff = gpuArray(Pupil_NAeff);

Gaussian = exp(-(fxx_c.^2+fyy_c.^2)/(2*((NA_obj+NAs)*0.95/lambda)^2));
Gaussian = gpuArray(Gaussian/max(Gaussian(:)));

Pupil_NAs = zeros(Np,Mp);
r_s=(fxxp.^2+fyyp.^2).^(1/2);
Pupil_NAs(find(r_s<NAs/lambda))=1;
Pupil_NAs = gpuArray(Pupil_NAs);

obj = iF(F(obj).*Pupil_NAeff);
field_p_whole = iF(F(field_p_whole).*Pupil_NAs);

% Zernike polynomial

n = [0 1 1 2 2 2 3 3 3 3 4 4 4 4 4 5 5 5 5 5 5 6 6 6 6 6 6 6];
m = [0 -1 1 -2 0 2 -3 -1 1 3 -4 -2 0 2 4 -5 -3 -1 1 3 5 -6 -4 -2 0 2 4 6];



N_idx = length(n);
zerpoly = zeros(Nc,Mc,N_idx);
for i =1:N_idx
    [theta,rr] = cart2pol(fxx_c/NA_obj*lambda,fyy_c/NA_obj*lambda);
    idx = rr<=1;
    z = zeros(size(fxx_c));
    z(idx) = zernfun(n(i),m(i),rr(idx),theta(idx));
    z = z/max(z(:));
    zerpoly(:,:,i) = z;
end
zerpoly = gpuArray(zerpoly);

% iteration number 

itr = 20;

% cost function

err = zeros(1,itr+1);

% calculate the initial cost function value

% for j = 1:Nimg
%     for m = 1:N_defocus
%         I_temp = abs(iF(Pupil_obj.*H_coherent(:,:,m).*F(gpuArray(field_p(:,:,j,m)).*obj))).^2;
%         I_forward(:,:,j,m) = gather(I_temp(N_bound_pad+1:N_bound_pad+N,N_bound_pad+1:N_bound_pad+M));
%     end
% end
% 
% for m = 1:N_defocus
%     err(1) = err(1)+ sum(sum(sum(abs(I_image_up(:,:,:,m)-I_forward(:,:,:,m)).^2)));
% end

%% Iterative algorithm

tic;
fprintf('| Iter  |   error    | Elapsed time (sec) |\n');
for i = 1:itr
    
    % Sequential update
    
    for j = 1:Nimg
       
        
        for m = 1:N_defocus
            
            fieldp_shift = iF(F(field_p_whole).*exp(1j*2*pi*ps*(fxxp.*xshift(j,m) + fyyp.*yshift(j,m))));
            field_p_gpu = fieldp_shift(1+yshift_max:Nc+yshift_max,1+xshift_max:Mc+xshift_max);
            I_image_up_current_sqrt = sqrt(gpuArray(I_image_up(:,:,j,m)));
%             field_p(:,:,j,m) = gather(field_p_gpu);
            
            field_f = F(field_p_gpu.*obj);
            z_temp = iF(Pupil_obj.*H_coherent(:,:,m).*field_f); 
            z_temp_crop_abs = abs(z_temp(N_bound_pad+1:N_bound_pad+N,N_bound_pad+1:N_bound_pad+M));
            
            err(i+1) = err(i+1)+ gather(sum(sum(abs(I_image_up_current_sqrt - z_temp_crop_abs).^2)));
            
            residual = F( z_temp./(abs(z_temp)+eps).*padarray(I_image_up_current_sqrt - z_temp_crop_abs,[N_bound_pad,N_bound_pad],0) );
            I_temp = iF(conj(Pupil_obj.*H_coherent(:,:,m)).*residual);

            grad_Iobj = -conj(field_p_gpu).*I_temp;
            grad_Ip = -iF(F(padarray(conj(obj).*I_temp,[yshift_max,xshift_max],0)).*exp(-1j*2*pi*ps*(fxxp.*xshift(j,m) + fyyp.*yshift(j,m))));
            grad_P = -conj(H_coherent(:,:,m).*field_f).*residual;

            obj = obj - grad_Iobj/max(max(abs(field_p_gpu)))^2;
            field_p_whole = field_p_whole - grad_Ip/max(abs(obj(:)))^2;
%             Pupil_obj = Pupil_obj - grad_P/max(abs(field_f(:)))^2/10;
            Pupil_obj = Pupil_obj - grad_P/max(abs(field_f(:))).*abs(field_f)./(abs(field_f).^2+1e-3)/5;
            % shift estimation
            
%             if j  ~= 1
                Ip_shift_fx = iF(F(field_p_whole).*(1j*2*pi*fxxp).*exp(1j*2*pi*ps*(fxxp.*xshift(j,m) + fyyp.*yshift(j,m))));
                Ip_shift_fy = iF(F(field_p_whole).*(1j*2*pi*fyyp).*exp(1j*2*pi*ps*(fxxp.*xshift(j,m) + fyyp.*yshift(j,m))));

                Ip_shift_fx = Ip_shift_fx(1+yshift_max:Nc+yshift_max,1+xshift_max:Mc+xshift_max);
                Ip_shift_fy = Ip_shift_fy(1+yshift_max:Nc+yshift_max,1+xshift_max:Mc+xshift_max);

                grad_xshift = -real(sum(sum(conj(I_temp).*obj.*Ip_shift_fx)));
                grad_yshift = -real(sum(sum(conj(I_temp).*obj.*Ip_shift_fy)));

                xshift(j,m) = xshift(j,m) - gather(grad_xshift/N/M/max(abs(obj(:)))^2);
                yshift(j,m) = yshift(j,m) - gather(grad_yshift/N/M/max(abs(obj(:)))^2);
%             end
            
            
         end
    end
    
    obj = iF(F(obj).*Gaussian);
    field_p_whole = iF(F(field_p_whole).*Pupil_NAs);
    Pupil_angle = angle(Pupil_obj);
    Pupil_angle = Pupil_angle - sum(sum(zerpoly(:,:,2).*Pupil_angle))/sum(sum(zerpoly(:,:,2).^2)).*zerpoly(:,:,2)...
        - sum(sum(zerpoly(:,:,3).*Pupil_angle))/sum(sum(zerpoly(:,:,3).^2)).*zerpoly(:,:,3);
    Pupil_obj = abs(Pupil_obj).*exp(1j*Pupil_angle);

% 
%     for j = 1:Nimg
%         for m = 1:N_defocus
%             I_temp = abs(iF(Pupil_obj.*H_coherent(:,:,m).*F(gpuArray(field_p(:,:,j,m)).*obj))).^2;
%             I_forward(:,:,j,m) = gather(I_temp(N_bound_pad+1:N_bound_pad+N,N_bound_pad+1:N_bound_pad+M));
%         end
%     end
% 
%     for m = 1:N_defocus
%         err(i+1) = err(i+1)+ sum(sum(sum(abs(I_image_up(:,:,:,m)-I_forward(:,:,:,m)).^2)));
%     end


    
    if mod(i,1) == 0
        fprintf('|  %2d   |  %.2e  |        %.2f      |\n', i, err(i+1),toc);
        figure(39);
        subplot(2,3,1),imagesc(abs(obj));colormap gray;axis square;
        subplot(2,3,4),imagesc(angle(obj));colormap gray;axis square;

        subplot(2,3,2),imagesc(abs(field_p_whole).^2);colormap gray;axis square;
        subplot(2,3,5),imagesc(angle(field_p_whole));colormap gray;axis square;
        
        
        subplot(2,3,3),plot(xshift,yshift);colormap gray;axis image;
        subplot(2,3,6),imagesc(fftshift(angle(Pupil_obj)));colormap gray;axis image;
        

        pause(0.001);
    end

end
% save(['recon'],'obj','field_p_whole','xshift','yshift','err')

%% Plot result

% figure;imagesc(mean(I_image_up,3));colormap gray;axis square;
% title('mean raw data');
% 
% figure;imagesc(I_obj);colormap gray;axis square;
% title('Reconstructed image');
% 
% % figure;imagesc(I);colormap gray;axis square;
% % title('Ground truth');
% 
% figure;imagesc(NAx,NAy,log10(abs(F(I_obj))));colormap jet;axis square;
% title('Fourier transform of reconstructed image');
% hold on;circle(0,0,4*NA_obj);
% 
% 
% figure;imagesc(NAx,NAy,log10(abs(F(mean(I_image_up,3)))));colormap jet;axis square;
% title('Fourier transform of mean raw data');
% hold on;circle(0,0,2*NA_obj);


