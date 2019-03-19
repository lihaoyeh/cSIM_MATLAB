% Generate simulation data for shifted speckle illumination 
% Coded by Li-Hao Yeh 2016.08.13
% Last update 2017.02.22
% Add coherent simulation

clear all;
set(0,'DefaultFigureWindowStyle','docked');



F = @(x) fftshift(fft2(ifftshift(x)));
iF = @(x) fftshift(ifft2(ifftshift(x)));

I = double(imread('resolution.jpg'));

% I = I(:,:,2);
% I = padarray(I(129:516,345:732,2),[6,6]);
% I = I(323:506,358:541,2);

N_bound_pad = 30;

I = padarray(I(323:506,358:541,2),[N_bound_pad,N_bound_pad]);

I = I/max(I(:));

amp = (1-I)+0.5;
amp = amp/max(amp(:));
amp = ones(size(amp));
ph = I*0.3;

T_c = amp.*exp(1j*ph);



%% --------Experiment Setup----------

lambda = 0.605; k=2*pi/lambda; % wavelength (micons) and wave number
mag = 16;
pscrop = 6.5/mag; % Pixels size (microns)
NA_obj = 0.1;
NAs = 0.4;

undersamp_factor = 1;
ps = pscrop/undersamp_factor;


[N,M] = size(I);

Ncrop = N/undersamp_factor;
Mcrop = M/undersamp_factor;

xh = (-M/2:(M/2-1)).*ps; yh = (-N/2:(N/2-1)).*ps;
fx = (-M/2:(M/2-1))./(ps*M); fy = (-N/2:(N/2-1))./(ps*N);
NAx = fx*lambda; NAy = fy*lambda;
[xhh,yhh] = meshgrid(xh,yh);
[fxx,fyy] = meshgrid(fx,fy);

%% -------- Propagation kernel --------

xs = (-M*4/2:(M*4/2-1)).*ps; ys = (-N*4/2:(N*4/2-1)).*ps;
fxs = (-M*4/2:(M*4/2-1))./(ps*4*M); fys = (-N*4/2:(N*4/2-1))./(ps*4*N);
NAxs = fxs.*lambda; NAys = fys.*lambda;
[xss,yss] = meshgrid(xs,ys);
[fxxs,fyys] = meshgrid(fxs,fys);
NAxx = fxxs.*lambda;
NAyy = fyys.*lambda;


r_prop=(fxxs.^2+fyys.^2).^(1/2);
Pupil_prop = zeros(N*4,M*4);
Pupil_prop(find(r_prop<NAs/lambda))=1;


Pupil_2NA = zeros(N*4,M*4);
Pupil_2NA(find(r_prop<2*NAs/lambda))=1;

%% Pixel shift in group

N_pattern = 1;

N_shiftx = 25;
N_shifty = 25;

Nimg = N_shiftx*N_shifty*N_pattern;
N_shift = N_shiftx*N_shifty;

% pixel_step = [2;2;2;2;2;2;2;2;2;2];
pixel_step = 2*ones(N_pattern,1);
pixel_shift_stack = zeros(2,N_shift,N_pattern);


for i = 1:N_pattern
    pixel_shiftx = (-(N_shiftx-1)/2:(N_shiftx-1)/2).*pixel_step(i);
    pixel_shifty = (-(N_shifty-1)/2:(N_shifty-1)/2).*pixel_step(i);
    [pixel_shiftyy,pixel_shiftxx] = meshgrid(pixel_shifty,pixel_shiftx);
    
    pixel_shift_stack(1,:,i) = round((pixel_shiftxx(:)*cos(45/180*pi) - pixel_shiftyy(:)*sin(45/180*pi))/sqrt(2) + randn(size(pixel_shiftyy(:)))*0);% + randn(N_shiftx*N_shifty,1)*0.7);
    pixel_shift_stack(2,:,i) = round((pixel_shiftxx(:)*cos(45/180*pi) + pixel_shiftyy(:)*sin(45/180*pi))/sqrt(2) + randn(size(pixel_shiftyy(:)))*0);%+((1:(N_shiftx*N_shifty))/N_shiftx/N_shifty*12)');
end



%% Pattern generation with random phase mask

rng(49);

speckle_intensity = zeros(4*N,4*M,N_pattern);
speckle_field = zeros(4*N,4*M,N_pattern);

for i = 1:N_pattern
    random_mapf = exp(1j*rand(4*N,4*M)*100);
    
    temp = iF(random_mapf.*Pupil_prop);
%     temp = abs(iF(F(temp).*Pupil_2NA));
    speckle_field(:,:,i) = temp/max(abs(temp(:)));
    speckle_intensity(:,:,i) = abs(speckle_field(:,:,i)).^2;
end

speckle_intensity_crop = speckle_intensity(1.5*N+1:2.5*N,1.5*M+1:2.5*M,1);
speckle_intensity_cropf = F(speckle_intensity_crop);
% speckle_field = rand(4*N,4*N);

figure;imagesc(xh,yh,speckle_intensity_crop); colormap gray;axis square;
% % figure;imagesc(NAx,NAy,log10(abs(F(speckle_field_crop))),[1 5]);colormap jet;axis square;
figure;imagesc(NAx,NAy,log10(abs(speckle_intensity_cropf))/max(max(log10(abs(speckle_intensity_cropf)))),[0 1]); colormap jet; axis square;
hold on;circle(0,0,2*NAs);


%% Setup Zernike polynomials/aberration function

N_conv_pad = 30;
N_aug = N + 2*N_conv_pad;
M_aug = M + 2*N_conv_pad;

fx_aug = (-M_aug/2:(M_aug/2-1))./(ps*M_aug); fy_aug = (-N_aug/2:(N_aug/2-1))./(ps*N_aug);
[fxx_aug,fyy_aug] = meshgrid(fx_aug,fy_aug);
% 

rng(41);

% n = [0 1 1 2 2 2 3 3 3 3 4 4 4 4 4 5 5 5 5 5 5 6 6 6 6 6 6 6];
% m = [0 -1 1 -2 0 2 -3 -1 1 3 -4 -2 0 2 4 -5 -3 -1 1 3 5 -6 -4 -2 0 2 4 6];

n = [2 2 2 3 3 3 3 4 4 4 4 4 5 5 5 5 5 5 6 6 6 6 6 6 6];
m = [-2 0 2 -3 -1 1 3 -4 -2 0 2 4 -5 -3 -1 1 3 5 -6 -4 -2 0 2 4 6];


N_idx = length(n);
zerpoly = zeros(N_aug,M_aug,N_idx);
for i =1:N_idx
[theta,rr] = cart2pol(fxx_aug/NA_obj*lambda,fyy_aug/NA_obj*lambda);
idx = rr<=1;
z = zeros(size(fxx_aug));
z(idx) = zernfun(n(i),m(i),rr(idx),theta(idx));
z = z/max(z(:));
zerpoly(:,:,i) = z;
end

zern_para = randn(N_idx,1)*0.4;
zern_para2 = zern_para + randn(N_idx,1)*0.1;
aberration = zeros(N_aug,M_aug);
aberration2 = zeros(N_aug,M_aug);

for i = 1:N_idx
    aberration = aberration + zern_para(i)*zerpoly(:,:,i);
    aberration2 = aberration2 + zern_para2(i)*zerpoly(:,:,i);    
end

% aberration = zerpoly(:,:,6);

% aberration = zeros(N,M);

figure;imagesc(aberration);colormap gray;axis image;
figure;imagesc(aberration2);colormap gray;axis image;
%% Data generation




Pupil_obj = zeros(N_aug,M_aug,2);
r_obj=(fxx_aug.^2+fyy_aug.^2).^(1/2);
Pupil_obj_temp =zeros(N_aug,M_aug);
Pupil_obj_temp(find(r_obj<NA_obj/lambda))=1;
Pupil_obj(:,:,1) = Pupil_obj_temp;
Pupil_obj(:,:,2) = Pupil_obj_temp;
% Pupil_obj(:,:,1) = Pupil_obj_temp.*exp(1j*aberration);
% Pupil_obj(:,:,2) = Pupil_obj_temp.*exp(1j*aberration2);


T_incoherent = abs(F(abs(iF(Pupil_obj(:,:,1))).^2));
T_incoherent_max = max(abs(T_incoherent(:)));
T_incoherent = T_incoherent/T_incoherent_max;

z_camera = [0;31.25];
N_defocus = size(z_camera,1);

speckle_intensity_shift_crop = zeros(N,M,Nimg);
speckle_field_shift_crop = zeros(N,M,Nimg);

Nb = Ncrop - 2*N_bound_pad/undersamp_factor;
Mb = Mcrop - 2*N_bound_pad/undersamp_factor;

I_image = zeros(Nb,Mb,Nimg);
Ic_image = zeros(Nb,Mb,Nimg,N_defocus);

cor_init = [0,0];
cor_end = [1,10];

cor_delta = (cor_end - cor_init)/(Nimg-1);
cor_x_shift = 0:cor_delta(2):cor_delta(2)*(Nimg-1);
cor_y_shift = 0:cor_delta(1):cor_delta(1)*(Nimg-1);

% z_shift = 3;
% cor_z_shift = 0:(z_shift/(Nimg-1)):z_shift;

cor_x_shift = zeros(Nimg,1); 
cor_y_shift = zeros(Nimg,1);
% z_speckle = randn(Nimg,1)*3;
% z_speckle = (0:(Nimg-1))/Nimg*20;
I_fluctuation = 1 + 0.05*randn(Nimg,2);
% I_fluctuation = 0.7 + 0.3*(0:(2*Nimg-1))/Nimg/2;
% I_fluctuation = [I_fluctuation(1:2:end);I_fluctuation(1,2:2:end)]';



for i = 1:Nimg
    
    idx_ps = mod(i-1,N_shift)+1;
    idx_pt = floor((i-1)/N_shift)+1;
    

    
    
%     I_shift = I;
    I_shift = max(0,real(iF(F(I).*exp(-1j*2*pi*ps*(fxx*cor_x_shift(i) + fyy*cor_y_shift(i))))));

    
%     temp_speckle_field = iF(F(speckle_field).*Pupil_prop.*exp(-1j*pi*lambda*z_speckle(i)*(fxxs.^2 + fyys.^2)));
    
    speckle_intensity_shift_crop(:,:,i) = speckle_intensity(1.5*N+1+pixel_shift_stack(1,idx_ps,idx_pt):2.5*N+pixel_shift_stack(1,idx_ps,idx_pt),1.5*M+1+pixel_shift_stack(2,idx_ps,idx_pt):2.5*M+pixel_shift_stack(2,idx_ps,idx_pt),idx_pt);
    
    
    speckle_field_shift_crop(:,:,i) = speckle_field(1.5*N+1+pixel_shift_stack(1,idx_ps,idx_pt):2.5*N+pixel_shift_stack(1,idx_ps,idx_pt),1.5*M+1+pixel_shift_stack(2,idx_ps,idx_pt):2.5*M+pixel_shift_stack(2,idx_ps,idx_pt),idx_pt);
    
    Itemp = iF(F(padarray(speckle_intensity_shift_crop(:,:,i).*I_shift,[N_conv_pad,N_conv_pad],0)).*T_incoherent);
    Itemp2 = F(Itemp(1+N_conv_pad:N+N_conv_pad,1+N_conv_pad:M+N_conv_pad));
    Itemp3 = abs(iF(Itemp2(N/2+1-Ncrop/2:N/2+1+Ncrop/2-1,M/2+1-Mcrop/2:M/2+1+Mcrop/2-1)));
    I_image(:,:,i) = Itemp3(N_bound_pad/undersamp_factor+1:N_bound_pad/undersamp_factor+Nb,N_bound_pad/undersamp_factor+1:N_bound_pad/undersamp_factor+Mb);
    
    
    
    for j = 1:N_defocus
%         for m = 1:N_mode
%             a = randn;
%             speckle_field_shift = iF(F(speckle_field).*exp(-1j*2*pi*ps*(fxxs*(pixel_shift_stack(2,idx_ps,idx_pt)+0*a) + fyys*(pixel_shift_stack(1,idx_ps,idx_pt)+0*a))));
%             speckle_field_shift_crop(:,:,i) = speckle_field_shift(1.5*N+1:2.5*N,1.5*M+1:2.5*M);

            Ic_temp = abs(iF(F(padarray(speckle_field_shift_crop(:,:,i).*T_c,[N_conv_pad,N_conv_pad],0)).*Pupil_obj(:,:,j).*exp(-1j*pi*lambda*(z_camera(j))*(fxx_aug.^2+fyy_aug.^2)))).^2;
            Ic_temp2 = F(Ic_temp(1+N_conv_pad:N+N_conv_pad,1+N_conv_pad:M+N_conv_pad));
            Ic_temp3 = abs(iF(Ic_temp2(N/2+1-Ncrop/2:N/2+1+Ncrop/2-1,M/2+1-Mcrop/2:M/2+1+Mcrop/2-1)));
            Ic_image(:,:,i,j) = Ic_temp3(N_bound_pad/undersamp_factor+1:N_bound_pad/undersamp_factor+Nb,N_bound_pad/undersamp_factor+1:N_bound_pad/undersamp_factor+Mb);

%             shift_fluctuation(m,i,j) = a;
%         end
    end
    
    if mod(i,100) == 0 || i == Nimg
        fprintf('Data generating process (%2d / %2d)\n',i,Nimg);
    end

end

%% Poisson + background noise

dark_current = 0;
photon_count = 3000;

I_image = I_image/max(I_image(:))*photon_count;
Ic_image = Ic_image/max(Ic_image(:))*photon_count;

% I_image = imnoise(( I_image/max(I_image(:)).*photon_count + dark_current)*1e-12,'poisson')*1e12;
% Ic_image = abs(imnoise(( Ic_image/max(Ic_image(:)).*photon_count)*1e-12,'poisson')*1e12 + dark_current + dark_current*randn(size(Ic_image)));


%% Save file
savefile='res_speckle_shift_15x15_dual_coherent_camera';
save(savefile,'pscrop','lambda','NA_obj','NAs','z_camera','I_image','I','speckle_intensity','pixel_shift_stack','speckle_intensity_shift_crop','Ic_image','speckle_field_shift_crop','-v7.3');

