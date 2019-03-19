function obj_rot = rotate_re(obj,theta)
    
    gpu_indicator = isa(obj,'gpuArray');
    
    [Ncrop,Mcrop] = size(obj);
    obj = padarray(obj,[Ncrop/2,Mcrop/2]);
    
    [N,M] = size(obj);
    
    if gpu_indicator == 0
        x = -M/2:(M/2-1);
        y = -N/2:(N/2-1);

        fx = ifftshift((-M/2:(M/2-1))/M);
        fy = ifftshift((-N/2:(N/2-1))/N);
    else
        x = gpuArray(-M/2:(M/2-1));
        y = gpuArray(-N/2:(N/2-1));

        fx = gpuArray(ifftshift((-M/2:(M/2-1))/M));
        fy = gpuArray(ifftshift((-N/2:(N/2-1))/N));
    end
    
    if abs(theta) <= pi/2
        
        alpha = -tan(theta/2);
        beta = sin(theta);
        gamma = -tan(theta/2);

        obj_rot = ifft(fft(obj).*exp(-1j*2*pi*alpha*fy(:)*x));
        obj_rot = ifft(fft(obj_rot,[],2).*exp(-1j*2*pi*beta*y(:)*fx),[],2);
        obj_rot = ifft(fft(obj_rot).*exp(-1j*2*pi*gamma*fy(:)*x));
    
    else
        if theta>0
            alpha = -tan((pi/2)/2);
            beta = sin(pi/2);
            gamma = -tan((pi/2)/2);
            
            obj_rot = ifft(fft(obj).*exp(-1j*2*pi*alpha*fy(:)*x));
            obj_rot = ifft(fft(obj_rot,[],2).*exp(-1j*2*pi*beta*y(:)*fx),[],2);
            obj_rot = ifft(fft(obj_rot).*exp(-1j*2*pi*gamma*fy(:)*x));
            
            alpha = -tan((theta-pi/2)/2);
            beta = sin(theta-pi/2);
            gamma = -tan((theta-pi/2)/2);
            
            obj_rot = ifft(fft(obj_rot).*exp(-1j*2*pi*alpha*fy(:)*x));
            obj_rot = ifft(fft(obj_rot,[],2).*exp(-1j*2*pi*beta*y(:)*fx),[],2);
            obj_rot = ifft(fft(obj_rot).*exp(-1j*2*pi*gamma*fy(:)*x));
            
        else
            alpha = -tan(-(pi/2)/2);
            beta = sin(-pi/2);
            gamma = -tan(-(pi/2)/2);
            
            obj_rot = ifft(fft(obj).*exp(-1j*2*pi*alpha*fy(:)*x));
            obj_rot = ifft(fft(obj_rot,[],2).*exp(-1j*2*pi*beta*y(:)*fx),[],2);
            obj_rot = ifft(fft(obj_rot).*exp(-1j*2*pi*gamma*fy(:)*x));
            
            alpha = -tan((theta+pi/2)/2);
            beta = sin(theta+pi/2);
            gamma = -tan((theta+pi/2)/2);
            
            obj_rot = ifft(fft(obj_rot).*exp(-1j*2*pi*alpha*fy(:)*x));
            obj_rot = ifft(fft(obj_rot,[],2).*exp(-1j*2*pi*beta*y(:)*fx),[],2);
            obj_rot = ifft(fft(obj_rot).*exp(-1j*2*pi*gamma*fy(:)*x));
        end
    end
    
%     mean_energy = mean2(abs(obj));
    
%     obj_rot(abs(obj_rot)<0.05*mean_energy) = 0;
    obj_rot = obj_rot(N/2-Ncrop/2+1:N/2+Ncrop/2,M/2-Mcrop/2+1:M/2+Mcrop/2);
    
    
end