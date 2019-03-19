# cSIM with speckle
This is a MATLAB implementation of coherent structured illumination microscopy algorithm with speckle <br/>
```speckle_group_simulation_dual_coherent_camera.m```: Creating simulation data for cSIM processing <br/>
```cSIM_random_pattern_optimized_sim.m```: Processing dataset with simulation settings <br/>
```cSIM_random_pattern_optimized_exp.m```: Processing dataset with experiment settings <br/>
```dftregistration.m```: DFT registration code implemented by [1] <br/>
```zernfun.m```: Zernike polynomial function implemented by [2] <br/>
```rotate_re.m```: Lossless image rotation function based on [3] <br/>

[1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, "Efficient subpixel image registration algorithms," Opt. Lett. 33, 156-158 (2008). <br/>
[2] https://www.mathworks.com/matlabcentral/fileexchange/7687-zernike-polynomials <br/>
[3] Alan W. Paeth, "A fast algorithm for general raster rotation," Proceedings of Graphics Interface and Vision Interface '86, 77-81 (1986) 

