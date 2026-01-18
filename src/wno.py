"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). Wavelet Neural Operator for solving 
   parametric partialdifferential equations in computational mechanics problems.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from timeit import default_timer

""" Load required packages 
-- "PyWavelets"
    https://pywavelets.readthedocs.io/en/latest/install.html
    ($ conda install pywavelets)
"""

from torch.nn.parameter import Parameter


import ptwt, pywt
from ptwt.conv_transform_3 import wavedec3, waverec3
from pytorch_wavelets import DWT1D, IDWT1D
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from pytorch_wavelets import DWT, IDWT 

""" Def: 1d Wavelet convolutional layer """
class WaveConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet='db4', mode='symmetric', omega=4):
        super(WaveConv1d, self).__init__()

        """
        1D Wavelet layer. It does Wavelet Transform, linear transform, and
        Inverse Wavelet Transform. 
        
        Input parameters: 
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        level        : scalar, levels of wavelet decomposition
        size         : scalar, length of input 1D signal
        wavelet      : string, wavelet filter
        mode         : string, padding style for wavelet decomposition
        
        It initializes the kernel parameters: 
        -------------------------------------
        self.weights_approx : tensor, shape-[in_channels * out_channels * x]
                              kernel weights for Approximate wavelet coefficients
        self.weights_detail : tensor, shape-[in_channels * out_channels * x]
                              kernel weights for Detailed wavelet coefficients
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        if np.isscalar(size):
            self.size = size
        else:
            raise Exception("size: WaveConv1d accepts signal length in scalar only") 
        self.wavelet = wavelet 
        self.mode = mode
        dwt_ = DWT1D(wave=self.wavelet, J=self.level, mode=self.mode)
        dummy_data = torch.randn( 1,1,self.size ) 
        mode_data, _ = dwt_(dummy_data)
        self.modes = mode_data.shape[-1]
        self.omega = omega
        self.effective_modes = self.modes//self.omega+1
        
        # Parameter initilization
        self.scale = (1 / (in_channels*out_channels))
        self.weights_approx = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes, dtype=torch.cfloat))
        self.weights_detail = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes, dtype=torch.cfloat))

    # Element-wise multiplication
    def mul1d(self, input, weights):
        """
        Performs element-wise multiplication

        Input Parameters
        ----------------
        input   : tensor, shape-(batch * in_channel * x ) 
                  1D wavelet coefficients of input signal
        weights : tensor, shape-(in_channel * out_channel * x)
                  kernel weights of corresponding wavelet coefficients

        Returns
        -------
        convolved signal : tensor, shape-(batch * out_channel * x)
        """
        return torch.einsum("bix,iox->box", input, weights)
    
    # Spectral Convolution
    def spectralconv(self, waves, weights):
        """
        Performs spectral convolution using Fourier decomposition

        Input Parameters
        ----------
        waves : tensor, shape-[Batch * Channel * size(x)]
                signal to be convolved, here the wavelet coefficients.
        weights : tensor, shape-[in_channel * out_channel * size(x)]
                The weights/kernel of the neural network.

        Returns
        -------
        convolved signal : tensor, shape-[batch * out_channel * size(x)]

        """
        # Get the frequency componenets
        xw = torch.fft.rfft(waves)
        
        # Initialize the output
        conv_out = torch.zeros(waves.shape[0], self.out_channels, waves.shape[-1]//2+1, dtype=torch.cfloat, device=waves.device)
        
        # Perform Element-wise multiplication in spectral doamin
        conv_out[:,:,:self.effective_modes] = self.mul1d(xw[:,:,:self.effective_modes], weights)
        return torch.fft.irfft(conv_out, n=waves.shape[-1])

    def forward(self, x):
        """
        Input parameters: 
        -----------------
        x : tensor, shape-[Batch * Channel * x]
        
        Output parameters: 
        ------------------
        x : tensor, shape-[Batch * Channel * x]
        """
        if x.shape[-1] > self.size:
            factor = int(np.log2(x.shape[-1] // self.size))
            
            # Compute single tree Discrete Wavelet coefficients using some wavelet  
            dwt = DWT1D(wave=self.wavelet, J=self.level+factor, mode=self.mode).to(x.device)
            x_ft, x_coeff = dwt(x)
            
        elif x.shape[-1] < self.size:
            factor = int(np.log2(self.size // x.shape[-1]))
            
            # Compute single tree Discrete Wavelet coefficients using some wavelet  
            dwt = DWT1D(wave=self.wavelet, J=self.level-factor, mode=self.mode).to(x.device)
            x_ft, x_coeff = dwt(x)
            
        else:
            # Compute single tree Discrete Wavelet coefficients using some wavelet  
            dwt = DWT1D(wave=self.wavelet, J=self.level, mode=self.mode).to(x.device)
            x_ft, x_coeff = dwt(x)
            
        # Instantiate higher level coefficients as zeros
        out_ft = torch.zeros_like(x_ft, device= x.device)
        out_coeff = [torch.zeros_like(coeffs, device= x.device) for coeffs in x_coeff]
        
        # Multiply the final low pass wavelet coefficients
        out_ft = self.spectralconv(x_ft, self.weights_approx)
        
        # Multiply the final high pass wavelet coefficients
        out_coeff[-1] = self.spectralconv(x_coeff[-1].clone(), self.weights_detail)
    
        # Reconstruct the signal
        idwt = IDWT1D(wave=self.wavelet, mode=self.mode).to(x.device)
        x = idwt((out_ft, out_coeff)) 
        return x


""" Def: 2d Wavelet convolutional layer (discrete) """
class WaveConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet, mode='symmetric', omega=8):
        super(WaveConv2d, self).__init__()

        """
        2D Wavelet layer. It does DWT, linear transform, and Inverse dWT. 
        
        Input parameters: 
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        level        : scalar, levels of wavelet decomposition
        size         : scalar, length of input 1D signal
        wavelet      : string, wavelet filters
        mode         : string, padding style for wavelet decomposition
        
        It initializes the kernel parameters: 
        -------------------------------------
        self.weights1 : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for Approximate wavelet coefficients
        self.weights2 : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for Horizontal-Detailed wavelet coefficients
        self.weights3 : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for Vertical-Detailed wavelet coefficients
        self.weights4 : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for Diagonal-Detailed wavelet coefficients
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        if isinstance(size, list):
            if len(size) != 2:
                raise Exception('size: WaveConv2dCwt accepts the size of 2D signal in list with 2 elements')
            else:
                self.size = size
        else:
            raise Exception('size: WaveConv2dCwt accepts size of 2D signal is list')
        self.wavelet = wavelet       
        self.mode = mode
        dummy_data = torch.randn( 1,1,*self.size )        
        dwt_ = DWT(J=self.level, mode=self.mode, wave=self.wavelet)
        mode_data, _ = dwt_(dummy_data)
        self.modes1 = mode_data.shape[-2]
        self.modes2 = mode_data.shape[-1]
        self.omega = omega
        self.effective_modes_x = self.modes1//self.omega+1
        self.effective_modes_y = self.modes2//self.omega+1
        
        # Parameter initilization
        self.scale = (1 / (in_channels * out_channels))
        self.weights_a1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_x, dtype=torch.cfloat))
        self.weights_a2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_x, dtype=torch.cfloat))
        self.weights_h1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_x, dtype=torch.cfloat))
        self.weights_h2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_x, dtype=torch.cfloat))
        self.weights_v1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_x, dtype=torch.cfloat))
        self.weights_v2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_x, dtype=torch.cfloat))
        self.weights_d1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_x, dtype=torch.cfloat))
        self.weights_d2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_x, dtype=torch.cfloat))

    # Element-wise multiplication
    def mul2d(self, input, weights):
        """
        Performs element-wise multiplication

        Input Parameters
        ----------------
        input   : tensor, shape-(batch * in_channel * x * y )
                  2D wavelet coefficients of input signal
        weights : tensor, shape-(in_channel * out_channel * x * y)
                  kernel weights of corresponding wavelet coefficients

        Returns
        -------
        convolved signal : tensor, shape-(batch * out_channel * x * y)
        """
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    # Spectral Convolution
    def spectralconv(self, waves, weights1, weights2):
        """
        Performs spectral convolution using Fourier decomposition

        Input Parameters
        ----------
        waves : tensor, shape-[Batch * Channel * size(x)]
                signal to be convolved, here the wavelet coefficients.
        weights : tensor, shape-[in_channel * out_channel * size(x)]
                The weights/kernel of the neural network.

        Returns
        -------
        convolved signal : tensor, shape-[batch * out_channel * size(x)]

        """
        # Get the frequency componenets
        xw = torch.fft.rfft2(waves)
        
        # Initialize the output
        conv_out = torch.zeros(waves.shape[0], self.out_channels, waves.shape[-2], waves.shape[-1]//2+1, dtype=torch.cfloat, device=waves.device)
        
        # Perform Element-wise multiplication in spectral doamin
        conv_out[:,:,:self.effective_modes_x,:self.effective_modes_y] = self.mul2d(xw[:,:,:self.effective_modes_x,:self.effective_modes_y], weights1)
        conv_out[:,:,-self.effective_modes_x:,:self.effective_modes_y] = self.mul2d(xw[:,:,-self.effective_modes_x:,:self.effective_modes_y], weights2)
        return torch.fft.irfft2(conv_out, s=(waves.shape[-2], waves.shape[-1]))
    
    def forward(self, x):
        """
        Input parameters: 
        -----------------
        x : tensor, shape-[Batch * Channel * x * y]
        Output parameters: 
        ------------------
        x : tensor, shape-[Batch * Channel * x * y]
        """
        if x.shape[-1] > self.size[-1]:
            factor = int(np.log2(x.shape[-1] // self.size[-1]))
            
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT(J=self.level+factor, mode=self.mode, wave=self.wavelet).to(x.device)
            x_ft, x_coeff = dwt(x)
            
        elif x.shape[-1] < self.size[-1]:
            factor = int(np.log2(self.size[-1] // x.shape[-1]))
            
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT(J=self.level-factor, mode=self.mode, wave=self.wavelet).to(x.device)
            x_ft, x_coeff = dwt(x)
        
        else:
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT(J=self.level, mode=self.mode, wave=self.wavelet).to(x.device)
            x_ft, x_coeff = dwt(x)

        # Instantiate higher level coefficients as zeros
        out_ft = torch.zeros_like(x_ft, device= x.device)
        out_coeff = [torch.zeros_like(coeffs, device= x.device) for coeffs in x_coeff]
        
        # Multiply the final approximate Wavelet modes
        out_ft = self.spectralconv(x_ft, self.weights_a1, self.weights_a2)
        
        # Multiply the final detailed wavelet coefficients
        out_coeff[-1][:,:,0,:,:] = self.spectralconv(x_coeff[-1][:,:,0,:,:].clone(), self.weights_h1, self.weights_h2)
        out_coeff[-1][:,:,1,:,:] = self.spectralconv(x_coeff[-1][:,:,1,:,:].clone(), self.weights_v1, self.weights_v2)
        out_coeff[-1][:,:,2,:,:] = self.spectralconv(x_coeff[-1][:,:,2,:,:].clone(), self.weights_d1, self.weights_d2)
        
        # Return to physical space        
        idwt = IDWT(mode=self.mode, wave=self.wavelet).to(x.device)
        x = idwt((out_ft, out_coeff))
        return x

    
""" Def: 2d Wavelet convolutional layer (slim continuous) """
class WaveConv2dCwt(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet1, wavelet2, omega=8):
        super(WaveConv2dCwt, self).__init__()

        """
        !! It is computationally expensive than the discrete "WaveConv2d" !!
        2D Wavelet layer. It does SCWT (Slim continuous wavelet transform),
                                linear transform, and Inverse dWT. 
        
        Input parameters: 
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        level        : scalar, levels of wavelet decomposition
        size         : scalar, length of input 1D signal
        wavelet1     : string, Specifies the first level biorthogonal wavelet filters
        wavelet2     : string, Specifies the second level quarter shift filters
        mode         : string, padding style for wavelet decomposition
        
        It initializes the kernel parameters: 
        -------------------------------------
        self.weights0 : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for Approximate wavelet coefficients
        self.weights- 15r, 45r, 75r, 105r, 135r, 165r : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for REAL wavelet coefficients at 15, 45, 75, 105, 135, 165 angles
        self.weights- 15c, 45c, 75c, 105c, 135c, 165c : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for COMPLEX wavelet coefficients at 15, 45, 75, 105, 135, 165 angles
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        if isinstance(size, list):
            if len(size) != 2:
                raise Exception('size: WaveConv2dCwt accepts the size of 2D signal in list with 2 elements')
            else:
                self.size = size
        else:
            raise Exception('size: WaveConv2dCwt accepts size of 2D signal is list')
        self.wavelet_level1 = wavelet1
        self.wavelet_level2 = wavelet2        
        dummy_data = torch.randn( 1,1,*self.size ) 
        dwt_ = DTCWTForward(J=self.level, biort=self.wavelet_level1, qshift=self.wavelet_level2)
        mode_data, mode_coef = dwt_(dummy_data)
        self.modes1 = mode_data.shape[-2]
        self.modes2 = mode_data.shape[-1]
        self.modes21 = mode_coef[-1].shape[-3]
        self.modes22 = mode_coef[-1].shape[-2]
        self.omega = omega
        self.effective_modes_x = self.modes1//self.omega+1
        self.effective_modes_y = self.modes2//self.omega+1
        self.effective_modes_xx = self.modes21//self.omega+1
        self.effective_modes_yy = self.modes22//self.omega+1
        
        # Parameter initilization
        self.scale = (1 / (in_channels * out_channels))
        self.weights_01 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, dtype=torch.cfloat))
        self.weights_02 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, dtype=torch.cfloat))
        self.weights_15r1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_15r2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_15c1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_15c2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_45r1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_45r2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_45c1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_45c2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_75r1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_75r2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_75c1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_75c2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_105r1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_105r2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_105c1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_105c2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_135r1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_135r2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_135c1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_135c2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_165r1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_165r2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_165c1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_165c2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))

    # Convolution
    def mul2d(self, input, weights):
        """
        Performs element-wise multiplication

        Input Parameters
        ----------------
        input   : tensor, shape-(batch * in_channel * x * y )
                  2D wavelet coefficients of input signal
        weights : tensor, shape-(in_channel * out_channel * x * y)
                  kernel weights of corresponding wavelet coefficients

        Returns
        -------
        convolved signal : tensor, shape-(batch * out_channel * x * y)
        """
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    # Spectral Convolution
    def spectralconv(self, waves, weights1, weights2):
        """
        Performs spectral convolution using Fourier decomposition

        Input Parameters
        ----------
        waves : tensor, shape-[Batch * Channel * size(x)]
                signal to be convolved, here the wavelet coefficients.
        weights : tensor, shape-[in_channel * out_channel * size(x)]
                The weights/kernel of the neural network.

        Returns
        -------
        convolved signal : tensor, shape-[batch * out_channel * size(x)]

        """
        # Get the frequency componenets
        modes1, modes2 = weights2.shape[-2], weights2.shape[-1]
        xw = torch.fft.rfft2(waves)
        
        # Initialize the output
        conv_out = torch.zeros(waves.shape[0], self.out_channels, waves.shape[-2], waves.shape[-1]//2+1, dtype=torch.cfloat, device=waves.device)
        
        # Perform Element-wise multiplication in spectral doamin
        conv_out[:,:,:modes1,:modes2] = self.mul2d(xw[:,:,:modes1,:modes2], weights1)
        conv_out[:,:,-modes1:,:modes2] = self.mul2d(xw[:,:,-modes1:,:modes2], weights2)
        return torch.fft.irfft2(conv_out, s=(waves.shape[-2], waves.shape[-1]))

    def forward(self, x):
        """
        Input parameters: 
        -----------------
        x : tensor, shape-[Batch * Channel * x * y]
        Output parameters: 
        ------------------
        x : tensor, shape-[Batch * Channel * x * y]
        """      
        if x.shape[-1] > self.size[-1]:
            factor = int(np.log2(x.shape[-1] // self.size[-1]))
            
            # Compute dual tree continuous Wavelet coefficients
            cwt = DTCWTForward(J=self.level+factor, biort=self.wavelet_level1, qshift=self.wavelet_level2).to(x.device)
            x_ft, x_coeff = cwt(x)
            
        elif x.shape[-1] < self.size[-1]:
            factor = int(np.log2(self.size[-1] // x.shape[-1]))
            
            # Compute dual tree continuous Wavelet coefficients
            cwt = DTCWTForward(J=self.level-factor, biort=self.wavelet_level1, qshift=self.wavelet_level2).to(x.device)
            x_ft, x_coeff = cwt(x)            
        else:
            # Compute dual tree continuous Wavelet coefficients 
            cwt = DTCWTForward(J=self.level, biort=self.wavelet_level1, qshift=self.wavelet_level2).to(x.device)
            x_ft, x_coeff = cwt(x)
        
        # Instantiate higher level coefficients as zeros
        out_ft = torch.zeros_like(x_ft, device= x.device)
        out_coeff = [torch.zeros_like(coeffs, device= x.device) for coeffs in x_coeff]
        
        # Multiply the final approximate Wavelet modes
        out_ft = self.spectralconv(x_ft, self.weights_01, self.weights_02)
        # Multiply the final detailed wavelet coefficients        
        out_coeff[-1][:,:,0,:,:,0] = self.spectralconv(x_coeff[-1][:,:,0,:,:,0].clone(), self.weights_15r1, self.weights_15r2)
        out_coeff[-1][:,:,0,:,:,1] = self.spectralconv(x_coeff[-1][:,:,0,:,:,1].clone(), self.weights_15c1, self.weights_15c2)
        out_coeff[-1][:,:,1,:,:,0] = self.spectralconv(x_coeff[-1][:,:,1,:,:,0].clone(), self.weights_45r1, self.weights_45r2)
        out_coeff[-1][:,:,1,:,:,1] = self.spectralconv(x_coeff[-1][:,:,1,:,:,1].clone(), self.weights_45c1, self.weights_45c2)
        out_coeff[-1][:,:,2,:,:,0] = self.spectralconv(x_coeff[-1][:,:,2,:,:,0].clone(), self.weights_75r1, self.weights_75r2)
        out_coeff[-1][:,:,2,:,:,1] = self.spectralconv(x_coeff[-1][:,:,2,:,:,1].clone(), self.weights_75c1, self.weights_75c2)
        out_coeff[-1][:,:,3,:,:,0] = self.spectralconv(x_coeff[-1][:,:,3,:,:,0].clone(), self.weights_105r1, self.weights_105r2)
        out_coeff[-1][:,:,3,:,:,1] = self.spectralconv(x_coeff[-1][:,:,3,:,:,1].clone(), self.weights_105c1, self.weights_105c2)
        out_coeff[-1][:,:,4,:,:,0] = self.spectralconv(x_coeff[-1][:,:,4,:,:,0].clone(), self.weights_135r1, self.weights_135r2)
        out_coeff[-1][:,:,4,:,:,1] = self.spectralconv(x_coeff[-1][:,:,4,:,:,1].clone(), self.weights_135c1, self.weights_135c2)
        out_coeff[-1][:,:,5,:,:,0] = self.spectralconv(x_coeff[-1][:,:,5,:,:,0].clone(), self.weights_165r1, self.weights_165r2)
        out_coeff[-1][:,:,5,:,:,1] = self.spectralconv(x_coeff[-1][:,:,5,:,:,1].clone(), self.weights_165c1, self.weights_165c2)        
        
        # Reconstruct the signal
        icwt = DTCWTInverse(biort=self.wavelet_level1, qshift=self.wavelet_level2).to(x.device)
        x = icwt((out_ft, out_coeff))
        return x
    
    
""" Def: 3d Wavelet convolutional layer """
class WaveConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet='db4', mode='periodic', omega=6):
        super(WaveConv3d, self).__init__()

        """
        3D Wavelet layer. It does 3D DWT, linear transform, and Inverse dWT.    
        
        Input parameters: 
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        level        : scalar, levels of wavelet decomposition
        size         : scalar, length of input 1D signal
        wavelet      : string, Specifies the first level biorthogonal wavelet filters
        mode         : string, padding style for wavelet decomposition
        
        It initializes the kernel parameters: 
        -------------------------------------
        self.weights0 : tensor, shape-[in_channels * out_channels * x * y * z]
                        kernel weights for Approximate wavelet coefficients
        self.weights_ : tensor, shape-[in_channels * out_channels * x * y * z]
                        kernel weights for Detailed wavelet coefficients 
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        if isinstance(size, list):
            if len(size) != 3:
                raise Exception('size: WaveConv2dCwt accepts the size of 3D signal in list with 3 elements')
            else:
                self.size = size
        else:
            raise Exception('size: WaveConv2dCwt accepts size of 3D signal is list')
        self.wavelet = wavelet
        self.mode = mode
        dummy_data = torch.randn( [*self.size] ).unsqueeze(0)
        mode_data = wavedec3(dummy_data, pywt.Wavelet(self.wavelet), level=self.level, mode=self.mode)
        self.modes1 = mode_data[0].shape[-3]
        self.modes2 = mode_data[0].shape[-2]
        self.modes3 = mode_data[0].shape[-1]
        self.omega = omega
        self.effective_modes_x = self.modes1//self.omega+1
        self.effective_modes_y = self.modes2//self.omega+1
        self.effective_modes_z = self.modes3//self.omega+1

        self.scale = (1 / (in_channels * out_channels))
        self.weights_a1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_a2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_a3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_a4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_aad1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_aad2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_aad3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_aad4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_ada1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_ada2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_ada3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_ada4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_add1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_add2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_add3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_add4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_daa1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_daa2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_daa3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_daa4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_dad1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_dad2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_dad3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_dad4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_dda1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_dda2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_dda3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_dda4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_ddd1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_ddd2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_ddd3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_ddd4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))

    # Convolution
    def mul3d(self, input, weights):
        """
        Performs element-wise multiplication

        Input Parameters
        ----------------
        input   : tensor, shape-(in_channel * x * y * z)
                  3D wavelet coefficients of input signal
        weights : tensor, shape-(in_channel * out_channel * x * y * z)
                  kernel weights of corresponding wavelet coefficients

        Returns
        -------
        convolved signal : tensor, shape-(out_channel * x * y * z)
        """
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)
    
    # Spectral Convolution
    def spectralconv(self, waves, weights1, weights2, weights3, weights4):
        """
        Performs spectral convolution using Fourier decomposition

        Input Parameters
        ----------
        waves : tensor, shape-[Batch * Channel * size(x)]
                signal to be convolved, here the wavelet coefficients.
        weights : tensor, shape-[in_channel * out_channel * size(x)]
                The weights/kernel of the neural network.

        Returns
        -------
        convolved signal : tensor, shape-[batch * out_channel * size(x)]

        """
        # Get the frequency componenets
        xw = torch.fft.rfftn(waves, dim=[-3,-2,-1])
        
        # Initialize the output
        conv_out = torch.zeros(waves.shape[0], self.out_channels, waves.shape[-3], waves.shape[-2], waves.shape[-1]//2+1, dtype=torch.cfloat, device=waves.device)
        
        # Perform Element-wise multiplication in spectral doamin
        conv_out[:,:,:self.effective_modes_x,:self.effective_modes_y,:self.effective_modes_z] = self.mul3d(xw[:,:,:self.effective_modes_x,:self.effective_modes_y,:self.effective_modes_z], weights1)
        conv_out[:,:,-self.effective_modes_x:,:self.effective_modes_y,:self.effective_modes_z] = self.mul3d(xw[:,:,-self.effective_modes_x:,:self.effective_modes_y,:self.effective_modes_z], weights2)
        conv_out[:,:,:self.effective_modes_x,-self.effective_modes_y:,:self.effective_modes_z] = self.mul3d(xw[:,:,:self.effective_modes_x,-self.effective_modes_y:,:self.effective_modes_z], weights3)
        conv_out[:,:,-self.effective_modes_x:,-self.effective_modes_y:,:self.effective_modes_z] = self.mul3d(xw[:,:,-self.effective_modes_x:,-self.effective_modes_y:,:self.effective_modes_z], weights4)
        return torch.fft.irfftn(conv_out, s=(waves.shape[-3], waves.shape[-2], waves.shape[-1]))
    
    def forward(self, x):
        """
        Input parameters: 
        -----------------
        x : tensor, shape-[Batch * Channel * x * y * z]
        Output parameters: 
        ------------------
        x : tensor, shape-[Batch * Channel * x * y * z]
        """ 
        if x.shape[-1] > self.size[-1]:
            factor = int(np.log2(x.shape[-1] // self.size[-1]))
            
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            x_coeff = wavedec3(x, pywt.Wavelet(self.wavelet), level=self.level+factor, mode=self.mode)
        
        elif x.shape[-1] < self.size[-1]:
            factor = int(np.log2(self.size[-1] // x.shape[-1]))
            
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            x_coeff = wavedec3(x, pywt.Wavelet(self.wavelet), level=self.level-factor, mode=self.mode)        
        else:
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            x_coeff = wavedec3(x, pywt.Wavelet(self.wavelet), level=self.level, mode=self.mode)
        
        # Convert tuple to list
        x_coeff = list(x_coeff)
        
        # Multiply relevant Wavelet modes
        x_coeff[0] = self.spectralconv(x_coeff[0].clone(), self.weights_a1, self.weights_a2, self.weights_a3, self.weights_a4)
        x_coeff[1]['aad'] = self.spectralconv(x_coeff[1]['aad'].clone(), self.weights_aad1, self.weights_aad2, self.weights_aad3, self.weights_aad4)
        x_coeff[1]['ada'] = self.spectralconv(x_coeff[1]['ada'].clone(), self.weights_ada1, self.weights_ada2, self.weights_ada3, self.weights_ada4)
        x_coeff[1]['add'] = self.spectralconv(x_coeff[1]['add'].clone(), self.weights_add1, self.weights_add2, self.weights_add3, self.weights_add4)
        x_coeff[1]['daa'] = self.spectralconv(x_coeff[1]['daa'].clone(), self.weights_daa1, self.weights_daa2, self.weights_daa3, self.weights_daa4)
        x_coeff[1]['dad'] = self.spectralconv(x_coeff[1]['dad'].clone(), self.weights_dad1, self.weights_dad2, self.weights_dad3, self.weights_dad4)
        x_coeff[1]['dda'] = self.spectralconv(x_coeff[1]['dda'].clone(), self.weights_dda1, self.weights_dda2, self.weights_dda3, self.weights_dda4)
        x_coeff[1]['ddd'] = self.spectralconv(x_coeff[1]['ddd'].clone(), self.weights_ddd1, self.weights_ddd2, self.weights_ddd3, self.weights_ddd4)
            
        # Instantiate higher level coefficients as zeros
        for jj in range(2, self.level + 1):
            x_coeff[jj] = {key: torch.zeros([*x_coeff[jj][key].shape], device=x.device)
                            for key in x_coeff[jj].keys()}
        
        # Convert back to the tuple
        x_coeff = tuple(x_coeff)
        
        # Return to physical space        
        x = waverec3(x_coeff, pywt.Wavelet(self.wavelet))
        return x
    
class WNO1d(nn.Module):
    def __init__(self, width, level, layers, size, wavelet, in_channel, grid_range, omega, padding=0):
        super(WNO1d, self).__init__()

        """
        The WNO network. It contains l-layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. l-layers of the integral operators v(j+1)(x) = g(K.v + W.v)(x).
            --> W is defined by self.w; K is defined by self.conv.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        Input : 2-channel tensor, Initial condition and location (a(x), x)
              : shape: (batchsize * x=s * c=2)
        Output: Solution of a later timestep (u(x))
              : shape: (batchsize * x=s * c=1)
              
        Input parameters:
        -----------------
        width : scalar, lifting dimension of input
        level : scalar, number of wavelet decomposition
        layers: scalar, number of wavelet kernel integral blocks
        size  : scalar, signal length
        wavelet: string, wavelet filter
        in_channel: scalar, channels in input including grid
        grid_range: scalar (for 1D), right support of 1D domain
        padding   : scalar, size of zero padding
        """

        self.level = level
        self.width = width
        self.layers = layers
        self.size = size
        self.wavelet = wavelet
        self.omega = omega
        self.in_channel = in_channel
        self.grid_range = grid_range 
        self.padding = padding
        
        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()
        
        self.fc0 = nn.Linear(self.in_channel, self.width) # input channel is 2: (a(x), x)
        for i in range( self.layers ):
            self.conv.append( WaveConv1d(self.width, self.width, self.level, size=self.size,
                                         wavelet=self.wavelet, omega=self.omega) )
            self.w.append( nn.Conv1d(self.width, self.width, 1) )
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)              # Shape: Batch * x * Channel
        x = x.permute(0, 2, 1)       # Shape: Batch * Channel * x
        if self.padding != 0:
            x = F.pad(x, [0,self.padding]) 
        
        for index, (convl, wl) in enumerate( zip(self.conv, self.w) ):
            x = convl(x) + wl(x) 
            if index != self.layers - 1:   # Final layer has no activation    
                x = F.mish(x)        # Shape: Batch * Channel * x 
                
        if self.padding != 0:
            x = x[..., :-self.padding] 
        x = x.permute(0, 2, 1)       # Shape: Batch * x * Channel
        x = F.mish( self.fc1(x) )    # Shape: Batch * x * Channel
        x = self.fc2(x)              # Shape: Batch * x * Channel
        return x

    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, self.grid_range, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)
    
# %%
""" The forward operation """
class WNO2d(nn.Module):
    def __init__(self, width, level, layers, size, wavelet, in_channel, out_channel, grid_range, omega, padding=0):
        super(WNO2d, self).__init__()

        """
        The WNO network. It contains l-layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. l-layers of the integral operators v(j+1)(x,y) = g(K.v + W.v)(x,y).
            --> W is defined by self.w; K is defined by self.conv.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        Input : 3-channel tensor, Initial input and location (a(x,y), x,y)
              : shape: (batchsize * x=width * x=height * c=3)
        Output: Solution of a later timestep (u(x,y))
              : shape: (batchsize * x=width * x=height * c=1)
              
        Input parameters:
        -----------------
        width : scalar, lifting dimension of input
        level : scalar, number of wavelet decomposition
        layers: scalar, number of wavelet kernel integral blocks
        size  : list with 2 elements (for 2D), image size
        wavelet: string, wavelet filter
        in_channel: scalar, channels in input including grid
        grid_range: list with 2 elements (for 2D), right supports of 2D domain
        padding   : scalar, size of zero padding
        """

        self.level = level
        self.width = width
        self.layers = layers
        self.size = size
        self.wavelet1 = wavelet[0]
        self.wavelet2 = wavelet[1]
        self.omega = omega
        self.in_channel = in_channel
        self.grid_range = grid_range 
        self.padding = padding
        
        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()
        
        self.fc0 = nn.Linear(self.in_channel, self.width) # input channel is 3: (a(x, y), x, y)
        for i in range( self.layers ):
            self.conv.append( WaveConv2dCwt(self.width, self.width, self.level, size=self.size,
                                            wavelet1=self.wavelet1, wavelet2=self.wavelet2, omega=self.omega) )
            self.w.append( nn.Conv2d(self.width, self.width, 1) )
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channel)

    def forward(self, x):
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)    
        x = self.fc0(x)                      # Shape: Batch * x * y * Channel
        x = x.permute(0, 3, 1, 2)            # Shape: Batch * Channel * x * y
        if self.padding != 0:
            x = F.pad(x, [0,self.padding, 0,self.padding]) 
        
        for index, (convl, wl) in enumerate( zip(self.conv, self.w) ):
            x = convl(x) + wl(x) 
            if index != self.layers - 1:     # Final layer has no activation    
                x = F.mish(x)                # Shape: Batch * Channel * x * y
                
        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding]     
        x = x.permute(0, 2, 3, 1)            # Shape: Batch * x * y * Channel
        x = F.mish( self.fc1(x) )            # Shape: Batch * x * y * Channel
        x = self.fc2(x)                      # Shape: Batch * x * y * Channel
        return x
    
    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, self.grid_range[0], size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, self.grid_range[1], size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)