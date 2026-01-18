import torch
import torch.fft

class NoiseGeneratorNACA:
    """
    Generates various types of structured spatial noise for 2D data.
    Adapted for unstructured/body-fitted mesh.
    """
    def __init__(self, x_shape, device='cuda'):
        if len(x_shape) != 3:
            raise ValueError(f"Expected 3D shape (B, H, W), but got {x_shape}")
            
        self.B, self.H, self.W = x_shape
        self.device = device
        
        self._create_freq_grids()

    def _create_freq_grids(self):
        """Creates frequency grids for Fourier-space filtering."""
        freq_h = torch.fft.fftfreq(self.H, device=self.device)
        freq_w = torch.fft.fftfreq(self.W, device=self.device)
        
        grid_h, grid_w = torch.meshgrid(freq_h, freq_w, indexing='ij')
        
        self.freq_radius_sq = (grid_h**2 + grid_w**2).unsqueeze(0)
        self.max_freq_sq = torch.max(self.freq_radius_sq)

    def generate(self, x, noise_type, noise_std, **kwargs):
        """Main method to generate and apply noise."""
        if x.shape[0] != self.B:
            self.B = x.shape[0]
        
        if noise_type == 'partial_random':
            return self._noise_partial_random(x, noise_std, kwargs.get('k_fraction', 0.1))
        elif noise_type == 'harmonic_spatial':
            return self._noise_harmonic_spatial(x, noise_std)
        elif noise_type == 'bandlimited_lowpass':
            return self._noise_bandlimited(x, noise_std, 'lowpass', kwargs.get('cutoff_fraction', 0.1))
        elif noise_type == 'bandlimited_highpass':
            return self._noise_bandlimited(x, noise_std, 'highpass', kwargs.get('cutoff_fraction', 0.1))
        elif noise_type == 'spatial_block_noise':
            return self._noise_spatial_block(x, noise_std, kwargs.get('block_size_fraction', 0.25))
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")

    def _noise_partial_random(self, x, noise_std, k_fraction):
        """Applies white noise to random k% of features."""
        x_flat = x.reshape(self.B, -1)
        num_features = x_flat.shape[1]
        k = int(k_fraction * num_features)

        noise = torch.randn_like(x_flat) * noise_std
        mask_flat = torch.zeros_like(x_flat)
        
        indices = torch.topk(torch.rand_like(x_flat), k=k, dim=1).indices
        mask_flat.scatter_(dim=1, index=indices, value=1.0)
        
        x_noisy_flat = x_flat + (noise * mask_flat)
        
        return x_noisy_flat.reshape(x.shape), mask_flat.reshape(x.shape)

    def _noise_harmonic_spatial(self, x, noise_std):
        """Adds spatial sine waves."""
        grid_y = torch.linspace(0, 1, self.H, device=self.device).view(1, self.H, 1)
        grid_x = torch.linspace(0, 1, self.W, device=self.device).view(1, 1, self.W)
        
        k_y = torch.randint(1, 6, (self.B, 1, 1), device=self.device) * (2 * torch.pi)
        k_x = torch.randint(1, 6, (self.B, 1, 1), device=self.device) * (2 * torch.pi)
        phi = torch.rand(self.B, 1, 1, device=self.device) * (2 * torch.pi)
        
        noise = torch.sin(k_y * grid_y + k_x * grid_x + phi) * noise_std
        
        x_noisy = x + noise
        mask = torch.ones_like(x)
        return x_noisy, mask

    def _noise_bandlimited(self, x, noise_std, mode='lowpass', cutoff_fraction=0.1):
        """Band-limited noise."""
        noise = torch.randn_like(x)
        
        noise_fft = torch.fft.fftn(noise, dim=(1, 2))
        
        cutoff_freq_sq = cutoff_fraction**2 * self.max_freq_sq
        
        if mode == 'lowpass':
            filter_mask = (self.freq_radius_sq < cutoff_freq_sq).float()
        elif mode == 'highpass':
            filter_mask = (self.freq_radius_sq >= cutoff_freq_sq).float()
        else:
            raise ValueError("mode must be 'lowpass' or 'highpass'")

        filtered_fft = noise_fft * filter_mask
        filtered_noise = torch.fft.ifftn(filtered_fft, dim=(1, 2)).real
        
        std_dev = torch.std(filtered_noise, dim=(1, 2), keepdim=True)
        filtered_noise = (filtered_noise / (std_dev + 1e-6)) * noise_std
        
        x_noisy = x + filtered_noise
        mask = torch.ones_like(x)
        return x_noisy, mask

    def _noise_spatial_block(self, x, noise_std, block_size_fraction=0.25):
        """Adds a block of white noise to random location."""
        noise = torch.randn_like(x) * noise_std
        mask = torch.zeros_like(x)
        
        block_h = int(self.H * block_size_fraction)
        block_w = int(self.W * block_size_fraction)
        
        for b in range(self.B):
            y0 = torch.randint(0, self.H - block_h + 1, (1,)).item()
            x0 = torch.randint(0, self.W - block_w + 1, (1,)).item()
            
            mask[b, y0:y0+block_h, x0:x0+block_w] = 1.0
            
        x_noisy = x + (noise * mask)
        return x_noisy, mask
    
class NoiseGeneratorBurger:
    """
    Generates various types of structured noise for 1D tensor.
    Assumes input tensor has shape (B, s) where s is spatial resolution.
    """
    def __init__(self, x_shape, device='cuda'):
        if len(x_shape) != 2:
            raise ValueError(f"Expected 2D shape (B, s), but got {x_shape}")
            
        self.B, self.s = x_shape
        self.device = device
        
        self._create_grids()
        self._create_freq_grids()

    def _create_grids(self):
        """Creates normalized coordinate grid (0 to 1)."""
        self.grid_x = torch.linspace(0, 1, self.s, device=self.device).view(1, self.s)

    def _create_freq_grids(self):
        """Creates frequency grid for Fourier-space filtering."""
        freq_x = torch.fft.fftfreq(self.s, device=self.device)
        self.freq_radius = torch.abs(freq_x).unsqueeze(0)
        self.max_freq = torch.max(self.freq_radius)

    def generate(self, x, noise_type, noise_std, **kwargs):
        """
        Main method to generate and apply noise to a batch.
        
        Args:
            x: Clean input tensor (B, s)
            noise_type: Type of noise
            noise_std: Standard deviation of noise
            **kwargs: Additional parameters
        
        Returns:
            tuple: (x_noisy, mask)
        """
        if x.shape[0] != self.B:
            self.B = x.shape[0]
        
        if noise_type == 'partial_random':
            return self._noise_partial_random(x, noise_std, kwargs.get('k_fraction', 0.1))
        elif noise_type == 'harmonic_spatial':
            return self._noise_harmonic_spatial(x, noise_std)
        elif noise_type == 'bandlimited_lowpass':
            return self._noise_bandlimited(x, noise_std, 'lowpass', kwargs.get('cutoff_fraction', 0.1))
        elif noise_type == 'bandlimited_highpass':
            return self._noise_bandlimited(x, noise_std, 'highpass', kwargs.get('cutoff_fraction', 0.1))
        elif noise_type == 'spatial_block_noise':
            return self._noise_spatial_block(x, noise_std, kwargs.get('block_size_fraction', 0.25))
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")

    def _noise_partial_random(self, x, noise_std, k_fraction):
        """Applies white noise to a random k% of spatial features."""
        num_features = x.shape[1]
        k = int(k_fraction * num_features)

        noise = torch.randn_like(x) * noise_std
        mask = torch.zeros_like(x)
        
        indices = torch.topk(torch.rand_like(x), k=k, dim=1).indices
        mask.scatter_(dim=1, index=indices, value=1.0)
        
        x_noisy = x + (noise * mask)
        
        return x_noisy, mask

    def _noise_harmonic_spatial(self, x, noise_std):
        """Adds spatial sine waves."""
        k_x = torch.randint(1, 6, (self.B, 1), device=self.device) * (2 * torch.pi)
        phi = torch.rand(self.B, 1, device=self.device) * (2 * torch.pi)
        
        noise = torch.sin(k_x * self.grid_x + phi) * noise_std
        
        x_noisy = x + noise
        mask = torch.ones_like(x)
        return x_noisy, mask

    def _noise_bandlimited(self, x, noise_std, mode='lowpass', cutoff_fraction=0.1):
        """Band-limited noise (low-pass or high-pass)."""
        noise = torch.randn_like(x)
        
        noise_fft = torch.fft.fft(noise, dim=1)
        
        cutoff_freq = cutoff_fraction * self.max_freq
        
        if mode == 'lowpass':
            filter_mask = (self.freq_radius < cutoff_freq).float()
        elif mode == 'highpass':
            filter_mask = (self.freq_radius >= cutoff_freq).float()
        else:
            raise ValueError("mode must be 'lowpass' or 'highpass'")

        filtered_fft = noise_fft * filter_mask
        filtered_noise = torch.fft.ifft(filtered_fft, dim=1).real
        
        std_dev = torch.std(filtered_noise, dim=1, keepdim=True)
        filtered_noise = (filtered_noise / (std_dev + 1e-6)) * noise_std
        
        x_noisy = x + filtered_noise
        mask = torch.ones_like(x)
        return x_noisy, mask

    def _noise_spatial_block(self, x, noise_std, block_size_fraction=0.25):
        """Adds a block of white noise to a random spatial location."""
        noise = torch.randn_like(x) * noise_std
        mask = torch.zeros_like(x)
        
        block_size = int(self.s * block_size_fraction)
        
        for b in range(self.B):
            x0 = torch.randint(0, self.s - block_size + 1, (1,)).item()
            mask[b, x0:x0+block_size] = 1.0
            
        x_noisy = x + (noise * mask)
        return x_noisy, mask

class NoiseGeneratorDarcy:
    """
    Generates various types of structured spatial noise for 3D tensor (no time dimension).
    Assumes input tensor has shape (B, H, W).
    """
    def __init__(self, x_shape, device='cuda'):
        if len(x_shape) != 3:
            raise ValueError(f"Expected 3D shape (B, H, W), but got {x_shape}")
            
        self.B, self.H, self.W = x_shape
        self.device = device
        
        self._create_grids()
        self._create_freq_grids()

    def _create_grids(self):
        """Creates normalized coordinate grids (0 to 1)."""
        self.grid_y = torch.linspace(0, 1, self.H, device=self.device).view(1, self.H, 1)
        self.grid_x = torch.linspace(0, 1, self.W, device=self.device).view(1, 1, self.W)

    def _create_freq_grids(self):
        """Creates frequency grids for Fourier-space filtering."""
        freq_h = torch.fft.fftfreq(self.H, device=self.device)
        freq_w = torch.fft.fftfreq(self.W, device=self.device)
        
        grid_h, grid_w = torch.meshgrid(freq_h, freq_w, indexing='ij')
        
        self.freq_radius_sq = (grid_h**2 + grid_w**2).unsqueeze(0)
        self.max_freq_sq = torch.max(self.freq_radius_sq)

    def generate(self, x, noise_type, noise_std, **kwargs):
        if x.shape[0] != self.B:
            self.B = x.shape[0]
        
        if noise_type == 'partial_random':
            return self._noise_partial_random(x, noise_std, kwargs.get('k_fraction', 0.1))
        elif noise_type == 'harmonic_spatial':
            return self._noise_harmonic_spatial(x, noise_std)
        elif noise_type == 'bandlimited_lowpass':
            return self._noise_bandlimited(x, noise_std, 'lowpass', kwargs.get('cutoff_fraction', 0.1))
        elif noise_type == 'bandlimited_highpass':
            return self._noise_bandlimited(x, noise_std, 'highpass', kwargs.get('cutoff_fraction', 0.1))
        elif noise_type == 'spatial_block_noise':
            return self._noise_spatial_block(x, noise_std, kwargs.get('block_size_fraction', 0.25))
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")

    def _noise_partial_random(self, x, noise_std, k_fraction):
        x_flat = x.reshape(self.B, -1)
        num_features = x_flat.shape[1]
        k = int(k_fraction * num_features)

        noise = torch.randn_like(x_flat) * noise_std
        mask_flat = torch.zeros_like(x_flat)
        
        indices = torch.topk(torch.rand_like(x_flat), k=k, dim=1).indices
        mask_flat.scatter_(dim=1, index=indices, value=1.0)
        
        x_noisy_flat = x_flat + (noise * mask_flat)
        
        return x_noisy_flat.reshape(x.shape), mask_flat.reshape(x.shape)

    def _noise_harmonic_spatial(self, x, noise_std):
        k_y = torch.randint(1, 6, (self.B, 1, 1), device=self.device) * (2 * torch.pi)
        k_x = torch.randint(1, 6, (self.B, 1, 1), device=self.device) * (2 * torch.pi)
        phi = torch.rand(self.B, 1, 1, device=self.device) * (2 * torch.pi)
        
        noise = torch.sin(k_y * self.grid_y + k_x * self.grid_x + phi) * noise_std
        
        x_noisy = x + noise
        mask = torch.ones_like(x)
        return x_noisy, mask

    def _noise_bandlimited(self, x, noise_std, mode='lowpass', cutoff_fraction=0.1):
        noise = torch.randn_like(x)
        
        noise_fft = torch.fft.fftn(noise, dim=(1, 2))
        
        cutoff_freq_sq = cutoff_fraction**2 * self.max_freq_sq
        
        if mode == 'lowpass':
            filter_mask = (self.freq_radius_sq < cutoff_freq_sq).float()
        elif mode == 'highpass':
            filter_mask = (self.freq_radius_sq >= cutoff_freq_sq).float()
        else:
            raise ValueError("mode must be 'lowpass' or 'highpass'")

        filtered_fft = noise_fft * filter_mask
        filtered_noise = torch.fft.ifftn(filtered_fft, dim=(1, 2)).real
        
        std_dev = torch.std(filtered_noise, dim=(1, 2), keepdim=True)
        filtered_noise = (filtered_noise / (std_dev + 1e-6)) * noise_std
        
        x_noisy = x + filtered_noise
        mask = torch.ones_like(x)
        return x_noisy, mask

    def _noise_spatial_block(self, x, noise_std, block_size_fraction=0.25):
        noise = torch.randn_like(x) * noise_std
        mask = torch.zeros_like(x)
        
        block_h = int(self.H * block_size_fraction)
        block_w = int(self.W * block_size_fraction)
        
        for b in range(self.B):
            y0 = torch.randint(0, self.H - block_h + 1, (1,)).item()
            x0 = torch.randint(0, self.W - block_w + 1, (1,)).item()
            
            mask[b, y0:y0+block_h, x0:x0+block_w] = 1.0
            
        x_noisy = x + (noise * mask)
        return x_noisy, mask
