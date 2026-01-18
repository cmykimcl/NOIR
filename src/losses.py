import torch
import torch.nn as nn

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class ICALoss(nn.Module):
    """
    Independent Component Analysis (ICA) Loss Function (Negentropy Approx).
    
    Optimized for 3D Tensors:
    - Expects input shape: (actuation_dim, averaging_dim, source_dim)
    - Computes ICA loss (non-Gaussianity) for each (averaging_dim, source_dim) pair
      using 'actuation_dim' as the samples.
    - Averages the loss over averaging_dim.
    """
    
    def __init__(self, g_func='logcosh'):
        super(ICALoss, self).__init__()
        
        if g_func not in ['logcosh', 'kurtosis']:
            raise ValueError("g_func must be 'logcosh' or 'kurtosis'")
            
        self.g_func_name = g_func
        
        if self.g_func_name == 'logcosh':
            self.g = lambda x: torch.log(torch.cosh(x))
            e_g_v = 0.3745587
        elif self.g_func_name == 'kurtosis':
            self.g = lambda x: x**4
            e_g_v = 3.0
            
        self.register_buffer('E_G_v', torch.tensor(e_g_v, dtype=torch.float32))

    def forward(self, b):
        """
        Args:
            b: Input tensor of shape (actuation_dim, averaging_dim, source_dim)
               - actuation_dim: number of samples (was batch_size in old version)
               - averaging_dim: dimension to average loss over (was dim1*dim2 in old 4D version)
               - source_dim: number of source channels
        
        Returns:
            Scalar loss value
        """
        # Shape: (actuation_dim, averaging_dim, source_dim)
        actuation_dim, averaging_dim, source_dim = b.shape
        
        if actuation_dim <= 1:
            return torch.tensor(0.0, device=b.device, requires_grad=True)

        # --- Preprocessing (Center & Whiten) ---
        # Compute mean/std over the actuation dimension (dim=0)
        b_mean = torch.mean(b, dim=0, keepdim=True)  # Shape: (1, averaging_dim, source_dim)
        b_centered = b - b_mean
        
        b_std = torch.std(b_centered, dim=0, keepdim=True)  # Shape: (1, averaging_dim, source_dim)
        b_whitened = b_centered / (b_std + 1e-8)  # Shape: (actuation_dim, averaging_dim, source_dim)
        # Clamp to prevent overflow in cosh
        b_whitened = torch.clamp(b_whitened, min=-10, max=10)
        # --- Calculate E[G(b_k)] ---
        G_b = self.g(b_whitened)  # Shape: (actuation_dim, averaging_dim, source_dim)
        E_G_b = torch.mean(G_b, dim=0)  # Expectation over actuation_dim -> (averaging_dim, source_dim)

        # --- Compute final loss ---
        diff = E_G_b - self.E_G_v.to(E_G_b.dtype)  # Shape: (averaging_dim, source_dim)
        diff_sq = diff**2  # Shape: (averaging_dim, source_dim)
        # diff_sq = diff_sq / (torch.mean(E_G_b**2) + 1e-8)  # Prevent division by zero

        # Sum over source channels (dim=-1), then average over averaging positions
        loss_per_position = torch.sum(diff_sq, dim=-1)  # Shape: (averaging_dim,)
        loss = torch.mean(loss_per_position)  # Final scalar
        
        return -loss