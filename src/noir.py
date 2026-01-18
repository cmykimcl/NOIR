import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from fcn import FCN, Penultimate_FCN
from fno import SpectralConv1d

class OrthogonalLayerDeepONet(nn.Module):
    """
    Classical DeepONet (FCN-FCN) implementation using operator-style formulation.
    branch_arch: list of layer sizes for branch network, e.g., [input_dim, 64, 128]
    trunk_arch: list of layer sizes for trunk network
    d_new: The new intermediate dimension to transform to before summation
    num_outputs: number of output channels per trunk query (e.g., temperature, pressure, etc.)
    """

    def __init__(self, branch_arch, trunk_arch, p_ica=24, num_outputs=1, activation_fn=nn.ReLU):
        super(OrthogonalLayerDeepONet, self).__init__()

        self.num_outputs = num_outputs
        self.branch_output_size = branch_arch[-1]  # This is 'd'
        self.p_ica = p_ica

        # Build branch and trunk networks
        self.branch_net = Penultimate_FCN(branch_arch, activation_fn)

        # Automatically append output layer size to trunk_arch
        trunk_output_size = self.branch_output_size * self.num_outputs
        full_trunk_arch = trunk_arch + [trunk_output_size]

        self.trunk_net = FCN(full_trunk_arch, activation_fn)

        self.intermediate_transform = nn.Sequential(
            nn.Linear(self.branch_output_size, self.p_ica),
            # activation_fn()
        )

    def forward(self, branch_input, trunk_input):
        """
        branch_input: Tensor of shape (B, input_dim, in_channel)
        trunk_input: Tensor of shape (B, P, D_trunk)
        """
        B, P, _ = trunk_input.shape
        d = self.branch_output_size
        C = self.num_outputs

        # Branch: (B, d)
        # remove the channel dimension if it exists
        branch_input = branch_input.squeeze(-1) if branch_input.dim() == 3 else branch_input
        branch_output, penult = self.branch_net(branch_input)

        ## Trunk:
        trunk_input_reshaped = trunk_input.reshape(-1, trunk_input.shape[-1])  # (B*P, D_trunk)
        trunk_output = self.trunk_net(trunk_input_reshaped)  # (B*P, d*C)
        trunk_output = trunk_output.reshape(B, P, d, C)  # (B, P, d, C)
        
        # --- 2. Calculate Intermediate Components ---
        # Unsqueeze branch_output to (B, 1, d, 1) to make it broadcastable
        branch_broadcastable = branch_output.unsqueeze(1).unsqueeze(3)
        # This is Psi_k = b_k * t_k. Shape: (B, P, d, C)
        intermediate_components = branch_broadcastable * trunk_output
        
        # --- 3. Apply Intermediate Transformation ---
        # Permute to (B, P, C, d)
        x = intermediate_components.permute(0, 1, 3, 2)
        # Apply layer: (B, P, C, d) -> (B, P, C, d_new)
        x = self.intermediate_transform(x)
        # --- 4. Compute Final Output ---
        output = torch.sum(x, dim=-1)  # Shape: (B, P, C)
        # Return both
        return output, x 
    
from fno import SpectralConv2d

class FNO2d_layer_ortho(nn.Module):
    def __init__(self, modes1, modes2,  width, p_ica = 24, in_channels=12, out_channels=10):
        super(FNO2d_layer_ortho, self).__init__()

        """
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(in_channels, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        # self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        # self.w3 = nn.Conv1d(self.width, self.width, 1)


        self.fc1 = nn.Linear(self.width, 128)
        # We still need fc2 to get the weights and the out_channels dimension
        self.fc2 = nn.Linear(128, out_channels) 

        # This layer transforms the components from dimension 'C_in' (128) to 'ica_dim'
        self.ica_projection = nn.Sequential(
            nn.Linear(128, p_ica),  # 128 is self.fc2.in_features
        )


    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1_ = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1_ + x2
        x = F.relu(x)

        # x1 = self.conv3(x)
        # x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        # x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x_ = F.relu(x) # Shape: (B, H, W, 128)
        
        # W shape: (out_channels, in_channels) -> (C_out, 128)
        W = self.fc2.weight  # Get fc2 weights for orthogonalization
        
        # Reshape for broadcasting:
        # x_penult -> (B, H, W, 1, 128)
        # W        -> (1, 1, 1, C_out, 128)
        x_b = x_.unsqueeze(-2)
        W_b = W.view(1, 1, 1, self.fc2.out_features, self.fc2.in_features)
        
        # components shape: (B, H, W, C_out, 128)
        components = x_b * W_b

        # --- Apply ICA Projection ---
        # Apply layer: (B, H, W, C_out, 128) -> (B, H, W, C_out, ica_dim)
        transformed_components = self.ica_projection(components)
        
        # Sum over the new 'ica_dim' (the last dimension)
        # (B, H, W, C_out, ica_dim) -> (B, H, W, C_out)
        x = torch.sum(transformed_components, dim=-1)
        
        # Return final output and new transformed components
        return x, transformed_components

class FNO1d_layer_ortho(nn.Module):
    def __init__(self, modes, width, p_ica=24, in_channels=2, out_channels=1):
        super(FNO1d_layer_ortho, self).__init__()

        """
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=in_channels)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=out_channels)
        """

        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(in_channels, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

        self.ica_projection = nn.Sequential(
            nn.Linear(128, p_ica),
        )

    def forward(self, x):
        batchsize = x.shape[0]
        size_x = x.shape[1]

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x_ = F.relu(x)  # Shape: (B, L, 128)
        
        # W shape: (out_channels, in_channels) -> (C_out, 128)
        W = self.fc2.weight
        
        # Reshape for broadcasting:
        # x_ -> (B, L, 1, 128)
        # W  -> (1, 1, C_out, 128)
        x_b = x_.unsqueeze(-2)
        W_b = W.view(1, 1, self.fc2.out_features, self.fc2.in_features)
        
        # components shape: (B, L, C_out, 128)
        components = x_b * W_b

        # --- Apply ICA Projection ---
        # Apply layer: (B, L, C_out, 128) -> (B, L, C_out, ica_dim)
        transformed_components = self.ica_projection(components)
        
        # Sum over the new 'ica_dim' (the last dimension)
        # (B, L, C_out, ica_dim) -> (B, L, C_out)
        x = torch.sum(transformed_components, dim=-1)
        
        # Return final output and new transformed components
        return x, transformed_components