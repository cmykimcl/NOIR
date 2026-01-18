import torch
import torch.nn as nn
from fcn import FCN

class NOMAD(nn.Module):
    def __init__(self, 
                 branch_arch, 
                 trunk_arch, 
                 combined_arch,
                 num_outputs=1, 
                 activation_fn=nn.ReLU):
        """
        Initializes the NOMAD model with an architecture defined by layer lists,
        similar to the DeepONet input structure.

        Args:
            branch_arch (list): A list of integers defining the layer sizes for the branch network.
                                Example: [input_dim, hidden1, ..., output_dim]
            trunk_arch (list): A list of integers defining the layer sizes for the trunk network.
                               Example: [input_dim, hidden1, ..., output_dim]
            combined_arch (list): A list of integers defining the layer sizes for the combined network.
                                  The input dimension must match the sum of the branch and trunk output dimensions.
                                  Example: [branch_out + trunk_out, hidden1, ..., final_output_dim]
            activation_fn (nn.Module): The activation function to use between layers. Defaults to nn.ReLU.
        """
        super().__init__()

        # --- Input Validation ---
        if not all(isinstance(arch, list) and len(arch) >= 2 for arch in [branch_arch, trunk_arch, combined_arch]):
            raise ValueError("branch_arch, trunk_arch, and combined_arch must be lists with at least two values (input and output dim).")
        
        # The input to the combined network must be the concatenation of the trunk and branch outputs.
        expected_combined_input_dim = branch_arch[-1] * 2
        if combined_arch[0] != expected_combined_input_dim:
            raise ValueError(f"Input dimension of combined_arch ({combined_arch[0]}) must equal the sum of "
                             f"the output dimensions of branch_arch ({branch_arch[-1]}) and trunk_arch ({trunk_arch[-1]}), "
                             f"which is {expected_combined_input_dim}.")

        # --- Network Construction ---
        self.num_outputs = num_outputs
        trunk_output_size = branch_arch[-1] * self.num_outputs
        full_trunk_arch = trunk_arch + [trunk_output_size]

        self.trunk_net = self.build_mlp(full_trunk_arch, activation_fn)
        self.branch_net = self.build_mlp(branch_arch, activation_fn)
        self.combined_net = self.build_mlp(combined_arch, activation_fn)

    def build_mlp(self, arch, activation_fn):
        """Builds a multi-layer perceptron (MLP) from an architecture list."""
        layers = []
        # Iterate through the architecture list to create linear layers
        for i in range(len(arch) - 1):
            layers.append(nn.Linear(arch[i], arch[i + 1]))
            # Add an activation function after each layer except the last one
            if i < len(arch) - 2:
                layers.append(activation_fn())
        return nn.Sequential(*layers)

    def forward(self, branch_input, trunk_input):
        """
        Forward pass for the NOMAD model.

        Args:
            trunk_input (torch.Tensor): Input for the trunk net, shape [B, N, trunk_input_dim].
            branch_input (torch.Tensor): Input for the branch net, shape [B, branch_input_dim].
        
        Returns:
            torch.Tensor: The model output, shape [B, N, final_output_dim].
        """
        N = trunk_input.shape[1]

        # 1. Process inputs through their respective networks
        if branch_input.ndim > 2:
            # merge dimension 1 to later to a single dimension
            branch_input = branch_input.view(branch_input.shape[0], -1)
        trunk_feat = self.trunk_net(trunk_input)                                   # Shape: [B, N, trunk_output_dim]
        branch_feat = self.branch_net(branch_input)                                # Shape: [B, branch_output_dim]
        # 2. Expand branch features to match the spatial/temporal dimension of trunk features
        branch_feat_expanded = branch_feat.unsqueeze(1).repeat(1, N, 1)            # Shape: [B, N, branch_output_dim]
        # 3. Fuse the features by concatenation
        fused = torch.cat([trunk_feat, branch_feat_expanded], dim=-1)              # Shape: [B, N, trunk_out + branch_out]
        # 4. Process the fused features through the final combined network
        out = self.combined_net(fused)                                             # Shape: [B, N, final_output_dim]
        # if out.shape[-1] == 1:
        #     out = out.squeeze(-1)  # Remove last dim if single output for shape [B, N]
        return out