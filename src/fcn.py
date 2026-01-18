import torch
import torch.nn as nn

# Fully Connected Neural Network (FCNN) class to dynamically build networks
class FCN(nn.Module):
    """ 
    Fully Connected Neural Network (FCNN) class to dynamically build networks
    Args:
    - architecture (list): List of integers where each integer represents the number of neurons in a layer
    - activation_fn (torch.nn.Module): Activation function to apply after each layer

    Example:
    architecture = [2, 64, 64, 1]
    activation_fn = nn.ReLU
    fcn = FCN(architecture, activation_fn)
    """
    def __init__(self, architecture, activation_fn=nn.ReLU):
        super(FCN, self).__init__()
        layers = []
        # build the network layers except the last layer
        # The last layer is added separately to avoid activation function
        for i in range(len(architecture) - 2):
            layers.append(nn.Linear(architecture[i], architecture[i + 1]))
            layers.append(activation_fn())
        self.network = nn.Sequential(*layers)
        # Last layer without activation function
        self.last_hidden = nn.Linear(architecture[-2], architecture[-1], bias=False)  # Last layer without activation
        self.layer_norm = nn.LayerNorm(architecture[-1])  # Layer normalization for the last layer

    def forward(self, x):
        # get penultimate layer output
        x_penult = self.network(x)  # Apply all layers except the last one
        x = self.last_hidden(x_penult)  # Apply the last layer
        x = self.layer_norm(x)  # Apply layer normalization
        # Return the output of the last layer
        return x
    
# Fully Connected Neural Network (FCNN) class to dynamically build networks
class Penultimate_FCN(nn.Module):
    """ 
    Fully Connected Neural Network (FCNN) class to dynamically build networks
    Args:
    - architecture (list): List of integers where each integer represents the number of neurons in a layer
    - activation_fn (torch.nn.Module): Activation function to apply after each layer

    Example:
    architecture = [2, 64, 64, 1]
    activation_fn = nn.ReLU
    fcn = FCN(architecture, activation_fn)
    """
    def __init__(self, architecture, activation_fn=nn.ReLU):
        super(Penultimate_FCN, self).__init__()
        layers = []
        # build the network layers except the last layer
        # The last layer is added separately to avoid activation function
        for i in range(len(architecture) - 2):
            layers.append(nn.Linear(architecture[i], architecture[i + 1]))
            layers.append(activation_fn())
        self.network = nn.Sequential(*layers)
        self.layer_norm = nn.LayerNorm(architecture[-1])  # Layer normalization for the last layer
        # Last layer without activation function
        self.last_hidden = nn.Linear(architecture[-2], architecture[-1], bias=False)  # Last layer without activation

    def forward(self, x):
        # get penultimate layer output
        x_penult = self.network(x)  # Apply all layers except the last one
        x = self.last_hidden(x_penult)  # Apply the last layer
        x = self.layer_norm(x)  # Apply layer normalization
        # Return the output of the last layer
        return x, x_penult  # Return both the output of the last layer and the penultimate layer output