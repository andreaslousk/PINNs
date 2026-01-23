"""PINN model architecture for Black-Scholes"""
import torch
from torch import nn
from . import config


class BlackScholesPinn(nn.Module):
    """
    Physics-Informed Neural Network for Black-Scholes equation.
    
    Simple feedforward network with configurable depth.
    """
    
    def __init__(self, neurons=None, num_layers=None, activation='relu'):
        """
        Initialize PINN model.
        
        Args:
            neurons: Number of neurons per layer (default from config)
            num_layers: Number of hidden layers (default from config)
            activation: Activation function ('relu', 'tanh', 'gelu')
        """
        super().__init__()
        
        self.neurons = neurons if neurons is not None else config.HIDDEN_DIM
        self.num_layers = num_layers if num_layers is not None else config.NUM_LAYERS
        
        # Build network
        layers = []
        
        # Input layer
        layers.append(nn.Linear(config.INPUT_DIM, self.neurons))
        layers.append(self._get_activation(activation))
        
        # Hidden layers
        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(self.neurons, self.neurons))
            layers.append(self._get_activation(activation))
        
        # Output layer
        layers.append(nn.Linear(self.neurons, config.OUTPUT_DIM))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation):
        """Get activation function"""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _initialize_weights(self):
        """Initialize network weights using Kaiming initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, 2] where x[:, 0] = S, x[:, 1] = t
        
        Returns:
            Option price [batch_size, 1]
        """
        return self.network(x)
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Legacy model class for backward compatibility
class Model1(nn.Module):
    """
    Original 10-layer model architecture.
    
    Kept for backward compatibility with old code.
    """
    
    def __init__(self, neurons):
        super().__init__()
        
        self.neurons = neurons
        
        self.layer1 = nn.Linear(2, self.neurons)
        self.act1 = nn.ReLU()
        
        self.layer2 = nn.Linear(self.neurons, self.neurons)
        self.act2 = nn.ReLU()
        
        self.layer3 = nn.Linear(self.neurons, self.neurons)
        self.act3 = nn.ReLU()
        
        self.layer4 = nn.Linear(self.neurons, self.neurons)
        self.act4 = nn.ReLU()
        
        self.layer5 = nn.Linear(self.neurons, self.neurons)
        self.act5 = nn.ReLU()
        
        self.layer6 = nn.Linear(self.neurons, self.neurons)
        self.act6 = nn.ReLU()
        
        self.layer7 = nn.Linear(self.neurons, self.neurons)
        self.act7 = nn.ReLU()
        
        self.layer8 = nn.Linear(self.neurons, self.neurons)
        self.act8 = nn.ReLU()
        
        self.layer9 = nn.Linear(self.neurons, self.neurons)
        self.act9 = nn.ReLU()
        
        self.layer10 = nn.Linear(self.neurons, 1)
    
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.act4(self.layer4(x))
        x = self.act5(self.layer5(x))
        x = self.act6(self.layer6(x))
        x = self.act7(self.layer7(x))
        x = self.act8(self.layer8(x))
        x = self.act9(self.layer9(x))
        x = self.layer10(x)
        return x
