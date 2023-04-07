import torch.nn as nn
import torch.nn.functional as F

class ResMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers,device):
        super(ResMLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input embedding
        self.fc_in = nn.Linear(input_size, hidden_size)
        
        # ResMLP block
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size).to(device),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, hidden_size).to(device),
                nn.LeakyReLU()
            )
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.fc_out = nn.Linear(hidden_size, output_size).to(device)

    def forward(self, x,device):
        # Input embedding
        x = self.fc_in(x).to(device)
        
        # ResMLP block
        for i in range(self.num_layers):
            residual = x
            x = self.res_blocks[i](x)
            x += residual
            
        # Output layer
        x = self.fc_out(x)
        return x
