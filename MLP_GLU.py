import torch
import torch.nn as nn

class MLP_GLU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(MLP_GLU, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.fc_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GLU(),
                nn.Linear(hidden_size, hidden_size)
            )
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Embedding
        x = nn.functional.relu(nn.Linear(self.input_size, self.hidden_size)(x))

        # MLP+GLU block
        for i in range(self.num_layers):
            x = self.fc_layers[i](x)

        # Output
        x = self.fc_out(x)
        return x
