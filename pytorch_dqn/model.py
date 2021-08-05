import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvPolicy(nn.Module):
    """https://github.com/DLR-RM/stable-baselines3/blob/3845bf9f3209173195f90752e341bbc45a44571b/stable_baselines3/common/torch_layers.py#L51"""
    
    def __init__(self, num_stacked_frames, action_space_dim):
        super(ConvPolicy, self).__init__()

        self.num_stacked_frames=num_stacked_frames
        self.action_space_dim=action_space_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(num_stacked_frames, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate the flatten dim with a forward pass
        with torch.no_grad():
            mock_data = torch.rand((1, num_stacked_frames, 84, 84))
            n_flatten = self.cnn(mock_data).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(
                n_flatten,
                self.action_space_dim
            ),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.linear(self.cnn(x))
        

class DensePolicy(nn.Module):
    def __init__(self, state_space_dim, action_space_dim, num_hidden=64):
        super(DensePolicy, self).__init__()
        self.state_space_dim = state_space_dim
        self.action_space_dim = action_space_dim
        self.num_hidden = num_hidden

        self.inp = nn.Linear(self.state_space_dim, self.num_hidden)
        self.h1 = nn.Linear(self.num_hidden, self.num_hidden)
        self.h2 = nn.Linear(self.num_hidden, self.num_hidden)
        self.out = nn.Linear(self.num_hidden, self.action_space_dim)

    
    def forward(self, x):    
        x = self.inp(x)
        x = F.relu(x)
        x = self.h1(x)
        x = F.relu(x)
        x = self.h2(x)
        x = F.relu(x)
        out = self.out(x)
        return out
    
    