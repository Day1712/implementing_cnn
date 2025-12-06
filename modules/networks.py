import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, stride=1)
        self.fc1 = nn.Linear(64 * 3 * 3, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        return x
    

class VarResNet(nn.Module):
    def __init__(self, n_channels=81, pooling='max'): # N=81 from Q15
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(32, n_channels, 3, padding=1, stride=1)
        self.fc1 = nn.Linear(n_channels, 10)

        self.pooling = pooling

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        
        # for Q16, compare global max and global mean pooling
        if self.pooling == 'max':
            x = x.amax(dim=(2, 3)) 
        elif self.pooling == 'mean':
            x = x.mean(dim=(2, 3))
        
        x = self.fc1(x)
        return x