import torch
import torch.nn as nn

class Multi_CNN_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size = (1, 8))
        self.conv2 = nn.Conv2d(1, 1, kernel_size = (1, 16))
        self.conv3 = nn.Conv2d(1, 1, kernel_size = (1, 32))
        self.linear1 = nn.Linear(64 - 8 + 1, 24)
        self.linear2 = nn.Linear(64 - 16 + 1, 24)
        self.linear3 = nn.Linear(64 - 32 + 1, 24)
        self.linear = nn.Linear(24 * 3, 24)


    
    def forward(self, x):
        # x -> B, 5, 64
        x1 = self.conv1(x.unsqueeze(1)).squeeze(1) # x1 -> B, 5, 64 - 8 + 1
        x2 = self.conv2(x.unsqueeze(1)).squeeze(1)
        x3 = self.conv3(x.unsqueeze(1)).squeeze(1)

        x1 = self.linear1(x1)
        x2 = self.linear2(x2)
        x3 = self.linear3(x3)

        x_out = torch.cat([x1, x2, x3], dim = -1)
        return self.linear(x_out)
