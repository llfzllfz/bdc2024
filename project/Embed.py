import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)

class DataEmbedding_inverted_CNN(nn.Module):
    def __init__(self, c_in, d_model, dropout = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size = (1, 8))
        self.conv2 = nn.Conv2d(1, 1, kernel_size = (1, 16))
        self.conv3 = nn.Conv2d(1, 1, kernel_size = (1, 32))

        self.linear1 = nn.Linear(c_in - 8 + 1, d_model)
        self.linear2 = nn.Linear(c_in - 16 + 1, d_model)
        self.linear3 = nn.Linear(c_in - 32 + 1, d_model)

        self.value_embedding = nn.Linear(c_in, d_model)
        # self.linear1 = nn.Linear(37, 37)
        # self.conv2 = nn.Conv2d(4, 1, (1, 24))
        # self.value_embedding = nn.Linear(c_in - 24 + 1, d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, x_mark):
        # x -> B, T, (4, 9)
        # x_mark -> B, T, 1
        x = x.unsqueeze(1) # B, 1, T, (4, 9)
        x = x.transpose(-1, -2) # B, 1, 36, T
        
        x1 = self.conv1(x).squeeze(1)
        x2 = self.conv2(x).squeeze(1)
        x3 = self.conv3(x).squeeze(1)

        return self.dropout(self.linear1(x1)), self.dropout(self.linear2(x2)), self.dropout(self.linear3(x3)), self.dropout(self.value_embedding(x_mark))
        x = self.value_embedding(x)
        return self.dropout(x)

class DataEmbedding_inverted_LSTM(nn.Module):
    def __init__(self, c_in, d_model, dropout = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(46, hidden_size=16, batch_first=True)
        self.fc = nn.Linear(16, 46)
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, x_mark):
        # x -> B, T, (4, 9 + 1)
        B, T, C = x.size()
        # x = x.transpose(-1, -2).contiguous()
        # x = x.view(B*C, T).unsqueeze(2)
        x, _ = self.lstm(x)
        x = self.fc(x).squeeze(-1)
        # x = x.view(B, C, T)
        x = x.permute(0, 2, 1)
        x = self.value_embedding(x)
        return self.dropout(x)