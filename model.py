from __future__ import print_function
import math
import torch
import torch.nn as nn
import torch.nn.init as init 
import torch.nn.functional as F
import torch.optim as optim


class MLPNet(nn.Module):
    def __init__(self, num_features=78, num_classes=15):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(num_features, 256)  
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)  

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

# class MLPNet(nn.Module):
#     def __init__(self, num_features=78, num_classes=15):
#         super(MLPNet, self).__init__()
#         self.fc1 = nn.Linear(num_features, 256)  
#         self.fc2 = nn.Linear(256, 512)
#         self.fc3 = nn.Linear(512, 128)
#         self.fc4 = nn.Linear(128, num_classes)  

#     def forward(self, x):
#         x = F.leaky_relu(self.fc1(x))
#         x = F.leaky_relu(self.fc2(x))
#         x = F.leaky_relu(self.fc3(x))
#         x = self.fc4(x)
#         return x



# class MLPNet(nn.Module):
#     def __init__(self, num_features=78, num_classes=15, dropout_rate=0.5):
#         super(MLPNet, self).__init__()
#         self.fc1 = nn.Linear(num_features, 256)
#         self.bn1 = nn.BatchNorm1d(256)  
#         self.fc2 = nn.Linear(256, 128)
#         self.bn2 = nn.BatchNorm1d(128)  
#         self.fc3 = nn.Linear(128, num_classes)
#         self.dropout = nn.Dropout(dropout_rate)  

#     def forward(self, x):
#         x = F.leaky_relu(self.bn1(self.fc1(x)))
#         x = self.dropout(x) 
#         x = F.leaky_relu(self.bn2(self.fc2(x)))
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x
