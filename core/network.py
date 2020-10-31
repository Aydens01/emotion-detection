#! /usr/bin/env python
#-*- coding: utf-8 -*-

"""
author  : Adrien Lafage\n
date    : february 2020
Convolutional Neural Network class
===========
"""

############| IMPORTS |#############
import torch.nn as nn
import torch.nn.functional as F
####################################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=16*13*13,out_features = 120)
        self.fc2 = nn.Linear(in_features=120,out_features=84)
        self.fc3 = nn.Linear(in_features=84,out_features=8)
    
    def forward(self,x):
        """
            x: batch_size x 1 x 64 x 64
        """
        x = self.conv1(x) # batch_size x 6 x 60 x 60
        x = self.pool(F.relu(x)) # batch_size x 6 x 30 x 30
        x = self.conv2(x) # batch_size x 16 x 26 x 26
        x = self.pool(F.relu(x)) # batch_size x 16 x 13 x 13
        x = x.view(-1, 16 * 13 * 13) # flatten the output for each image
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#########################################################################

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        # Convolution part
        self.features = nn.Sequential(
            nn.Conv2d(1, 3, 5),       # (N, 1, 64, 64) -> (N, 3, 60, 60)
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2), # (N, 3, 60, 60) -> (N, 3, 30, 30)
            nn.Conv2d(3,6,3),         # (N, 3, 30, 30) -> (N, 6, 28, 28)
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2) # (N, 6, 28,28) -> (N, 6, 14, 14)
        )
        # Fast forward part
        self.classifer = nn.Sequential(
            nn.Linear(1176, 200),     # (N, 6*14*14) -> (N, 200)
            nn.ReLU(),
            nn.Linear(200, 80),       # (N, 200) -> (N, 80)
            nn.ReLU(),
            nn.Linear(80, 8)          # (N, 80) -> (N, 8)
        )

    def forward(self, x):
        """
            x: (N, 1, 64, 64)
        """
        x = self.features(x)
        x = x.view(-1, 1176)
        x = self.classifer(x)
        return(x)

#########################################################################

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        # Convolution part
        self.features = nn.Sequential(
            nn.Conv2d(1, 3, 5),       # (N, 1, 64, 64) -> (N, 3, 60, 60)
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2), # (N, 3, 60, 60) -> (N, 3, 30, 30)
            nn.Conv2d(3,6,3),         # (N, 3, 30, 30) -> (N, 6, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2,2),        # (N, 6, 28, 28) -> (N, 6, 14, 14)
            nn.Conv2d(6,9,3),         # (N, 6, 14, 14) -> (N, 9, 12, 12)
            nn.BatchNorm2d(9),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2) # (N, 9, 12, 12) -> (N, 9, 6, 6)
        )
        # Fast forward part
        self.classifer = nn.Sequential(
            nn.Linear(324, 200),      # (N, 9*6*6) -> (N, 200)
            nn.ReLU(),
            nn.Linear(200, 80),       # (N, 200) -> (N, 80)
            nn.ReLU(),
            nn.Linear(80, 8)          # (N, 80) -> (N, 8)
        )

    def forward(self, x):
        """
            x: (N, 1, 64, 64)
        """
        x = self.features(x)
        x = x.view(-1, 324)
        x = self.classifer(x)
        return(x)

#########################################################################

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        # Convolution part
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5),         # (N, 1, 64, 64) -> (N, 6, 60, 60)
            nn.ReLU(),                  
            nn.MaxPool2d(2, stride=2),  # (N, 6, 60, 60) -> (N, 6, 30, 30)
            nn.Conv2d(6, 12, 3),        # (N, 6, 30, 30) -> (N, 12, 28, 28)
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)   # (N, 12, 28, 28) -> (N, 12, 14, 14)
        )
        # Fast forward part
        self.classifier = nn.Sequential(
            nn.Linear(2352, 200),       # (N, 12*14*14) -> (N, 200)
            nn.ReLU(),
            nn.Linear(200, 80),
            nn.ReLU(),
            nn.Linear(80, 8)       
        )
    
    def forward(self, x):
        """
            x: (N, 1, 64, 64)
        """
        x = self.features(x)
        x = x.view(-1, 2352)
        x = self.classifier(x)
        return(x)

#########################################################################

class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        # Convolution part
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 5),         # (N, 1, 64, 64) -> (N, 8, 60, 60)
            nn.ReLU(),                  
            nn.MaxPool2d(2, stride=2),  # (N, 8, 60, 60) -> (N, 8, 30, 30)
            nn.Conv2d(8, 16, 3),        # (N, 8, 30, 30) -> (N, 16, 28, 28)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)   # (N, 16, 28, 28) -> (N, 16, 14, 14)
        )
        # Fast forward part
        self.classifier = nn.Sequential(
            nn.Linear(3136, 200),       # (N, 16*14*14) -> (N, 200)
            nn.ReLU(),
            nn.Linear(200, 80),
            nn.ReLU(),
            nn.Linear(80, 8)       
        )
    
    def forward(self, x):
        """
            x: (N, 1, 64, 64)
        """
        x = self.features(x)
        x = x.view(-1, 3136)
        x = self.classifier(x)
        return(x)

#########################################################################

class Net5(nn.Module):
    def __init__(self):
        super(Net5, self).__init__()
        # Convolution part
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, 5),         # (N, 1, 64, 64) -> (N, 10, 60, 60)
            nn.ReLU(),                  
            nn.MaxPool2d(2, stride=2),  # (N, 10, 60, 60) -> (N, 10, 30, 30)
            nn.Conv2d(10, 20, 3),        # (N, 10, 30, 30) -> (N, 20, 28, 28)
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)   # (N, 20, 28, 28) -> (N, 20, 14, 14)
        )
        # Fast forward part
        self.classifier = nn.Sequential(
            nn.Linear(3920, 200),       # (N, 20*14*14) -> (N, 200)
            nn.ReLU(),
            nn.Linear(200, 80),
            nn.ReLU(),
            nn.Linear(80, 8)       
        )
    
    def forward(self, x):
        """
            x: (N, 1, 64, 64)
        """
        x = self.features(x)
        x = x.view(-1, 3920)
        x = self.classifier(x)
        return(x)

#########################################################################

class Net6(nn.Module):
    def __init__(self):
        super(Net6, self).__init__()
        # Convolution part
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, 5),         # (N, 1, 64, 64) -> (N, 10, 60, 60)
            nn.ReLU(),                  
            nn.MaxPool2d(2, stride=2),  # (N, 10, 60, 60) -> (N, 10, 30, 30)
            nn.Conv2d(10, 20, 3),        # (N, 10, 30, 30) -> (N, 20, 28, 28)
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)   # (N, 20, 28, 28) -> (N, 20, 14, 14)
        )
        # Fast forward part
        self.classifier = nn.Sequential(
            nn.Linear(3920, 200),       # (N, 20*14*14) -> (N, 200)
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(200, 80),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(80, 8)       
        )
    
    def forward(self, x):
        """
            x: (N, 1, 64, 64)
        """
        x = self.features(x)
        x = x.view(-1, 3920)
        x = self.classifier(x)
        return(x)

#########################################################################

class Net7(nn.Module):
    def __init__(self):
        super(Net7, self).__init__()
        # Convolution part
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, 5),         # (N, 1, 64, 64) -> (N, 10, 60, 60)
            nn.ReLU(),                  
            nn.MaxPool2d(2, stride=2),  # (N, 10, 60, 60) -> (N, 10, 30, 30)
            nn.Conv2d(10, 20, 3),        # (N, 10, 30, 30) -> (N, 20, 28, 28)
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)   # (N, 20, 28, 28) -> (N, 20, 14, 14)
        )
        # Fast forward part
        self.classifier = nn.Sequential(
            nn.Linear(3920, 200),       # (N, 20*14*14) -> (N, 200)
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(200, 80),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(80, 8)       
        )
    
    def forward(self, x):
        """
            x: (N, 1, 64, 64)
        """
        x = self.features(x)
        x = x.view(-1, 3920)
        x = self.classifier(x)
        return(x)