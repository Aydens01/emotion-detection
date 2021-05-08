# -*- coding: utf-8 -*-

# fmt: off
import torch.nn as nn

# fmt: on
class Fernet(nn.Module):
    def __init__(self):
        super(Fernet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=7,
                stride=2,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=2,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        self.classifiers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(
                in_features=2048,
                out_features=512,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=512,
                out_features=7,
            ),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifiers(x)
        return x

