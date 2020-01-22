import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=16*13*13,out_features = 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,8)
    
    def forward(self,x):
        x = self.conv1(x) # batch_size x 6 x 60 x 60
        x = self.pool(F.relu(x)) # batch_size x 6 x 30 x 30
        x = self.conv2(x) # batch_size x 16 x 26 x 26
        x = self.pool(F.relu(x)) # batch_size x 16 x 13 x 13
        x = x.view(-1, 16 * 13 * 13) # flatten the output for each image
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
