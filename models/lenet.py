import torch
import torch.nn as nn
import torch.nn.functional as F

class Lenet(nn.Module):
    def __init__(self, num_classes=10):
        super(Lenet, self).__init__()
        self.in_plane = 1
        self.conv1 = nn.Conv2d(in_channels=self.in_plane, out_channels=6, kernel_size=5, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = nn.ReLU(out)
        out = self.max_pool1(out)
        out = self.conv2(out)
        out = nn.ReLU(out)
        out = self.max_pool2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = nn.ReLU(out)
        out = self.fc2(out)

        return out