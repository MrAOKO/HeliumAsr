import torch
import torch.nn as nn
import torch.nn.functional as F

class DFCNN(nn.Module):
    def __init__(self, num_classes):
        super(DFCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d((2, 2))
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d((2, 2))
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(512*5,1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_out = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = self.dropout(x)
        x = x.permute(0, 3, 1, 2)  # (batch, time, channels, freq)
        b, t, c, f = x.shape
        x = x.reshape(b, t, c * f)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)
        return x