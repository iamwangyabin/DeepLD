import torch
from torch import nn

class Descriptor(nn.Module):

    def __init__(self,
            out_dim=128,init_num_channels=64,
            num_conv_layers=3,use_bias=False,
            conv_ksize=3):
        super(Descriptor, self).__init__()
        in_channel=2
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=conv_ksize, stride=2,padding=1, bias=use_bias)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=conv_ksize, stride=2,padding=1, bias=use_bias)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=conv_ksize, stride=2, padding=1, bias=use_bias)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.

    def forward(self, x):
        residual = x

        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out