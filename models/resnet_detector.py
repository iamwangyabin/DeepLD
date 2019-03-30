import torch
from torch import nn
from torch.nn import functional as f
import numpy as np

# build resnet blocks
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, use_bias=True,downsample=None):
        super(BasicBlock, self).__init__()
        self.bn0=nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=5, stride=stride,padding=2, bias=use_bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=5, stride=stride,padding=2, bias=use_bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



class DetectorModel(torch.nn.Module):

    def __init__(self, num_block=3, num_channels=16,conv_ksize=5,
                 use_bias=True, min_scale=2**-3, max_scale=1, num_scales=9):

        self.inplanes = num_channels
        self.num_blocks=num_block
        self.min_scale = min_scale
        self.max_scale=max_scale
        self.num_scales=num_scales

        super(DetectorModel, self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=conv_ksize, stride=1, padding=1,
                               bias=use_bias)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.layer=BasicBlock(self.inplanes, self.inplanes, stride=1, use_bias=True)
        self.soft_conv=nn.Conv2d(16, 1, kernel_size=conv_ksize, stride=1, padding=1,
                               bias=use_bias)

        self.ori_layer=nn.Conv2d(self.inplanes,2,kernel_size=conv_ksize, stride=1, padding=1,
                                bias=True )


    def forward(self, x):
        num_conv = 0

        x=self.conv1(x)
        num_conv+=1
        for i in range(self.num_blocks):
            x=self.layer(x)
        x=self.bn1(x)

        if self.num_scales == 1:
            scale_factors = [1.0]
        else:
            scale_log_factors = np.linspace(np.log(self.max_scale), np.log(self.min_scale), self.num_scales)
            scale_factors = np.exp(scale_log_factors)
        score_maps_list = []

        base_height_f=x.shape[0]
        base_width_f = x.shape[1]

        for i, s in enumerate(scale_factors):
            # scale are defined by extracted patch size (s of s*default_patch_size) so we need use inv-scale for resizing images
            inv_s=1.0/s
            feat_height=base_height_f * inv_s+0.5
            feat_width = base_width_f * inv_s + 0.5
            rs_feat_maps=torch.nn.functional.interpolate(x,torch.stack([feat_height, feat_width]))
            score_maps = self.soft_conv(x)
            score_maps_list.append(score_maps)

        # orientation (initial map start from 0.0)
        # ori_W_init=torch.nn.init.zeros_()
        # ori_b_init = tf.constant(np.array([1, 0], dtype=np.float32))  # init with 1 for cos(q), 0 for sin(q)
        ori_b_init=torch.nn.init.constant(np.array([1,0], dtype=np.float32))

        self.ori_layer.bias.data.fill_(ori_b_init)
        ori_maps=self.ori_layer(x)
        ori_maps=f.normalize(ori_maps,dim=-1)

        endpoints={}
        endpoints['ori_maps'] = ori_maps
        endpoints['mso'] = True
        endpoints['multi_scores'] = True
        endpoints['scale_factors'] = scale_factors
        endpoints['pad_size'] = num_conv * (self.conv_ksize//2)
        return score_maps_list,endpoints