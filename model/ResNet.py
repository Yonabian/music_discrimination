import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentionFusion(nn.Module):
    def __init__(self, inter_channels,r):
        super(AttentionFusion, self).__init__()

        self.small_att = nn.Sequential(
        nn.Conv2d(int(inter_channels/2), inter_channels, 1, padding=0, bias=True),
        nn.BatchNorm2d(inter_channels),
        )

        self.large_att = nn.Sequential(
        nn.Conv2d(inter_channels, int(inter_channels/r), 1, padding=0, bias=True),
        nn.BatchNorm2d(int(inter_channels/r)),
        nn.ReLU(inplace=True),
        nn.Conv2d(int(inter_channels/r), inter_channels, 1, padding=0, bias=True),
        nn.BatchNorm2d(inter_channels),
        )

    def forward(self, x1, x2):

        new_x1 = self.small_att(x1)
        new_x2 = self.large_att(x2)

        new_x12 = new_x1 + new_x2
        weight = torch.sigmoid(new_x12)

        p2d = (0, 0, 0, 0, 0, x2.shape[1] - x1.shape[1])
        x1 = F.pad(x1, p2d, "constant", 0)
        x = torch.mul(x1, weight) + torch.mul(x2, 1 - weight)
        return x


class PaddingFusion(nn.Module):

    def __init__(self):
        super(PaddingFusion, self).__init__()

    def forward(self, x1, x2):
        p2d = (0,0,0,0,0,x2.shape[1]-x1.shape[1])
        x1 = F.pad(x1, p2d, "constant", 0)
        x12 = x1 + x2
        return x12


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=1, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=dilation, groups=1, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, fusionType='PaddingFusion', r=4):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, 2)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.globalavgpool = nn.AdaptiveAvgPool2d((1,1))
        if fusionType == 'Attention':
            self.fusionMode1 = AttentionFusion(128,r)
            self.fusionMode2 = AttentionFusion(256,r)
            self.fusionMode3 = AttentionFusion(512,r)
        else:
            self.fusionMode1 = PaddingFusion()
            self.fusionMode2 = PaddingFusion()
            self.fusionMode3 = PaddingFusion()
        self.logistic = nn.Linear(512, 2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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

        featureMap1 = self.layer1(x)
        featureMap2 = self.layer2(featureMap1)
        featureMap3 = self.layer3(featureMap2)
        featureMap4 = self.layer4(featureMap3)

        # GAP
        vectors1 = self.globalavgpool(featureMap1)
        vectors2 = self.globalavgpool(featureMap2)
        vectors3 = self.globalavgpool(featureMap3)
        vectors4 = self.globalavgpool(featureMap4)

        # Feature fusion
        new_data1 = self.fusionMode1(vectors1, vectors2)
        new_data2 = self.fusionMode2(new_data1, vectors3)
        new_data3 = self.fusionMode3(new_data2, vectors4)

        # logistic
        y = self.logistic(new_data3.reshape(new_data3.shape[0],new_data3.shape[1]))
        y = torch.sigmoid(y)

        return y
