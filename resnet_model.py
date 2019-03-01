import torch.nn as nn
import torch
import math
from collections import OrderedDict

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    # "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, last_layer=False):
        super(BasicBlock, self).__init__()
        m = OrderedDict()
        m['conv1'] = conv3x3(inplanes, planes, stride)
        m['bn1'] = nn.BatchNorm2d(planes)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = conv3x3(planes, planes)
        m['bn2'] = nn.BatchNorm2d(planes)
        self.group1 = nn.Sequential(m)

        self.relu = nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

        self.last_layer = last_layer

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        feature = self.group1(x)
        out = feature + residual



        out = self.relu(out)
        if self.last_layer == False:
            return out
        else:
            return out, out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, last_layer=False):
        super(Bottleneck, self).__init__()
        m  = OrderedDict()
        m['conv1'] = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        m['bn1'] = nn.BatchNorm2d(planes)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        m['bn2'] = nn.BatchNorm2d(planes)
        m['relu2'] = nn.ReLU(inplace=True)
        m['conv3'] = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        m['bn3'] = nn.BatchNorm2d(planes * 4)
        self.group1 = nn.Sequential(m)

        self.relu= nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

        self.last_layer = last_layer

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.group1(x) + residual
        out = self.relu(out)
        if self.last_layer == False:
            return out
        else:
            return out, out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()

        m = OrderedDict()
        m['conv1'] = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        m['bn1'] = nn.BatchNorm2d(64)
        m['relu1'] = nn.ReLU(inplace=True)
        m['maxpool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.group1 = nn.Sequential(m)

        self.layer1 = self._make_layer(block, 64, layers[0], isAttn=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, isAttn=True)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, isAttn=True)
        self.layer4_0 = self._make_layer4(block, 512, stride=2, flag=True)
        self.layer4_1 = self._make_layer4(block, 512, stride=1, flag=True)
        self.layer4_2 = self._make_layer4(block, 512, stride=1, flag=False)



    def _make_layer(self, block, planes, blocks, stride=1, isAttn=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if isAttn == True:
                if i == blocks - 1:
                    layers.append(block(self.inplanes, planes, last_layer=True))
                else:
                    layers.append(block(self.inplanes, planes, last_layer=False))
            else:
                    layers.append(block(self.inplanes, planes, last_layer=False))

        return nn.Sequential(*layers)

    def _make_layer4(self, block, planes, stride=1, flag=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, last_layer=flag))
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.group1(x)

        x = self.layer1(x)
        x, fea2 = self.layer2(x)
        x, fea3 = self.layer3(x)
        x, fea4_0 = self.layer4_0(x)
        x, fea4_1 = self.layer4_1(x)
        fea4_2 = self.layer4_2(x)

        return fea4_0, fea4_1, fea4_2



def load_state_dict(model, model_root):
    from torch import nn
    import re
    from collections import OrderedDict
    own_state_old = model.state_dict()
    own_state = OrderedDict() # remove all 'group' string
    for k, v in own_state_old.items():
        k = re.sub('group\d+\.', '', k)
        own_state[k] = v

    state_dict = torch.load(model_root)

    for name, param in state_dict.items():
        if name not in own_state:
            if 'fc' in name:
                continue
            if 'layer4.0' in name or 'layer4.1' in name or 'layer4.2' in name:
                name = 'layer4_' + name.split(".", 2)[1] + '.0.' + name.split(".", 2)[2]
            else:
                print(own_state.keys())
                raise KeyError('unexpected key "{}" in state_dict'
                               .format(name))
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)




def resnet18(pretrained=False, model_root=None):
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        load_state_dict(model, model_root=model_root)
    return model


def resnet34(pretrained=False, model_root=None):
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        load_state_dict(model, model_root=model_root)
    return model


def resnet50(pretrained=False, model_root=None):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        load_state_dict(model, model_root=model_root)
    return model


def resnet101(pretrained=False, model_root=None):
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    if pretrained:
        load_state_dict(model, model_root=model_root)
    return model


def resnet152(pretrained=False, model_root=None):
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    if pretrained:
        load_state_dict(model, model_root=model_root)
    return model