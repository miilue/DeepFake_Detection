import torch.nn as nn
import torch
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
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
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 1)
        self.fc1 = nn.Linear(512 * block.expansion, 256)
        self.fc2 = nn.Linear(256, 1)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
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
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # x = self.fc1(x)
        # x = self.fc2(x)

        return x


class ASPP(nn.Module):
    def __init__(self, inplanes, planes, kernel_size):
        super().__init__()
        div_planes = int(planes/4)
        padding = int((kernel_size-1)/2)
        self.convd6 = nn.Conv2d(inplanes, div_planes, kernel_size=kernel_size, padding=padding * 6, dilation=6)
        self.convd12 = nn.Conv2d(inplanes, div_planes, kernel_size=kernel_size, padding=padding * 12, dilation=12)
        self.convd18 = nn.Conv2d(inplanes, div_planes, kernel_size=kernel_size, padding=padding * 18, dilation=18)
        self.convd24 = nn.Conv2d(inplanes, div_planes, kernel_size=kernel_size, padding=padding * 24, dilation=24)

    def forward(self, X):
        out = torch.cat([self.convd6(X), self.convd12(X), self.convd18(X), self.convd24(X)], dim=1)
        return out


class ASPP_block(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1):
        super().__init__()
        # self.ASPP = ASPP(inplanes, planes, kernel_size)
        self.conv0 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, padding=int((kernel_size - 1) / 2), stride=1)
        self.conv = nn.Conv2d(planes, planes, kernel_size=kernel_size, padding=int((kernel_size-1)/2), stride=stride)
        self.bn = nn.BatchNorm2d(planes)
        # self.conv1x1 = conv1x1(planes, 1)
        # self.conv1x1 = conv1x1(planes, int(planes / 2))
        # self.conv1x1 = conv1x1(planes, planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        # out = self.relu(self.conv1x1(self.bn(self.conv(self.ASPP(X)))))
        out = self.relu(self.bn(self.conv(self.conv0(X))))
        return out


class DotProductAttention(nn.Module):
    def __init__(self, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, keys, values):
        # key为ASPP块的输出，values为残差块的输出
        c = keys.shape[1]
        h = values.shape[-2]
        w = values.shape[-1]
        keys = keys.reshape(keys.shape[0], keys.shape[1], keys.shape[2] * keys.shape[3])
        # keys = keys.squeeze(1)
        queries = keys
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(h * w)
        # scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = nn.functional.softmax(scores, dim=-1)  # !!!
        values = values.reshape(values.shape[0], values.shape[1], h * w)
        # out = torch.bmm(values, self.dropout(self.attention_weights))
        out = torch.bmm(self.attention_weights, values)
        out = nn.Sigmoid()(out)
        out = out.reshape(out.shape[0], out.shape[1], h, w)
        # out = keys + values
        return out


class ASPP_A_Net(nn.Module):
    def __init__(self, block, layers):
        super(ASPP_A_Net, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.attention = DotProductAttention()

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.aspp_blk1 = ASPP_block(self.inplanes, 64, stride=1)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.aspp_blk2 = ASPP_block(64, 128, stride=2)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.aspp_blk3 = ASPP_block(128, 256, stride=2)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.aspp_blk4 = ASPP_block(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 1)
        self.fc1 = nn.Linear(512 * block.expansion, 256)
        self.fc2 = nn.Linear(256, 1)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # keys = x
        x = self.layer1(x)
        # keys = self.aspp_blk1(keys)
        # x = self.attention(keys=keys, values=x)

        # keys = x
        x = self.layer2(x)
        # keys = self.aspp_blk2(keys)
        # x = self.attention(keys=keys, values=x)

        # keys = x
        x = self.layer3(x)
        # keys = self.aspp_blk3(keys)
        # x = self.attention(keys=keys, values=x)

        keys = x
        x = self.layer4(x)
        keys = self.aspp_blk4(keys)
        x = self.attention(keys=keys, values=x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # x = self.fc1(x)
        # x = self.fc2(x)

        return x


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def aspp_a_net():
    return ASPP_A_Net(BasicBlock, [2, 2, 2, 2])
