import math
import torch
from torch import nn


# 残差块
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return self.relu(Y)


def Resnet_18():
    b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels, num_channels))
        return blk

    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))  # *--把list展开
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))

    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(),
                        nn.Linear(512, 256),
                        nn.Linear(256, 1))
    return net


class ASPP(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super().__init__()
        self.convd6 = nn.Conv2d(input_channels, int(output_channels/4), kernel_size=kernel_size, padding=int((kernel_size-1)/2*6), dilation=6)
        self.convd12 = nn.Conv2d(input_channels, int(output_channels/4), kernel_size=kernel_size, padding=int((kernel_size-1)/2*12), dilation=12)
        self.convd18 = nn.Conv2d(input_channels, int(output_channels/4), kernel_size=kernel_size, padding=int((kernel_size-1)/2*18), dilation=18)
        self.convd24 = nn.Conv2d(input_channels, int(output_channels/4), kernel_size=kernel_size, padding=int((kernel_size-1)/2*24), dilation=24)

    def forward(self, X):
        Y = torch.cat([self.convd6(X), self.convd12(X), self.convd18(X), self.convd24(X)], dim=1)
        return Y


class ASPP_block(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1):
        super().__init__()
        self.ASPP = ASPP(input_channels, output_channels, kernel_size)
        self.conv = nn.Conv2d(output_channels, output_channels, kernel_size=kernel_size, padding=int((kernel_size-1)/2), stride=stride)
        self.bn = nn.BatchNorm2d(output_channels)
        self.conv1x1 = nn.Conv2d(output_channels, int(output_channels/2), kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = self.relu(self.conv1x1(self.bn(self.conv(self.ASPP(X)))))
        return Y


class DotProductAttention(nn.Module):
    def __init__(self, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, keys, values):
        c = keys.shape[1]
        d = values.shape[-1]
        keys = keys.reshape(keys.shape[0], keys.shape[1], keys.shape[2] * keys.shape[3])
        queries = keys
        scores = torch.bmm(queries.transpose(1, 2), keys) / math.sqrt(c)
        self.attention_weights = nn.functional.softmax(scores, dim=1)  # !!!
        values = values.reshape(values.shape[0], values.shape[1], d * d)
        # Y = torch.bmm(values, self.dropout(self.attention_weights))
        Y = torch.bmm(values, self.attention_weights)
        Y = Y.reshape(Y.shape[0], Y.shape[1], d, d)
        return Y


class myBlockhead(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=7, padding=3, stride=2)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.ASPPblk = ASPP_block(input_channels, output_channels, kernel_size=7, stride=2)
        self.attention = DotProductAttention()

    def forward(self, X):
        # queries = self.ASPPblk(X)
        Y = self.relu(self.bn(self.conv(X)))
        # Y = self.attention(keys=queries, values=self.relu(self.bn(self.conv(X))))
        # Y = self.relu(self.bn(self.conv(X))) + queries
        return Y


class myBlock1(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.resnet_blk = Residual(input_channels, output_channels, strides=1)
        self.ASPPblk = ASPP_block(input_channels, output_channels, kernel_size=3, stride=1)
        self.attention = DotProductAttention()

    def forward(self, X):
        queries = self.ASPPblk(X)
        Y = self.attention(keys=queries, values=self.resnet_blk(X))
        # Y = self.resnet_blk(X) + queries
        return Y


class myBlock2(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.resnet_blk1 = Residual(input_channels, output_channels, use_1x1conv=True, strides=2)
        self.resnet_blk2 = Residual(output_channels, output_channels, strides=1)
        self.ASPPblk = ASPP_block(input_channels, output_channels, kernel_size=3, stride=2)
        self.attention = DotProductAttention()

    def forward(self, X):
        queries = self.ASPPblk(X)
        Y = self.attention(keys=queries, values=self.resnet_blk2(self.resnet_blk1(X)))
        # Y = self.resnet_blk2(self.resnet_blk1(X)) + queries
        return Y


def myNet():
    b1 = nn.Sequential(myBlockhead(input_channels=3, output_channels=64),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b2 = nn.Sequential(myBlock1(input_channels=64, output_channels=64))
    b3 = nn.Sequential(myBlock1(input_channels=64, output_channels=64))
    b4 = nn.Sequential(myBlock2(input_channels=64, output_channels=128))
    b5 = nn.Sequential(myBlock2(input_channels=128, output_channels=256))
    b6 = nn.Sequential(myBlock2(input_channels=256, output_channels=512))
    net = nn.Sequential(b1, b2, b3, b4, b5, b6,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(),
                        nn.Linear(512, 256),
                        nn.Linear(256, 1))
    return net
