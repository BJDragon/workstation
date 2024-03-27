import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicNN(nn.Module):
    def __init__(self, layer_sizes):
        super(DynamicNN, self).__init__()

        # 根据给定的数组构建网络层
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.Tanh())  # 添加激活函数，除了最后一层

        self.model = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.model(x))


class FCNN(nn.Module):
    """全连接网络"""

    def __init__(self, input_size, output_size):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 500)
        self.fc3 = nn.Linear(500, 200)
        self.fc4 = nn.Linear(200, 50)
        self.fc5 = nn.Linear(50, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.res3 = nn.Linear(500, 200)
        self.res4 = nn.Linear(200, 50)

    def resblock(self, x, res):
        """残差块"""
        return x + res

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.resblock(self.fc3(x), self.res3(x))
        x = self.tanh(x)
        x = self.resblock(self.fc4(x), self.res4(x))
        x = self.tanh(x)
        x = self.fc5(x)
        return self.softmax(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            # nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(out_channel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(out_channel)
            )
        self.leakyReLU = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = self.leakyReLU(out)
        return out


class _resnet(nn.Module):
    def __init__(self, ResidualBlock, block_num, num_classes):
        super(_resnet, self).__init__()
        self.in_channel = 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 32, block_num[0], stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 64, block_num[1], stride=1)
        self.layer3 = self.make_layer(ResidualBlock, 128, block_num[2], stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 256, block_num[3], stride=2)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(256, num_classes),
                                nn.Softmax(dim=1))

    def make_layer(self, block, channels, num_blocks, stride):
        # strides=[1, 1] or [2, 1]
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, channels, stride))
            self.in_channel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18(num_classes=4, block_num=[2, 2, 2, 2]):
    return _resnet(ResidualBlock, num_classes=num_classes, block_num=block_num)
