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
        out = self.model(x)

        return out

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


class FCNN2(nn.Module):
    """全连接网络"""
    def __init__(self, input_size, output_size):
        super(FCNN2, self).__init__()
        self.fc1 = nn.Linear(input_size, 20)
        self.fc2 = nn.Linear(20, 20)
        self.sigmoid = nn.Sigmoid()
        self.fc3 = nn.Linear(20, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        return x


