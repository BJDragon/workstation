"""
对于不同模型可能需要更改的内容：
1.加载数据集过后的transform部分，需要对数据进行整形重造，位于get_loader当中；
2.main当中的网络model与summary中的inputsize需要调整一下，以及criterion；
3.train与val部分设置了两种类型，regression与classification两种类型，定义了回归任务与分类任务两种类型的需要计算的公式；
4.model save模型保存的名称；
5.结果后处理展示部分，保存结果的路径

痛点：
位置太过于分散，不便于数据监测
作为main函数的输入投入进去运行。
"""
import json
import random
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as tf
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import *

from tqdm.auto import tqdm
from torchsummary import summary
from matplotlib import pyplot as plt
import matplotlib.ticker as m_tick

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = './model'
if not os.path.exists(model_path):
    # 如果路径不存在，则创建它
    os.makedirs(model_path)
    print("路径已创建:", model_path)
else:
    print("路径已存在:", model_path)


def set_seed(seed: int):
    # 随机种子设定
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    pass


def init_weights(m: nn.Module):
    # 初始化网络参数层权重
    class_name = m.__class__.__name__
    if class_name.find('Conv2d') != -1 or class_name.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif class_name.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class CustomImageDataset(Dataset):
    # 只是将输入的x与y输出称为dataset类型
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        xi = self.x[idx]
        yi = self.y[idx]
        return xi, yi


def get_loader(args: argparse.Namespace):
    # 数据加载并填装进入loader
    # x = np.load(args.x_path)
    # y = np.load(args.y_path)
    #
    # x, y = transform(x, y)

    # train_data = pd.read_excel('训练集.xlsx', sheet_name='norm')
    # x_train = train_data.iloc[:, 0:2].values
    # y_train = train_data.iloc[:, 2:4].values
    # val_data = pd.read_excel('验证集.xlsx', sheet_name='norm')
    # x_val = val_data.iloc[:, 0:2].values
    # y_val = val_data.iloc[:, 2:4].values
    x = pd.read_excel('X.xlsx', sheet_name='Sheet1').values
    y = pd.read_excel('Y.xlsx', sheet_name='Sheet1').values

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=args.test_size, random_state=42)

    train_dataset = CustomImageDataset(x_train, y_train)
    val_dataset = CustomImageDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader


def regression_train(epoch: int, train_loader: DataLoader, model, optimizer, criterion, args: argparse.Namespace):
    model.train()
    train_loss_lis = np.array([])
    for batch in tqdm(train_loader):
        x, y = batch
        x, y = x.to(device), y.to(device)

        out = model(x)
        loss = criterion(out, y)

        # 梯度清楚
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_lis = np.append(train_loss_lis, loss.item())

    train_loss = sum(train_loss_lis) / len(train_loss_lis)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{args.epochs:03d} ] loss = {train_loss:.5f}")
    return train_loss


def regression_validate(epoch: int, val_loader: DataLoader, model, criterion, args: argparse.Namespace):
    model.eval()
    val_loss_lis = np.array([])
    for batch in tqdm(val_loader):
        x, y = batch

        with torch.no_grad():
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            val_loss_lis = np.append(val_loss_lis, loss.item())

    val_loss = sum(val_loss_lis) / len(val_loss_lis)

    # Print the information.
    print(f"[ Validation | {epoch + 1:03d}/{args.epochs:03d} ]  acc = {val_loss:.5f}")
    return val_loss


def draw_regression_loss(train_losses, val_losses):
    train_losses = train_losses.reshape(-1)
    val_losses = val_losses.reshape(-1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


def classification_train(epoch: int, train_loader: DataLoader, model, optimizer, criterion,
                         args: argparse.Namespace):
    model.train()
    train_loss_lis = np.array([])
    train_acc_lis = np.array([])

    for batch in tqdm(train_loader):
        x, y = batch
        x, y = x.to(device), y.to(device)

        out = model(x)
        loss = criterion(out, y)
        batch_acc = accuracy_score(torch.argmax(out, dim=1).cpu(), torch.argmax(y, dim=1).cpu())

        # 梯度清除
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_lis = np.append(train_loss_lis, loss.item())
        train_acc_lis = np.append(train_acc_lis, batch_acc)
    train_loss = sum(train_loss_lis) / len(train_loss_lis)
    train_acc = sum(train_acc_lis) / len(train_acc_lis)

    # Print the information.
    print(f"[Train | {epoch + 1:03d}/{args.epochs:03d} ] loss = {train_loss: .5f}, accuracy = {train_acc: .5f}")
    return train_loss, train_acc


def classification_validate(epoch: int, val_loader: DataLoader, model, criterion,
                            args: argparse.Namespace):
    model.eval()
    val_loss_lis = np.array([])
    val_acc_lis = np.array([])

    for batch in tqdm(val_loader):
        x, y = batch
        x, y = x.to(device), y.to(device)

        out = model(x)
        loss = criterion(out, y)
        batch_acc = accuracy_score(torch.argmax(out, dim=1).cpu(), torch.argmax(y, dim=1).cpu())

        val_loss_lis = np.append(val_loss_lis, loss.item())
        val_acc_lis = np.append(val_acc_lis, batch_acc)
    val_loss = sum(val_loss_lis) / len(val_loss_lis)
    val_acc = sum(val_acc_lis) / len(val_acc_lis)

    # Print the information.
    print(f"[Validation | {epoch + 1: 03d}/{args.epochs: 03d} ] loss = {val_loss: .5f}, accuracy = {val_acc: .5f}")
    return val_loss, val_acc


def save_result(train_loader, val_loader, model, json_file='data.json', device='cuda'):
    # 1.保存训练集与数据集。
    model.eval()
    train_x = []
    train_y = []
    with torch.no_grad():
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # pred = model(x)
            train_x.extend(x.cpu().tolist())
            train_y.extend(y.cpu().tolist())

    val_x = []
    val_y = []
    val_pred = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            batch_val_pred = model(x)
            val_x.extend(x.cpu().tolist())
            val_y.extend(y.cpu().tolist())
            val_pred.extend(batch_val_pred.cpu().tolist())

    # 2.损失函数保存

    # 3.验证结果保存
    data_to_save = {
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "val_pred": val_pred
    }

    with open(json_file, 'w') as f:
        json.dump(data_to_save, f)
    print(f'保存数据进入{json_file}')

    plt.plot(np.array(val_y)[:, 0], label= 'val_y_1')
    plt.plot(np.array(val_pred)[:, 0], label='val_pred_2')
    plt.title('val_y vs val_pred -1')
    plt.legend()
    plt.show()
    plt.plot(np.array(val_y)[:, 1], label= 'val_y_2')
    plt.plot(np.array(val_pred)[:, 1], label='val_pred_2')
    plt.title('val_y vs val_pred -2')
    plt.legend()
    plt.show()
    # plt.plot((np.array(val_pred)[:, 0] - np.array(val_y)[:, 0]) / np.array(val_y)[:, 0])
    # plt.plot((np.array(val_pred)[:, 1] - np.array(val_y)[:, 1]) / np.array(val_y)[:, 1])
    # plt.title('relative_error')
    # plt.show()


def transform(x, y):
    # 函数内的函数，用于对数据转换变形等操作
    x = x.reshape(-1, 1, 5, 5)
    # y = y[:, 1::2]

    # 用于转换为onehot编码
    y = torch.Tensor(y).long()
    y = torch.zeros(y.shape[0], 3).scatter_(1, y, 1)
    return x, y


def main(args: argparse.Namespace):
    print('---------Train on: ' + device + '----------')

    if args.seed is not None:
        set_seed(args.seed)

    train_loader, val_loader = get_loader(args)

    # Create model
    # model = DynamicNN([2, 50, 100, 50, 2]).to(device)
    # model = ResNet18(num_classes=3).to(device)
    # model = FCNN(25, 3).to(device)
    model = FCNN2(2, 2).to(device)
    # Visualize model
    # summary(model, input_size=(1, 5, 5))
    # Define Optimizer and Loss
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=int(0.5 * args.epochs), gamma=0.25)
    # Define list to record acc & loss for plt
    train_loss = np.array([])
    train_acc = np.array([])
    val_loss = np.array([])
    val_acc = np.array([])

    for epoch in range(args.epochs):
        scheduler.step()

        # 1.regression_train & regression_validate---------------------
        # 1.1 regression_train
        train_batch_loss = regression_train(epoch, train_loader, model, optimizer, criterion, args)
        train_loss = np.append(train_loss, train_batch_loss)
        # 1.2 regression_validate
        val_batch_acc = regression_validate(epoch, val_loader, model, criterion, args)
        val_loss = np.append(val_loss, val_batch_acc)

        # # 2.classification_train & classification_validate--------------------
        # # 1.1 classification_train
        # train_batch_loss, _ = classification_train(epoch, train_loader, model, optimizer, criterion, args)
        # train_loss = np.append(train_loss, train_batch_loss)
        # # 1.2 classification_validate
        # val_batch_acc, _ = classification_validate(epoch, val_loader, model, criterion, args)
        # val_loss = np.append(val_loss, val_batch_acc)

        # Save model
        if train_batch_loss == np.min(train_loss):
            torch.save(model, 'model/lkh.pt')
    # Draw loss & acc
    draw_regression_loss(train_loss, val_loss)
    save_result(train_loader, val_loader, model, json_file='test_data_lkh.json', device='cuda')

    return np.mean(val_loss[-10:-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Source for ImageNet Classification')
    parser.add_argument('-sd', '--seed', default=42, type=int, help='seed for initializing training. ')

    # dataset parameters
    parser.add_argument('-tp', '--x_path', default='data/x.npy',
                        help='the path of x data.')
    parser.add_argument('-vp', '--y_path', default='data/category.npy',
                        help='the path of y data.')

    # train parameters
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='the size of batch.')
    parser.add_argument('-ep', '--epochs', type=int, default=200, help='the num of epochs.')
    parser.add_argument('-ts', '--test_size', type=float, default=0.08, help='the percent of val_data.')

    # model parameters
    parser.add_argument('-lr', '--lr', type=float, default=0.005, help='initial learning rate', dest='lr')
    parser.add_argument('-mm', '--momentum', type=float, default=0.9, help='initial momentum')
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-5, help='initial momentum')

    args = parser.parse_args()

    import optuna
    import torch.optim as optim

    # 需要优化的函数
    # optuna_model = True
    optuna_model = False
    if optuna_model != True:
        min = main(args)
        print(min)
    else:
        import numpy as np

        goal = np.array([1000])

        np.save('goal.npy', goal)
        def objective(trial):

            print('---------Train on: ' + device + '----------')

            if args.seed is not None:
                set_seed(args.seed)
            batch_size = trial.suggest_int("batch_size", 4, 72)
            args.batch_size = batch_size
            train_loader, val_loader = get_loader(args)

            # Create model
            # model = DynamicNN([2, 50, 100, 50, 2]).to(device)
            # model = ResNet18(num_classes=3).to(device)
            # model = FCNN(25, 3).to(device)
            model = FCN().to(device)

            # Generate the optimizers.
            optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD", "Adagrad"])
            lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
            optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

            # Visualize model
            # summary(model, input_size=(1, 5, 5))
            # Define Optimizer and Loss
            # criterion = nn.CrossEntropyLoss()
            criterion = nn.MSELoss()
            # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
            # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
            step = trial.suggest_float('step_size', 0.1, 0.6)
            gamma = trial.suggest_float('gamma', 0.01, 0.5)
            scheduler = StepLR(optimizer, step_size=int(step * args.epochs), gamma=gamma)
            # Define list to record acc & loss for plt
            train_loss = np.array([])
            train_acc = np.array([])
            val_loss = np.array([])
            val_acc = np.array([])
            epochs = trial.suggest_int('epochs', 300, 2000)

            for epoch in range(epochs):

                # 1.regression_train & regression_validate---------------------
                # 1.1 regression_train
                train_batch_loss = regression_train(epoch, train_loader, model, optimizer, criterion, args)
                train_loss = np.append(train_loss, train_batch_loss)
                # 1.2 regression_validate
                val_batch_acc = regression_validate(epoch, val_loader, model, criterion, args)
                val_loss = np.append(val_loss, val_batch_acc)
                scheduler.step()

            # Draw loss & acc
            # draw_regression_loss(train_loss, val_loss)
            # save_result(train_loader, val_loader, model, json_file='test_data_lkh.json', device='cuda')
            goal = np.load('goal.npy')
            if val_loss[-1] < goal:
                np.save('goal.npy', val_loss[-1])
                torch.save(model, 'model/lkh.pt')
            return val_loss[-1]


        # 使用study对象的optimize来优化，里面的参数是上面定义的方法，以及迭代次数

        study = optuna.create_study(sampler=optuna.samplers.TPESampler(), storage='sqlite:///db.sqlite3')
        study.optimize(objective, n_trials=30)

        print(study.best_params)