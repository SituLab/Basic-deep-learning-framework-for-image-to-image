# Development of deep learning

import os
import cv2
import torch
import argparse
import numpy as np
from torch import nn
from tqdm import tqdm
from PIL import Image
from model import UNet
from torch import optim
from utils import save_images
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

os.makedirs('./dataset/input/', exist_ok=True)
os.makedirs('./dataset/label/', exist_ok=True)
os.makedirs('./dataset/test/', exist_ok=True)
os.makedirs('./weights/', exist_ok=True)
os.makedirs('./results/', exist_ok=True)


class LoadDataset(Dataset):
    def __init__(self, image_dir='./dataset/input', label_dir='./dataset/label', transform=None, train_num=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = os.listdir(image_dir)[:train_num]    # 使用固定数量的数据,默认为全部
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 读取输入图像
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        input = cv2.imread(img_path,-1).astype(np.float32)/65535
        
        # 读取对应的标签图像
        label_path = os.path.join(self.label_dir, self.image_files[idx])
        label = cv2.imread(label_path,-1).astype(np.float32)/65535

        # 进行预处理
        if self.transform:
            image = self.transform(input)
            label = self.transform(label)

        return image, label

def train(args):
    print('training begin:')
    # 创建当前训练文件夹，以args.running_name命名
    os.makedirs(os.path.join('./weights/', args.running_name), exist_ok=True)
    os.makedirs(os.path.join('./results/', args.running_name, 'eval'), exist_ok=True)

    # 加载数据
    dataset = LoadDataset(transform=args.transform, train_num=100)
    train_ds, valid_ds = random_split(dataset, [int(0.9*len(dataset)), int(0.1*len(dataset))])
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)
    
    # 配置网络
    net = args.net
    mse = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters(), lr=args.lr)

    # 训练网络
    min_loss, best_epoch, t_loss_set, v_loss_set = 999, 0, [], []    # 用于保存最优模型和记录loss
    for epoch in range(args.epochs):
        # 训练集训练网络
        net.train()
        loss_sum = 0
        for input, label in tqdm(train_dl, desc=f"training epoch {epoch}:"):
            output = net(input.to(args.device))
            loss = mse(output, label.to(args.device))
            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        t_loss_set.append(loss_sum / len(train_dl))

        # 验证集保存最优模型
        net.eval()
        loss_sum = 0
        for input, label in tqdm(valid_dl, desc=f"validing epoch {epoch}:"):
            output = net(input.to(args.device))
            loss = mse(output, label.to(args.device))
            if loss_sum == 0:
                save_images((output*255).type(torch.uint8), os.path.join('results', args.running_name, 'eval', f'{epoch}.png'))
            loss_sum += loss.item()
        tmp_loss = loss_sum / len(valid_dl)
        v_loss_set.append(tmp_loss)
        if tmp_loss < min_loss:
            best_epoch = epoch
            min_loss = tmp_loss
            torch.save(net.state_dict(), os.path.join('weights', args.running_name, f'best_{args.running_name}.pth'))
        # 打印此epoch的信息
        print(f'epoch{epoch}, t_loss: {t_loss_set[-1]:.6f}, v_loss: {v_loss_set[-1]:.6f}, best_epoch {best_epoch}: {min_loss:.6f}')
        with open(f'log_{args.running_name}.txt', 'a') as f:   # 在末尾保存训练信息
            f.write(f'epoch{epoch:05}, t_loss: {t_loss_set[-1]:.6f}, v_loss: {v_loss_set[-1]:.6f}, best_epoch {best_epoch}: {min_loss:.6f}\n')
    # 保存loss信息
    with open(f'log_{args.running_name}.txt', 'a') as f:
        f.write('t_loss_set:\n'+str(t_loss_set)+'\n')
        f.write('v_loss_set:\n'+str(v_loss_set)+'\n')
    plt.plot(t_loss_set)
    plt.plot(v_loss_set)
    plt.title(f'loss_{args.running_name}.png')
    plt.savefig(f'loss_{args.running_name}.png')
    plt.show()
    pass

def test(args):
    print('testing')
    pass


def launch():
    parser = argparse.ArgumentParser()
    parser.add_argument('--running_name', '-i', type=str, required=True, default='demo1', help="输入文件路径")
    parser.add_argument('--model_path', type=str, default='./weights/demo1/best_model.pth')
    parser.add_argument('--is_train', type=bool, default=True)
    parser.add_argument('--is_test', type=bool, default=True)
    
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.epochs = 200
    args.batch_size = 8
    args.lr = 3e-4
    # 数据预处理，包括尺寸调整和归一化
    args.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        # transforms.Normalize(mean=[0.5], std=[0.5]) # 灰度，归一化到[-1,1]
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])   # 彩色
        ])
    args.net = UNet().to(args.device)

    # 记录每次训练的配置信息
    with open('log_'+args.running_name+'.txt', 'w') as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")  # 每个键值对后添加换行符


    if args.is_train :
        print(args.running_name)
        train(args)

    if args.is_test :
        test(args)

    return args

if __name__ == '__main__':
    launch()    
