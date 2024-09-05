# 创建网络模型
# 内容：double、
import torch
import torch.nn as nn
import torch.nn.functional as F

#************ UNet模型 ************#
class DoubleConv(nn.Module):
    """定义UNET网络中的卷积块，由两个卷积层组成"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    """定义UNET网络的架构"""

    def __init__(self, in_channels=1, out_channels=1, dim=16):
        super().__init__()
        self.down_conv1 = DoubleConv(in_channels, dim)
        self.down_conv2 = DoubleConv(dim, dim * 2)
        self.down_conv3 = DoubleConv(dim * 2, dim * 4)
        self.down_conv4 = DoubleConv(dim * 4, dim * 8)
        self.down_conv5 = DoubleConv(dim * 8, dim * 16)
        self.up_transpose1 = nn.ConvTranspose2d(dim * 16, dim * 8, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(dim * 16, dim * 8)
        self.up_transpose2 = nn.ConvTranspose2d(dim * 8, dim * 4, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(dim * 8, dim * 4)
        self.up_transpose3 = nn.ConvTranspose2d(dim * 4, dim * 2, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(dim * 4, dim * 2)
        self.up_transpose4 = nn.ConvTranspose2d(dim * 2, dim, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(dim * 2, dim)
        self.out_conv = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # 编码器部分
        x1 = self.down_conv1(x)
        x2 = self.down_conv2(F.max_pool2d(x1, kernel_size=2, stride=2))
        x3 = self.down_conv3(F.max_pool2d(x2, kernel_size=2, stride=2))
        x4 = self.down_conv4(F.max_pool2d(x3, kernel_size=2, stride=2))
        x5 = self.down_conv5(F.max_pool2d(x4, kernel_size=2, stride=2))
        # 解码器部分
        x = self.up_transpose1(x5)
        x = self.up_conv1(torch.cat([x, x4], dim=1))
        x = self.up_transpose2(x)
        x = self.up_conv2(torch.cat([x, x3], dim=1))
        x = self.up_transpose3(x)
        x = self.up_conv3(torch.cat([x, x2], dim=1))
        x = self.up_transpose4(x)
        x = self.up_conv4(torch.cat([x, x1], dim=1))
        # 输出层
        x = nn.Sigmoid()(self.out_conv(x))
        return x