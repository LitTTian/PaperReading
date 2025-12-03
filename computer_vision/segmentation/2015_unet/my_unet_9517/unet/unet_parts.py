""" Parts of the U-Net model """
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    (convolution卷积 => [BN]批归一化 => ReLU激活函数) * 2
    假设输入图像尺寸:2016x1512 调用DoubleConv(3, 64)
    输入: [B, 3, 2016, 1512] -> [B, 64, 2016, 1512] -> [B, 64, 2016, 1512]
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        # 序列模块：两次(卷积+BN+ReLU)
        self.double_conv = nn.Sequential(
            # 第一层卷积：输入通道 -> 中间通道，3×3卷积（捕捉局部特征），padding=1（保持特征图尺寸不变）
            # H_out = (H_in - kernel_size + 2*padding) // stride + 1
            # NT: Conv2d中的参数计算: in_channels * kernel_size ** 2 * out_channels + out_channels (bias=True时)
            # NT: BatchNorm2d中的参数计算: 2 * n_channels (gamma和beta),均值和方差为统计量
            # NT: ReLU: 无参数
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 第二层卷积：中间通道 -> 输出通道，同样3×3卷积+padding=1
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            # NT: 无参数 MaxPooling(2)下采样：2x2窗口，步长2，尺寸减半
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样和双卷积"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:  # 使用双线性插值上采样
            # NT: bilinear通过周围4个像素的加权平均值计算新像素值，align_corners保持角点对齐
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:  # 使用转置卷积上采样(论文中使用这个)
            # NT: 转置卷积(反卷积)，能更好保留细节，但参数更多且可能过拟合
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # 假设x1[B, 1024, 52, 52], x2[B, 512, 136, 136]
        x1 = self.up(x1) # 上采样 x1 -> [B, 512, 104, 104]
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # HL: 这里是让后面的x1(padding)适配前面特征图x2的尺寸（论文中则相反(crop)），这样做的好处是保留了更多像素点信息，且最终输出的特征图尺寸和最初输入一致
        x = torch.cat([x2, x1], dim=1)  # 拼接 => [B, 1024, 136, 136]
        return self.conv(x)  # 双卷积 => [B, 512, 136, 136]


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)