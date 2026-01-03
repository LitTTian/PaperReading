import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetPP(nn.Module):
    """
    UNet++ 核心网络结构
    支持嵌套密集连接和深度可配置
    """
    def __init__(self, in_channels=3, num_classes=1, deep_supervision=True, init_features=32):
        """
        参数说明：
        - in_channels: 输入通道数（如RGB图像为3，灰度图为1）
        - num_classes: 分割类别数（二分类为1，多分类为对应数量）
        - deep_supervision: 是否启用深度监督（UNet++核心特性）
        - init_features: 初始卷积核数量（控制网络宽度）
        """
        super(UNetPP, self).__init__()
        self.deep_supervision = deep_supervision
        features = init_features

        # 编码器模块（下采样路径）
        self.x00 = DoubleConv(in_channels, features)
        self.x10 = DoubleConv(features, features * 2)
        self.x20 = DoubleConv(features * 2, features * 4)
        self.x30 = DoubleConv(features * 4, features * 8)
        self.x40 = DoubleConv(features * 8, features * 16)

        # 嵌套密集连接模块（横向连接）
        self.x01 = DoubleConv(features + features * 2, features)
        self.x11 = DoubleConv(features * 2 + features * 4, features * 2)
        self.x21 = DoubleConv(features * 4 + features * 8, features * 4)
        self.x31 = DoubleConv(features * 8 + features * 16, features * 8)

        self.x02 = DoubleConv(features + features + features * 2, features)
        self.x12 = DoubleConv(features * 2 + features * 2 + features * 4, features * 2)
        self.x22 = DoubleConv(features * 4 + features * 4 + features * 8, features * 4)

        self.x03 = DoubleConv(features + features + features + features * 2, features)
        self.x13 = DoubleConv(features * 2 + features * 2 + features * 2 + features * 4, features * 2)

        self.x04 = DoubleConv(features * 4 + features * 2, features)

        # 下采样（最大池化）
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 上采样（转置卷积）
        self.up1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)

        # 输出层（深度监督）
        self.out1 = nn.Conv2d(features, num_classes, kernel_size=1)
        self.out2 = nn.Conv2d(features, num_classes, kernel_size=1)
        self.out3 = nn.Conv2d(features, num_classes, kernel_size=1)
        self.out4 = nn.Conv2d(features, num_classes, kernel_size=1)

    def forward(self, x):
        # 编码器路径
        x00 = self.x00(x)                          # 第一层，原始尺寸
        x10 = self.x10(self.pool(x00))             # 第二层，1/2尺寸
        x20 = self.x20(self.pool(x10))             # 第三层，1/4尺寸
        x30 = self.x30(self.pool(x20))             # 第四层，1/8尺寸
        x40 = self.x40(self.pool(x30))             # 第五层，1/16尺寸

        # 第一层嵌套连接（x01）
        x01 = self.x01(torch.cat([x00, self.up1(x10)], dim=1))
        
        # 第二层嵌套连接（x11, x02）
        x11 = self.x11(torch.cat([x10, self.up2(x20)], dim=1))
        x02 = self.x02(torch.cat([x00, x01, self.up1(x11)], dim=1))
        
        # 第三层嵌套连接（x21, x12, x03）
        x21 = self.x21(torch.cat([x20, self.up3(x30)], dim=1))
        x12 = self.x12(torch.cat([x10, x11, self.up2(x21)], dim=1))
        x03 = self.x03(torch.cat([x00, x01, x02, self.up1(x12)], dim=1))
        
        # 第四层嵌套连接（x31, x22, x13, x04）
        x31 = self.x31(torch.cat([x30, self.up4(x40)], dim=1))
        x22 = self.x22(torch.cat([x20, x21, self.up3(x31)], dim=1))
        x13 = self.x13(torch.cat([x10, x11, x12, self.up2(x22)], dim=1))
        x04 = self.x04(torch.cat([x01, x02, x03, self.up1(x13)], dim=1))

        # 输出层（深度监督）
        out1 = self.out1(x01)
        out2 = self.out2(x02)
        out3 = self.out3(x03)
        out4 = self.out4(x04)

        # 如果启用深度监督，返回所有输出；否则仅返回最后一个输出
        if self.deep_supervision:
            return [out1, out2, out3, out4]
        else:
            return out4
