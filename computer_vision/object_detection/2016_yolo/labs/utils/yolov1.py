import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOv1(nn.Module):
    def __init__(self, num_classes=20, num_bboxes=2):
        super(YOLOv1, self).__init__()
        self.num_classes = num_classes
        self.num_bboxes = num_bboxes  # 每个网格预测2个框（论文设定）
        self.grid_size = 7            # 7×7网格

        # ------------------- 主干卷积层（Backbone）-------------------
        self.backbone = nn.Sequential(
            # (B, 3, 448, 448) → (B, 64, 224, 224)
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # (B, 64, 224, 224) → (B, 192, 112, 112)
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            # (B, 192, 112, 112) → (B, 192, 56, 56)
            nn.MaxPool2d(kernel_size=2, stride=2),

            # (B, 192, 56, 56) → (B, 128, 56, 56)
            nn.Conv2d(192, 128, kernel_size=1),
            nn.LeakyReLU(0.1),
            # (B, 128, 56, 56) → (B, 256, 56, 56)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            # (B, 256, 56, 56) → (B, 256, 56, 56)
            nn.Conv2d(256, 256, kernel_size=1),
            nn.LeakyReLU(0.1),
            # (B, 256, 56, 56) → (B, 512, 56, 56)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            *[nn.Sequential(  # (B, 512, 28, 28) → (B, 512, 28, 28)
                nn.Conv2d(512, 256, kernel_size=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1)
            ) for _ in range(4)],  # 4次重复
            # (B, 512, 28, 28) → (B, 512, 28, 28)
            nn.Conv2d(512, 512, kernel_size=1),
            nn.LeakyReLU(0.1),
            # (B, 512, 28, 28) → (B, 1024, 28, 28)
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            # (B, 1024, 28, 28) → (B, 1024, 14, 14)
            nn.MaxPool2d(kernel_size=2, stride=2),

            *[nn.Sequential(  # (B, 1024, 14, 14) → (B, 1024, 14, 14)
                nn.Conv2d(1024, 512, kernel_size=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1)
            ) for _ in range(2)],  # 2次重复
            # (B, 1024, 14, 14) → (B, 1024, 14, 14)
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            # (B, 1024, 14, 14) → (B, 1024, 7, 7)
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),

            # (B, 1024, 7, 7) → (B, 1024, 7, 7)
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            # (B, 1024, 7, 7) → (B, 1024, 7, 7)
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1)
        )

        # ------------------- 全连接层（Head）-------------------
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # 展平为一维向量
            nn.Linear(1024 * 7 * 7, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),  # 论文中使用Dropout防止过拟合
            nn.Linear(4096, self.grid_size * self.grid_size * (5 * self.num_bboxes + self.num_classes)),
            # 输出维度: 7×7×(5×2+20)=7×7×30=1470
            nn.Sigmoid()  # 归一化到0-1（坐标/置信度/类别概率）
        )

    def forward(self, x):
        # 前向传播：卷积提取特征 → 全连接输出
        x = self.backbone(x)
        x = self.fc_layers(x)
        # 重塑输出为 [batch_size, 7, 7, 30]
        x = x.view(-1, self.grid_size, self.grid_size, 5 * self.num_bboxes + self.num_classes)
        return x