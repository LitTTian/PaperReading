import os
import torch
import numpy as np
from PIL import Image
# 示例调色板
CITYSCAPE_PALETTE = np.asarray([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]], dtype=np.uint8)
TRIPLET_PALETTE = np.asarray([
    [0, 0, 0, 255],
    [217, 83, 79, 255],
    [91, 192, 222, 255]], dtype=np.uint8)
SINGLE_CHANNEL = ["gray", "grayscale", "mask"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def tensor2numpy(tensor, mode="rgb", convert=True):  # 返回numpy数组，
    # detech
    if tensor.requires_grad:
        tensor = tensor.detach()
    image_np = tensor.cpu().numpy()
    # print("tensor2numpy 输入形状：", image_np.shape)
    if mode in SINGLE_CHANNEL:  # 维度只有2
        pass
    else:
        image_np = tensor.cpu().numpy().transpose(1, 2, 0)  # 转换成(H,W,C)
        if image_np.shape[2] == 1:  # 单通道
            image_np = image_np[:, :, 0]
    # 如果所有数值都在0-1之间，则转换到0-255
    if convert and (image_np.max() <= 1.0 or mode=="rgb"):
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
    else:
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    # print("tensor2numpy 输出形状：", image_np.shape)
    return image_np

def tensor2image(tensor, palette=None, mode="rgb"):  # 返回numpy数组，
    if len(tensor.shape) == 4 or (tensor.dim() == 3 and mode in SINGLE_CHANNEL):
        # show warning
        # print("Warning: tensor2image received a 4D tensor. Calling tensorb2image instead.")
        return tensorb2images(tensor, palette=palette, mode=mode)
    image_np = tensor2numpy(tensor, mode=mode)
    # image = Image.fromarray(image_np)
    if palette is not None:
        # clip(0, len(palette)-1)
        image_np = np.clip(image_np, 0, len(palette) - 1)
        image = Image.fromarray(palette[image_np])
    else:
        image = Image.fromarray(image_np)
    return image

def tensorb2images(tensors, palette=None, mode="rgb"): # batch
    images = []
    for i in range(tensors.shape[0]):
        image = tensor2image(tensors[i], palette=palette, mode=mode)
        images.append(image)
    return images

def denormalize_tensor(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def get_max_channel(tensor):
    if tensor.dim() == 3:
        # C,H,W
        max_channel, _ = torch.max(tensor, dim=0)
    elif tensor.dim() == 4:
        # B,C,H,W
        max_channel, _ = torch.max(tensor, dim=1)
    else:
        raise ValueError("Input tensor must be 3D or 4D.")
    return max_channel


import torch


def get_max_avg_channel(tensor, abs=False):
    """
    返回所有通道中「平均值最高」的通道数据
    
    Args:
        tensor: 3D 张量 (C, H, W) 或 4D 张量 (B, C, H, W)
    Returns:
        平均值最高的通道数据：3D 输入返回 (H, W)，4D 输入返回 (B, H, W)
    """
    tensor = torch.abs(tensor) if abs else tensor
    if tensor.dim() == 3:
        # 3D: (C, H, W) → 计算每个通道的平均值（对 H×W 所有像素求平均）
        channel_means = tensor.mean(dim=[1, 2])  # 结果形状：(C,)
        # 找到平均值最大的通道索引
        max_mean_idx = torch.argmax(channel_means)
        # 提取该通道数据（形状：(H, W)）
        max_channel = tensor[max_mean_idx, :, :]
    
    elif tensor.dim() == 4:
        # 4D: (B, C, H, W) → 计算每个 batch 中每个通道的平均值（对 H×W 求平均）
        channel_means = tensor.mean(dim=[2, 3])  # 结果形状：(B, C)
        # 找到每个 batch 中平均值最大的通道索引（结果形状：(B,)）
        max_mean_idx = torch.argmax(channel_means, dim=1)
        # 提取每个 batch 中对应通道的数据（形状：(B, H, W)）
        # 用 gather 实现批量索引提取（避免 for 循环）
        batch_size = tensor.shape[0]
        # 构造索引：(B, 1, 1) → 适配通道维度的索引
        max_mean_idx = max_mean_idx.view(batch_size, 1, 1)
        # 扩展索引到 (B, 1, H, W) → 匹配 tensor 后两个维度
        max_mean_idx = max_mean_idx.expand(-1, 1, tensor.shape[2], tensor.shape[3])
        # 提取通道数据（gather 沿通道维度 dim=1 提取）
        max_channel = tensor.gather(dim=1, index=max_mean_idx).squeeze(1)  # 去掉多余的通道维度
    
    else:
        raise ValueError("Input tensor must be 3D (C, H, W) or 4D (B, C, H, W).")
    
    return max_channel

# def save_colorful_images(predictions, filenames, output_dir, palettes=CITYSCAPE_PALETTE):
#    """
#    Saves a given (B x C x H x W) into an image file.
#    If given a mini-batch tensor, will save the tensor as a grid of images.
#    """
#    for ind in range(len(filenames)):
#        im = Image.fromarray(palettes[predictions[ind].squeeze()])
#        fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
#        out_dir = split(fn)[0]
#        if not exists(out_dir):
#            os.makedirs(out_dir)
#        im.save(fn)

# def predict_image_mask(model, image, returnImage=False, device='cpu'):
#     input_tensor = torch.unsqueeze(image, 0)  # 添加批次维度
#     input_tensor = input_tensor.to(device)
#     model.eval()
#     with torch.no_grad():
#         output = model(input_tensor)
#         output = torch.sigmoid(output)
#         output = (output > 0.5).float()
#         output_mask = output.squeeze(0).squeeze(0)  # 去掉批次维度和通道维度
#     print(output_mask.shape)
#     if returnImage:
#         return tensor2image(output_mask)
#     else:
#         return output_mask



