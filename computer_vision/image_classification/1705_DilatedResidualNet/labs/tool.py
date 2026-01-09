import numpy as np
import torch.nn as nn

def create_equivalent_conv_kernel(orig_kernel, dilation=None):
    """
    手动构造与空洞卷积等价的普通卷积核
    支持二维不同空洞率（如(2,3)）
    
    orig_kernel: 原始卷积核（如3×3的Tensor/ndarray，或4维[out_ch, in_ch, h, w]）
    dilation: 空洞率，可选，格式为(int, int) 或 int。若为None且orig_kernel是Parameter，自动从其提取
    return: 等价的普通卷积核（格式与输入一致，k_eff × k_eff 或 [out_ch, in_ch, k_eff, k_eff]）
    """
    # 处理输入核格式：支持2D（h,w）或4D（out_ch, in_ch, h, w）
    if isinstance(orig_kernel, nn.Module): # nn.conv2d
        orig_kernel = orig_kernel.weight.detach().cpu().numpy() if isinstance(orig_kernel.weight, nn.Parameter) else orig_kernel.weight
        dilation = orig_kernel.dilation
    elif isinstance(orig_kernel, nn.Parameter):
        orig_kernel = orig_kernel.detach().cpu().numpy()
    is_4d = orig_kernel.ndim == 4
    if is_4d:
        out_ch, in_ch, kh_orig, kw_orig = orig_kernel.shape
    else:
        kh_orig, kw_orig = orig_kernel.shape
        out_ch, in_ch = 1, 1  # 为统一处理补全维度
    
    # 统一dilation为二维元组（支持行、列不同空洞率）
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    assert len(dilation) == 2, "dilation必须是int或长度为2的元组"
    dil_h, dil_w = dilation  # 行空洞率、列空洞率分别处理
    
    # 计算等效普通核的尺寸（行和列可能不同）
    k_eff_h = kh_orig + (kh_orig - 1) * (dil_h - 1)
    k_eff_w = kw_orig + (kw_orig - 1) * (dil_w - 1)
    
    # 初始化等效核（保持与输入相同的维度格式）
    if is_4d:
        eq_kernel = np.zeros((out_ch, in_ch, k_eff_h, k_eff_w), dtype=orig_kernel.dtype)
    else:
        eq_kernel = np.zeros((k_eff_h, k_eff_w), dtype=orig_kernel.dtype)
    
    # 手动映射：原始核 → 等价核（分别处理行和列的空洞率）
    for och in range(out_ch):
        for ich in range(in_ch):
            # 取出当前通道的2D核
            kernel_2d = orig_kernel[och, ich] if is_4d else orig_kernel
            
            for kh in range(kh_orig):
                for kw in range(kw_orig):
                    # 计算等效核中的坐标（关键：行用行空洞率，列用列空洞率）
                    eq_kh = kh * dil_h
                    eq_kw = kw * dil_w
                    
                    # 赋值（保持原始核的数值分布）
                    if is_4d:
                        eq_kernel[och, ich, eq_kh, eq_kw] = kernel_2d[kh, kw]
                    else:
                        eq_kernel[eq_kh, eq_kw] = kernel_2d[kh, kw]
    
    # 还原输出格式（如果输入是2D，就返回2D）
    if not is_4d:
        eq_kernel = eq_kernel.squeeze()
    
    return eq_kernel