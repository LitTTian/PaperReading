import torch
import torch.nn as nn
from torchsummary import summary

def register_hooks(model, layer_name_prefix=""):
    """为模型的每一层注册前向Hook，打印输入输出尺寸"""
    hooks = []
    
    def hook_fn(module, input, output, name):
        # 处理输入（可能是元组）
        input_str = []
        for idx, inp in enumerate(input):
            if isinstance(inp, torch.Tensor):
                input_str.append(f"input[{idx}]: {list(inp.size())}")
        
        # 处理输出（可能是元组）
        output_str = []
        if isinstance(output, torch.Tensor):
            output_str.append(f"output: {list(output.size())}")
        elif isinstance(output, (tuple, list)):
            for idx, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    output_str.append(f"output[{idx}]: {list(out.size())}")
        
        print(f"[{name}] -> {' | '.join(input_str)} | {' | '.join(output_str)}")
    
    # 遍历模型的每一层
    for name, module in model.named_modules():
        # 跳过空名称（根模块）和Sequential容器（避免重复打印）
        if name and not isinstance(module, nn.Sequential):
            full_name = f"{layer_name_prefix}.{name}" if layer_name_prefix else name
            hook = module.register_forward_hook(lambda m, i, o, n=full_name: hook_fn(m, i, o, n))
            hooks.append(hook)
    
    return hooks