import functools

def add_logging(module, enabled=True):
    """给任意 nn.Module 的 forward 动态加 log"""
    orig_forward = module.forward

    def logged_forward(*args, **kwargs):
        if enabled:
            print(f"[LOG] {module.__class__.__name__}.forward 输入:", 
                  [arg.shape for arg in args if hasattr(arg, "shape")])
        out = orig_forward(*args, **kwargs)
        if enabled and hasattr(out, "shape"):
            print(f"[LOG] {module.__class__.__name__}.forward 输出:", out.shape)
        return out

    module.forward = logged_forward
    return module

def log(func):
    """装饰器: 打印类名, 方法名, 输入输出形状"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        class_name = self.__class__.__name__
        method_name = func.__name__

        # 输入 shape
        in_shapes = []
        for arg in args:
            if hasattr(arg, "shape"):
                in_shapes.append(str(arg.shape))
        print(f"[LOG] {class_name}.{method_name} - 输入: {in_shapes}")

        # 调用原始方法
        out = func(self, *args, **kwargs)

        # 输出 shape
        if hasattr(out, "shape"):
            print(f"[LOG] {class_name}.{method_name} - 输出: {out.shape}")
        else:
            print(f"[LOG] {class_name}.{method_name} - 输出: (非Tensor类型)")

        return out
    return wrapper
