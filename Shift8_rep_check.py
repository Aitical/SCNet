import torch
from torch import nn as nn
from torch.nn import functional as F

class Shift8(nn.Module):
    def __init__(self, groups=4, stride=1, mode="constant", reparam=False) -> None:
        super().__init__()
        self.g = groups
        self.mode = mode
        self.stride = stride
        self.reparam_conv = None
        
        if reparam:
            self.reparameterize()

    def forward(self, x):
        # 如果已经重参数化，使用卷积
        if self.reparam_conv is not None:
            return self.reparam_conv(x)
        
        # 否则使用原始的shift操作
        b, c, h, w = x.shape
        out = torch.zeros_like(x)

        pad_x = F.pad(x, pad=[self.stride for _ in range(4)], mode=self.mode)
        assert c == self.g * 8

        cx, cy = self.stride, self.stride
        stride = self.stride
        
        out[:, 0 * self.g : 1 * self.g, :, :] = pad_x[
            :, 0 * self.g : 1 * self.g, cx - stride : cx - stride + h, cy : cy + w
        ]
        out[:, 1 * self.g : 2 * self.g, :, :] = pad_x[
            :, 1 * self.g : 2 * self.g, cx + stride : cx + stride + h, cy : cy + w
        ]
        out[:, 2 * self.g : 3 * self.g, :, :] = pad_x[
            :, 2 * self.g : 3 * self.g, cx : cx + h, cy - stride : cy - stride + w
        ]
        out[:, 3 * self.g : 4 * self.g, :, :] = pad_x[
            :, 3 * self.g : 4 * self.g, cx : cx + h, cy + stride : cy + stride + w
        ]
        out[:, 4 * self.g : 5 * self.g, :, :] = pad_x[
            :,
            4 * self.g : 5 * self.g,
            cx + stride : cx + stride + h,
            cy + stride : cy + stride + w,
        ]
        out[:, 5 * self.g : 6 * self.g, :, :] = pad_x[
            :,
            5 * self.g : 6 * self.g,
            cx + stride : cx + stride + h,
            cy - stride : cy - stride + w,
        ]
        out[:, 6 * self.g : 7 * self.g, :, :] = pad_x[
            :,
            6 * self.g : 7 * self.g,
            cx - stride : cx - stride + h,
            cy + stride : cy + stride + w,
        ]
        out[:, 7 * self.g : 8 * self.g, :, :] = pad_x[
            :,
            7 * self.g : 8 * self.g,
            cx - stride : cx - stride + h,
            cy - stride : cy - stride + w,
        ]
        
        return out
    
    def reparameterize(self):
        """将Shift操作转换为等效的深度卷积"""
        kernel_size = 2 * self.stride + 1
        channels = self.g * 8
        
        # 创建深度卷积 (每个通道独立卷积)
        conv = nn.Conv2d(
            channels, 
            channels, 
            kernel_size=kernel_size,
            stride=1,
            padding=self.stride,
            groups=channels,  # 深度卷积
            bias=False
        )
        
        # 初始化卷积核权重
        with torch.no_grad():
            weight = torch.zeros(channels, 1, kernel_size, kernel_size)
            center = self.stride
            
            # 为每个通道组设置对应的shift模式
            for i in range(8):
                start_ch = i * self.g
                end_ch = (i + 1) * self.g
                
                if i == 0:  # 上移
                    weight[start_ch:end_ch, 0, center - self.stride, center] = 1.0
                elif i == 1:  # 下移
                    weight[start_ch:end_ch, 0, center + self.stride, center] = 1.0
                elif i == 2:  # 左移
                    weight[start_ch:end_ch, 0, center, center - self.stride] = 1.0
                elif i == 3:  # 右移
                    weight[start_ch:end_ch, 0, center, center + self.stride] = 1.0
                elif i == 4:  # 右下移
                    weight[start_ch:end_ch, 0, center + self.stride, center + self.stride] = 1.0
                elif i == 5:  # 左下移
                    weight[start_ch:end_ch, 0, center + self.stride, center - self.stride] = 1.0
                elif i == 6:  # 右上移
                    weight[start_ch:end_ch, 0, center - self.stride, center + self.stride] = 1.0
                elif i == 7:  # 左上移
                    weight[start_ch:end_ch, 0, center - self.stride, center - self.stride] = 1.0
            
            conv.weight.data = weight

        conv.weight.requires_grad = False
        conv.eval()
        
        self.reparam_conv = conv
        return conv


# 为整个模型添加重参数化方法
def switch_to_deploy(model):
    """将模型中所有Shift8层转换为卷积"""
    for module in model.modules():
        if isinstance(module, Shift8):
            module.reparameterize()
    return model


# 测试代码
if __name__ == "__main__":
    # 创建Shift8层
    shift = Shift8(groups=8, stride=1)
    x = torch.randn(1, 64, 32, 32)
    
    # 原始输出
    with torch.no_grad():
        out1 = shift(x)
    
    # 重参数化后的输出
    shift.reparameterize()
    with torch.no_grad():
        out2 = shift(x)
    
    # 验证结果一致性
    print(f"输出差异: {torch.abs(out1 - out2).max().item()}")
    print(f"相对误差: {(torch.abs(out1 - out2) / (torch.abs(out1) + 1e-8)).mean().item()}")
    
    # 性能测试
    import time
    
    # 原始Shift8
    shift_original = Shift8(groups=8, stride=1)
    x = torch.randn(4, 64, 256, 256).cuda()
    shift_original = shift_original.cuda()
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = shift_original(x)
    torch.cuda.synchronize()
    original_time = time.time() - start
    
    # 重参数化版本
    shift_reparam = Shift8(groups=8, stride=1)
    shift_reparam.reparameterize()
    shift_reparam = shift_reparam.cuda()
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = shift_reparam(x)
    torch.cuda.synchronize()
    reparam_time = time.time() - start
    
    print(f"\n原始Shift8耗时: {original_time:.4f}s")
    print(f"重参数化卷积耗时: {reparam_time:.4f}s")
    print(f"加速比: {original_time / reparam_time:.2f}x")