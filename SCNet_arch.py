import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import default_init_weights, make_layer


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


class ResidualBlockShift(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-Shift-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockShift, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, kernel_size=1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.shift = Shift8(groups=num_feat//8, stride=1)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.shift(self.conv1(x))))
        return identity + out * self.res_scale

    
class UpShiftPixelShuffle(nn.Module):
    def __init__(self, dim, scale=2) -> None:
        super().__init__()

        self.up_layer = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.LeakyReLU(0.02),
            Shift8(groups=dim//8),
            nn.Conv2d(dim, dim*scale*scale, kernel_size=1),
            nn.PixelShuffle(upscale_factor=scale)
        )
    def forward(self, x):
        out = self.up_layer(x)
        return out

class UpShiftMLP(nn.Module):
    def __init__(self, dim, mode='bilinear', scale=2) -> None:
        super().__init__()

        self.up_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode=mode, align_corners=False),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.LeakyReLU(0.02),
            Shift8(groups=dim//8),
            nn.Conv2d(dim, dim, kernel_size=1)
        )
    def forward(self, x):
        out = self.up_layer(x)
        return out

@ARCH_REGISTRY.register()
class SCNet(nn.Module):
    """ SCNet (https://arxiv.org/abs/2307.16140) based on the Modified SRResNet.
    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_block (int): Block number in the body network. Default: 16.
        upscale (int): Upsampling factor. Support x2, x3 and x4. Default: 4.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(SCNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 1)
        self.body = make_layer(ResidualBlockShift, num_block, num_feat=num_feat)

        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = UpShiftMLP(num_feat, scale=self.upscale)

        elif self.upscale == 4:
            self.upconv1 = UpShiftMLP(num_feat)
            self.upconv2 = UpShiftMLP(num_feat)
        elif self.upscale == 8:
            self.upconv1 = UpShiftMLP(num_feat)
            self.upconv2 = UpShiftMLP(num_feat)
            self.upconv3 = UpShiftMLP(num_feat)
        # freeze infrence
        self.pixel_shuffle = nn.Identity()

        self.conv_hr = nn.Conv2d(num_feat, num_feat, kernel_size=1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, kernel_size=1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.upconv1, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        out = self.body(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        elif self.upscale == 8:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv3(out)))

        out = self.conv_last(self.lrelu(self.conv_hr(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out

if __name__ == '__main__':
    model = SCNet(upscale=4)
    load_dict = torch.load('SCNet-T-x4.pth')
    model.load_state_dict(load_dict['params'])
