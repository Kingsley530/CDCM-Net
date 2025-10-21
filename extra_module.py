import torch
import torch.nn as nn
from functools import reduce
import numpy as np


######################################## FreqSpatial start ########################################

class ScharrConv(nn.Module):
    def __init__(self, channel):
        super(ScharrConv, self).__init__()
        
        # 定义Scharr算子的水平和垂直卷积核
        scharr_kernel_x = np.array([[3,  0, -3],
                                    [10, 0, -10],
                                    [3,  0, -3]], dtype=np.float32)
        
        scharr_kernel_y = np.array([[3, 10, 3],
                                    [0,  0, 0],
                                    [-3, -10, -3]], dtype=np.float32)
        
        # 将Scharr核转换为PyTorch张量并扩展为通道数
        scharr_kernel_x = torch.tensor(scharr_kernel_x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
        scharr_kernel_y = torch.tensor(scharr_kernel_y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
        
        # 扩展为多通道
        self.scharr_kernel_x = scharr_kernel_x.expand(channel, 1, 3, 3)  # (channel, 1, 3, 3)
        self.scharr_kernel_y = scharr_kernel_y.expand(channel, 1, 3, 3)  # (channel, 1, 3, 3)

        # 定义卷积层，但不学习卷积核，直接使用Scharr核
        self.scharr_kernel_x_conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.scharr_kernel_y_conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        
        # 将卷积核的权重设置为Scharr算子的核
        self.scharr_kernel_x_conv.weight.data = self.scharr_kernel_x.clone()
        self.scharr_kernel_y_conv.weight.data = self.scharr_kernel_y.clone()

        # 禁用梯度更新
        self.scharr_kernel_x_conv.requires_grad = False
        self.scharr_kernel_y_conv.requires_grad = False

    def forward(self, x):
        # 对输入的特征图进行Scharr卷积（水平和垂直方向）
        grad_x = self.scharr_kernel_x_conv(x)
        grad_y = self.scharr_kernel_y_conv(x)
        
        # 计算梯度幅值
        edge_magnitude = grad_x * 0.5 + grad_y * 0.5
        
        return edge_magnitude

class FreqSpatial(nn.Module):
    def __init__(self, in_channels):
        super(FreqSpatial, self).__init__()

        self.sed = ScharrConv(in_channels)
        
        # 时域卷积部分
        self.spatial_conv1 = Conv(in_channels, in_channels)
        self.spatial_conv2 = Conv(in_channels, in_channels)

        # 频域卷积部分
        self.fft_conv = Conv(in_channels * 2, in_channels * 2, 3)
        self.fft_conv2 = Conv(in_channels, in_channels, 3)
        
        self.final_conv = Conv(in_channels, in_channels, 1)

    def forward(self, x):
        batch, c, h, w = x.size()
        # 时域提取
        spatial_feat = self.sed(x)
        spatial_feat = self.spatial_conv1(spatial_feat)
        spatial_feat = self.spatial_conv2(spatial_feat + x)

        # 频域卷积
        # 1. 先转换到频域
        fft_feat = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(fft_feat), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(fft_feat), dim=-1)
        fft_feat = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        fft_feat = rearrange(fft_feat, 'b c h w d -> b (c d) h w').contiguous()

        # 2. 频域卷积处理
        fft_feat = self.fft_conv(fft_feat)

        # 3. 还原回时域
        fft_feat = rearrange(fft_feat, 'b (c d) h w -> b c h w d', d=2).contiguous()
        fft_feat = torch.view_as_complex(fft_feat)
        fft_feat = torch.fft.irfft2(fft_feat, s=(h, w), norm='ortho')
        
        fft_feat = self.fft_conv2(fft_feat)

        # 合并时域和频域特征
        out = spatial_feat + fft_feat
        return self.final_conv(out)

######################################## FreqSpatial end ########################################

######################################## ContextGuideFusionModule begin ########################################

class SEAttention(nn.Module):
    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class ContextGuideFusionModule(nn.Module):
    def __init__(self, inc) -> None:
        super().__init__()
        
        self.adjust_conv = nn.Identity()
        if inc[0] != inc[1]:
            self.adjust_conv = Conv(inc[0], inc[1], k=1)
        
        self.se = SEAttention(inc[1] * 2)
    
    def forward(self, x0, x1):  # 直接接受两个输入张量
        x0 = self.adjust_conv(x0)
        x_concat = torch.cat([x0, x1], dim=1)  # n c h w
        x_concat = self.se(x_concat)
        x0_weight, x1_weight = torch.split(x_concat, [x0.size()[1], x1.size()[1]], dim=1)
        x0_weight = x0 * x0_weight
        x1_weight = x1 * x1_weight
        return torch.cat([x0 + x1_weight, x1 + x0_weight], dim=1)

        
######################################## ContextGuideFusionModule end ########################################

######################################## EMA  ########################################

class EMSConv(nn.Module):
    # Efficient Multi-Scale Conv
    def __init__(self, channel=256, kernels=[3, 5]):
        super().__init__()
        self.groups = len(kernels)
        min_ch = channel // 4
        assert min_ch >= 16, f'channel must Greater than {64}, but {channel}'
        
        self.convs = nn.ModuleList([])
        for ks in kernels:
            self.convs.append(Conv(c1=min_ch, c2=min_ch, k=ks))
        self.conv_1x1 = Conv(channel, channel, k=1)
        
    def forward(self, x):
        _, c, _, _ = x.size()
        x_cheap, x_group = torch.split(x, [c // 2, c // 2], dim=1)
        x_group = rearrange(x_group, 'bs (g ch) h w -> bs ch h w g', g=self.groups)
        x_group = torch.stack([self.convs[i](x_group[..., i]) for i in range(len(self.convs))])
        x_group = rearrange(x_group, 'g bs ch h w -> bs (g ch) h w')
        x = torch.cat([x_cheap, x_group], dim=1)
        x = self.conv_1x1(x)
        
        return x

class EMSConvP(nn.Module):
    # Efficient Multi-Scale Conv Plus
    def __init__(self, channel=256, kernels=[1, 3, 5, 7]):
        super().__init__()
        self.groups = len(kernels)
        min_ch = channel // self.groups
        assert min_ch >= 16, f'channel must Greater than {16 * self.groups}, but {channel}'
        
        self.convs = nn.ModuleList([])
        for ks in kernels:
            self.convs.append(Conv(c1=min_ch, c2=min_ch, k=ks))
        self.conv_1x1 = Conv(channel, channel, k=1)
        
    def forward(self, x):
        x_group = rearrange(x, 'bs (g ch) h w -> bs ch h w g', g=self.groups)
        x_convs = torch.stack([self.convs[i](x_group[..., i]) for i in range(len(self.convs))])
        x_convs = rearrange(x_convs, 'g bs ch h w -> bs (g ch) h w')
        x_convs = self.conv_1x1(x_convs)
        
        return x_convs

class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=16):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Conv(channels // self.groups, channels // self.groups)
        )
        self.conv_pool_hw = Conv(channels // self.groups, channels // self.groups, 1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        # print('group_x =' , group_x.size())    # group_x = 32 16 32 32 
        x_pool_ch = self.gap(group_x)
        # print('x_pool_ch =' , x_pool_ch.size())   # x_pool_ch = 
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        # print('hw =' , hw.size())  # hw = [32, 16, 64, 1]
        x_pool_hw_weight = self.conv_pool_hw(hw).sigmoid()
        x_pool_ch = x_pool_ch * torch.mean(x_pool_hw_weight, dim=2, keepdim=True)
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        # print('x_h = ', x_h.size(), 'x_w = ', x_w.size())    # x_h = [32, 16, 32, 1],  x_w = [32, 16, 32, 1]
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        # print('x1 = ', x1.size())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        weights = (group_x * weights.sigmoid()).reshape(b, c, h, w)
        x_pool_h, x_pool_w = torch.split(hw, [h, w], dim=2)
        x_pool_h_weight, x_pool_w_weight = torch.split(x_pool_hw_weight, [h, w], dim=2)
        x_pool_h, x_pool_w = x_pool_h * x_pool_h_weight, x_pool_w * x_pool_w_weight
        x_pool = (group_x * x_pool_h.sigmoid() * x_pool_w.permute(0, 1, 3, 2).sigmoid() * x_pool_ch.sigmoid()).reshape(b, c, h, w)
        return (weights * x_pool)

######################################## EMA ########################################

######################################## Multi-Attention Start #####################################################


# ---------------------- 基础注意力模块 ----------------------
class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y  # 像素注意力输出（无残差）

class SKConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        super(SKConv, self).__init__()
        d = max(in_channels//r, L)
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.ModuleList()
        for i in range(M):
            self.conv.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, 
                         padding=1+i, dilation=1+i, groups=8, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(out_channels, d, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Conv2d(d, out_channels*M, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        output = []
        for conv in self.conv:
            output.append(conv(x))
        U = reduce(lambda x, y: x+y, output)
        s = self.global_pool(U)
        z = self.fc1(s)
        a_b = self.fc2(z)
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1)
        a_b = self.softmax(a_b)
        a_b = list(a_b.chunk(self.M, dim=1))
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1, 1), a_b))
        V = list(map(lambda x, y: x*y, output, a_b))
        V = reduce(lambda x, y: x+y, V)
        return V  # 通道注意力输出（无残差）

class SobelConv(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()
        
        sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sobel_kernel_y = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)
        sobel_kernel_x = torch.tensor(sobel.T, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)
        
        self.sobel_kernel_x_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.sobel_kernel_y_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        
        self.sobel_kernel_x_conv3d.weight.data = sobel_kernel_x.clone()
        self.sobel_kernel_y_conv3d.weight.data = sobel_kernel_y.clone()
        
        self.sobel_kernel_x_conv3d.weight.requires_grad = False
        self.sobel_kernel_y_conv3d.weight.requires_grad = False


        # self.sobel_kernel_x_conv3d.requires_grad = False
        # self.sobel_kernel_y_conv3d.requires_grad = False

    def forward(self, x):
        return (self.sobel_kernel_x_conv3d(x[:, :, None, :, :]) + self.sobel_kernel_y_conv3d(x[:, :, None, :, :]))[:, :, 0]

class EIEM(nn.Module):
    def __init__(self, inc, ouc):
        super().__init__()
        self.sobel_branch = SobelConv(inc)
        self.conv_branch = Conv(inc, inc, 3)
        self.conv1 = Conv(inc*2, inc, 1)
        self.conv2 = Conv(inc, ouc, 1)  # 移除残差连接

    def forward(self, x):
        x_sobel = self.sobel_branch(x)
        x_conv = self.conv_branch(x)
        x_concat = torch.cat([x_sobel, x_conv], dim=1)
        x_feature = self.conv1(x_concat)
        return self.conv2(x_feature)  # 空间注意力输出（无残差）


class MultiAttention(nn.Module):
    def __init__(self, channel):
        super().__init__()
        # 三个子分支保持不变（只输出注意力权重，而不是特征）
        self.pa = PALayer(channel)   # 像素注意力
        self.sk = SKConv(channel, channel)  # 通道注意力
        self.eiem = EIEM(channel, channel)  # 空间边缘注意力

        # 统一 sigmoid 映射到权重范围
        self.sigmoid = nn.Sigmoid()

        # 最后残差映射
        self.residual_conv = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 各分支分别生成权重 mask（与 PPSA 类似）
        pa_mask = self.sigmoid(self.pa(x))      # bs, c, h, w
        sk_mask = self.sigmoid(self.sk(x))      # bs, c, h, w
        eiem_mask = self.sigmoid(self.eiem(x))  # bs, c, h, w

        # 并行加权到输入特征（解耦）
        pa_out = x * pa_mask
        sk_out = x * sk_mask
        eiem_out = x * eiem_mask

        # 融合方式：直接相加（类似 PPSA 的 channel_out + spatial_out）
        out = pa_out + sk_out + eiem_out

        # 残差增强（保持稳定训练）
        out = self.residual_conv(out + x)
        return out

######################################## EMSConv+EMSConvP begin ########################################

class EMSConv(nn.Module):
    # Efficient Multi-Scale Conv
    def __init__(self, channel=256, kernels=[3, 5]):
        super().__init__()
        self.groups = len(kernels)
        min_ch = channel // 4
        assert min_ch >= 16, f'channel must Greater than {64}, but {channel}'
        
        self.convs = nn.ModuleList([])
        for ks in kernels:
            self.convs.append(Conv(c1=min_ch, c2=min_ch, k=ks))
        self.conv_1x1 = Conv(channel, channel, k=1)
        
    def forward(self, x):
        _, c, _, _ = x.size()
        x_cheap, x_group = torch.split(x, [c // 2, c // 2], dim=1)
        x_group = rearrange(x_group, 'bs (g ch) h w -> bs ch h w g', g=self.groups)
        x_group = torch.stack([self.convs[i](x_group[..., i]) for i in range(len(self.convs))])
        x_group = rearrange(x_group, 'g bs ch h w -> bs (g ch) h w')
        x = torch.cat([x_cheap, x_group], dim=1)
        x = self.conv_1x1(x)
        
        return x

class EMSConvP(nn.Module):
    # Efficient Multi-Scale Conv Plus + MultiAttention per branch
    def __init__(self, channel=256, kernels=[1, 3, 5, 7]):
        super().__init__()
        self.groups = len(kernels)
        min_ch = channel // self.groups
        assert min_ch >= 16, f'channel must Greater than {16 * self.groups}, but {channel}'
        
        self.branches = nn.ModuleList([])
        for ks in kernels:
            self.branches.append(
                nn.Sequential(
                    Conv(c1=min_ch, c2=min_ch, k=ks),
                    MultiAttention(min_ch)   # 👈 每个分支加注意力
                )
            )
        self.conv_1x1 = Conv(channel, channel, k=1)

    def forward(self, x):
        # 通道分组
        x_group = rearrange(x, 'bs (g ch) h w -> bs ch h w g', g=self.groups)
        # 每个分支分别做 conv + attention
        x_convs = torch.stack([self.branches[i](x_group[..., i]) for i in range(len(self.branches))])
        # 融合
        x_convs = rearrange(x_convs, 'g bs ch h w -> bs (g ch) h w')
        x_convs = self.conv_1x1(x_convs)
        return x_convs
