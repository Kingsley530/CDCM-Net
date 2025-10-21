import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2
from torch.nn import init  
import numpy as np
from einops import rearrange
from extra_module import FreqSpatial, EMSConvP, ContextGuideFusionModule, MultiAttention


class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        the_three_features = self.features[:7](x)  # 72*72*32
        the_four_features = self.features[:11](x)  # 36*36*64
        x = self.features[4:](low_level_features)
        return low_level_features, the_three_features, the_four_features, x

    # -----------------------------------------#




class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch2 = EMSConvP(c1=dim_in, c2=256, n=1, c3k=True)
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 2, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.mulatten = MultiAttention(256)

    def forward(self, x):
        [b, c, row, col] = x.size()

        emscp = self.branch2(x)

        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        feature_cat = torch.cat([emscp, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result

class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilenet", pretrained=False, downsample_factor=16):
        super(DeepLab, self).__init__()
        if backbone == "xception":

            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone == "mobilenet":

            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
            the_three_channels = 32
            the_four_channels = 64
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))


        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16 // downsample_factor)
        self.ema1 = EMA(48, 48)
        self.ema2 = EMA(96, 96)
        self.ema3 = EMA(256, 256)
        self.mulatten = MultiAttention(256)


        self.cgfm1 = ContextGuideFusionModule(inc=(the_three_channels, low_level_channels))  
        self.cgfm2 = ContextGuideFusionModule(inc=(64, 48))  

        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(120, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(96 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

        self.freqspatial1 = FreqSpatial(c1=48, c2=48)
        self.freqspatial2 = FreqSpatial(c1=96, c2=96)

    def forward(self, x):
        H, W = x.size(2), x.size(3)

        low_level_features, the_three_features, the_four_features, x = self.backbone(x)
        x = self.aspp(x)
        # x = self.ema3(x)
        
        the_three_features_up = F.interpolate(the_three_features, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
        # combined_features1 = self.cgfm1([the_three_features_up, low_level_features])
        combined_features1 = self.cgfm1(the_three_features_up, low_level_features)
        combined_features1 = self.freqspatial1(combined_features1)
        combined_features1 = self.ema1(combined_features1)
        the_four_features_up = F.interpolate(the_four_features, size=(combined_features1.size(2), combined_features1.size(3)), mode='bilinear', align_corners=True)
        # combined_features2 = self.cgfm2([the_four_features_up, combined_features1])
        combined_features2 = self.cgfm2(the_four_features_up, combined_features1)
        combined_features2 = self.freqspatial2(combined_features2)
        combined_features2 = self.ema2(combined_features2)
        x_up = F.interpolate(x, size=(combined_features2.size(2), combined_features2.size(3)), mode='bilinear', align_corners=True)
        x = self.cat_conv(torch.cat((x_up, combined_features2), dim=1))#144*144*(256+48)-144*144*256
        x = self.mulatten(x)
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

