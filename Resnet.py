"""
Detail:
Ref:
Project:my_python_script
Time:2022/3/10 19:52
Author:WRZ
"""
import torch
from torchvision import models
import torch.nn as nn


class GlobalAVGMAXPooling(nn.Module):
    def __init__(self, in_ch=20, out_ch=30, pool=1):
        """
        Args:
                in_ch:  liner input
                out_ch: liner output
                pool: double channels
        """
        super(GlobalAVGMAXPooling, self).__init__()

        self.maxpool = nn.AdaptiveMaxPool2d((pool, pool))
        self.avgpool = nn.AdaptiveAvgPool2d((pool, pool))
        self.linear = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        return self.linear(torch.flatten(torch.cat((self.maxpool(x), self.avgpool(x)), 1), start_dim=1))


def _upsample_like(src, tar):

    src = nn.functional.interpolate(
        src, size=tar, mode='bilinear', align_corners=True)

    return src


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3,
                 dirate=1, kernel_size=3):

        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(
            in_ch, out_ch, kernel_size, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)

        self.conv_b1 = nn.Conv2d(
            in_ch, out_ch, 1, padding=0, dilation=1 * dirate)
        self.bn_b1 = nn.BatchNorm2d(out_ch)

        self.relu_s1 = nn.ReLU(inplace=True)
        self.conv_s2 = nn.Conv2d(
            out_ch, out_ch, kernel_size, padding=2 * dirate, dilation=2 * dirate)
        self.bn_s2 = nn.BatchNorm2d(out_ch)

        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x):
        xin = self.bn_b1(self.conv_b1(x))

        xout = self.bn_s2(self.conv_s2(
            self.relu_s1(self.bn_s1(self.conv_s1(x)))))
        xout2 = _upsample_like(xout + xin, tar=xout.shape[2] * 2)

        return self.relu_out(xout2)


class classifer(nn.Module):
    def __init__(self, in_ch=1024, out_ch=1):
        super(classifer, self).__init__()
        self.CBR0 = REBNCONV(in_ch, in_ch // 2, 1)
        self.CB0 = REBNCONV(in_ch, in_ch // 2, 1, kernel_size=3)
        self.CBR1 = REBNCONV(in_ch // 2, in_ch // 4, 1)
        self.CBR2 = REBNCONV(in_ch // 4, in_ch // 8, 1)  # 128,256,256
        self.CBR3 = REBNCONV(in_ch // 8, in_ch // 16, 1)

        self.poolliner = GlobalAVGMAXPooling(in_ch // 8, in_ch // 32, 1)
        self.linear_out = nn.Linear(in_ch // 32, 1)

        self.Conv_out = nn.Conv2d(
            in_ch // 16, out_ch, 3, padding=1, dilation=1)
        self.BN_out = nn.BatchNorm2d(out_ch)
        self.Sigmoid_out = nn.Sigmoid()

    def forward(self, x):
        # 512,64
        C0 = self.CBR0(x)

        # 256，128
        C1 = self.CBR1(C0)

        # 128，256
        C2 = self.CBR2(C1)

        # 64，512
        C3 = self.CBR3(C2)

        liner_out = self.Sigmoid_out(self.linear_out(self.poolliner(C3)))

        Conv_out = self.Conv_out(C3)
        BN_out = self.BN_out(Conv_out)
        Image_out = self.Sigmoid_out(BN_out)

        return [Image_out, liner_out]


class Resnet50_rebuild(nn.Module):
    def __init__(self, in_ch=1024, out_ch=1, pretrained=True):
        super(Resnet50_rebuild, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)

        self.backbone = nn.Sequential(*list(self.model.children())[:-3])

        self.classifer_out = nn.Sequential(self.backbone,
                                           classifer(in_ch=in_ch, out_ch=out_ch))

    def forward(self, x):

        return self.classifer_out(x)
