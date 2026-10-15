import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                 stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias), 
        nn.PReLU(num_parameters=out_channels)
    )

class multkernelconv(nn.Module):
    def __init__(self, channels, dilation=1, bias=True):
        super(multkernelconv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, 
                    stride=1, padding=dilation, dilation=dilation, bias=bias, groups=channels), 
            nn.PReLU(num_parameters=channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=5, 
                    stride=1, padding=2*dilation, dilation=dilation, bias=bias, groups=channels), 
            nn.PReLU(num_parameters=channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=7, 
                    stride=1, padding=3*dilation, dilation=dilation, bias=bias, groups=channels), 
            nn.PReLU(num_parameters=channels)
        )
        self.Pconv = nn.Conv2d(in_channels=3*channels, out_channels=channels, kernel_size=1, 
                            stride=1, padding=0, bias=bias)
        self.prelu = nn.PReLU(num_parameters=channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        out = torch.cat(tensors=[x1, x2, x3], dim=1)
        out = self.Pconv(out)
        out = self.prelu(x + out)
        return out

class ResBlock(nn.Module):
    def __init__(self, channels, dilation=1, bias=True):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, 
                     stride=1, padding=dilation, dilation=dilation, bias=bias), 
            nn.PReLU(num_parameters=channels)
        )
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, 
                              stride=1, padding=dilation, dilation=dilation, bias=bias)
        self.prelu = nn.PReLU(num_parameters=channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.prelu(x + out)
        return out

class CALayer(nn.Module):
    def __init__(self, n_dim, reduction=4):
        super(CALayer, self).__init__()
        #nChannels_ = nChannels
        self.n_dim = n_dim
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.convX = nn.Conv2d(n_dim, n_dim//reduction,kernel_size=1)
        self.conv1 = nn.Conv2d(n_dim, n_dim//reduction,kernel_size=1)
        self.conv2 = nn.Conv2d(n_dim//reduction, n_dim//reduction,kernel_size=1)
    def forward(self, x):
        y = self.pool(x)
        x = self.convX(x)
        
        
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.sigmoid(y)
        return x * y

class Ferreti(nn.Module):
    def __init__(self):
        super(Ferreti, self).__init__()
        
        self.conv0 = nn.Sequential(
            convrelu(in_channels=6, out_channels=20),
            convrelu(in_channels=20, out_channels=20)
        )
        self.conv1 = nn.Sequential(
            convrelu(in_channels=6, out_channels=40),
            convrelu(in_channels=40, out_channels=40)
        )
        self.conv2 = nn.Sequential(
            convrelu(in_channels=6, out_channels=20),
            convrelu(in_channels=20, out_channels=20)
        )

        self.alignment1 = nn.Sequential(
            multkernelconv(channels=60, dilation=2),
            CALayer(n_dim=60, reduction=3)
        )

        self.alignment2 = nn.Sequential(
            multkernelconv(channels=60, dilation=2),
            CALayer(n_dim=60, reduction=3)
        )

        self.convSC = nn.Sequential(
            convrelu(in_channels=40, out_channels=80),
            convrelu(in_channels=80, out_channels=80),
            CALayer(n_dim=80, reduction=1)
        )

        self.convfeat1 = nn.Sequential(
            convrelu(in_channels=40, out_channels=80),
            convrelu(in_channels=80, out_channels=80),
            CALayer(n_dim=80, reduction=2)
        )
        

        self.resblock1 = multkernelconv(channels=80, dilation=1)
        self.resblock2 = multkernelconv(channels=80, dilation=2)
        self.resblock3 = multkernelconv(channels=80, dilation=1)

        self.sigmoid = nn.Sigmoid()

        
        self.conv3 = nn.Conv2d(in_channels=80, out_channels=3, kernel_size=3, 
                              stride=1, padding=1, dilation=1, bias=True)
        
    def forward(self, img0_c, img1_c, img2_c, train):
        feat0 = self.conv0(img0_c)
        feat1 = self.conv1(img1_c)
        feat2 = self.conv2(img2_c)
        
        feat0 = torch.cat(tensors=[feat0, feat1], dim=1)
        feat0 = self.alignment1(feat0)

        feat2 = torch.cat(tensors=[feat2, feat1], dim=1)
        feat2 = self.alignment2(feat2)

        SC = self.convSC(feat1)
        feat1 = self.convfeat1(feat1)
        

        feat = torch.cat(tensors=[feat0, feat1, feat2], dim=1)
        # feat = self.resblock1(feat)
        # feat = self.resblock2(feat)
        # feat = self.resblock3(feat)
        
        res = self.conv3(feat+SC)
        res = self.sigmoid(res)
        img_hdr_r = torch.clamp(res, 0, 1)
        img_hdr_m = img_hdr_r
        
        return img_hdr_m, img_hdr_r
