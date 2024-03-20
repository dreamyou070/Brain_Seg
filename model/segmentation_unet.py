import torch
import torch.nn as nn
import torch.nn.functional as F
# https://github.com/milesial/Pytorch-UNet
class Org_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))



        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, use_batchnorm = True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        if use_batchnorm :
            self.double_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                                             nn.BatchNorm2d(mid_channels),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                             nn.BatchNorm2d(out_channels),
                                             nn.ReLU(inplace=True))
        else :
            #self.double_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            #                                 nn.LayerNorm(mid_channels),
            #                                 nn.ReLU(inplace=True),
            #                                 nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1,bias=False),
            #                                 nn.LayerNorm(out_channels),
            #                                 nn.ReLU(inplace=True))
            self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
            self.act1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1,bias=False)
            self.act2 = nn.ReLU(inplace=True)

        self.use_batchnorm = use_batchnorm

    def forward(self, x):
        if self.use_batchnorm :
            return self.double_conv(x)
        else :
            x = self.conv1(x)
            b, d, r, p = x.shape
            layer_norm = nn.LayerNorm([d, r, p]).to(x.device)
            x = layer_norm(x)
            x = self.act1(x)
            x = self.conv2(x)
            b, d, r, p = x.shape
            layer_norm = nn.LayerNorm([d, r, p]).to(x.device)
            x = layer_norm(x)
            x = self.act2(x)
            return x




class Down(nn.Module):
    """ (1) resolution down by maxpool
        (2) channel down by convolution """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2),
                                          DoubleConv(in_channels, out_channels))
    def forward(self, x):
        return self.maxpool_conv(x)

class Mid(nn.Module):

    """ (1) just double convoblution """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels,out_channels)

    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, use_batchnorm = True ):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels,
                                   use_batchnorm = use_batchnorm)

    def forward(self, x1, x2):

        # [1] x1
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # [2] concat
        x = torch.cat([x2, x1], dim=1) # concatenation
        # [3] out conv
        x = self.conv(x)
        return x

class Up_conv(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, bilinear=True, use_batchnorm=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x1):

        # [1] x1
        x = self.up(x1)
        return x
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Segmentation_Head(nn.Module):

    def __init__(self,  n_classes, bilinear=False, use_batchnorm=True):
        super(Segmentation_Head, self).__init__()

        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.up1 = (Up(1280, 640 // factor, bilinear, use_batchnorm))
        self.up2 = (Up(640, 320 // factor, bilinear, use_batchnorm))
        self.up3 = (Up_conv(320, bilinear))
        self.outc = (OutConv(160, n_classes))

    def forward(self, x16_out, x32_out, x64_out):

        x = self.up1(x16_out,x32_out)  # 1,640,32,32
        x = self.up2(x, x64_out)    # 1,320,64,64
        x3_out = self.up3(x)        # 1,160,128,128
        logits = self.outc(x3_out)  # 1,4, 128,128
        return logits
