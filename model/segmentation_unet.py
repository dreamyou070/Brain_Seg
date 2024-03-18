import torch
import torch.nn as nn
import torch.nn.functional as F
# https://github.com/milesial/Pytorch-UNet
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
    def forward(self, x):
        return self.double_conv(x)


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

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

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

    def __init__(self, in_channels, bilinear=True):
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

class UNet(nn.Module):
    def __init__(self, n_classes, bilinear=False):
        super(UNet, self).__init__()

        self.n_classes = n_classes
        self.bilinear = bilinear

        self.down1 = (Down(40, 80))  # self.down1 = (Down(40,80))
        self.down2 = (Down(80, 160)) # self.down2 = (Down(80,160))
        self.down3 = (Down(160, 320)) # self.down3 = (Down(160,320))
        factor = 2 if bilinear else 1
        self.up1 = (Up(320, 160 // factor, bilinear))
        self.up2 = (Up(160, 80 // factor, bilinear))
        self.up3 = (Up_conv(80, bilinear))
        self.outc = (OutConv(40, n_classes))

    def forward(self, x1, x2, x3):

        x1_out = self.down1(x1)
        x2_in = x2 + x1_out         # 1,80,32,32
        x2_out = self.down2(x2_in)  # out channel = 160
        x3_in = x3 + x2_out         # 1, 160, 16,16
        x3_out = self.down3(x3_in)  # 1, 320, 8, 8
        x = self.up1(x3_out, x2_out)
        x = self.up2(x, x1_out)     # 1,80, 32, 32
        x = self.up3(x)             # 1, 40, 64,64
        logits = self.outc(x)
        return logits

class UNet2(nn.Module):

    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet2, self).__init__()

        self.n_classes = n_classes
        self.bilinear = bilinear
        self.init_conv = (DoubleConv(n_channels, 40))
        self.down1 = (Down(40, 80))  # self.down1 = (Down(40,80))
        self.down2 = (Down(80, 160))  # self.down2 = (Down(80,160))
        self.down3 = (Down(160, 320))  # self.down3 = (Down(160,320))
        factor = 2 if bilinear else 1
        self.up1 = (Up(320, 160 // factor, bilinear))
        self.up2 = (Up(160, 80 // factor, bilinear))
        self.up3 = (Up_conv(80, bilinear))
        self.outc = (OutConv(40, n_classes))

    def forward(self, x, x1, x2, x3):
        # x = [1,4,64,64]
        x_out = self.init_conv(x)  # 1,40,64,64
        x1_in = x_out + x1         # 1,40,64,64
        x1_out = self.down1(x1_in)    # 1,80,32,32
        x2_in = x1_out + x2        # 1,80,32,32
        x2_out = self.down2(x2_in) # 1,160,16,16
        x3_in = x2_out + x3        # 1,160,16,16
        x3_out = self.down3(x3_in) # 1,320,8,8
        x = self.up1(x3_out, x2_out)
        x = self.up2(x, x1_out)     # 1,80, 32, 32
        x = self.up3(x)             # 1, 40, 64,64
        logits = self.outc(x)
        return logits

class UNet3(nn.Module):

    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet3, self).__init__()

        self.n_classes = n_classes
        self.bilinear = bilinear
        self.init_conv = (DoubleConv(n_channels, 40))
        self.down1 = (Down(40, 80))  # self.down1 = (Down(40,80))
        self.down2 = (Down(80, 160))  # self.down2 = (Down(80,160))
        self.down3 = (Down(160, 320))  # self.down3 = (Down(160,320))
        factor = 2 if bilinear else 1
        self.up1 = (Up(320, 160 // factor, bilinear))
        self.up2 = (Up(160, 80 // factor, bilinear))
        self.up3 = (Up_conv(80, bilinear))
        self.outc = (OutConv(40, n_classes))

    def forward(self, x):
        # x = [1,4,64,64]
        x_out = self.init_conv(x)     # 1,40,64,64
        x1_out = self.down1(x_out)    # 1,80,32,32
        x2_out = self.down2(x1_out)   # 1,160,16,16
        x3_out = self.down3(x2_out)   # 1,320,8,8
        x = self.up1(x3_out, x2_out)
        x = self.up2(x, x1_out)       # 1,80, 32, 32
        x = self.up3(x)               # 1, 40, 64,64
        logits = self.outc(x)
        return x_out, x1_out, x2_out, logits

"""
# x1 = torch.randn(1,40,64,64)
class UNet2(nn.Module):
    def __init__(self, n_classes, bilinear=False):
        super(UNet, self).__init__()

        self.n_classes = n_classes
        self.bilinear = bilinear

        self.down1 = (Down(40, 80))  # self.down1 = (Down(40,80))
        self.down2 = (Down(80, 160)) # self.down2 = (Down(80,160))
        self.down3 = (Down(160, 320)) # self.down3 = (Down(160,320))
        factor = 2 if bilinear else 1
        self.up1 = (Up(320, 160 // factor, bilinear))
        self.up2 = (Up(160, 80 // factor, bilinear))
        self.up3 = (Up_conv(80, bilinear))
        self.outc = (OutConv(40, n_classes))

    def forward(self, x1, x2, x3):

        x1_out = self.down1(x1)
        x2_in = x2 + x1_out         # 1,80,32,32
        x2_out = self.down2(x2_in)  # out channel = 160
        x3_in = x3 + x2_out         # 1, 160, 16,16
        x3_out = self.down3(x3_in)  # 1, 320, 8, 8
        x = self.up1(x3_out, x2_out)
        x = self.up2(x, x1_out)     # 1,80, 32, 32
        x = self.up3(x)             # 1, 40, 64,64
        logits = self.outc(x)
        return logits
"""
"""
x = torch.randn(1,4,64,64)
x1 = torch.randn(1,40, 64,64)
x2 = torch.randn(1,80, 32,32)
x3 = torch.randn(1,160,16,16)
unet1 = UNet(n_classes = 4)
#logit1 = unet1(x, x1,x2,x3)


unet2 = UNet2(n_channels=4, n_classes=4)
logit2 = unet2(x, x1,x2,x3)
print(logit2.shape)
"""