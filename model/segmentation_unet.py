# https://github.com/milesial/Pytorch-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
x16_out = torch.randn(1, 1280, 16, 16)
x32_out = torch.randn(1, 640, 32, 32)
x64_out = torch.randn(1, 320, 64, 64)
model = Segmentation_Head_c_with_binary(4)
model(x16_out, x32_out, x64_out)
"""

class DoubleConv(nn.Module):
    """ double convolution block, CONV - Norm - Non linear """
    def __init__(self, in_channels, out_channels, mid_channels=None, norm_type = 'layer_norm', nonlinear_type = 'relu'):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        # [1] non linear function
        self.nonlinear_type = nonlinear_type
        nonlinearity = nn.ReLU(inplace=True)
        if self.nonlinear_type == 'leaky_relu':
            nonlinearity = nn.LeakyReLU(inplace=True)
        elif self.nonlinear_type == 'silu':
            nonlinearity = nn.SiLU(inplace=True)

        # [2] normalization function
        self.norm_type = norm_type
        normalization = nn.BatchNorm2d(mid_channels)
        if norm_type == 'layer_norm':
            normalization = nn.LayerNorm([mid_channels, int(20480 / mid_channels), int(20480 / mid_channels)])
        elif norm_type == 'instance_norm':
            normalization = nn.InstanceNorm2d(mid_channels)

        self.double_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                                         normalization, nonlinearity,
                                         nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                         normalization, nonlinearity)
    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """ Upsample is simple interpolation module,
        ConvTranspose2d is Convolution with kernel """
    def __init__(self, in_channels, out_channels, mid_channels,
                 bilinear=False,
                 norm_type = 'layer_norm', nonlinear_type = 'relu'):
        super().__init__()
        # what is difference between Upsample and ConvTranspose2d ?
        self.up = nn.ConvTranspose2d(in_channels, int(in_channels//2), kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels = in_channels,
                               mid_channels = mid_channels,
                               out_channels = out_channels,
                               norm_type = norm_type,
                               nonlinear_type=nonlinear_type)
    def forward(self, x1, x2):

        # [1] x1
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # [2] concat
        x = torch.cat([x2, x1], dim=1)

        # [3] out conv
        x = self.conv(x)
        return x

class UpSingle(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=2, res=128, use_nonlinearity = False, nonlinear_type = 'relu'):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = nn.ConvTranspose2d(in_channels = in_channels,
                                     out_channels = out_channels,
                                     kernel_size=kernel_size,
                                     stride=kernel_size)

        self.use_nonlinearity = use_nonlinearity
        if self.use_nonlinearity :
            self.layernorm = nn.LayerNorm([out_channels,res, res])
            self.nonlinear = nn.ReLU(inplace=True)
            if nonlinear_type == 'leaky_relu':
                self.nonlinear = nn.LeakyReLU(inplace=True)
            elif nonlinear_type == 'silu':
                self.nonlinear = nn.SiLU(inplace=True)
    def forward(self, x1):
        x = self.up(x1)
        if self.use_nonlinearity :
            x = self.layernorm(x)
            x = self.nonlinear(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Segmentation_Head_a(nn.Module):

    def __init__(self,
                 n_classes,
                 mask_res = 128,
                 norm_type = 'layer_norm',
                 use_nonlinearity = False,
                 nonlinear_type = 'relu'):
        super(Segmentation_Head_a, self).__init__()

        self.n_classes = n_classes
        self.mask_res = mask_res
        self.up1 = Up(1280, 640, 640, norm_type, use_nonlinearity)
        self.up2 = Up(640, 320, 320, norm_type, use_nonlinearity)
        self.up3 = UpSingle(in_channels = 320,
                            out_channels = 160,
                            kernel_size=2,
                            res = 128,
                            use_nonlinearity = use_nonlinearity,
                            nonlinear_type=nonlinear_type) # 64 -> 128 , channel 320 -> 160
        if self.mask_res == 256 :
            self.up4 = UpSingle(in_channels = 160,
                                out_channels = 160,
                                kernel_size=2,
                                res=256,
                                use_nonlinearity = use_nonlinearity,
                                nonlinear_type=nonlinear_type)  # 128 -> 256
        self.outc = OutConv(160, n_classes)

    def forward(self, x16_out, x32_out, x64_out):

        x = self.up1(x16_out,x32_out)  # 1,640,32,32 -> 640*32
        x = self.up2(x, x64_out)    # 1,320,64,64
        x3_out = self.up3(x)        # 1,160,128,128
        x_in = x3_out
        if self.mask_res == 256 :
            x4_out = self.up4(x3_out)
            x_in = x4_out
        logits = self.outc(x_in)  # 1,4, 128,128
        return logits

class Segmentation_Head_a_with_binary(nn.Module):

    def __init__(self,
                 n_classes,
                 mask_res = 128,
                 norm_type = 'layer_norm',
                 use_nonlinearity = False,
                 nonlinear_type = 'relu'):
        super(Segmentation_Head_a_with_binary, self).__init__()

        self.n_classes = n_classes
        self.mask_res = mask_res
        self.up1 = Up(1280, 640, 640, norm_type, use_nonlinearity)
        self.up2 = Up(640, 320, 320, norm_type, use_nonlinearity)
        self.up3 = UpSingle(in_channels=320,
                            out_channels=160,
                            kernel_size=2,
                            res=128,
                            use_nonlinearity=use_nonlinearity,
                            nonlinear_type=nonlinear_type)  # 64 -> 128 , channel 320 -> 160
        global_res = 128
        if self.mask_res == 256:
            self.up4 = UpSingle(in_channels=160,
                                out_channels=160,
                                kernel_size=2,
                                res=256,
                                use_nonlinearity=use_nonlinearity,
                                nonlinear_type=nonlinear_type)  # 128 -> 256
            global_res = 256
        self.binary_up = UpSingle(in_channels=160,
                                 out_channels=80,
                                 res=global_res,
                                 kernel_size=1,
                                 use_nonlinearity=use_nonlinearity,
                                 nonlinear_type = nonlinear_type)
        self.segment_up = UpSingle(in_channels=160,
                                  out_channels=80,
                                  res=global_res,
                                  kernel_size=1,
                                  use_nonlinearity=use_nonlinearity,
                                  nonlinear_type = nonlinear_type)
        self.outc_b = OutConv(80, 2)
        self.outc_s = OutConv(160, n_classes)

    def forward(self, x16_out, x32_out, x64_out):

        x = self.up1(x16_out,x32_out)  # 1,640,32,32 -> 640*32
        x = self.up2(x, x64_out)    # 1,320,64,64
        x3_out = self.up3(x)        # 1,160,128,128
        x_in = x3_out
        if self.mask_res == 256 :
            x4_out = self.up4(x3_out)
            x_in = x4_out
        x_out_b = self.binary_up(x_in)
        binary_logits = self.outc_b(x_out_b)  # 1,2,256,256
        x_out_s = self.segment_up(x_in)
        seg_in = torch.cat([x_out_b, x_out_s], dim=1)
        segment_logits = self.outc_s(seg_in)
        return binary_logits, segment_logits


class Segmentation_Head_b(nn.Module):

    def __init__(self,
                 n_classes,
                 mask_res = 128,
                 norm_type='layer_norm',
                 use_nonlinearity=False,
                 nonlinear_type='relu'):
        super(Segmentation_Head_b, self).__init__()

        self.n_classes = n_classes
        self.mask_res = mask_res
        self.up1 = Up(1280, 640, 640, norm_type, use_nonlinearity)
        self.up2 = Up(640, 320, 320, norm_type, use_nonlinearity)
        self.up3 = Up(640, 320, 320, norm_type, use_nonlinearity)
        self.up4 = UpSingle(in_channels = 320,
                             out_channels=160,
                             kernel_size=2,
                             res = 128,
                             use_nonlinearity = use_nonlinearity,
                             nonlinear_type = nonlinear_type)
        if self.mask_res == 256 :
            self.up5 = UpSingle(in_channels = 160,
                                out_channels = 160,
                                kernel_size=2,
                                res = 256,
                                use_nonlinearity = use_nonlinearity,
                                nonlinear_type = nonlinear_type)
        self.outc = OutConv(160, n_classes)

    def forward(self, x16_out, x32_out, x64_out):

        x1_out = self.up1(x16_out, x32_out)  # 1,640,32,32
        x2_out = self.up2(x32_out, x64_out)  # 1,320,64,64
        x3_out = self.up3(x1_out, x2_out)    # 1,320,64,64
        x4_out = self.up4(x3_out)            # 1,160, 256,256
        x_in = x4_out
        if self.mask_res == 256 :
            x5_out = self.up5(x4_out)        # 1,160,256,256
            x_in = x5_out
        logits = self.outc(x_in)  # 1,3,256,256
        return logits

class Segmentation_Head_b_with_binary(nn.Module):

    def __init__(self,
                 n_classes,
                 mask_res = 128,
                 norm_type='layer_norm',
                 use_nonlinearity=False,
                 nonlinear_type='relu'):
        super(Segmentation_Head_b_with_binary, self).__init__()

        self.n_classes = n_classes
        self.mask_res = mask_res
        self.up1 = Up(1280, 640, 640, norm_type, use_nonlinearity)
        self.up2 = Up(640, 320, 320, norm_type, use_nonlinearity)
        self.up3 = Up(640, 320, 320, norm_type, use_nonlinearity)
        self.up4 = UpSingle(in_channels=320,
                            out_channels=160,
                            kernel_size=2,
                            res=128,
                            use_nonlinearity=use_nonlinearity,
                            nonlinear_type=nonlinear_type)
        global_res = 128
        if self.mask_res == 256:
            self.up5 = UpSingle(in_channels=160,
                                out_channels=160,
                                kernel_size=2,
                                res=256,
                                use_nonlinearity=use_nonlinearity,
                                nonlinear_type=nonlinear_type)
            global_res = 256
        self.binary_up = UpSingle(in_channels=160,
                                 out_channels=160,
                                 kernel_size=1,
                                 res = global_res,
                                 use_nonlinearity = use_nonlinearity,
                                  nonlinear_type = nonlinear_type)
        self.segment_up = UpSingle(in_channels=160,
                                  out_channels=160,
                                  kernel_size=1,
                                  res = global_res,
                                  use_nonlinearity = use_nonlinearity,
                                   nonlinear_type = nonlinear_type)
        self.outc_b = OutConv(160, 2)
        self.outc_s = OutConv(320, n_classes)

    def forward(self, x16_out, x32_out, x64_out):

        x1_out = self.up1(x16_out, x32_out)  # 1,640,32,32
        x2_out = self.up2(x32_out, x64_out)  # 1,320,64,64
        x3_out = self.up3(x1_out, x2_out)    # 1,320,64,64
        x4_out = self.up4(x3_out)            # 1,160, 256,256
        x_in = x4_out
        if self.mask_res == 256 :
            x5_out = self.up5(x4_out)        # 1,160,256,256
            x_in = x5_out
        x_out_b = self.binary_up(x_in)
        binary_logits = self.outc_b(x_out_b)  # 1,2,256,256
        x_out_s = self.segment_up(x_in)
        seg_in = torch.cat([x_out_b, x_out_s], dim=1)
        segment_logits = self.outc_s(seg_in)
        return binary_logits, segment_logits




class Segmentation_Head_c(nn.Module):

    def __init__(self,
                 n_classes,
                 mask_res=128,
                 norm_type='layer_norm',
                 use_nonlinearity=False,
                 nonlinear_type='relu'):
        super(Segmentation_Head_c, self).__init__()

        self.n_classes = n_classes
        self.mask_res = mask_res
        self.up1 = Up(1280, 640, 640, norm_type, use_nonlinearity)
        self.up2 = Up(640, 320, 320, norm_type, use_nonlinearity)
        self.up3 = Up(640, 320, 320, norm_type, use_nonlinearity)
        self.up4 = UpSingle(in_channels = 640,
                            out_channels = 320,
                            kernel_size=2,
                            res = 128,
                            use_nonlinearity = use_nonlinearity,
                            nonlinear_type = nonlinear_type)
        if self.mask_res == 256 :
            self.up5 = UpSingle(in_channels = 320,
                                out_channels = 320,
                                kernel_size=2,
                                res = 256,
                                use_nonlinearity = use_nonlinearity,
                                nonlinear_type = nonlinear_type)
        self.outc = OutConv(320, n_classes)

    def forward(self, x16_out, x32_out, x64_out):

        x1_out = self.up1(x16_out, x32_out)     # 1,640,32,32
        x2_out = self.up2(x32_out, x64_out)     # 1,320,64,64
        x3_out = self.up3(x1_out, x2_out)       # 1,320,64,64
        x = torch.cat([x3_out, x64_out], dim=1) # 1,640,64,64
        x4_out = self.up4(x)                    # 1,320,128,128
        x_in = x4_out
        if self.mask_res == 256 :
            x5_out = self.up5(x4_out)            # 1,320,256,256
            x_in = x5_out
        logits = self.outc(x_in)  # 1,3,256,256
        return logits

class Segmentation_Head_c_with_binary(nn.Module):

    def __init__(self,  n_classes, mask_res = 128, use_nonlinearity = False,
                    norm_type = 'layer_norm',
                 nonlinear_type = 'relu'):
        super(Segmentation_Head_c_with_binary, self).__init__()

        self.n_classes = n_classes
        self.mask_res = mask_res
        self.up1 = Up(1280, 640, 640, norm_type, use_nonlinearity)
        self.up2 = Up(640, 320, 320, norm_type, use_nonlinearity)
        self.up3 = Up(640, 320, 320, norm_type, use_nonlinearity)
        self.up4 = UpSingle(in_channels=640,
                            out_channels=320,
                            kernel_size=2,
                            res=128,
                            use_nonlinearity=use_nonlinearity,
                            nonlinear_type=nonlinear_type)
        global_res = 128
        if self.mask_res == 256:
            global_res = 256
            self.up5 = UpSingle(in_channels=320,
                                out_channels=320,
                                kernel_size=2,
                                res=256,
                                use_nonlinearity=use_nonlinearity,
                                nonlinear_type=nonlinear_type)
        self.binary_up = UpSingle(in_channels = 320,
                                out_channels = 160,
                                kernel_size=1,
                                 res = global_res,
                                 use_nonlinearity = use_nonlinearity,
                                  nonlinear_type = nonlinear_type)
        self.segment_up = UpSingle(in_channels = 320,
                                out_channels = 160,
                                kernel_size=1,
                                  res = global_res,
                                 use_nonlinearity = use_nonlinearity,
                                   nonlinear_type = nonlinear_type)
        self.outc_b = (OutConv(160, 2))
        self.outc_s = (OutConv(320, n_classes))

    def forward(self, x16_out, x32_out, x64_out):

        x1_out = self.up1(x16_out, x32_out)     # 1,640,32,32
        x2_out = self.up2(x32_out, x64_out)     # 1,320,64,64
        x3_out = self.up3(x1_out, x2_out)       # 1,320,64,64
        x = torch.cat([x3_out, x64_out], dim=1) # 1,640,64,64
        x4_out = self.up4(x)                    # 1,320,128,128
        x_in = x4_out
        if self.mask_res == 256 :
            x5_out = self.up5(x4_out)            # 1,320,256,256
            x_in = x5_out
        x_out_b = self.binary_up(x_in)
        binary_logits = self.outc_b(x_out_b)     # 1,2,256,256
        x_out_s = self.segment_up(x_in)
        seg_in = torch.cat([x_out_b, x_out_s], dim=1)
        segment_logits = self.outc_s(seg_in)
        return binary_logits, segment_logits