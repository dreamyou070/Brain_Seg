# https://github.com/milesial/Pytorch-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import trunc_normal_, DropPath
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper

#x16_out = torch.randn(1, 1280, 16, 16)
#x32_out = torch.randn(1, 640, 32, 32)
#x64_out = torch.randn(1, 320, 64, 64)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, norm_type = 'layernorm', nonlinear_type = 'relu'):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.norm_type =norm_type
        self.nonlinear_type = nonlinear_type
        if self.nonlinear_type == 'relu':
            nonlinearity = nn.ReLU(inplace=True)
        elif self.nonlinear_type == 'leaky_relu':
            nonlinearity = nn.LeakyReLU(inplace=True)
        elif self.nonlinear_type == 'silu':
            nonlinearity = nn.SiLU(inplace=True)

        if norm_type == 'batch_norm':
            self.double_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                                             nn.BatchNorm2d(mid_channels),
                                             nonlinearity,
                                             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                             nn.BatchNorm2d(out_channels),
                                             nonlinearity)
        elif norm_type == 'layer_norm':
            self.double_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                                             nn.LayerNorm([mid_channels, int(20480/mid_channels),int(20480/mid_channels)]),
                                             nonlinearity,
                                             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1,bias=False),
                                             nn.LayerNorm([mid_channels, int(20480 / mid_channels), int(20480 / mid_channels)]),
                                             nonlinearity)
        elif norm_type == 'instance_norm':
            self.double_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                                             nn.InstanceNorm2d(mid_channels),
                                             nonlinearity,
                                             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                             nn.InstanceNorm2d(out_channels),
                                             nonlinearity)
    def forward(self, x):
        return self.double_conv(x)

class DoubleConv_res(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, use_batchnorm = True, res=16, nonlinear_type = 'relu'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.nonlinear_type = nonlinear_type
        if self.nonlinear_type == 'relu':
            nonlinearity = nn.ReLU(inplace=True)
        elif self.nonlinear_type == 'leaky_relu':
            nonlinearity = nn.LeakyReLU(inplace=True)
        elif self.nonlinear_type == 'silu':
            nonlinearity = nn.SiLU(inplace=True)

        if use_batchnorm :
            self.double_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                                             nn.BatchNorm2d(mid_channels),
                                             nonlinearity,
                                             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                             nn.BatchNorm2d(out_channels),
                                             nonlinearity)
        else :
            self.double_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                                             nn.LayerNorm([mid_channels, res,res]),
                                             nonlinearity,
                                             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1,bias=False),
                                             nn.LayerNorm([mid_channels, res,res]),
                                             nonlinearity)
    def forward(self, x):
        return self.double_conv(x)

class Up_special(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, use_batchnorm = True, res=16, nonlinear_type = 'relu'):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv_res(int(in_channels+in_channels // 2),
                                       out_channels,
                                       use_batchnorm = use_batchnorm,
                                       res=res,
                                       nonlinear_type=nonlinear_type)

    def forward(self, x1, x2):

        # [1] x1
        x1 = self.up(x1) # 1,640, 16,16
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # [2] concat
        x = torch.cat([x2, x1], dim=1) # concatenation 1,1920,

        # [3] out conv
        x = self.conv(x)
        return x

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, norm_type = 'layer_norm', nonlinear_type = 'relu'):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, norm_type = norm_type, nonlinear_type = nonlinear_type)

    def forward(self, x1, x2):

        # [1] x1
        x1 = self.up(x1)
        print(f'x1.shape = {x1.shape}')
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

    def __init__(self, in_channels, out_channels, kernel_size=2, res=128, use_nonlinearity = False, linear_type = 'relu'):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = nn.ConvTranspose2d(in_channels = in_channels,
                                     out_channels = out_channels,
                                     kernel_size=kernel_size,
                                     stride=kernel_size)

        self.use_nonlinearity = use_nonlinearity
        if self.use_nonlinearity :
            self.layernorm = nn.LayerNorm([out_channels,res, res])
            if linear_type == 'relu':
                self.nonlinear = nn.ReLU(inplace=True)
            elif linear_type == 'leaky_relu':
                self.nonlinear = nn.LeakyReLU(inplace=True)
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


class Segmentation_Head_a_with_binary(nn.Module):

    def __init__(self,
                 n_classes,
                 bilinear=False,
                 use_batchnorm=True,
                 mask_res = 128,
                 use_nonlinearity = False,
                 nonlinearity_type = 'relu'):
        super(Segmentation_Head_a_with_binary, self).__init__()

        self.n_classes = n_classes
        self.mask_res = mask_res
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.up1 = Up(1280, 640 // factor, bilinear, use_batchnorm, nonlinearity_type)
        self.up2 = Up(640, 320 // factor, bilinear, use_batchnorm, nonlinearity_type)
        self.up3 = Up_conv(in_channels = 320,
                           out_channels = 160,
                           res = 128,
                           kernel_size=2,
                           use_nonlinearity = use_nonlinearity,
                           nonlinear_type = nonlinearity_type)
        global_res = 128
        if self.mask_res == 256 :
            self.up4 = Up_conv(in_channels = 160,
                               out_channels = 160,
                               res = 256,
                               kernel_size=2,
                               use_nonlinearity = use_nonlinearity,
                               nonlinear_type = nonlinearity_type)
            global_res = 256
        self.binary_up = Up_conv(in_channels=160,
                                 out_channels=80,
                                 res=global_res,
                                 kernel_size=1,
                                 use_nonlinearity=use_nonlinearity,
                                 nonlinear_type = nonlinearity_type)
        self.segment_up = Up_conv(in_channels=160,
                                  out_channels=80,
                                  res=global_res,
                                  kernel_size=1,
                                  use_nonlinearity=use_nonlinearity,
                                  nonlinear_type = nonlinearity_type)
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

    def __init__(self,  n_classes, bilinear=False, use_batchnorm=True, mask_res = 128, use_nonlinearity = False):
        super(Segmentation_Head_b, self).__init__()

        self.n_classes = n_classes
        self.mask_res = mask_res
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.up1 = (Up(1280, 640 // factor, bilinear, use_batchnorm))
        self.up2 = (Up(640, 320 // factor, bilinear, use_batchnorm))
        self.up3 = (Up(640, 320 // factor, bilinear, use_batchnorm))
        self.up4 = Up_conv(in_channels = 320,
                           out_channels=160,
                           kernel_size=2,
                           res = 128,
                           use_nonlinearity = use_nonlinearity)
        if self.mask_res == 256 :
            self.up5 = Up_conv(in_channels = 160,
                               out_channels = 160,
                               kernel_size=2,
                               res = 256,
                               use_nonlinearity = use_nonlinearity)
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

    def __init__(self,  n_classes, bilinear=False, use_batchnorm=True, mask_res = 128, use_nonlinearity = False):
        super(Segmentation_Head_b_with_binary, self).__init__()

        self.n_classes = n_classes
        self.mask_res = mask_res
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.up1 = (Up(1280, 640 // factor, bilinear, use_batchnorm))
        self.up2 = (Up(640, 320 // factor, bilinear, use_batchnorm))
        self.up3 = (Up(640, 320 // factor, bilinear, use_batchnorm))
        self.up4 = Up_conv(in_channels = 320,
                            out_channels=160,
                            kernel_size=2,
                            res = 128,
                            use_nonlinearity = use_nonlinearity)
        global_res = 128
        if self.mask_res == 256 :
            self.up5 = Up_conv(in_channels = 160,
                                out_channels = 160,
                                kernel_size=2,
                                res = 256,
                                use_nonlinearity = use_nonlinearity)
            global_res = 256
        self.binary_up = Up_conv(in_channels=160,
                                 out_channels=160,
                                 kernel_size=1,
                                 res = global_res,
                                 use_nonlinearity = use_nonlinearity)
        self.segment_up = Up_conv(in_channels=160,
                                  out_channels=160,
                                  kernel_size=1,
                                  res = global_res,
                                  use_nonlinearity = use_nonlinearity)
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

    def __init__(self,  n_classes, bilinear=False, use_batchnorm=True, mask_res = 128, use_nonlinearity = False):
        super(Segmentation_Head_c, self).__init__()

        self.n_classes = n_classes
        self.mask_res = mask_res
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.up1 = Up(1280, 640 // factor, bilinear, use_batchnorm)
        self.up2 = Up(640, 320 // factor, bilinear, use_batchnorm)
        self.up3 = Up(640, 320 // factor, bilinear, use_batchnorm)
        self.up4 = Up_conv(in_channels = 640,
                            out_channels = 320,
                            kernel_size=2,
                            res = 128,
                            use_nonlinearity = use_nonlinearity)
        if self.mask_res == 256 :
            self.up5 = Up_conv(in_channels = 320,
                                out_channels = 320,
                                kernel_size=2,
                                res = 256,
                                use_nonlinearity = use_nonlinearity)
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

    def __init__(self,  n_classes, bilinear=False, use_batchnorm=True, mask_res = 128, use_nonlinearity = False):
        super(Segmentation_Head_c_with_binary, self).__init__()

        self.n_classes = n_classes
        self.mask_res = mask_res
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.up1 = Up(1280, 640 // factor, bilinear, use_batchnorm)
        self.up2 = Up(640, 320 // factor, bilinear, use_batchnorm)
        self.up3 = Up(640, 320 // factor, bilinear, use_batchnorm)
        self.up4 = Up_conv(in_channels = 640,
                            out_channels = 320,
                            kernel_size=2,
                            res = 128,
                            use_nonlinearity = use_nonlinearity)
        global_res = 128
        if self.mask_res == 256 :
            self.up5 = Up_conv(in_channels = 320,
                               out_channels = 320,
                               kernel_size=2,
                               res = 256,
                               use_nonlinearity = use_nonlinearity) # 1,320,256,256
            global_res = 256
        self.binary_up = Up_conv(in_channels = 320,
                                out_channels = 160,
                                kernel_size=1,
                                 res = global_res,
                                 use_nonlinearity = use_nonlinearity)
        self.segment_up = Up_conv(in_channels = 320,
                                out_channels = 160,
                                kernel_size=1,
                                  res = global_res,
                                 use_nonlinearity = use_nonlinearity)
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


# --------------------------------------------------------------------------------------------------------------------- #
class Segmentation_Head_a(nn.Module):

    def __init__(self,
                 n_classes,
                 bilinear=False,
                 mask_res = 128,
                 norm_type = 'layer_norm',
                 use_nonlinearity = False):
        super(Segmentation_Head_a, self).__init__()

        self.n_classes = n_classes
        self.mask_res = mask_res
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.up1 = Up(1280,
                      640 // factor,
                      bilinear,
                      norm_type)
        self.up2 = Up(640,
                      320 // factor,
                      bilinear,
                      norm_type)
        self.up3 = Up_conv(in_channels = 320,
                           out_channels = 160,
                           kernel_size=2,
                           res = 128,
                           use_nonlinearity = use_nonlinearity) # 64 -> 128 , channel 320 -> 160
        if self.mask_res == 256 :
            self.up4 = Up_conv(in_channels = 160,
                               out_channels = 160,
                               kernel_size=2,
                               res=256,
                               use_nonlinearity = use_nonlinearity)  # 128 -> 256
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


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, register_hook=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_grad_checkpointing=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention( dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if use_grad_checkpointing:
            self.attn = checkpoint_wrapper(self.attn)
            self.mlp = checkpoint_wrapper(self.mlp)

    def forward(self, x, register_hook=False):
        x = x + self.drop_path(self.attn(self.norm1(x), register_hook=register_hook))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):

    def __init__(self,
                 embed_dim=960,
                 depth=3,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None,
                 use_grad_checkpointing=False, ckpt_layer=0,
                 use_nonlinearity = False,
                 mask_res = 128,
                 n_classes=4):

        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.up11 = Up_conv(in_channels=1280,
                            out_channels=640,
                            kernel_size=2,
                            res=16,
                            use_nonlinearity=False,
                            linear_type='relu')
        self.up12 = Up_conv(in_channels=640,
                            out_channels=320,
                            kernel_size=2,
                            res=32,
                            use_nonlinearity=False,
                            linear_type='relu')

        self.up21 = Up_conv(in_channels=640,
                            out_channels=320,
                            kernel_size=2,
                            res=32,
                            use_nonlinearity=False,
                            linear_type='relu')

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # droup out rate
        self.blocks = nn.ModuleList([Block(dim=embed_dim,
                                           num_heads=num_heads,
                                           mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                                           use_grad_checkpointing=(use_grad_checkpointing and i >= depth - ckpt_layer)) for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.up_128 = Up_conv(in_channels = 960,
                              out_channels = 160,
                              kernel_size=2,
                              res = 128,
                              use_nonlinearity = use_nonlinearity) # 64 -> 128 , channel 320 -> 160
        self.mask_res = mask_res
        if self.mask_res == 256:
            self.up_256 = Up_conv(in_channels=160,
                                  out_channels=160,
                                  kernel_size=2,
                                  res=256,
                                  use_nonlinearity=use_nonlinearity)  # 128 -> 256
        self.n_classes = n_classes
        self.outc = OutConv(160, n_classes)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x16_out, x32_out, x64_out, register_blk=-1):

        x16_out = self.up12(self.up11(x16_out)) # 1, 320, 64,64
        x32_out = self.up21(x32_out)
        x = torch.cat([x16_out, x32_out, x64_out], dim=1) # 1,960,64,64
        B, C, H, W = x.shape
        x = x.view(B, H*W, C) # 1,960,64*64
        for i, blk in enumerate(self.blocks):
            x = blk(x, register_blk == i)
        x = self.norm(x)
        # B, (H*W) , C -> B, H, W, C
        x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[2], int(x.shape[1] ** 0.5), int(x.shape[1] ** 0.5))

        x = self.up_128(x)  # 1,160,128,128
        if self.mask_res == 256:
            x = self.up_256(x)
        logits = self.outc(x)  # 1,4, 128,128
        return logits
