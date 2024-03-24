# https://github.com/milesial/Pytorch-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F

#x16_out = torch.randn(1, 1280, 16, 16)
#x32_out = torch.randn(1, 640, 32, 32)
#x64_out = torch.randn(1, 320, 64, 64)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, norm_type = 'layernorm'):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.norm_type =norm_type

        if norm_type == 'batch_norm':
            self.double_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                                             nn.BatchNorm2d(mid_channels),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                             nn.BatchNorm2d(out_channels),
                                             nn.ReLU(inplace=True))
        elif norm_type == 'layer_norm':
            self.double_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                                             nn.LayerNorm([mid_channels, int(20480/mid_channels),int(20480/mid_channels)]),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1,bias=False),
                                             nn.LayerNorm([mid_channels, int(20480 / mid_channels), int(20480 / mid_channels)]),
                                             nn.ReLU(inplace=True))
        elif norm_type == 'instance_norm':
            self.double_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                                             nn.InstanceNorm2d(mid_channels),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                             nn.InstanceNorm2d(out_channels),
                                             nn.ReLU(inplace=True))
    def forward(self, x):
        return self.double_conv(x)

class DoubleConv_res(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, use_batchnorm = True, res=16):
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
            self.double_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                                             nn.LayerNorm([mid_channels, res,res]),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1,bias=False),
                                             nn.LayerNorm([mid_channels, res,res]),
                                             nn.ReLU(inplace=True))
    def forward(self, x):
        return self.double_conv(x)

class Up_special(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, use_batchnorm = True, res=16 ):
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
                                       res=res)

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
    def __init__(self, in_channels, out_channels, bilinear=True, norm_type = 'layer_norm' ):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, norm_type = norm_type)

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

    def __init__(self, in_channels, out_channels, kernel_size=2, res=128, use_nonlinearity = False):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = nn.ConvTranspose2d(in_channels = in_channels,
                                     out_channels = out_channels,
                                     kernel_size=kernel_size, stride=kernel_size)

        self.use_nonlinearity = use_nonlinearity
        if self.use_nonlinearity :
            self.layernorm = nn.LayerNorm([out_channels,res, res])
            self.nonlinear = nn.ReLU(inplace=True)
        # Layernorm 와 Relu 는 왜 쓰는 걸까?
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
                 use_nonlinearity = False
                 ):
        super(Segmentation_Head_a_with_binary, self).__init__()

        self.n_classes = n_classes
        self.mask_res = mask_res
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.up1 = (Up(1280, 640 // factor, bilinear, use_batchnorm))
        self.up2 = (Up(640, 320 // factor, bilinear, use_batchnorm))
        self.up3 = Up_conv(in_channels = 320,
                           out_channels = 160,
                           res = 128,
                           kernel_size=2,
                           use_nonlinearity = use_nonlinearity)
        global_res = 128
        if self.mask_res == 256 :
            self.up4 = Up_conv(in_channels = 160,
                               out_channels = 160,
                               res = 256,
                               kernel_size=2,
                               use_nonlinearity = use_nonlinearity)
            global_res = 256
        self.binary_up = Up_conv(in_channels=160,
                                 out_channels=80,
                                 res=global_res,
                                 kernel_size=1,
                                 use_nonlinearity=use_nonlinearity)
        self.segment_up = Up_conv(in_channels=160,
                                  out_channels=80,
                                  res=global_res,
                                  kernel_size=1,
                                  use_nonlinearity=use_nonlinearity)
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
"""
model = Segmentation_Head_a(n_classes=3,
                            bilinear=False,
                            mask_res=128,
                            norm_type='instance_norm',
                            use_nonlinearity=False)
x16_out = torch.randn(1, 1280, 16, 16)
x32_out = torch.randn(1, 640, 32, 32)
x64_out = torch.randn(1, 320, 64, 64)
output = model(x16_out, x32_out, x64_out)
"""
class ViTPatchEmbeddings(nn.Module):

    def __init__(self, image_size, patch_size, num_channels, hidden_size):
        super().__init__()
        image_size, patch_size = image_size, patch_size
        num_channels, hidden_size = num_channels, hidden_size
        patch_size = (1,1)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.projection = nn.Conv2d(num_channels,
                                    hidden_size,
                                    kernel_size=patch_size, stride=patch_size)
    def forward(self,
                pixel_values: torch.Tensor) -> torch.Tensor:
        embeddings = self.projection(pixel_values)
        return embeddings

class Segmentation_Head_a_position_embedding(nn.Module):

    def __init__(self,
                 n_classes,
                 bilinear=False,
                 mask_res = 128,
                 norm_type = 'layer_norm',
                 use_nonlinearity = False):
        super(Segmentation_Head_a_position_embedding, self).__init__()

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

        self.patch_embeddings_16 = nn.Parameter(torch.randn(1, 16 * 16, 1280), requires_grad=True)
        self.patch_embeddings_32 = nn.Parameter(torch.randn(1, 32 * 32, 640), requires_grad=True)
        self.patch_embeddings_64 = nn.Parameter(torch.randn(1, 64 * 64, 320), requires_grad=True)

    def forward(self, x16_out, x32_out, x64_out):

        import einops

        # [1] x_16_out preparing
        x16 = einops.rearrange(x16_out, 'b c h w -> b (h w) c')  # B,H*W,C
        b_size = x16.shape[0]
        x16_pe = self.patch_embeddings_16.expand(b_size, -1, -1).to(x16.device)
        x16_out = einops.rearrange(((x16 + x16_pe).view(b_size, 16, 16, -1)), 'b h w c -> b c h w')

        # [2] x_32_out preparing
        x32 = einops.rearrange(x32_out, 'b c h w -> b (h w) c')  # B,H*W,C
        x32_pe = self.patch_embeddings_32.expand(b_size, -1, -1).to(x32.device)
        x32_out = einops.rearrange(((x32 + x32_pe).view(b_size, 32, 32, -1)), 'b h w c -> b c h w')

        # [3] x_64_out preparing
        x64 = einops.rearrange(x64_out, 'b c h w -> b (h w) c')  # B,H*W,C
        x64_pe = self.patch_embeddings_64.expand(b_size, -1, -1).to(x64.device)
        x64_out = einops.rearrange(((x64 + x64_pe).view(b_size, 64, 64, -1)), 'b h w c -> b c h w')

        x = self.up1(x16_out,x32_out)  # 1,640,32,32 -> 640*32
        x = self.up2(x, x64_out)    # 1,320,64,64
        x3_out = self.up3(x)        # 1,160,128,128
        x_in = x3_out
        if self.mask_res == 256 :
            x4_out = self.up4(x3_out)
            x_in = x4_out
        logits = self.outc(x_in)  # 1,4, 128,128
        return logits

"""
x16_out = torch.randn(1, 1280, 16, 16)
x32_out = torch.randn(1, 640, 32, 32)
x64_out = torch.randn(1, 320, 64, 64)
model = Segmentation_Head_a_position_embedding(n_classes=3,
                                                  bilinear=False,
                                                  mask_res=128,
                                                  norm_type='instance_norm',
                                                  use_nonlinearity=False)
model(x16_out, x32_out, x64_out)
"""