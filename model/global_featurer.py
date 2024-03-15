import torch
import torch.nn as nn

class SingleConv2d(nn.Module):
    def __init__(self,
                 channel,
                 inner_dim,):
        super().__init__()
        """
        self.conv_in = nn.Conv2d(in_channels = channel,
                                 out_channels = inner_dim,
                                 kernel_size = kernel_size,
                                 stride = 1)
        self.conv_out = nn.Conv2d(in_channels=inner_dim,
                                  out_channels=channel,
                                  kernel_size=kernel_size,
                                  stride = 1)
        """
        self.linear_in = nn.Linear(channel, inner_dim,)
        self.linear_out = nn.Linear(inner_dim, channel)
    def forward(self, x: torch.Tensor):

        b, pix_num, d = x.shape
        """
        res = int(pix_num ** 0.5)
        x2d = x.permute(0,2,1).reshape(b,d,res,res)
        out = self.conv_in(x2d)
        out = self.conv_out(out)
        """
        out = self.linear_out(self.linear_in(x))
        return out

class AllConv2d(nn.Module):

    """ position embedding added local feature """
    """ non position embedding added global feature """

    layer_names_res_dim = {'down_blocks_0_attentions_0_transformer_blocks_0_attn2': (64, 320),
                           'down_blocks_0_attentions_1_transformer_blocks_0_attn2': (64, 320),

                           'down_blocks_1_attentions_0_transformer_blocks_0_attn2': (32, 640),
                           'down_blocks_1_attentions_1_transformer_blocks_0_attn2': (32, 640),

                           'down_blocks_2_attentions_0_transformer_blocks_0_attn2': (16, 1280),
                           'down_blocks_2_attentions_1_transformer_blocks_0_attn2': (16, 1280),

                           'mid_block_attentions_0_transformer_blocks_0_attn2': (8, 1280),

                           'up_blocks_1_attentions_0_transformer_blocks_0_attn2': (16, 1280),
                           'up_blocks_1_attentions_1_transformer_blocks_0_attn2': (16, 1280),
                           'up_blocks_1_attentions_2_transformer_blocks_0_attn2': (16, 1280),

                           'up_blocks_2_attentions_0_transformer_blocks_0_attn2': (32, 640),
                           'up_blocks_2_attentions_1_transformer_blocks_0_attn2': (32, 640),
                           'up_blocks_2_attentions_2_transformer_blocks_0_attn2': (32, 640),

                           'up_blocks_3_attentions_0_transformer_blocks_0_attn2': (64, 320),
                           'up_blocks_3_attentions_1_transformer_blocks_0_attn2': (64, 320),
                           'up_blocks_3_attentions_2_transformer_blocks_0_attn2': (64, 320), }
    def __init__(self) -> None:
        super().__init__()

        self.global_featuring = {}
        for layer_name in self.layer_names_res_dim.keys() :
            res, dim = self.layer_names_res_dim[layer_name]
            self.global_featuring[layer_name] = {}
            self.global_featuring[layer_name] = SingleConv2d(channel = dim,
                                                             inner_dim = 40,)
                                                                 #kernel_size = res,
                                                                 #stride = 1)

    def forward(self, x: torch.Tensor, layer_name):
        device = x.device
        if layer_name in self.layer_names_res_dim.keys() :
            global_featurer = self.global_featuring[layer_name].to(device)
            global_feature = global_featurer(x)
            return global_feature
        else :
            return x

