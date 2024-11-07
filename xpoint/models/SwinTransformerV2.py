# --------------------------------------------------------
# Swin Transformer V2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np

import warnings
warnings.filterwarnings('ignore') # setting ignore as a parameter


class Mlp(nn.Module):
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


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


   

class WindowAttention(nn.Module):

    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, input_resolution,pretrained_img_size=None,qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        self.input_resolution = input_resolution
        self.pretrained_img_size = pretrained_img_size

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        relative_coords_table = self.get_relative_coords_table(window_size, pretrained_window_size)
        relative_position_index = self.get_relative_position_index(window_size)
        self.register_buffer("relative_coords_table", relative_coords_table)
        self.register_buffer("relative_position_index", relative_position_index)
        #print("relative_coords_table and relative_position_index registered!")
            

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def get_relative_coords_table(self,window_size, pretrained_window_size):
        relative_coords_h = torch.arange(-(window_size[0] - 1), window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(window_size[1] - 1), window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (window_size[1] - 1)
        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / np.log2(8)
        return relative_coords_table


    def get_relative_position_index(self,window_size):
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01).to(x.device))).exp() #to device is added by me !
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        pretrained_img_size (tuple[int]): Pretrained image size.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, dim, input_resolution,pretrained_img_size, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.pretrained_img_size = pretrained_img_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,input_resolution=self.input_resolution,pretrained_img_size=self.pretrained_img_size,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        #print("attn_mask registered!")
    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x): #HERE x_size
        H, W = self.input_resolution
        B, L, C = x.shape
        #assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size #HERE x_size!
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C


        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, halve=True ,norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim) #HERE 4*dim in swinv1 ??
        self.halve = halve

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        if self.halve:
            jump = 2
            x0 = x[:, 0::jump, 0::jump, :]  # B H/2 W/2 C
            x1 = x[:, 1::jump, 0::jump, :]  # B H/2 W/2 C
            x2 = x[:, 0::jump, 1::jump, :]  # B H/2 W/2 C
            x3 = x[:, 1::jump, 1::jump, :]  # B H/2 W/2 C
            x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        else:
            x = torch.cat([x, x, x, x], -1)  #
        
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += H * W * self.dim // 2
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        pretrained_img_size (tuple[int]): Pretrained image size.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(self, dim, input_resolution,pretrained_img_size, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0,halve=True):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.pretrained_img_size = pretrained_img_size
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,pretrained_img_size = self.pretrained_img_size,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer,halve=halve)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x,self.input_resolution)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
        downsample_ratio_beginning (int, optional): Downsample ratio at the beginning of the network. Default: 2
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None,downsample_ratio_beginning=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=downsample_ratio_beginning, stride=downsample_ratio_beginning) #HERE  kernel and stride are patch_size = 4, but lets try 2 since  i want division by 8 at the end.
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        #I FIXED YOU
        x = self.proj(x) #HERE self.proj is not in swinIR
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C 
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    # def forward(self, x):
    #     B, HW, C = x.shape
    #     #print(x.shape)
    #     x = x.transpose(1,2).view(B,self.embed_dim,  self.img_size[0], self.img_size[1])  # B Ph*Pw C
    #     #print(x.shape)
        
    #     return x
    def depth_to_space(self,x, block_size):
        N, C, H, W = x.size()
        x = x.view(N, block_size, block_size, C // (block_size ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (block_size ** 2), H * block_size, W * block_size)  # (N, C//bs^2, H * bs, W * bs)
        return x
    def forward(self, x):
        #print(x.shape,self.img_size)
        x = x.view(x.shape[0], self.img_size[0],self.img_size[1], -1).permute(0, 3, 1, 2) #//32
        #print(x.shape)
        x = self.depth_to_space(x, 4)
        #print(x.shape)
        return x

    def flops(self):
        flops = 0
        return flops

class SwinTransformerV2(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        input_img_size (int | tuple(int)): Input image size. Default 224 
        pretrained_img_size (int | tuple(int)): Pretrained image size. Default 224 #HERE ---- pretrained_img_size ::::))))
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        downsample_ratio_beginning (int, optional): Downsample ratio at the beginning of the network. Default: 2
    """

    def __init__(self, input_image_size =224,pretrained_img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], downsample_ratio_beginning=2 , **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size
        self.in_chans = in_chans
        self.pretrained_img_size = pretrained_img_size
        self.downsample_ratio_beginning = downsample_ratio_beginning
        self.overall_downsample_ratio = 2**(len(depths)-1)*self.downsample_ratio_beginning #2**(len(depths)//2)*self.downsample_ratio_beginning #HERE IMPORTANT LOOK
        self.input_image_size = input_image_size
        self.tuned_image_size = tuple(dim // self.overall_downsample_ratio * self.overall_downsample_ratio for dim in input_image_size) # IT IS ADDED FOR removing padding properly for every image size
        self.padded_input_image_size = np.array(self.check_image_size(torch.zeros(1,in_chans,*to_2tuple(input_image_size)),get_hw=True))

        print("Overall downsample ratio is ",self.overall_downsample_ratio," and encoded image shape will be :", np.array([*self.input_image_size]) ,"-->",np.array([*self.tuned_image_size]) ,"-->",np.array([*self.input_image_size])//self.overall_downsample_ratio)
        print("Padded input image size is ",np.array(self.input_image_size),"-->",np.array([*self.tuned_image_size]) ,"-->",self.padded_input_image_size)
        self.register_buff = to_2tuple(self.input_image_size) == to_2tuple(self.pretrained_img_size)
        print("Register buffer is ",self.register_buff)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=self.padded_input_image_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            downsample_ratio_beginning=self.downsample_ratio_beginning)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image #HERE PatchUnEmbed
        self.patch_unembed = PatchUnEmbed(
            img_size=self.padded_input_image_size // self.overall_downsample_ratio , patch_size=patch_size, in_chans=embed_dim, embed_dim=int(embed_dim * 2 ** (len(depths)-1)),
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        exponent = 0
        for i_layer in range(self.num_layers):
            if i_layer>0: #i_layer==1: #i_layer%2 != 0: #LOOK
                exponent += 1
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(self.padded_input_image_size[0] // self.downsample_ratio_beginning // (2 ** exponent),
                                                 self.padded_input_image_size[1] // self.downsample_ratio_beginning // (2 ** exponent)),
                               pretrained_img_size = self.pretrained_img_size,
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer],
                               halve = True if i_layer>=0 else False, #if (i_layer % 2 == 0) else False, LOOK
                               )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        #self.avgpool = nn.AdaptiveAvgPool1d(1)
        #self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

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
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}
    
    def check_image_size(self, x,get_hw=False): #HERE
        _, _, h, w = x.size()
        rate = self.overall_downsample_ratio*self.window_size//2 #LOOK
        mod_pad_h = int(np.ceil(h/rate)*rate-h) #(self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = int(np.ceil(w/rate)*rate-w) #(self.window_size - w % self.window_size) % self.window_size
        #print(rate,mod_pad_h,mod_pad_w)
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        if get_hw:
            return x.shape[2:]
        return x
    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3]) #HERE
        x = self.patch_embed(x)
        #print("after patch embed : ", x.shape)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        i=1
        for layer in self.layers:
            #print("before layer {} : {}".format(i,x.shape))
            x = layer(x) #HERE
            #print("after layer {} : {}".format(i,x.shape))
            i +=1

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x) #HERE
        #x = self.avgpool(x.transpose(1, 2))  # B C 1
        #x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        if(self.in_chans == 3 and x.shape[1] == 1):
            x = torch.cat((x,x,x),dim=1)
        B,C,H, W = x.shape #HERE
        assert C == self.in_chans, f"Input channel ({C}) should be equal to 'in_chans' : {self.in_chans}."

        #print("BEFORE check image size : ", x.shape)
        x = self.check_image_size(x)#HERE
        #print("AFTER check image size : ", x.shape)
        #print(self.input_image_size, self.pretrained_img_size)

        x = self.forward_features(x)
        #x = self.head(x) #HERE

        h,w = np.array(self.input_image_size) // self.overall_downsample_ratio*4 #4 comes from depth_to_space LOOK
        x = x[..., :h, :w] #dont forget each dimension is reduced by 32, 512 512 --> 16 16
        #print("YES, IT WORKS : ",x.shape)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


if __name__ == "__main__":

    # pretrained_input_shape = (224,224)
    # B,C,H,W = 1,3,512,640

    # input = torch.randn(B,C,H,W)
    # print("Input shape : ", input.shape)
    # print("Pretrained input shape : ", pretrained_input_shape)
    # model = SwinTransformer(    input_image_size=(H,W),
    #                             pretrained_img_size=pretrained_input_shape,
    #                             patch_size=4,
    #                             in_chans=3,
    #                             num_classes=1000,
    #                             embed_dim=96,
    #                             depths=[2, 2, 6, 2],
    #                             num_heads=[ 3, 6, 12, 24 ],
    #                             window_size=7,
    #                             #mlp_ratio=config.MODEL.SWINV2.MLP_RATIO,
    #                             #qkv_bias=config.MODEL.SWINV2.QKV_BIAS,
    #                             #drop_rate=config.MODEL.DROP_RATE,
    #                             drop_path_rate=0.2,
    #                             #ape=config.MODEL.SWINV2.APE,
    #                             #patch_norm=config.MODEL.SWINV2.PATCH_NORM,
    #                             #use_checkpoint=config.TRAIN.USE_CHECKPOINT,
    #                             downsample_ratio_beginning=4,
    #                             )
    
    # output = model(input)

    # tiny_p4_w7_224 = torch.load("swin_tiny_patch4_window7_224.pth",map_location=torch.device('cpu'))["model"]
    # for name,value in tiny_p4_w7_224.items():
    #     print(name,value.shape)
    # for name,value in model.state_dict().items():
    #     print("---",name,value.shape)
    # model.load_state_dict(tiny_p4_w7_224, strict=False)
    
    # print("Output shape : ", output.shape)




    pretrained_input_shape = (256,256)
    B,C,H,W = 1,3,512,640

    input = torch.randn(B,C,H,W)
    print("Input shape : ", input.shape)
    print("Pretrained input shape : ", pretrained_input_shape)
    model = SwinTransformerV2(    input_image_size=(H,W),
                                pretrained_img_size=pretrained_input_shape,
                                patch_size=4,
                                in_chans=3,
                                num_classes=1000,
                                embed_dim=128,
                                depths=[2, 2, 18, 2],
                                num_heads=[ 4, 8, 16, 32 ],
                                window_size=16,
                                #mlp_ratio=config.MODEL.SWINV2.MLP_RATIO,
                                #qkv_bias=config.MODEL.SWINV2.QKV_BIAS,
                                #drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=0.2,
                                #ape=config.MODEL.SWINV2.APE,
                                #patch_norm=config.MODEL.SWINV2.PATCH_NORM,
                                #use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                downsample_ratio_beginning=4,
                                pretrained_window_sizes=[12,12,12,6]
                                )
    
    output = model(input)
    #print("Input shape : ", input.shape)
    #print("Output shape : ", output.shape)

    base_p4_w12_384 = torch.load("swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth",map_location=torch.device('cpu'))["model"]
    # for name,value in base_p4_w12_384.items():
    #     print(name,value.shape)
    # for name,value in model.state_dict().items():
    #     print("---",name,value.shape)
    # model.load_state_dict(base_p4_w12_384, strict=False)
    
    print("Output shape : ", output.shape)


    #print("\n\n\n")

    # input2 = torch.randn(B,C,501,578)
    # model2 = SwinTransformer(input_image_size=(H,W),
    #                          pretrained_img_size=(128,128),
    #                             patch_size=4,
    #                             in_chans=3,
    #                             num_classes=1000,
    #                             embed_dim=96,
    #                             depths=[2, 2, 18, 2],
    #                             num_heads=[4,8,12,16],
    #                             window_size=16,
    #                             #mlp_ratio=config.MODEL.SWINV2.MLP_RATIO,
    #                             #qkv_bias=config.MODEL.SWINV2.QKV_BIAS,
    #                             #drop_rate=config.MODEL.DROP_RATE,
    #                             drop_path_rate=0.2,
    #                             #ape=config.MODEL.SWINV2.APE,
    #                             #patch_norm=config.MODEL.SWINV2.PATCH_NORM,
    #                             #use_checkpoint=config.TRAIN.USE_CHECKPOINT,
    #                             )
    
    # output = model2(input2)


    # params2 = {}
    # for name,value in model2.state_dict().items():
    #     params2[name] = value.shape
        

    # for key,value in params1.items():
    #     if key in params2.keys():
    #         if(params2[key] == value):
    #             #print("yess : ",key,value,params2[key])
    #             pass
    #         else:
                
    #             print("noo : ",key,value,params2[key])
    #     else:
    #         print(key, " is not in params2")