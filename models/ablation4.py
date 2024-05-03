import torch.nn as nn
import torch
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init

# code from https://github.com/PaddlePaddle/PaddleClas/blob/bf5f021b0e9d162e77af86b5ba1604ab122c321e/ppcls/arch/backbone/model_zoo/mixformer.py
# and https://github.com/Kakakakakakah/MU-Net

### Ablation study no channel interaction

def window_partition(x, window_size: int):
    """
    Args:   x: (B, H, W, C)
            window_size (int): window size(M)
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int,C):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def window_partition2(x, window_size):
    """ Split the feature map to windows.
    B, C, H, W --> B * H // win * W // win x win*win x C
    Args:
        x: (B, C, H, W)
        window_size (tuple[int]): window size
    Returns:
        windows: (num_windows*B, window_size * window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view([B, C, H // window_size[0], window_size[0],W // window_size[1], window_size[1]])
    windows = x.permute([0, 2, 4, 3, 5, 1]).contiguous().view([-1, window_size[0] * window_size[1], C])
    return windows  # [B*(H/WS)*(W/WS),WS*WS,C]


def window_reverse2(windows, window_size, H, W, C):
    """ Windows reverse to feature map.
    B * H // win * W // win x win*win x C --> B, C, H, W
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, C, H, W)
    """
    x = windows.view([-1, H // window_size[0], W // window_size[1],window_size[0], window_size[1], C])
    x = x.permute([0, 5, 1, 3, 2, 4]).contiguous().view([-1, C, H, W])
    return x  # (B, C, H, W)


class MixingAttention(nn.Module):
    r""" Mixing Attention Module.
    Modified from Window based multi-head self attention (W-MSA) module
    with relative position bias.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        dwconv_kernel_size (int): The kernel size for dw-conv
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to
            query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale
            of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self,
                 dim,window_size,dwconv_kernel_size, num_heads,
                 qkv_bias=True,qk_scale=None,
                 attn_drop=0.,proj_drop=0.):
        super().__init__()
        self.dim = dim
        attn_dim = dim // 2
        self.window_size = window_size  # Wh, Ww
        self.dwconv_kernel_size = dwconv_kernel_size
        self.num_heads = num_heads
        head_dim = attn_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5


        #   define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]

        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        # prev proj layer
        self.proj_attn = nn.Linear(dim, dim // 2)
        self.proj_attn_norm = nn.LayerNorm(dim // 2)

        self.proj_cnn = nn.Linear(dim, dim)
        self.proj_cnn_norm = nn.LayerNorm(dim)

        # conv branch
        self.dwconv3x3 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=self.dwconv_kernel_size, padding=self.dwconv_kernel_size // 2, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.channel_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim // 2, kernel_size=1),
        )
        self.projection = nn.Conv2d(dim, dim // 2, kernel_size=1)
        self.conv_norm = nn.BatchNorm2d(dim // 2)

        # window-attention branch
        self.qkv = nn.Linear(dim // 2, dim // 2 * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1)
        )
        self.attn_norm = nn.LayerNorm(dim // 2)

        # final projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            H: the height of the feature map
            W: the width of the feature map
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww)
                or None
        """
        # B * H // win * W // win x win*win x C
        x_atten = self.proj_attn_norm(self.proj_attn(x))
        x_cnn = self.proj_cnn_norm(self.proj_cnn(x))
        # B * H // win * W // win x win*win x C --> B, C, H, W
        x_cnn = window_reverse2(x_cnn, self.window_size, H, W, x_cnn.shape[-1])

        # conv branch
        x_cnn = self.dwconv3x3(x_cnn)
        channel_interaction = self.channel_interaction(F.adaptive_avg_pool2d(x_cnn, output_size=1))
        x_cnn = self.projection(x_cnn)

        # attention branch
        B_, N, C = x_atten.shape
        qkv = self.qkv(x_atten).reshape(
            [B_, N, 3, self.num_heads, C // self.num_heads]).permute([2, 0, 3, 1, 4])
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # channel interaction
        # x_cnn2v = torch.sigmoid(channel_interaction).reshape([-1, 1, self.num_heads, 1, C // self.num_heads])
        # v = v.reshape([x_cnn2v.shape[0], -1, self.num_heads, N, C // self.num_heads])
        # v = v * x_cnn2v
        # v = v.reshape([-1, self.num_heads, N, C // self.num_heads])

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape([B_ // nW, nW, self.num_heads, N, N]) + \
                   mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape([-1, self.num_heads, N, N])
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x_atten = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        # spatial interaction
        x_spatial = window_reverse2(x_atten, self.window_size, H, W, C)
        spatial_interaction = self.spatial_interaction(x_spatial)
        x_cnn = torch.sigmoid(spatial_interaction) * x_cnn
        x_cnn = self.conv_norm(x_cnn)
        # B, C, H, W --> B * H // win * W // win x win*win x C
        x_cnn = window_partition2(x_cnn, self.window_size)

        # concat
        x_atten = self.attn_norm(x_atten)
        x = torch.concat([x_atten, x_cnn], dim=-1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        #   out_features=in_features
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MixingBlock(nn.Module):
    r""" Mixing Block in MixFormer.
    Modified from Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        dwconv_kernel_size (int): kernel size for depth-wise convolution.
        shift_size (int): Shift size for SW-MSA.
            We do not use shift in MixFormer. Default: 0
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to
            query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Layer, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Layer, optional): Normalization layer.
            Default: nn.LayerNorm
    """

    def __init__(self, dim,num_heads,window_size=7,dwconv_kernel_size=3,shift_size=0,
                 mlp_ratio=4.,qkv_bias=True,qk_scale=None,
                 drop=0.,attn_drop=0.,drop_path=0.,act_layer=nn.GELU,norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert self.shift_size == 0, "No shift in MixFormer"

        self.norm1 = norm_layer(dim)
        self.attn = MixingAttention(
            dim,window_size=to_2tuple(self.window_size),
            dwconv_kernel_size=dwconv_kernel_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop)
        self.H = None
        self.W = None
    def forward(self, x, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, [0, pad_l, 0, pad_b, 0, pad_r, 0, pad_t])
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), axis=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)               # nW*B, window_size, window_size, C
        x_windows = x_windows.view([-1, self.window_size * self.window_size,C])  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        # nW*B, window_size*window_size, C
        attn_windows = self.attn(x_windows, Hp, Wp, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view([-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(attn_windows, self.window_size, Hp,Wp, C)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x,shifts=(self.shift_size, self.shift_size),axis=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :]

        x = x.reshape([B, H * W, C])
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# reshape -> view; transpose -> permute
class ConvMerging(nn.Module):
    r""" Conv Merging Layer.

    Args:
        dim (int): Number of input channels.
        out_dim (int): Output channels after the merging layer.
        norm_layer (nn.Module, optional): Normalization layer.
            Default: nn.LayerNorm
    """

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.reduction = nn.Conv2d(dim, out_dim, kernel_size=2, stride=2)
        self.norm = nn.BatchNorm2d(dim)# changed from: nn.BatchNorm(dim)

    def forward(self, x, H, W):
        """
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view([B, H, W, C]).permute([0, 3, 1, 2])

        x = self.norm(x)
        # B, C, H, W -> B, H*W, C
        x = self.reduction(x).flatten(2).permute([0, 2, 1])
        return x


class BasicLayer(nn.Module):
    """ A basic layer for one stage in MixFormer.
    Modified from Swin Transformer BasicLayer.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        dwconv_kernel_size (int): kernel size for depth-wise convolution.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to
            query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate.
            Default: 0.0
        norm_layer (nn.Layer, optional): Normalization layer.
            Default: nn.LayerNorm
        downsample (nn.Layer | None, optional): Downsample layer at the end
            of the layer. Default: None
        out_dim (int): Output channels for the downsample layer. Default: 0.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 dwconv_kernel_size=3,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 out_dim=0):
        super().__init__()
        self.window_size = window_size
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([ # changed from nn.LayerList
            MixingBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                dwconv_kernel_size=dwconv_kernel_size,
                shift_size=0,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i]
                if isinstance(drop_path, (np.ndarray, list)) else drop_path,
                norm_layer=norm_layer) for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, None)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return H, W, x_down, Wh, Ww
        else:
            return H, W, x, H, W


class ConvEmbed(nn.Module):
    r""" Image to Conv Stem Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels.
            Default: 96.
        norm_layer (nn.Module, optional): Normalization layer.
            Default: None
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3,
                      stride=patch_size[0] // 2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
        )
        self.proj = nn.Conv2d(embed_dim // 2, embed_dim,
                              kernel_size=patch_size[0] // 2,
                              stride=patch_size[0] // 2)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        if W % self.patch_size[1] != 0:
            x = F.pad(
                x, [0, self.patch_size[1] - W % self.patch_size[1], 0, 0])
        if H % self.patch_size[0] != 0:
            x = F.pad(
                x, [0, 0, 0, self.patch_size[0] - H % self.patch_size[0]])

        x = self.stem(x)
        x = self.proj(x)
        if self.norm is not None:
            _, _, Wh, Ww = x.shape
        x = x.flatten(2).permute([0, 2, 1])  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        x = x.permute([0, 2, 1])
        x = x.view([-1, self.embed_dim, Wh, Ww])
        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input
    

class MixFormer(nn.Module):
    """ A Pytorch impl impl of MixFormer based on PaddlePaddle code:
    MixFormer: Mixing Features across Windows and Dimensions (CVPR 2022, Oral)

    Modified from Swin Transformer.

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head.
            Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        dwconv_kernel_size (int): kernel size for depth-wise convolution.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
            Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
            Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Layer): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the
            patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding.
            Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory.
            Default: False
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 class_num=1000,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 dwconv_kernel_size=3,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 **kwargs):
        super(MixFormer, self).__init__()
        self.num_classes = num_classes = class_num
        self.num_layers = len(depths)
        if isinstance(embed_dim, int):
            embed_dim = [embed_dim * 2 ** i_layer
                         for i_layer in range(self.num_layers)]
        assert isinstance(embed_dim, list) and \
            len(embed_dim) == self.num_layers
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(self.embed_dim[-1])
        self.mlp_ratio = mlp_ratio

        # split image into patches
        self.patch_embed = ConvEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim[0],
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim[0]))
            trunc_normal_(self.absolute_pos_embed)

        # if self.ape:
        #     self.absolute_pos_embed = self.create_parameter(
        #         shape=(1, num_patches, self.embed_dim[0]),
        #         default_initializer=zeros_)
        #     self.add_parameter(
        #         "absolute_pos_embed", self.absolute_pos_embed)
        #     trunc_normal_(self.absolute_pos_embed)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate,
                          sum(depths)).tolist()

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(self.embed_dim[i_layer]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                dwconv_kernel_size=dwconv_kernel_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=ConvMerging
                if (i_layer < self.num_layers - 1) else None,
                out_dim=int(self.embed_dim[i_layer + 1])
                if (i_layer < self.num_layers - 1) else 0)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.last_proj = nn.Linear(self.num_features, 1280)
        self.activate = nn.GELU()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(
            1280,
            num_classes) if self.num_classes > 0 else Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                init.constant_(m.bias, 0.) # changed from zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.bias, 0.) # zeros_(m.bias)
            init.constant_(m.weight, 1.) # ones_(m.weight)   

    def forward_features(self, x):
        x = self.patch_embed(x)
        _, _, Wh, Ww = x.shape
        x = x.flatten(2).permute([0, 2, 1])
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            H, W, x, Wh, Ww = layer(x, Wh, Ww)

        x = self.norm(x)  # B L C
        x = self.last_proj(x)
        x = self.activate(x)
        x = self.avgpool(x.permute([0, 2, 1]))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def MixFormer_B0_AB4(**kwargs):
    model = MixFormer(
        embed_dim=24,
        depths=[1, 2, 6, 6],
        num_heads=[3, 6, 12, 24],
        drop_path_rate=0.,
        **kwargs)
    return model


def MixFormer_B1_AB4(**kwargs):
    model = MixFormer(
        embed_dim=32,
        depths=[1, 2, 6, 6],
        num_heads=[2, 4, 8, 16],
        drop_path_rate=0.,
        **kwargs)
    return model


def MixFormer_B2_AB4(**kwargs):
    model = MixFormer(
        embed_dim=32,
        depths=[2, 2, 8, 8],
        num_heads=[2, 4, 8, 16],
        drop_path_rate=0.05,
        **kwargs)
    return model


def MixFormer_B3_AB4(**kwargs):
    model = MixFormer(
        embed_dim=48,
        depths=[2, 2, 8, 6],
        num_heads=[3, 6, 12, 24],
        drop_path_rate=0.1,
        **kwargs)
    return model


def MixFormer_B4_AB4(**kwargs):
    model = MixFormer(
        embed_dim=64,
        depths=[2, 2, 8, 8],
        num_heads=[4, 8, 16, 32],
        drop_path_rate=0.2,
        **kwargs)
    return model


def MixFormer_B5_AB4(**kwargs):
    model = MixFormer(
        embed_dim=96,
        depths=[1, 2, 8, 6],
        num_heads=[6, 12, 24, 48],
        drop_path_rate=0.3,
        **kwargs)
    return model


def MixFormer_B6_AB4(**kwargs):
    model = MixFormer(
        embed_dim=96,
        depths=[2, 4, 16, 12],
        num_heads=[6, 12, 24, 48],
        drop_path_rate=0.5,
        **kwargs)
    return model