# from models.mixformer_pytorch import *
# from models.lambda_networks import LambdaLayer
# from torchinfo import summary
import torch.nn as nn
import torch
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from models.mixformer_pytorch import *

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

class LambdaMixingAttention(nn.Module):
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
                 E: nn.Parameter,
                 k: int,
                 qkv_bias=True,qk_scale=None,
                 attn_drop=0.,proj_drop=0.):
        super().__init__()
        attn_dim = dim // 2
        self.window_size = window_size  # Wh, Ww
        self.dwconv_kernel_size = dwconv_kernel_size
        self.num_heads = num_heads
        head_dim = attn_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.head_dim = head_dim

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

        # self.n = window_size[0] * window_size[1]
        self.d = dim // 2 # d (output dimension)
        self.m = window_size[0] * window_size[1] #* head_dim# m

        self.k = k

        self.v = dim // 2  # k

        # self.h = num_heads
        # self.h = head_dim  # h
        self.h = 1

        # Initialisation of the embedding matrices
        self.E = E # n-m-k

        # Linear mapping in the form of NN for the query, key and value computation.
        self.toqueries = nn.Linear(self.d, self.k * self.h, bias=False)
        self.tokeys = nn.Linear(self.d, self.k, bias=False)
        self.tovalues = nn.Linear(self.d, self.v, bias=False)

        # Create batch normalization layers
        self.bn_values = nn.BatchNorm1d(self.m)
        self.bn_queries = nn.BatchNorm2d(self.k)

        # Keys softmax function for the keys
        self.keys_softmax = nn.Softmax(dim=1)

        self.reset_params()


    def reset_params(self):
        """
        Initialize network parameters.
        """
        std_kv = 1/np.sqrt(self.d)  # standard deviation for the key and value
        std_q = 1/np.sqrt(self.d * self.k)  # standard deviation for the query
        torch.nn.init.normal_(self.toqueries.weight, mean=0.0, std=std_q)  # initialise of the query projection matrix
        torch.nn.init.normal_(self.tokeys.weight, mean=0.0, std=std_kv)  # initialise of the k projection matrix
        torch.nn.init.normal_(self.tovalues.weight, mean=0.0, std=std_kv)  # initialise of the v projection matrix

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

        # ######################################################################################

        context = torch.reshape(x_atten, [B_, self.m, C])

        k = self.tokeys(context)                   # b-m-k
        softmax_keys = self.keys_softmax(k)       # b-m-k

        v = self.tovalues(context) # b-m-v

        # <start> channel interaction
        x_cnn2v = torch.sigmoid(channel_interaction).reshape([-1, 1, self.num_heads, 1, C // self.num_heads])
        v = v.reshape([x_cnn2v.shape[0], -1, self.num_heads, N, C // self.num_heads])
        v = v * x_cnn2v
        v = v.reshape([-1, N, C])
        # <end> channel interaction

        v = self.bn_values(v)   

        q = torch.reshape(self.toqueries(x_atten), [B_, N, self.k, self.h])  # b-n-k-h
        q = q * self.scale
        q = torch.transpose(q, 1, 2)  # b-k-n-h
        q = self.bn_queries(q)  # b-k-n-h

        content_lambda = torch.einsum('bmk, bmv->bkv', softmax_keys, v)    # b-k-v
        position_lambdas = torch.einsum('nmk, bmv->bnkv', self.E, v)       # b-n-k-v
        content_output = torch.einsum('bknh, bkv->bhvn', q, content_lambda)   # b-h-v-n
        position_output = torch.einsum('bknh, bnkv->bhvn', q, position_lambdas)   # b-h-v-n

        attn = torch.reshape(content_output + position_output, [B_, N, C]) #update: b-n-d, not b-d-n

        # ######################################################################################

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape([B_ // nW, nW, self.num_heads, N, N]) + \
                   mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape([-1, self.num_heads, N, N])
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        x_atten = self.attn_drop(attn)

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



class LambdaMixingBlock(nn.Module):
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

    def __init__(self, dim,num_heads,E: nn.Parameter, k: int,window_size=7,dwconv_kernel_size=3,shift_size=0,
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
        self.attn = LambdaMixingAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            dwconv_kernel_size=dwconv_kernel_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            E = E,
            k = k
            )
        
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
                 E: nn.Parameter,
                 k: int,
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
            LambdaMixingBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                dwconv_kernel_size=dwconv_kernel_size,
                shift_size=0,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                E = E,
                k = k,
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


class LambdaMixFormer(nn.Module):
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
        super(LambdaMixFormer, self).__init__()
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

            dim = int(self.embed_dim[i_layer])
            # ###########################################################################
            m = window_size ** 2
            k = dim//24 # from 1 to 24 for images of size 32x32

            embedding = nn.Parameter(torch.Tensor(m, m, k), requires_grad=True)
            torch.nn.init.normal_(embedding, mean=0.0, std=1.0)
            # ###########################################################################

            layer = BasicLayer(
                dim=dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                dwconv_kernel_size=dwconv_kernel_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                E = embedding,
                k=k,
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
    

if __name__ == "__main__":
    hybrid_model = LambdaMixFormer(
        embed_dim=24,
        depths=[1, 2, 6, 6],
        num_heads=[3, 6, 12, 24],
        drop_path_rate=0.,
        class_num=10,
    )
    import torch
    # summary(hybrid_model, input_size=(1,3,32,32))
    img_size = 32
    # print(hybrid_model)

    optimizer = torch.optim.AdamW(hybrid_model.parameters(),lr = 1.)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    hybrid_model.to(device)

    for i in range(1):
        hybrid_model.train()

        inputs = torch.rand(10, 3,img_size,img_size)
        labels = torch.randint(0, 10, (10,))  # labels should be integers

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = hybrid_model(inputs)  # assuming your hybrid_model returns raw scores
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()