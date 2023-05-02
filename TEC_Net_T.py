"""
CiT-Net-Tiny
            stage1  stage2  stage3  stage4
    size    56x56   28x28   14x14   7x7
Unet dim    96      192     384     768
Swin dim    96      192     384     768
     head   3       6       12      24
     num    2       2       6       2
"""
import torch
import math
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from thop import *
from torch.nn import init
import torch.nn.functional as F
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from DDConv import DDConv
from einops.layers.torch import Rearrange


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.conv2patch = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=4, stride=4),
            nn.GELU(),
            # nn.ReLU(),
            nn.BatchNorm2d(embed_dim)
        )

    def forward(self, x):
        # # FIXME look at relaxing size constraints
        x = self.conv2patch(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

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

class oneXone_conv(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(oneXone_conv, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features)
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(out_features)
        )
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.Conv1(x)
        x = self.drop(x)
        x = self.Conv2(x)
        x = self.drop(x)
        return x

class GhostModule(nn.Module):
    def __init__(self, inp, oup=None, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        oup = oup or inp
        init_channels = math.ceil(oup // ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out

class GhostModule_Up(nn.Module):
    def __init__(self, inp, oup=None, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule_Up, self).__init__()
        oup = oup or inp
        init_channels = inp
        new_channels = init_channels

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class CAM_Module(Module):
    def __init__(self, in_dim, dim, num_heads, qk_scale=None, C_lambda=1e-4, attn_drop=0., proj_drop=0.):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.softmax = Softmax(dim=-1)
        self.c_lambda = C_lambda
        self.activaton = nn.Sigmoid()

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Sequential(nn.Conv2d(dim//12, dim, kernel_size=3, padding=1, stride=1, bias=False, groups=dim//12), nn.GELU())
        self.proj_drop = nn.Dropout(proj_drop)

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//12, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//12, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//12, kernel_size=1)

        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x, mask=None):
        m_batchsize, N, C = x.size()
        height = int(N ** .5)
        width = int(N ** .5)

        x = x.view(m_batchsize, C, height, width)
        proj_query = self.query_conv(x).view(m_batchsize, C//12, -1)
        proj_key = self.key_conv(x).view(m_batchsize, C//12, -1).permute(0, 2, 1)
        proj_value = self.value_conv(x).view(m_batchsize, C//12, -1)

        q = proj_query * self.scale
        attn = (q @ proj_key)

        if mask is not None:
            nW = mask.shape[0]  # num_windows
            attn = attn.view(m_batchsize // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ proj_value).reshape(m_batchsize, C//12, height, width)
        x = self.proj(x)
        x = x.reshape(m_batchsize, C, N).transpose(1, 2)
        x = self.proj_drop(x)

        out = self.gamma * x + x
        return out

class PAM_Module(Module):
    def __init__(self, in_dim, dim, num_heads, qk_scale=None, P_lambda=1e-4, attn_drop=0., proj_drop=0.):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Sequential(nn.Conv2d(dim//12, dim, kernel_size=3, padding=1, stride=1, bias=False, groups=dim//12), nn.GELU())
        self.proj_drop = nn.Dropout(proj_drop)

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//12, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//12, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//12, kernel_size=1)
        self.softmax = Softmax(dim=-1)


        self.p_lambda = P_lambda
        self.activaton = nn.Sigmoid()
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x, mask=None):
        m_batchsize, N, C = x.size()
        height = int(N ** .5)
        width = int(N ** .5)

        x = x.view(m_batchsize, C, height, width)
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)

        q = proj_query * self.scale
        attn = (q @ proj_key)

        if mask is not None:
            nW = mask.shape[0]  # num_windows
            attn = attn.view(m_batchsize // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ proj_value).reshape(m_batchsize, C//12, height, width)
        x = self.proj(x)
        x = x.reshape(m_batchsize, C, N).transpose(1, 2)
        x = self.proj_drop(x)

        out = self.gamma * x + x
        return out

class CHAM_Module(Module):
    def __init__(self, in_dim, dim, num_heads, qk_scale=None, P_lambda=1e-4, attn_drop=0., proj_drop=0.):
        super(CHAM_Module, self).__init__()
        self.chanel_in = in_dim

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Sequential(nn.Conv2d(dim//12, dim, kernel_size=3, padding=1, stride=1, bias=False, groups=dim//12), nn.GELU())
        self.proj_drop = nn.Dropout(proj_drop)

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//12, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//12, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//12, kernel_size=1)
        self.softmax = Softmax(dim=-1)

        self.p_lambda = P_lambda
        self.activaton = nn.Sigmoid()
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x, mask=None):
        m_batchsize, N, C = x.size()
        height = int(N ** .5)
        width = int(N ** .5)

        x = x.view(m_batchsize, C, height, width)
        proj_query = self.query_conv(x).view(m_batchsize, C//12 * height, -1)
        proj_key = self.key_conv(x).view(m_batchsize, C//12 * height, -1).permute(0, 2, 1)
        proj_value = self.value_conv(x).view(m_batchsize, C//12 * height, -1)

        q = proj_query * self.scale
        attn = (q @ proj_key)

        if mask is not None:
            nW = mask.shape[0]  # num_windows
            attn = attn.view(m_batchsize // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ proj_value).reshape(m_batchsize, C//12, height, width)
        x = self.proj(x)
        x = x.reshape(m_batchsize, C, N).transpose(1, 2)
        x = self.proj_drop(x)

        out = self.gamma * x + x
        return out


class CWAM_Module(Module):
    def __init__(self, in_dim, dim, num_heads, qk_scale=None, P_lambda=1e-4, attn_drop=0., proj_drop=0.):
        super(CWAM_Module, self).__init__()
        self.chanel_in = in_dim

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, bias=False, groups=dim), nn.GELU())
        self.proj_drop = nn.Dropout(proj_drop)

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//12, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//12, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=-1)

        self.p_lambda = P_lambda
        self.activaton = nn.Sigmoid()
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x, mask=None):
        m_batchsize, N, C = x.size()
        height = int(N ** .5)
        width = int(N ** .5)

        proj_query = x.view(m_batchsize, C * width, -1)
        proj_key = x.view(m_batchsize, C * width, -1).permute(0, 2, 1)
        proj_value = x.view(m_batchsize, C * width, -1)

        q = proj_query * self.scale
        attn = (q @ proj_key)

        if mask is not None:
            nW = mask.shape[0]  # num_windows
            attn = attn.view(m_batchsize // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ proj_value).reshape(m_batchsize, C, height, width)
        x = self.proj(x)
        x = x.reshape(m_batchsize, C, N).transpose(1, 2)
        x = self.proj_drop(x)

        out = self.gamma * x + x
        return out

class WindowAttention_ACAM(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim

        self.C_C = CAM_Module(self.dim, dim=dim, num_heads=num_heads)
        self.H_W = PAM_Module(self.dim, dim=dim, num_heads=num_heads)
        self.C_H = CHAM_Module(self.dim, dim=dim, num_heads=num_heads)
        self.C_W = CWAM_Module(self.dim, dim=dim, num_heads=num_heads)

        self.gamma1 = Parameter(torch.zeros(1))
        self.gamma2 = Parameter(torch.zeros(1))
        self.gamma3 = Parameter(torch.ones(1) * 0.5)
        self.gamma4 = Parameter(torch.ones(1) * 0.5)

    def _build_projection(self, dim_in, kernel_size=3, stride=1, padding=1):
        proj = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size, padding=padding, stride=stride, bias=False, groups=dim_in),
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(dim_in))
        return proj

    def forward(self, x, mask=None):
        x_out1 = self.C_C(x)

        x_out2 = self.H_W(x)

        x_out3 = self.C_H(x)

        x_out4 = self.C_W(x)

        x_out = (self.gamma1 * x_out1) + (self.gamma2 * x_out2) + (self.gamma3 * x_out3) + (self.gamma4 * x_out4)

        return x_out
""" =============================================================================================================== """

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio


        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer([dim, input_resolution[0], input_resolution[1]])

        self.attn = WindowAttention_ACAM(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
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

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = GhostModule(inp=dim)

    def forward(self, x):
        B, C, H, W = x.shape

        shortcut1 = x
        x = x.view(B, H, W, C)
        x = self.norm1(x)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, C, H, W)
        x = shortcut1 + self.drop_path(x)
        shortcut2 = x

        x = self.norm2(x)

        x = shortcut2 + self.drop_path(self.mlp(x))
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        h, w = input_resolution
        h = int(h/2)
        w = int(w/2)
        self.dim = dim
        self.norm = norm_layer([4*dim, h, w])
        self.reduction = GhostModule(inp=4 * dim, oup=2 * dim, ratio=4)

    def forward(self, x):
        H, W = self.input_resolution
        B, C, H, W = x.shape

        x0 = x[:, :, 0::2, 0::2]  # B C H/2 W/2
        x1 = x[:, :, 1::2, 0::2]  # B C H/2 W/2
        x2 = x[:, :, 0::2, 1::2]  # B C H/2 W/2
        x3 = x[:, :, 1::2, 1::2]  # B C H/2 W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # B 4*C H/2 W/2

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
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

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = GhostModule_Up(inp=dim) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer([dim // dim_scale, input_resolution[0]*2, input_resolution[1]*2])

    def forward(self, x):
        B, C, H, W  = x.shape
        x = self.expand(x)
        x = rearrange(x, 'b (p1 p2 c) h w -> b c (h p1) (w p2)', p1=2, p2=2, c=C // 2)
        x = self.norm(x)

        return x

class BasicLayer_up(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = oneXone_conv(in_features = dim, out_features = 16 * dim) if dim_scale == 2 else nn.Identity()
        self.output_dim = dim
        self.norm = norm_layer([6, input_resolution[0]*4, input_resolution[1]*4])

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.expand(x)
        x = rearrange(x, 'b (p1 p2 c) h w -> b c (h p1) (w p2)', p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale ** 2))
        x = self.norm(x)
        return x

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class conv_block_DDConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_DDConv, self).__init__()
        self.conv = nn.Sequential(
            DDConv(ch_in, ch_out, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),

            DDConv(ch_out, ch_out, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv_DDConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv_DDConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),

            DDConv(ch_in, ch_out, kernel_size=1),

            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class ConvMixerLayer(nn.Module):
    def __init__(self, dim, kernel_size=9):
        super().__init__()
        self.Resnet = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, groups=dim, padding=4),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        self.Conv_1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
    def forward(self, x):
        x = x + self.Resnet(x)
        x = self.Conv_1x1(x)
        return x


class ConvMixer(nn.Module):
    def __init__(self, dim=512, depth=1, kernel_size=9, patch_size=4, n_classes=1000):
        super().__init__()
        self.conv2d1 = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        self.ConvMixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.ConvMixer_blocks.append(ConvMixerLayer(dim=dim, kernel_size=kernel_size))

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dim, n_classes)
        )

    def forward(self, x):
        x = self.conv2d1(x)

        for ConvMixer_block in self.ConvMixer_blocks:
            x = ConvMixer_block(x)

        x = x
        return x

class CIT(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, out_chans=1,
                 embed_dim=96, depths=[2, 2, 6, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=True, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()

        self.out_channel = out_chans
        self.num_layers = len(depths)
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.size = int(img_size/(2 ** (self.num_layers + 1)))
        self.size_out = int(img_size/4)
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = nn.LayerNorm

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]


        if self.final_upsample == "expand_first":
            self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
                                          dim_scale=4, dim=embed_dim)
            self.output = nn.Conv2d(in_channels=6, out_channels=1, kernel_size=1, bias=False)

        self.apply(self._init_weights)

        self.embed_dim = 96
        self.num_heads = 3
        self.depth35 = 6
        self.drop_path3 = dpr[4:10]
        self.drop_path4 = dpr[10:12]

        print("CiT-Net-T----embed_dim:{}; num_heads:{}; depths:{}".format(self.embed_dim, num_heads, depths))

        self.layer1 = BasicLayer(dim=self.embed_dim * 1,
                            input_resolution=(56, 56),
                            depth=2,
                            num_heads=self.num_heads * 1,
                            window_size=self.window_size,  # 7
                            mlp_ratio=self.mlp_ratio,  # 4.
                            qkv_bias=self.qkv_bias,  # True
                            qk_scale=self.qk_scale,  # None
                            drop=self.drop_rate,  # 0.
                            attn_drop=self.attn_drop_rate,  # 0.
                            drop_path=dpr[0:2],
                            norm_layer=self.norm_layer,
                            downsample=PatchMerging,
                            use_checkpoint=False)

        self.layer2 = BasicLayer(dim=self.embed_dim * 2,
                            input_resolution=(28, 28),
                            depth=2,
                            num_heads=self.num_heads * 2,
                            window_size=self.window_size,  # 7
                            mlp_ratio=self.mlp_ratio,  # 4.
                            qkv_bias=self.qkv_bias,  # True
                            qk_scale=self.qk_scale,  # None
                            drop=self.drop_rate,  # 0.
                            attn_drop=self.attn_drop_rate,  # 0.
                            drop_path=dpr[2:4],
                            norm_layer=self.norm_layer,
                            downsample=PatchMerging,
                            use_checkpoint=False)

        self.layer3 = BasicLayer(dim=self.embed_dim * 4,
                            input_resolution=(14, 14),
                            depth=self.depth35,
                            num_heads=self.num_heads * 4,
                            window_size=self.window_size,  # 7
                            mlp_ratio=self.mlp_ratio,  # 4.
                            qkv_bias=self.qkv_bias,  # True
                            qk_scale=self.qk_scale,  # None
                            drop=self.drop_rate,  # 0.
                            attn_drop=self.attn_drop_rate,  # 0.
                            drop_path=self.drop_path3,
                            norm_layer=self.norm_layer,
                            downsample=PatchMerging,
                            use_checkpoint=False)

        self.layer4 = BasicLayer(dim=self.embed_dim * 8,
                            input_resolution=(7, 7),
                            depth=2,
                            num_heads=self.num_heads * 8,
                            window_size=self.window_size,  # 7
                            mlp_ratio=self.mlp_ratio,  # 4.
                            qkv_bias=self.qkv_bias,  # True
                            qk_scale=self.qk_scale,  # None
                            drop=self.drop_rate,  # 0.
                            attn_drop=self.attn_drop_rate,  # 0.
                            drop_path=self.drop_path4,
                            norm_layer=self.norm_layer,
                            downsample=None,
                            use_checkpoint=False)

        self.norm = norm_layer([self.num_features, self.size, self.size])

        self.Patch_Expand1 = PatchExpand(input_resolution=(7, 7),
                                        dim=self.embed_dim * 8,
                                        dim_scale=2,
                                        norm_layer=norm_layer)


        self.concat_linear1 = GhostModule(inp=self.embed_dim * 8, oup=self.embed_dim * 4)

        self.layer5 = BasicLayer_up(dim=self.embed_dim * 4,
                                    input_resolution=(14, 14),
                                    depth=self.depth35,
                                    num_heads=self.num_heads * 4,

                                    window_size=self.window_size,  # 7
                                    mlp_ratio=self.mlp_ratio,  # 4.
                                    qkv_bias=self.qkv_bias,  # True
                                    qk_scale=self.qk_scale,  # None
                                    drop=self.drop_rate,  # 0.
                                    attn_drop=self.attn_drop_rate,  # 0.

                                    drop_path=self.drop_path3,
                                    norm_layer=norm_layer,
                                    upsample=PatchExpand,
                                    use_checkpoint=False)

        self.concat_linear2 = GhostModule(inp=self.embed_dim * 4, oup=self.embed_dim * 2)

        self.layer6 = BasicLayer_up(dim=self.embed_dim * 2,
                                    input_resolution=(28, 28),
                                    depth=2,
                                    num_heads=self.num_heads * 2,

                                    window_size=self.window_size,  # 7
                                    mlp_ratio=self.mlp_ratio,  # 4.
                                    qkv_bias=self.qkv_bias,  # True
                                    qk_scale=self.qk_scale,  # None
                                    drop=self.drop_rate,  # 0.
                                    attn_drop=self.attn_drop_rate,  # 0.

                                    drop_path=dpr[2:4],
                                    norm_layer=norm_layer,
                                    upsample=PatchExpand,
                                    use_checkpoint=False)

        self.concat_linear3 = GhostModule(inp=self.embed_dim * 2, oup=self.embed_dim * 1)

        self.layer7 = BasicLayer_up(dim=self.embed_dim * 1,
                                    input_resolution=(56, 56),
                                    depth=2,
                                    num_heads=self.num_heads * 1,

                                    window_size=self.window_size,  # 7
                                    mlp_ratio=self.mlp_ratio,  # 4.
                                    qkv_bias=self.qkv_bias,  # True
                                    qk_scale=self.qk_scale,  # None
                                    drop=self.drop_rate,  # 0.
                                    attn_drop=self.attn_drop_rate,  # 0.

                                    drop_path=dpr[0:2],
                                    norm_layer=norm_layer,
                                    upsample=None,
                                    use_checkpoint=False)

        self.norm_up = norm_layer([self.embed_dim, self.size_out, self.size_out])
        self.patch = ConvMixer(dim=48, depth=5)  # 修改ConvMixer层数
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1e = conv_block(ch_in=48, ch_out=self.embed_dim * 1)
        self.Conv1s = conv_block(ch_in=48, ch_out=self.embed_dim * 1)
        self.Conv2e = conv_block_DDConv(ch_in=self.embed_dim * 1, ch_out=self.embed_dim * 2)
        self.Conv3e = conv_block_DDConv(ch_in=self.embed_dim * 2, ch_out=self.embed_dim * 4)
        self.Conv4e = conv_block_DDConv(ch_in=self.embed_dim * 4, ch_out=self.embed_dim * 8)
        self.Up4d = up_conv_DDConv(ch_in=self.embed_dim * 8, ch_out=self.embed_dim * 4)
        self.Up_conv4d = conv_block(ch_in=self.embed_dim * 8, ch_out=self.embed_dim * 4)
        self.Up3d = up_conv_DDConv(ch_in=self.embed_dim * 4, ch_out=self.embed_dim * 2)
        self.Up_conv3d = conv_block(ch_in=self.embed_dim * 4, ch_out=self.embed_dim * 2)
        self.Up2d = up_conv_DDConv(ch_in=self.embed_dim * 2, ch_out=self.embed_dim * 1)
        self.Up_conv2d = conv_block(ch_in=self.embed_dim * 2, ch_out=self.embed_dim * 1)
        self.Mid_Conv1 = nn.Conv2d(self.embed_dim * 2, self.embed_dim * 1, kernel_size=1, stride=1, padding=0)
        self.Mid_Conv2 = nn.Conv2d(self.embed_dim * 4, self.embed_dim * 2, kernel_size=1, stride=1, padding=0)
        self.Mid_Conv3 = nn.Conv2d(self.embed_dim * 8, self.embed_dim * 4, kernel_size=1, stride=1, padding=0)
        self.BN = nn.BatchNorm2d(1)
        self.CiT_Conv = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0)

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
        return {'relative_position_bias_table'}


    def up_x4(self, x):

        B, C, H, W = x.shape
        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = self.output(x)

        return x

    def forward(self, x):   # 1,3,224,224
        x = self.patch(x)
        Cnn = x
        Swin = x

        Cnn = self.Conv1e(Cnn)     # 1,96,56,56
        Swin = self.Conv1s(Swin)   # 1,96,56,56
        Cnn1 = Cnn
        Swin1 = Swin
        Mid1 = torch.cat((Cnn1, Swin1), dim=1)
        Mid1 = self.Mid_Conv1(Mid1)

        Cnn = self.maxpool(Cnn)
        Cnn = self.Conv2e(Cnn)
        Swin = self.layer1(Swin)   # 28,28
        Cnn2 = Cnn
        Swin2 = Swin
        Mid2 = torch.cat((Cnn2, Swin2), dim=1)
        Mid2 = self.Mid_Conv2(Mid2)

        Cnn = self.maxpool(Cnn)
        Cnn = self.Conv3e(Cnn)
        Swin = self.layer2(Swin)  # 14,14
        Cnn3 = Cnn
        Swin3 = Swin
        Mid3 = torch.cat((Cnn3, Swin3), dim=1)
        Mid3 = self.Mid_Conv3(Mid3)

        Cnn = self.maxpool(Cnn)
        Cnn = self.Conv4e(Cnn)
        Swin = self.layer3(Swin)  # 7,7
        Swin = self.layer4(Swin)  # 7,7
        Swin = self.norm(Swin)  # B L C  (1, 768, 7, 7)
        Cnn4 = Cnn
        Swin4 = Swin

        Cnn = self.Up4d(Cnn)
        Cnn = torch.cat((Cnn, Mid3), dim=1)
        Cnn = self.Up_conv4d(Cnn)
        Swin = self.Patch_Expand1(Swin)
        Swin = torch.cat([Swin, Mid3], 1)
        Swin = self.concat_linear1(Swin)  # 14,14
        Cnn5 = Cnn
        Swin5 = Swin

        Cnn = self.Up3d(Cnn)
        Cnn = torch.cat((Cnn, Mid2), dim=1)
        Cnn = self.Up_conv3d(Cnn)
        Swin = self.layer5(Swin)
        Swin = torch.cat([Swin, Mid2], 1)
        Swin = self.concat_linear2(Swin)  # 28,28
        Cnn6 = Cnn
        Swin6 = Swin

        Cnn7 = self.Up2d(Cnn)
        Cnn7 = torch.cat((Cnn7, Mid1), dim=1)
        Cnn7 = self.Up_conv2d(Cnn7)

        Swin7 = self.layer6(Swin)  # 56,56
        Swin7 = torch.cat([Swin7, Mid1], 1)  # 56,56
        Swin7 = self.concat_linear3(Swin7)  # 56,56
        Swin7 = self.layer7(Swin7)  # 56,56

        CNN = self.up_x4(Cnn7)  # 224,224
        Swin = self.norm_up(Swin7)  # B L C   1,96,56,56
        SWIN = self.up_x4(Swin)  # 224,224

        CNN_out = CNN
        Trans_out = SWIN

        CNN = self.BN(CNN)
        SWIN = self.BN(SWIN)
        CiT = torch.cat((CNN, SWIN), dim=1)
        CiT = self.CiT_Conv(CiT)

        CiT = torch.sigmoid(CiT)
        CNN_out = torch.sigmoid(CNN_out)
        Trans_out = torch.sigmoid(Trans_out)

        return CiT, CNN_out, Trans_out



    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


if __name__ == "__main__":
    with torch.no_grad():
        input = torch.rand(1, 1, 224, 224).to("cpu")
        model =CIT().to("cpu")

        out_result, _, _ = model(input)
        print(out_result.shape)

        flops, params = profile(model, (input,))

        print("-" * 50)
        print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
        print('Params = ' + str(params / 1000 ** 2) + ' M')