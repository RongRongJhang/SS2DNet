import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import math
from typing import Optional, Any
from functools import partial

try:
    from csms6s import selective_scan_fn
except:
    from .csms6s import selective_scan_fn

class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, merge=True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32).view(1, -1).repeat(d_inner, 1)
        A_log = torch.log(A)
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, merge=True):
        D = torch.ones(d_inner)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))
        del dt_projs
            
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True)
        return A_logs, Ds, dt_projs_weight, dt_projs_bias


class SS2D(nn.Module):
    def __init__(
        self,
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        forward_type="v2",
        channel_first=False,
        **kwargs,
    ):
        super().__init__()
        self.k_group = 4
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = nn.Linear

        self.in_proj = Linear(d_model, self.d_inner * 2, bias=bias)
        self.act = act_layer()
        
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
            )

        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
            self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=self.k_group,
        )

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = Linear(self.d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1)
        z = self.act(z)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x) if self.with_dconv else x
        x = self.act(x)

        B, D, H, W = x.shape
        L = H * W

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        Bs = Bs.contiguous()
        Cs = Cs.contiguous()
        
        As = -self.A_logs.float().exp()
        Ds = self.Ds.float()
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        selective_scan = partial(selective_scan_fn, backend="mamba")

        out_y = selective_scan(xs, dts, As, Bs, Cs, Ds, delta_bias=dt_projs_bias, delta_softplus=True).view(B, self.k_group, -1, L)
        
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        
        y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)

        y = y * z
        out = self.dropout(self.out_proj(y))
        return out

class SS2DAttention(nn.Module):
    def __init__(self, embed_size, num_heads, d_state=16, ssm_ratio=2.0, d_conv=3):
        super(SS2DAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        
        # SS2D 模組
        self.ss2d = SS2D(
            d_model=embed_size,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank="auto",
            act_layer=nn.SiLU,
            d_conv=d_conv,
            conv_bias=True,
            dropout=0.0,
            bias=False,
            forward_type="v2",
            channel_first=False
        )

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, embed_size, height, width)
        output: Tensor of shape (batch_size, embed_size, height, width)
        """
        batch_size, embed_size, height, width = x.shape
        assert embed_size == self.embed_size, f"Expected {self.embed_size}, got {embed_size}"
        
        # 轉換格式以適應 SS2D: (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1).contiguous()

        # SS2D 運算
        x = self.ss2d(x)

        # 轉換回原始格式: (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x

class LayerNormalization(nn.Module):
    def __init__(self, dim):
        super(LayerNormalization, self).__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # Rearrange the tensor for LayerNorm (B, C, H, W) to (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # Rearrange back to (B, C, H, W)
        return x.permute(0, 3, 1, 2)

class SEBlock(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(input_channels, input_channels // reduction_ratio)
        self.fc2 = nn.Linear(input_channels // reduction_ratio, input_channels)
        self._init_weights()

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        y = self.pool(x).reshape(batch_size, num_channels)
        y = F.relu(self.fc1(y))
        y = torch.tanh(self.fc2(y))
        y = y.reshape(batch_size, num_channels, 1, 1)
        return x * y
    
    def _init_weights(self):
        init.kaiming_uniform_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.fc2.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.fc1.bias, 0)
        init.constant_(self.fc2.bias, 0)

class MSEFBlock(nn.Module):
    def __init__(self, filters):
        super(MSEFBlock, self).__init__()
        self.layer_norm = LayerNormalization(filters)
        self.depthwise_conv = nn.Conv2d(filters, filters, kernel_size=3, padding=1, groups=filters)
        self.se_attn = SEBlock(filters)
        self._init_weights()

    def forward(self, x):
        x_norm = self.layer_norm(x)
        x1 = self.depthwise_conv(x_norm)
        x2 = self.se_attn(x_norm)
        x_fused = x1 * x2
        x_out = x_fused + x
        return x_out
    
    def _init_weights(self):
        init.kaiming_uniform_(self.depthwise_conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.depthwise_conv.bias, 0)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        assert embed_size % num_heads == 0
        self.head_dim = embed_size // num_heads
        self.query_dense = nn.Linear(embed_size, embed_size)
        self.key_dense = nn.Linear(embed_size, embed_size)
        self.value_dense = nn.Linear(embed_size, embed_size)
        self.combine_heads = nn.Linear(embed_size, embed_size)
        self._init_weights()

    def split_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        x = x.reshape(batch_size, height * width, -1)

        query = self.split_heads(self.query_dense(x), batch_size)
        key = self.split_heads(self.key_dense(x), batch_size)
        value = self.split_heads(self.value_dense(x), batch_size)
        
        attention_weights = F.softmax(torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        attention = torch.matmul(attention_weights, value)
        attention = attention.permute(0, 2, 1, 3).contiguous().reshape(batch_size, -1, self.embed_size)
        
        output = self.combine_heads(attention)
        
        return output.reshape(batch_size, height, width, self.embed_size).permute(0, 3, 1, 2)

    def _init_weights(self):
        init.xavier_uniform_(self.query_dense.weight)
        init.xavier_uniform_(self.key_dense.weight)
        init.xavier_uniform_(self.value_dense.weight)
        init.xavier_uniform_(self.combine_heads.weight)
        init.constant_(self.query_dense.bias, 0)
        init.constant_(self.key_dense.bias, 0)
        init.constant_(self.value_dense.bias, 0)
        init.constant_(self.combine_heads.bias, 0)

class Denoiser(nn.Module):
    def __init__(self, num_filters, kernel_size=3, activation='relu'):
        super(Denoiser, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        # self.bottleneck = MultiHeadSelfAttention(embed_size=num_filters, num_heads=4)
        self.bottleneck = SS2DAttention(embed_size=num_filters, num_heads=4)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.output_layer = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=1)
        self.res_layer = nn.Conv2d(num_filters, 1, kernel_size=kernel_size, padding=1)
        self.activation = getattr(F, activation)
        self._init_weights()

    def forward(self, x):
        x1 = self.activation(self.conv1(x))
        x2 = self.activation(self.conv2(x1))
        x3 = self.activation(self.conv3(x2))
        x4 = self.activation(self.conv4(x3))
        x = self.bottleneck(x4)
        x = self.up4(x)
        x = self.up3(x + x3)
        x = self.up2(x + x2)
        x = x + x1
        x = self.res_layer(x)
        return torch.tanh(self.output_layer(x + x))
    
    def _init_weights(self):
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.output_layer, self.res_layer]:
            init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                init.constant_(layer.bias, 0)

class LYT(nn.Module):
    def __init__(self, filters=32):
        super(LYT, self).__init__()
        self.process_y = self._create_processing_layers(filters)
        self.process_cb = self._create_processing_layers(filters)
        self.process_cr = self._create_processing_layers(filters)

        self.denoiser_cb = Denoiser(filters // 2)
        self.denoiser_cr = Denoiser(filters // 2)
        self.lum_pool = nn.MaxPool2d(8)
        # self.lum_mhsa = MultiHeadSelfAttention(embed_size=filters, num_heads=4)
        self.lum_mhsa = SS2DAttention(embed_size=filters, num_heads=4)
        self.lum_up = nn.Upsample(scale_factor=8, mode='nearest')
        self.lum_conv = nn.Conv2d(filters, filters, kernel_size=1, padding=0)
        self.ref_conv = nn.Conv2d(filters * 2, filters, kernel_size=1, padding=0)
        self.msef = MSEFBlock(filters)
        self.recombine = nn.Conv2d(filters * 2, filters, kernel_size=3, padding=1)
        self.final_adjustments = nn.Conv2d(filters, 3, kernel_size=3, padding=1)
        self._init_weights()

    def _create_processing_layers(self, filters):
        return nn.Sequential(
            nn.Conv2d(1, filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def _rgb_to_ycbcr(self, image):
        r, g, b = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
    
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b + 0.5
        v = 0.615 * r - 0.51499 * g - 0.10001 * b + 0.5
        
        yuv = torch.stack((y, u, v), dim=1)
        return yuv

    def forward(self, inputs):
        ycbcr = self._rgb_to_ycbcr(inputs)
        y, cb, cr = torch.split(ycbcr, 1, dim=1)
        cb = self.denoiser_cb(cb) + cb
        cr = self.denoiser_cr(cr) + cr

        y_processed = self.process_y(y)
        cb_processed = self.process_cb(cb)
        cr_processed = self.process_cr(cr)

        ref = torch.cat([cb_processed, cr_processed], dim=1)
        lum = y_processed
        lum_1 = self.lum_pool(lum)
        lum_1 = self.lum_mhsa(lum_1)
        lum_1 = self.lum_up(lum_1)
        lum = lum + lum_1

        ref = self.ref_conv(ref)
        shortcut = ref
        ref = ref + 0.2 * self.lum_conv(lum)
        ref = self.msef(ref)
        ref = ref + shortcut

        recombined = self.recombine(torch.cat([ref, lum], dim=1))
        output = self.final_adjustments(recombined)
        return torch.sigmoid(output)
    
    def _init_weights(self):
        for module in self.children():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)
                    