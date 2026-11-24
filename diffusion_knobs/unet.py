import math

import torch
from torch import nn
import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.theta) * torch.arange(half, device = t.device) / half)
        args = 1000 * t[:, None] * freqs[None]
        return torch.cat((args.sin(), args.cos()), dim = -1)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_dim, groups = 8):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, dim_out))
        self.block1 = nn.Sequential(nn.GroupNorm(groups, dim), nn.SiLU(), nn.Conv2d(dim, dim_out, 3, padding = 1))
        self.block2 = nn.Sequential(nn.GroupNorm(groups, dim_out), nn.SiLU(), nn.Conv2d(dim_out, dim_out, 3, padding = 1))
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h = h + self.time_mlp(time_emb)[..., None, None]
        h = self.block2(h)
        return h + self.res_conv(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(1, dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(dim = 1)
        out = F.scaled_dot_product_attention(q.mT, k.mT, v.mT)
        return x + self.to_out(out.mT.reshape(b, c, h, w))

class Unet(nn.Module):
    """28x28 -> 14x14 -> 7x7 and back, conditioned on t in [0, 1]"""

    def __init__(self, dim = 64, dim_mults = (1, 2, 2), channels = 1):
        super().__init__()
        dims = [dim, *(dim * m for m in dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))
        self.init_conv = nn.Conv2d(channels, dim, 3, padding = 1)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i == len(in_out) - 1
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in, time_dim),
                nn.Conv2d(dim_in, dim_out, 3, padding = 1) if is_last else nn.Conv2d(dim_in, dim_out, 4, stride = 2, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_dim)
        self.mid_attn = Attention(mid_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_dim)

        for i, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = i == len(in_out) - 1
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out + dim_in, dim_out, time_dim),
                nn.Conv2d(dim_out, dim_in, 3, padding = 1) if is_last else nn.Sequential(nn.Upsample(scale_factor = 2), nn.Conv2d(dim_out, dim_in, 3, padding = 1))
            ]))

        self.final_block = ResnetBlock(dim * 2, dim, time_dim)
        self.final_conv = nn.Conv2d(dim, channels, 1)

    def forward(self, x, time):
        t = self.time_mlp(time)
        x = self.init_conv(x)
        r = x.clone()

        skips = []
        for block, downsample in self.downs:
            x = block(x, t)
            skips.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block, upsample in self.ups:
            x = block(torch.cat((x, skips.pop()), dim = 1), t)
            x = upsample(x)

        x = self.final_block(torch.cat((x, r), dim = 1), t)
        return self.final_conv(x)
