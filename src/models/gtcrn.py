"""GTCRN 骨干（改编自 https://github.com/Xiaobin-Rong/gtcrn ，commit 502ebfab，MIT License，
Copyright (c) 2024 Rong Xiaobin）。

与原版的区别（按 BeamFusion 设置，CLAUDE.md 5.1 节）：
- 输入通道数可配置（BeamFusion：P 个波束输出的实部、虚部、幅度，共 3P 个通道），输出通道数可配置；
- encoder 输出 / decoder 输入 / G-DPRNN 隐层维度由 16 改为 width=32、hidden=32；
- 最后一层的 Tanh 换为可配置的激活（融合头输出 logits，由上层做 softmax 等）；
- intra-frame（沿频率）GRU 是否双向可配置。GTCRN 默认双向（沿频率，不破坏时间因果性）；
  BeamFusion 论文写 G-DPRNN 使用单向 GRU，两种都提供，默认保持 GTCRN 原设置；
- 去掉 einops 依赖；其余结构与超参（ERB 65+64、SFE、卷积核、膨胀率 1/2/5、TRA、BN、PReLU）保持不变。
时间方向上全部因果：时间卷积只在前面补零，TRA 与 inter-frame GRU 为单向。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ERB(nn.Module):
    def __init__(self, erb_subband_1=65, erb_subband_2=64, nfft=512, high_lim=8000, fs=16000):
        super().__init__()
        erb_filters = self.erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft // 2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_fc = nn.Linear(nfreqs - erb_subband_1, erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs - erb_subband_1, bias=False)
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T.contiguous(), requires_grad=False)

    @staticmethod
    def hz2erb(freq_hz):
        return 21.4 * np.log10(0.00437 * freq_hz + 1)

    @staticmethod
    def erb2hz(erb_f):
        return (10 ** (erb_f / 21.4) - 1) / 0.00437

    def erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        low_lim = erb_subband_1 / nfft * fs
        erb_points = np.linspace(self.hz2erb(low_lim), self.hz2erb(high_lim), erb_subband_2)
        bins = np.round(self.erb2hz(erb_points) / fs * nfft).astype(np.int32)
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)
        erb_filters[0, bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) / (bins[1] - bins[0] + 1e-12)
        for i in range(erb_subband_2 - 2):
            erb_filters[i + 1, bins[i]:bins[i + 1]] = (np.arange(bins[i], bins[i + 1]) - bins[i] + 1e-12) \
                / (bins[i + 1] - bins[i] + 1e-12)
            erb_filters[i + 1, bins[i + 1]:bins[i + 2]] = (bins[i + 2] - np.arange(bins[i + 1], bins[i + 2]) + 1e-12) \
                / (bins[i + 2] - bins[i + 1] + 1e-12)
        erb_filters[-1, bins[-2]:bins[-1] + 1] = 1 - erb_filters[-2, bins[-2]:bins[-1] + 1]
        erb_filters = erb_filters[:, erb_subband_1:]
        return torch.from_numpy(np.abs(erb_filters))

    def bm(self, x):
        """x: (B,C,T,F) -> (B,C,T,F_erb)"""
        return torch.cat([x[..., :self.erb_subband_1], self.erb_fc(x[..., self.erb_subband_1:])], dim=-1)

    def bs(self, x_erb):
        """x: (B,C,T,F_erb) -> (B,C,T,F)"""
        return torch.cat([x_erb[..., :self.erb_subband_1], self.ierb_fc(x_erb[..., self.erb_subband_1:])], dim=-1)


class SFE(nn.Module):
    """Subband Feature Extraction"""

    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=(1, kernel_size), stride=(1, stride), padding=(0, (kernel_size - 1) // 2))

    def forward(self, x):
        return self.unfold(x).reshape(x.shape[0], x.shape[1] * self.kernel_size, x.shape[2], x.shape[3])


class TRA(nn.Module):
    """Temporal Recurrent Attention"""

    def __init__(self, channels):
        super().__init__()
        self.att_gru = nn.GRU(channels, channels * 2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels * 2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x):
        zt = torch.mean(x.pow(2), dim=-1)  # (B,C,T)
        at = self.att_gru(zt.transpose(1, 2))[0]
        at = self.att_act(self.att_fc(at).transpose(1, 2))
        return x * at[..., None]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, use_deconv=False,
                 act: nn.Module | None = None, use_bn=True):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.PReLU() if act is None else act

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GTConvBlock(nn.Module):
    """Group Temporal Convolution"""

    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False):
        super().__init__()
        self.pad_size = (kernel_size[0] - 1) * dilation[0]
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.sfe = SFE(kernel_size=3, stride=1)
        self.point_conv1 = conv_module(in_channels // 2 * 3, hidden_channels, 1)
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        self.point_act = nn.PReLU()
        self.depth_conv = conv_module(hidden_channels, hidden_channels, kernel_size, stride=stride, padding=padding,
                                      dilation=dilation, groups=hidden_channels)
        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        self.depth_act = nn.PReLU()
        self.point_conv2 = conv_module(hidden_channels, in_channels // 2, 1)
        self.point_bn2 = nn.BatchNorm2d(in_channels // 2)
        self.tra = TRA(in_channels // 2)

    @staticmethod
    def shuffle(x1, x2):
        """(B,C,T,F) x2 -> (B,2C,T,F)，通道交错（等价于原版的 rearrange 'b c g t f -> b (c g) t f'）"""
        x = torch.stack([x1, x2], dim=2)  # (B,C,2,T,F)
        return x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])

    def forward(self, x):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x1 = self.sfe(x1)
        h1 = self.point_act(self.point_bn1(self.point_conv1(x1)))
        h1 = nn.functional.pad(h1, [0, 0, self.pad_size, 0])
        h1 = self.depth_act(self.depth_bn(self.depth_conv(h1)))
        h1 = self.point_bn2(self.point_conv2(h1))
        h1 = self.tra(h1)
        return self.shuffle(h1, x2)


class GRNN(nn.Module):
    """Grouped RNN"""

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn1 = nn.GRU(input_size // 2, hidden_size // 2, num_layers, batch_first=batch_first,
                           bidirectional=bidirectional)
        self.rnn2 = nn.GRU(input_size // 2, hidden_size // 2, num_layers, batch_first=batch_first,
                           bidirectional=bidirectional)

    def forward(self, x):
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        y1 = self.rnn1(x1.contiguous())[0]
        y2 = self.rnn2(x2.contiguous())[0]
        return torch.cat([y1, y2], dim=-1)


class DPGRNN(nn.Module):
    """Grouped Dual-path RNN"""

    def __init__(self, input_size, width, hidden_size, intra_bidirectional=True):
        super().__init__()
        self.width = width
        self.hidden_size = hidden_size
        if intra_bidirectional:
            self.intra_rnn = GRNN(input_size=input_size, hidden_size=hidden_size // 2, bidirectional=True)
        else:
            self.intra_rnn = GRNN(input_size=input_size, hidden_size=hidden_size, bidirectional=False)
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)
        self.inter_rnn = GRNN(input_size=input_size, hidden_size=hidden_size, bidirectional=False)
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

    def forward(self, x):
        """x: (B, C, T, F)"""
        x = x.permute(0, 2, 3, 1)  # (B,T,F,C)
        B, T, F, C = x.shape
        intra_x = self.intra_fc(self.intra_rnn(x.reshape(B * T, F, C)))
        intra_x = self.intra_ln(intra_x.reshape(B, T, F, self.hidden_size))
        intra_out = x + intra_x
        x = intra_out.permute(0, 2, 1, 3)  # (B,F,T,C)
        inter_x = self.inter_fc(self.inter_rnn(x.reshape(B * F, T, C)))
        inter_x = self.inter_ln(inter_x.reshape(B, F, T, self.hidden_size).permute(0, 2, 1, 3))
        return (intra_out + inter_x).permute(0, 3, 1, 2)  # (B,C,T,F)


class GTCRNBackbone(nn.Module):
    """特征 [B, in_ch, T, F=257] -> 输出 [B, out_ch, T, F=257]（logits，未经激活）。"""

    def __init__(self, in_ch: int, out_ch: int, width: int = 32, hidden: int = 32, intra_bidirectional=True,
                 last_bn: bool = True):
        super().__init__()
        C = width
        self.erb = ERB(65, 64)
        self.sfe = SFE(3, 1)
        self.en_convs = nn.ModuleList([
            ConvBlock(in_ch * 3, C, (1, 5), stride=(1, 2), padding=(0, 2)),
            ConvBlock(C, C, (1, 5), stride=(1, 2), padding=(0, 2), groups=2),
            GTConvBlock(C, C, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1)),
            GTConvBlock(C, C, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1)),
            GTConvBlock(C, C, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(5, 1)),
        ])
        self.dpgrnn1 = DPGRNN(C, 33, hidden, intra_bidirectional)
        self.dpgrnn2 = DPGRNN(C, 33, hidden, intra_bidirectional)
        self.de_convs = nn.ModuleList([
            GTConvBlock(C, C, (3, 3), stride=(1, 1), padding=(2 * 5, 1), dilation=(5, 1), use_deconv=True),
            GTConvBlock(C, C, (3, 3), stride=(1, 1), padding=(2 * 2, 1), dilation=(2, 1), use_deconv=True),
            GTConvBlock(C, C, (3, 3), stride=(1, 1), padding=(2 * 1, 1), dilation=(1, 1), use_deconv=True),
            ConvBlock(C, C, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=True),
            ConvBlock(C, out_ch, (1, 5), stride=(1, 2), padding=(0, 2), use_deconv=True, act=nn.Identity(),
                      use_bn=last_bn),
        ])

    def forward(self, feat):
        x = self.sfe(self.erb.bm(feat))
        en_outs = []
        for layer in self.en_convs:
            x = layer(x)
            en_outs.append(x)
        x = self.dpgrnn2(self.dpgrnn1(x))
        n = len(self.de_convs)
        for i, layer in enumerate(self.de_convs):
            x = layer(x + en_outs[n - 1 - i])
        return self.erb.bs(x)

    @property
    def output_layer(self) -> ConvBlock:
        return self.de_convs[-1]
