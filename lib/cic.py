import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalInformationCouplingModule(nn.Module):
    """Conditional Information Coupling (CIC) Module.

    Parameters
    ----------
    in_channels : int
        通道数，对应输入和条件特征的维度。
    inter_channels : int, optional
        中间特征维度，默认取 ``in_channels // 2``。
    dimension : int
        1/2/3 维输入的选择。
    sub_sample : bool
        是否在 ``k`` 和 ``v`` 上进行下采样。
    bn_layer : bool
        是否在 ``W`` 后接 BatchNorm。
    mask_mode : str
        ``"cosine"``(默认) 使用余弦相似度， ``"gate"`` 使用可学习门控。
    init_zero : bool
        若为 ``True``， ``W`` 的权重以 0 初始化；否则使用 xavier 初始化。
    """

    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True,
                 bn_layer=True, mask_mode="cosine", init_zero=True):
        super().__init__()
        assert dimension in [1, 2, 3]
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels or max(in_channels // 2, 1)
        self.mask_mode = mask_mode
        self.init_zero = init_zero

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
            gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
            gap = nn.AdaptiveAvgPool2d((1, 1))
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=2)
            bn = nn.BatchNorm1d
            gap = nn.AdaptiveAvgPool1d(1)

        self.v = conv_nd(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.mask_conv = None
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(self.inter_channels, self.in_channels, kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            if self.init_zero:
                nn.init.constant_(self.W[1].weight, 0)
                nn.init.constant_(self.W[1].bias, 0)
            else:
                nn.init.xavier_uniform_(self.W[0].weight)
                nn.init.zeros_(self.W[0].bias)
                nn.init.zeros_(self.W[1].weight)
                nn.init.zeros_(self.W[1].bias)
        else:
            self.W = conv_nd(self.inter_channels, self.in_channels, kernel_size=1, stride=1, padding=0)
            if self.init_zero:
                nn.init.constant_(self.W.weight, 0)
                nn.init.constant_(self.W.bias, 0)
            else:
                nn.init.xavier_uniform_(self.W.weight)
                nn.init.zeros_(self.W.bias)

        if self.mask_mode == "gate":
            self.mask_conv = conv_nd(self.in_channels, 1, kernel_size=1)

        self.q = conv_nd(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.k = conv_nd(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.gap = gap

        if sub_sample:
            self.v = nn.Sequential(self.v, max_pool_layer)
            self.k = nn.Sequential(self.k, max_pool_layer)

    def forward(self, x, kv_x):
        return self.forward_single(x, kv_x)

    def forward_single(self, x, kv_x):
        batch_size = x.size(0)
        v_x = self.v(kv_x).view(batch_size, self.inter_channels, -1)
        v_x = v_x.permute(0, 2, 1)

        q_x = self.q(x).view(batch_size, self.inter_channels, -1)
        q_x = q_x.permute(0, 2, 1)

        k_x = self.k(kv_x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(q_x, k_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, v_x)
        y = y.permute(0, 2, 1).contiguous()
        if self.dimension == 1:
            y = y.view(batch_size, self.inter_channels, x.size(2))
        else:
            y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)

        kv_x_gap = self.gap(kv_x)
        if self.dimension == 1:
            kv_x_gap = kv_x_gap.view(kv_x.size(0), -1, 1)
        else:
            kv_x_gap = kv_x_gap.view(kv_x.size(0), -1, 1, 1)
        kv_x_gap = kv_x_gap.expand_as(x)
        if self.mask_mode == "gate":
            mask = torch.sigmoid(self.mask_conv(x + kv_x_gap))
        else:
            mask = torch.cosine_similarity(x, kv_x_gap, dim=1).unsqueeze(1)
        z = W_y * mask + x
        return z
