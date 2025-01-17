import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import Attention


class BottleneckPruned(nn.Module):
    # Pruned bottleneck
    def __init__(self, cv1in, cv1out, cv2out, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        self.cv1 = Conv(cv1in, cv1out, k[0], 1)
        self.cv2 = Conv(cv1out, cv2out, k[1], 1, g=g)
        self.add = shortcut and cv1in == cv2out

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3Pruned(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, cv1cv2in, cv1out, cv2out, inner_cv1outs, cv3out, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        self.cv1 = Conv(cv1cv2in, cv1out, 1, 1)
        self.cv2 = Conv(cv1cv2in, cv2out, 1, 1)
        cv3in = cv1out + cv2out
        self.cv3 = Conv(cv3in, cv3out, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(
            BottleneckPruned(
                cv1out, inner_cv1outs[i], cv1out, shortcut, g, k=((1, 1), (3, 3)), e=1.0)
            for i in range(n)
        ))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3kPruned(C3Pruned):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, cv1cv2in, cv1out, cv2out, inner_cv1outs, cv3out, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(
            cv1cv2in, cv1out, cv2out, inner_cv1outs, cv3out, n, shortcut, g, e
        )
        self.m = nn.Sequential(*(
            BottleneckPruned(
                cv1out, inner_cv1outs[i], cv1out, shortcut, g, k=(k, k), e=1.0)
            for i in range(n)
        ))


class C2fPruned(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, cv1in, cv1out, cv1_split_sections, inner_cv1outs, inner_cv2outs, cv2out, c3k_cv3outs=None,
                 n=1, shortcut=False, g=1, e=0.5, use_c3k=False):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.cv1_split_sections = cv1_split_sections
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        # 如果是不使用c3k的情况下，才会真正去实例化出来self.m，否则的话，就只是给self.m赋值一个空的ModuleList
        if (not use_c3k):
            if shortcut:
                self.cv2 = Conv(sum(self.cv1_split_sections) +
                                n * self.cv1_split_sections[1], cv2out, 1)
                # 如果是shortcut的情况下, 那么在Bottlenet内部的每一个cv2的输出必须和内部的cv1的输入通道数相同
                for i in range(n):
                    assert (
                        inner_cv2outs[i] == self.cv1_split_sections[1]), "Shortcut channels must match"
                self.m = nn.ModuleList(
                    BottleneckPruned(
                        self.cv1_split_sections[1], inner_cv1outs[i], self.cv1_split_sections[1],
                        shortcut, g, k=((3, 3), (3, 3)), e=1.0
                    )
                    for i in range(n)
                )
            else:  # 在yolov11中, C3k2中的结构都是具有残差连接的, 所以实际上这个else是不生效的
                self.c = self.cv1_split_sections[1]
                cv2_inchannels = cv1out + sum(inner_cv2outs)
                self.cv2 = Conv(cv2_inchannels, cv2out, 1)
                # 如果不是shortcut的情况下, 那么在Bottlenet内部的每一个cv2的输出和内部的cv1的输入通道数不一定相等
                self.m = nn.ModuleList()
                for i in range(n):
                    self.m.append(
                        BottleneckPruned(
                            self.c, inner_cv1outs[i], inner_cv2outs[i],
                            shortcut, g, k=((3, 3), (3, 3)), e=1.0
                        )
                    )
                    self.c = inner_cv2outs[i]
        else:
            self.cv2 = Conv(sum(self.cv1_split_sections) +
                            c3k_cv3outs[-1], cv2out, 1)
            self.m = nn.ModuleList()

    def forward(self, x):
        """
        在head部分的C2f层中, 由于没有shortcut残差结构, 因此C2f结构中的第一个cv1层是可以被剪枝的
        但是剪完以后是不一定对称的, 因此要重新计算比例
        例如, C2f结构中的第一个cv1层剪枝前输出通道数为256, chunck以后左右各式128,
        但是剪枝后, cv1层输出通道数可能为120, 但是其中80落在左半区, 40落在右半区
        """
        # y = list(self.cv1(x).chunk(2, 1))
        y = list(self.cv1(x).split(self.cv1_split_sections, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2Pruned(C2fPruned):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self, c2f_cv1in, c2f_cv1out, c2f_cv1_split_sections, c2f_cv2out,
        bottle_inner_cv1outs, bottle_inner_cv2outs,
        c3k_cv1outs, c3k_cv2outs, c3k_inner_cv1outs, c3k_cv3outs,
        n=1, c3k=False, e=0.5, g=1, shortcut=True,
    ):
        """
        Treat this module as three parts:
        The C2fPruned part is like a shell, if c3k is False, then it is exactly the same with C2fPruned;
        Otherwise, use C3kPruned as the inner module instead of BottleneckPruned.

        c2f_cv1in, c2f_cv1out, c2f_cv1_split_sections, c2f_cv2out: outer C2fPruned params;
        bottle_inner_cv1outs, bottle_inner_cv2outs: inner BottleneckPruned params;

        """
        super().__init__(
            c2f_cv1in, c2f_cv1out, c2f_cv1_split_sections,
            bottle_inner_cv1outs, bottle_inner_cv2outs,
            c2f_cv2out, c3k_cv3outs, n, shortcut, g, e, c3k,
        )
        self.c3k = c3k
        if c3k:
            for i in range(n):
                c3k_cv1cv2in = (
                    c2f_cv1_split_sections[1] if i == 0 else c3k_cv3outs[i - 1])
                self.m.append(
                    C3kPruned(
                        c3k_cv1cv2in, c3k_cv1outs[i], c3k_cv2outs[i], c3k_inner_cv1outs[i], c3k_cv3outs[i],
                        2, shortcut, g
                    )
                )


class SPPFPruned(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, cv1in, cv1out, cv2out, k=5):
        super(SPPFPruned, self).__init__()
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        self.cv2 = Conv(cv1out * 4, cv2out, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class PSABlockPruned(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, ffn_cv1out, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(
            Conv(c, ffn_cv1out, 1), Conv(ffn_cv1out, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSAPruned(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, ffn_cv1outs, c2f_cv2out, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2f_cv2out, 1)

        self.m = nn.Sequential(
            *(PSABlockPruned(self.c, ffn_cv1outs[i], attn_ratio=0.5, num_heads=self.c // 64) for i in range(n))
        )

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


if __name__ == "__main__":
    # # ================Test BottleneckPruned================
    # cv1in = 112
    # cv1out = 22
    # cv2out = 111
    # model = BottleneckPruned(cv1in, cv1out, cv2out, shortcut=False)
    # print(model(torch.randn(1, cv1in, 64, 64)).shape)

    # # ================Test C2fPruned================
    # cv1in = 112
    # cv1out = 22
    # cv1_split_sections = [11, 11]
    # inner_cv1outs = [2]
    # inner_cv2outs = [cv1_split_sections[-1]]
    # cv2out = 111
    # model = C2fPruned(
    #     cv1in, cv1out, cv1_split_sections, inner_cv1outs, inner_cv2outs, cv2out,
    #     shortcut=True
    # )
    # print(model(torch.randn(1, cv1in, 64, 64)).shape)

    # # ================Test C3k2Pruned================
    # c2f_cv1in = 112
    # c2f_cv1out = 22
    # c2f_cv1_split_sections = [10, 12]
    # c2f_cv2out = 111
    # bottle_inner_cv1outs = None
    # bottle_inner_cv2outs = None
    # c3k_cv1outs = [55]
    # c3k_cv2outs = [66]
    # c3k_inner_cv1outs = [[2, 5]]
    # c3k_cv3outs = [67]
    # n = 1
    # c3k = True
    # model = C3k2Pruned(
    #     c2f_cv1in, c2f_cv1out, c2f_cv1_split_sections, c2f_cv2out, bottle_inner_cv1outs,
    #     bottle_inner_cv2outs, c3k_cv1outs, c3k_cv2outs, c3k_inner_cv1outs, c3k_cv3outs, n, c3k
    # )
    # print(model(torch.randn(1, c2f_cv1in, 64, 64)).shape)

    # ================Test C2PSAPruned================
    c1 = 512
    c2 = 256
    ffn_cv1outs = [22, 33]
    n = 1
    e = 0.5
    model = C2PSAPruned(c1, c2, ffn_cv1outs, n, e)
    print(model(torch.randn(1, c1, 64, 64)).shape)
