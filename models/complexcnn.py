import torch
import torch.nn as nn

class ComplexConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv_re = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_im = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        real, imag = x.chunk(2, dim=1)
        real_out = self.conv_re(real) - self.conv_im(imag)
        imag_out = self.conv_re(imag) + self.conv_im(real)
        return torch.cat([real_out, imag_out], dim=1)

class ComplexConv_trans(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv_re = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_im = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        real, imag = x.chunk(2, dim=1)
        real_out = self.conv_re(real) - self.conv_im(imag)
        imag_out = self.conv_re(imag) + self.conv_im(real)
        return torch.cat([real_out, imag_out], dim=1)