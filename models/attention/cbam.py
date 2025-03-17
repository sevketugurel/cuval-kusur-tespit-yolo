"""
Convolutional Block Attention Module (CBAM) implementation.
Reference: https://arxiv.org/abs/1807.06521
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return torch.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return torch.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        # Apply channel attention
        channel_att = self.channel_attention(x)
        x = x * channel_att
        
        # Apply spatial attention
        spatial_att = self.spatial_attention(x)
        x = x * spatial_att
        
        return x

def add_cbam_to_conv(conv, reduction_ratio=16, kernel_size=7):
    """
    Helper function to add CBAM to a convolutional layer.
    
    Args:
        conv: The convolutional layer to add CBAM to
        reduction_ratio: Reduction ratio for channel attention
        kernel_size: Kernel size for spatial attention
        
    Returns:
        A sequential module with the conv layer and CBAM
    """
    out_channels = conv.out_channels
    return nn.Sequential(
        conv,
        CBAM(out_channels, reduction_ratio, kernel_size)
    ) 