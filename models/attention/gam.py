"""
Global Attention Module (GAM) implementation.
Reference: https://arxiv.org/abs/2112.05561
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialGate(nn.Module):
    def __init__(self, in_channels):
        super(SpatialGate, self).__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.spatial(x)

class ChannelGate(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.channel(x)

class GAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(GAM, self).__init__()
        self.channel_gate = ChannelGate(in_channels, reduction_ratio)
        self.spatial_gate = SpatialGate(in_channels)
        
        # Global feature aggregation
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_global = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn_global = nn.BatchNorm2d(in_channels)
        self.act_global = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Global feature extraction
        global_feat = self.global_pool(x)
        global_feat = self.conv_global(global_feat)
        global_feat = self.bn_global(global_feat)
        global_feat = self.act_global(global_feat)
        
        # Channel attention
        channel_att = self.channel_gate(x)
        x_channel = x * channel_att
        
        # Spatial attention with global context
        x_with_global = x_channel + global_feat
        spatial_att = self.spatial_gate(x_with_global)
        
        # Final output
        out = x_channel * spatial_att
        
        return out

def add_gam_to_conv(conv, reduction_ratio=16):
    """
    Helper function to add Global Attention Module to a convolutional layer.
    
    Args:
        conv: The convolutional layer to add GAM to
        reduction_ratio: Reduction ratio for channel attention
        
    Returns:
        A sequential module with the conv layer and GAM
    """
    out_channels = conv.out_channels
    return nn.Sequential(
        conv,
        GAM(out_channels, reduction_ratio)
    ) 