"""
Coordinate Attention Module implementation.
Reference: https://arxiv.org/abs/2103.02907
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordAtt(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mid_channels = max(8, in_channels // reduction)
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)
        
        self.conv_h = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        identity = x
        
        n, c, h, w = x.size()
        # Height-wise pooling
        x_h = self.pool_h(x)  # [n, c, h, 1]
        # Width-wise pooling
        x_w = self.pool_w(x)  # [n, c, 1, w]
        x_w = x_w.permute(0, 1, 3, 2)  # [n, c, w, 1]
        
        # Concatenate along the spatial dimension
        y = torch.cat([x_h, x_w], dim=2)  # [n, c, h+w, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # Split the features
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # [n, c, 1, w]
        
        # Generate attention maps
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        # Apply attention
        out = identity * a_h * a_w
        
        return out

def add_coordatt_to_conv(conv, reduction=32):
    """
    Helper function to add Coordinate Attention to a convolutional layer.
    
    Args:
        conv: The convolutional layer to add CoordAtt to
        reduction: Reduction ratio for channel dimension
        
    Returns:
        A sequential module with the conv layer and CoordAtt
    """
    out_channels = conv.out_channels
    return nn.Sequential(
        conv,
        CoordAtt(out_channels, out_channels, reduction)
    ) 