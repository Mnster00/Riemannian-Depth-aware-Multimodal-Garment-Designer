"""
Depth-Aware U-Net with Hierarchical Geometric Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class DepthAwareUNet(nn.Module):
    """U-Net with depth-aware attention at multiple scales"""
    
    def __init__(
        self,
        in_channels: int = 512,
        depth_levels: List[int] = [64, 32, 16, 8],
        gamma1: float = 0.3,
        gamma2: float = 0.1
    ):
        super().__init__()
        self.depth_levels = depth_levels
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        
        # Encoder blocks
        self.enc1 = UNetBlock(in_channels, 256, down=True)
        self.enc2 = UNetBlock(256, 128, down=True)
        self.enc3 = UNetBlock(128, 64, down=True)
        self.enc4 = UNetBlock(64, 32, down=True)
        
        # Bottleneck
        self.bottleneck = UNetBlock(32, 32, down=False)
        
        # Decoder blocks with skip connections
        self.dec1 = UNetBlock(32 + 32, 64, down=False)
        self.dec2 = UNetBlock(64 + 64, 128, down=False)
        self.dec3 = UNetBlock(128 + 128, 256, down=False)
        self.dec4 = UNetBlock(256 + 256, in_channels, down=False)
        
        # Depth attention modules for each level
        self.depth_attentions = nn.ModuleList([
            DepthAttention(channels, level_size, gamma1, gamma2)
            for channels, level_size in zip([256, 128, 64, 32], depth_levels)
        ])
    
    def forward(self, x: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with depth-aware attention
        Args:
            x: [B, C, H, W] input features
            depth: [B, 1, H_orig, W_orig] depth map
        """
        # Encoder path with depth attention
        enc_features = []
        
        e1 = self.enc1(x)
        e1 = self.depth_attentions[0](e1, depth)
        enc_features.append(e1)
        
        e2 = self.enc2(e1)
        e2 = self.depth_attentions[1](e2, depth)
        enc_features.append(e2)
        
        e3 = self.enc3(e2)
        e3 = self.depth_attentions[2](e3, depth)
        enc_features.append(e3)
        
        e4 = self.enc4(e3)
        e4 = self.depth_attentions[3](e4, depth)
        enc_features.append(e4)
        
        # Bottleneck
        b = self.bottleneck(e4)
        
        # Decoder path with skip connections
        d1 = self.dec1(torch.cat([b, enc_features[3]], dim=1))
        d2 = self.dec2(torch.cat([d1, enc_features[2]], dim=1))
        d3 = self.dec3(torch.cat([d2, enc_features[1]], dim=1))
        d4 = self.dec4(torch.cat([d3, enc_features[0]], dim=1))
        
        return d4


class UNetBlock(nn.Module):
    """Basic U-Net block with optional downsampling"""
    
    def __init__(self, in_channels: int, out_channels: int, down: bool = False):
        super().__init__()
        self.down = down
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.norm1 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.activation = nn.ReLU(inplace=True)
        
        if down:
            self.downsample = nn.MaxPool2d(2)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First conv block
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)
        
        # Down/up sampling
        if self.down:
            out = self.downsample(out)
        elif hasattr(self, 'upsample'):
            out = self.upsample(out)
        
        return out


class DepthAttention(nn.Module):
    """Depth-based attention mechanism with geometric bias"""
    
    def __init__(
        self,
        channels: int,
        spatial_size: int,
        gamma1: float = 0.3,
        gamma2: float = 0.1
    ):
        super().__init__()
        self.channels = channels
        self.spatial_size = spatial_size
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        
        # Query, Key, Value projections
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(1, channels // 8, 1)  # Depth has 1 channel
        self.value = nn.Conv2d(channels, channels, 1)
        
        self.scale = (channels // 8) ** -0.5
        self.out_proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, features: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        Apply depth-aware attention
        Args:
            features: [B, C, H, W] feature map
            depth: [B, 1, H_orig, W_orig] depth map
        """
        B, C, H, W = features.shape
        
        # Resize depth to match feature resolution
        depth_resized = F.interpolate(depth, size=(H, W), mode='bilinear', align_corners=False)
        
        # Compute geometric bias from depth
        geo_bias = self.compute_geometric_bias(depth_resized)
        
        # Compute attention
        q = self.query(features).view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, C/8]
        k = self.key(depth_resized).view(B, -1, H * W)  # [B, C/8, HW]
        v = self.value(features).view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        
        # Attention scores with geometric bias
        attn = torch.bmm(q, k) * self.scale  # [B, HW, HW]
        attn = attn + geo_bias.view(B, 1, H * W).expand(-1, H * W, -1)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.bmm(attn, v)  # [B, HW, C]
        out = out.permute(0, 2, 1).view(B, C, H, W)
        out = self.out_proj(out)
        
        # Residual connection
        return features + out
    
    def compute_geometric_bias(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Compute geometric bias based on depth gradients and curvature
        Args:
            depth: [B, 1, H, W] depth map
        Returns:
            Geometric bias [B, H*W]
        """
        B, _, H, W = depth.shape
        
        # Compute depth gradients (surface normals approximation)
        grad_x = torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1])
        grad_y = torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :])
        
        # Pad to maintain dimensions
        grad_x = F.pad(grad_x, (0, 1), mode='replicate')
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')
        
        grad_norm = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        
        # Compute approximate curvature (Laplacian)
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 
                            dtype=depth.dtype, device=depth.device).view(1, 1, 3, 3)
        curvature = F.conv2d(depth, kernel, padding=1)
        curvature = torch.abs(curvature)
        
        # Combine gradient and curvature
        geometric_bias = self.gamma1 * grad_norm + self.gamma2 * curvature
        
        return geometric_bias.squeeze(1).view(B, -1)
