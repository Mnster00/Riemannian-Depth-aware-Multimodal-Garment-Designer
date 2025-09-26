"""
Riemannian Manifold Multi-Modal Alignment (RMMA) Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List


class RiemannianManifoldAlignment(nn.Module):
    """Riemannian Manifold Multi-Modal Alignment module"""
    
    def __init__(
        self,
        num_modalities: int = 5,
        manifold_dim: int = 256,
        latent_dim: int = 512,
        curvature_weight: float = 5e-4
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.manifold_dim = manifold_dim
        self.latent_dim = latent_dim
        self.curvature_weight = curvature_weight
        
        # Manifold embeddings for each modality
        self.manifold_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(latent_dim, manifold_dim, 1),
                nn.GroupNorm(32, manifold_dim),
                nn.ReLU(),
                nn.Conv2d(manifold_dim, manifold_dim, 1)
            ) for _ in range(num_modalities)
        ])
        
        # Metric tensors (learnable)
        self.metric_tensors = nn.ParameterList([
            nn.Parameter(torch.eye(manifold_dim) * 0.1 + torch.randn(manifold_dim, manifold_dim) * 0.01)
            for _ in range(num_modalities)
        ])
        
        # Cross-manifold attention
        self.cross_attention = ManifoldCrossAttention(manifold_dim)
        
        # Output projection
        self.output_proj = nn.Conv2d(manifold_dim * num_modalities, latent_dim, 1)
        
        self._alignment_loss = 0.0
    
    def forward(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        """
        Align multi-modal features through Riemannian geometry
        Args:
            modalities: List of tensors [B, C, H, W] for each modality
        Returns:
            Aligned features [B, C, H, W]
        """
        B, _, H, W = modalities[0].shape
        
        # Project to manifolds
        manifold_features = []
        for i, (modality, embedding) in enumerate(zip(modalities, self.manifold_embeddings)):
            # Ensure spatial dimensions match
            if modality.shape[-2:] != (H, W):
                modality = F.interpolate(modality, size=(H, W), mode='bilinear', align_corners=False)
            manifold_feat = embedding(modality)
            manifold_features.append(manifold_feat)
        
        # Compute geodesic distances and alignment
        aligned_features = []
        total_distance = 0.0
        
        for i in range(self.num_modalities):
            # Compute geodesic distance to other modalities
            distances = []
            for j in range(self.num_modalities):
                if i != j:
                    dist = self.geodesic_distance(
                        manifold_features[i], 
                        manifold_features[j],
                        self.metric_tensors[i],
                        self.metric_tensors[j]
                    )
                    distances.append(dist)
                    total_distance += dist
            
            # Parallel transport to reference manifold
            transported = self.parallel_transport(manifold_features[i], self.metric_tensors[i])
            aligned_features.append(transported)
        
        # Store alignment loss for training
        self._alignment_loss = total_distance / (self.num_modalities * (self.num_modalities - 1))
        
        # Cross-manifold attention
        attended_features = self.cross_attention(aligned_features)
        
        # Concatenate and project
        concatenated = torch.cat(attended_features, dim=1)
        output = self.output_proj(concatenated)
        
        # Add Ricci curvature regularization
        ricci_loss = self.compute_ricci_curvature()
        self._alignment_loss += self.curvature_weight * ricci_loss
        
        return output
    
    def geodesic_distance(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        g1: torch.Tensor,
        g2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute geodesic distance between features on different manifolds
        Simplified implementation using Euclidean approximation with metric tensor
        """
        # Flatten spatial dimensions
        B, C, H, W = x1.shape
        x1_flat = x1.view(B, C, -1).mean(dim=-1)  # [B, C]
        x2_flat = x2.view(B, C, -1).mean(dim=-1)  # [B, C]
        
        # Apply metric tensors
        x1_metric = torch.matmul(x1_flat, g1)  # [B, C]
        x2_metric = torch.matmul(x2_flat, g2)  # [B, C]
        
        # Compute distance
        distance = torch.norm(x1_metric - x2_metric, p=2, dim=1).mean()
        
        return distance
    
    def parallel_transport(
        self,
        features: torch.Tensor,
        metric: torch.Tensor
    ) -> torch.Tensor:
        """
        Parallel transport features along geodesic
        Simplified implementation
        """
        B, C, H, W = features.shape
        
        # Apply metric tensor transformation
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        transported = torch.matmul(features_flat, metric)
        transported = transported.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        return transported
    
    def compute_ricci_curvature(self) -> torch.Tensor:
        """
        Compute Ricci curvature for regularization
        Simplified scalar curvature computation
        """
        ricci_sum = 0.0
        for metric in self.metric_tensors:
            # Simplified Ricci scalar as trace of metric tensor deviation from identity
            deviation = metric - torch.eye(metric.shape[0], device=metric.device)
            ricci_sum += torch.trace(torch.matmul(deviation, deviation.T))
        
        return ricci_sum / self.num_modalities
    
    def compute_alignment_loss(self) -> torch.Tensor:
        """Return the computed alignment loss"""
        return self._alignment_loss


class ManifoldCrossAttention(nn.Module):
    """Cross-attention mechanism respecting manifold geometry"""
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Conv2d(dim, dim, 1)
        self.k_proj = nn.Conv2d(dim, dim, 1)
        self.v_proj = nn.Conv2d(dim, dim, 1)
        self.out_proj = nn.Conv2d(dim, dim, 1)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply cross-attention between manifold features
        Args:
            features: List of [B, C, H, W] tensors
        Returns:
            List of attended features
        """
        attended = []
        
        for i, feat in enumerate(features):
            B, C, H, W = feat.shape
            
            # Compute Q from current modality
            q = self.q_proj(feat).view(B, self.num_heads, self.head_dim, H*W)
            q = q.permute(0, 1, 3, 2)  # [B, heads, HW, head_dim]
            
            # Compute K, V from all modalities
            keys = []
            values = []
            for f in features:
                k = self.k_proj(f).view(B, self.num_heads, self.head_dim, H*W)
                v = self.v_proj(f).view(B, self.num_heads, self.head_dim, H*W)
                keys.append(k)
                values.append(v)
            
            keys = torch.cat(keys, dim=-1)  # [B, heads, head_dim, HW*num_modalities]
            values = torch.cat(values, dim=-1)  # [B, heads, head_dim, HW*num_modalities]
            
            # Attention
            attn = torch.matmul(q, keys) * self.scale  # [B, heads, HW, HW*num_modalities]
            attn = F.softmax(attn, dim=-1)
            
            # Apply attention
            out = torch.matmul(attn, values.permute(0, 1, 3, 2))  # [B, heads, HW, head_dim]
            out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
            out = self.out_proj(out)
            
            # Residual connection
            attended.append(feat + out)
        
        return attended
