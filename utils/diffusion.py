"""
Diffusion scheduler and utilities for RDMGD
"""

import torch
import numpy as np
from typing import Optional, Tuple


class DiffusionScheduler:
    """DDPM diffusion scheduler for training and inference"""
    
    def __init__(
        self,
        num_steps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = 'linear'
    ):
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Create beta schedule
        if schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_steps)
        elif schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(num_steps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        # Pre-compute useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_beta_schedule(self, num_steps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule as proposed in Improved DDPM"""
        steps = num_steps + 1
        x = torch.linspace(0, num_steps, steps)
        alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise(
        self,
        x_start: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Add noise to x_start for given timesteps"""
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps for training"""
        return torch.randint(0, self.num_steps, (batch_size,), device=device, dtype=torch.long)
    
    def p_mean_variance(
        self,
        model_output: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mean and variance for p(x_{t-1} | x_t)
        """
        batch_size = x.shape[0]
        
        # Extract values
        posterior_mean_coef1 = self._extract(self.posterior_mean_coef1, t, x.shape)
        posterior_mean_coef2 = self._extract(self.posterior_mean_coef2, t, x.shape)
        posterior_variance = self._extract(self.posterior_variance, t, x.shape)
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, t, x.shape)
        
        # Compute x_0 prediction
        if model_output.shape[1] == x.shape[1] * 2:
            # Model predicts both mean and variance
            model_mean, model_log_variance = torch.split(model_output, x.shape[1], dim=1)
        else:
            # Model predicts epsilon (noise)
            sqrt_recip_alphas_t = self._extract(torch.sqrt(1.0 / self.alphas), t, x.shape)
            sqrt_recipm1_alphas_t = self._extract(
                torch.sqrt(1.0 / self.alphas - 1.0), t, x.shape
            )
            model_mean = sqrt_recip_alphas_t * (x - sqrt_recipm1_alphas_t * model_output)
            model_log_variance = posterior_log_variance
        
        if clip_denoised:
            model_mean = torch.clamp(model_mean, -1, 1)
        
        return model_mean, posterior_variance, model_log_variance
    
    def p_sample(
        self,
        model_output: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True
    ) -> torch.Tensor:
        """Sample x_{t-1} from p(x_{t-1} | x_t)"""
        model_mean, _, model_log_variance = self.p_mean_variance(
            model_output, x, t, clip_denoised
        )
        
        noise = torch.randn_like(x)
        # No noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
    
    @torch.no_grad()
    def sample(
        self,
        model,
        shape: Tuple[int, ...],
        conditioning: dict,
        device: torch.device,
        guidance_scale: float = 1.0,
        clip_denoised: bool = True
    ) -> torch.Tensor:
        """
        Generate samples from the model using DDPM sampling
        Args:
            model: The RDMGD model
            shape: Shape of samples to generate
            conditioning: Dict containing conditioning inputs
            device: Device to run on
            guidance_scale: Classifier-free guidance scale
            clip_denoised: Whether to clip denoised samples
        """
        batch_size = shape[0]
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        for t in reversed(range(self.num_steps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Get model prediction
            model_output = model(
                img,
                conditioning['text'],
                conditioning['pose'],
                conditioning['depth'],
                conditioning['segmentation'],
                t,
                noise=None  # No noise during inference
            )
            
            # Classifier-free guidance
            if guidance_scale > 1.0:
                # Get unconditional prediction
                uncond_output = model(
                    img,
                    torch.zeros_like(conditioning['text']),
                    torch.zeros_like(conditioning['pose']),
                    torch.zeros_like(conditioning['depth']),
                    torch.zeros_like(conditioning['segmentation']),
                    t,
                    noise=None
                )
                
                # Apply guidance
                model_output = uncond_output + guidance_scale * (model_output - uncond_output)
            
            # Sample x_{t-1}
            img = self.p_sample(model_output, img, t_batch, clip_denoised)
        
        return img
    
    def _extract(self, values: torch.Tensor, t: torch.Tensor, shape: Tuple) -> torch.Tensor:
        """Extract values at timesteps t and reshape to broadcast shape"""
        batch_size = t.shape[0]
        out = values.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(shape) - 1)))
