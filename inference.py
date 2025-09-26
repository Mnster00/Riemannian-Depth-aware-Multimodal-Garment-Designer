"""
Inference script for RDMGD model
"""

import torch
import argparse
from PIL import Image
import numpy as np
import os
from torchvision import transforms

from models.rdmgd import RDMGD
from utils.diffusion import DiffusionScheduler


def load_model(checkpoint_path: str, device: torch.device) -> RDMGD:
    """Load trained RDMGD model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model arguments from checkpoint
    args = checkpoint['args']
    
    # Initialize model
    model = RDMGD(
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        num_modalities=5
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    return model


def preprocess_inputs(
    image_path: str,
    text: str,
    depth_path: Optional[str] = None,
    pose_path: Optional[str] = None,
    seg_path: Optional[str] = None,
    image_size: int = 512
) -> dict:
    """Preprocess multimodal inputs for inference"""
    
    # Image preprocessing
    image = Image.open(image_path).convert('RGB')
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image_tensor = image_transform(image).unsqueeze(0)
    
    # Text encoding (simplified - in practice would use CLIP)
    text_tensor = encode_text_simple(text)
    
    # Depth preprocessing
    if depth_path and os.path.exists(depth_path):
        depth = Image.open(depth_path).convert('L')
        depth_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        depth_tensor = depth_transform(depth).unsqueeze(0)
    else:
        # Generate placeholder depth if not provided
        depth_tensor = torch.ones(1, 1, image_size, image_size) * 0.5
    
    # Pose preprocessing (simplified)
    if pose_path and os.path.exists(pose_path):
        pose_data = np.load(pose_path)
        pose_tensor = torch.from_numpy(pose_data).float().unsqueeze(0)
    else:
        # Generate placeholder pose if not provided
        pose_tensor = torch.zeros(1, 18, image_size, image_size)
    
    # Segmentation preprocessing
    if seg_path and os.path.exists(seg_path):
        seg = Image.open(seg_path)
        seg = np.array(seg.resize((image_size, image_size), Image.NEAREST))
        seg_tensor = torch.zeros(1, 20, image_size, image_size)
        for c in range(20):
            seg_tensor[0, c] = torch.from_numpy((seg == c).astype(np.float32))
    else:
        # Generate placeholder segmentation if not provided
        seg_tensor = torch.zeros(1, 20, image_size, image_size)
        seg_tensor[0, 0] = 1.0  # Background
    
    return {
        'image': image_tensor,
        'text': text_tensor,
        'pose': pose_tensor,
        'depth': depth_tensor,
        'segmentation': seg_tensor
    }


def encode_text_simple(text: str, max_length: int = 77) -> torch.Tensor:
    """Simple text encoding for demo (in practice would use CLIP)"""
    # Create a simple hash-based encoding for demonstration
    words = text.lower().split()[:max_length]
    indices = [hash(word) % 5000 for word in words]
    
    # Pad to max length
    while len(indices) < max_length:
        indices.append(0)
    
    return torch.tensor(indices[:max_length], dtype=torch.long).unsqueeze(0)


def generate_image(
    model: RDMGD,
    conditioning: dict,
    diffusion: DiffusionScheduler,
    device: torch.device,
    num_inference_steps: int = 40,
    guidance_scale: float = 8.5
) -> np.ndarray:
    """Generate image using RDMGD model"""
    
    # Move conditioning to device
    for key in conditioning:
        conditioning[key] = conditioning[key].to(device)
    
    # Sample from model
    with torch.no_grad():
        # Get image shape from input
        B, C, H, W = conditioning['image'].shape
        shape = (B, C, H, W)
        
        # Generate
        generated = diffusion.sample(
            model,
            shape,
            conditioning,
            device,
            guidance_scale=guidance_scale,
            clip_denoised=True
        )
        
        # Denormalize
        generated = (generated + 1) / 2
        generated = torch.clamp(generated, 0, 1)
        
        # Convert to numpy
        generated = generated.cpu().numpy()
        generated = (generated[0].transpose(1, 2, 0) * 255).astype(np.uint8)
    
    return generated


def main(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint_path, device)
    
    # Initialize diffusion scheduler
    diffusion = DiffusionScheduler(
        num_steps=1000,
        beta_start=0.0001,
        beta_end=0.02
    )
    
    # Preprocess inputs
    print("Preprocessing inputs...")
    conditioning = preprocess_inputs(
        image_path=args.image_path,
        text=args.text,
        depth_path=args.depth_path,
        pose_path=args.pose_path,
        seg_path=args.seg_path,
        image_size=args.image_size
    )
    
    # Generate image
    print(f"Generating image with text: '{args.text}'")
    generated_image = generate_image(
        model,
        conditioning,
        diffusion,
        device,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale
    )
    
    # Save output
    output_image = Image.fromarray(generated_image)
    output_image.save(args.output_path)
    print(f"Saved generated image to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate fashion images with RDMGD')
    
    # Required arguments
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input model image')
    parser.add_argument('--text', type=str, required=True,
                        help='Text description of desired garment')
    parser.add_argument('--output_path', type=str, default='output.png',
                        help='Path to save generated image')
    
    # Optional modality inputs
    parser.add_argument('--depth_path', type=str, default=None,
                        help='Path to depth map (optional)')
    parser.add_argument('--pose_path', type=str, default=None,
                        help='Path to pose keypoints (optional)')
    parser.add_argument('--seg_path', type=str, default=None,
                        help='Path to segmentation mask (optional)')
    
    # Generation parameters
    parser.add_argument('--image_size', type=int, default=512,
                        help='Image size')
    parser.add_argument('--num_steps', type=int, default=40,
                        help='Number of diffusion steps')
    parser.add_argument('--guidance_scale', type=float, default=8.5,
                        help='Classifier-free guidance scale')
    
    args = parser.parse_args()
    main(args)
