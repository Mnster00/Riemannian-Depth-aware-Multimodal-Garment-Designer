
# RDMGD: Riemannian Depth-aware Multimodal Garment Designer

This is a official implementation of the RDMGD framework described in the paper "Depth-Aware Fashion Synthesis via Manifold-Aligned Multi-modal Diffusion"

## Overview

RDMGD integrates multiple modalities (text, pose, depth, segmentation, and images) through Riemannian manifold geometry to generate high-quality fashion images with geometrically consistent garments.

### Key Features
- **Riemannian Manifold Alignment**: Geometrically-principled multimodal fusion
- **Depth-Aware U-Net**: Hierarchical attention based on surface geometry
- **Multi-Modal Conditioning**: Support for text, pose, depth, segmentation, and image inputs
- **Diffusion-Based Generation**: High-quality image synthesis with DDPM

## Installation

```bash
# Clone the repository
git clone https://github.com/Mnster00/Riemannian-Depth-aware-Multimodal-Garment-Designer.git
cd rdmgd-official

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
rdmgd-minimal/
├── models/
│   ├── rdmgd.py          # Main RDMGD model
│   ├── rmma.py           # Riemannian manifold alignment module
│   └── depth_unet.py     # Depth-aware U-Net architecture
├── data/
│   └── dataset.py        # Multimodal dataset implementation
├── utils/
│   └── diffusion.py      # Diffusion scheduler and utilities
├── train.py              # Training script
├── inference.py          # Inference script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Dataset Preparation

The model expects data in the following structure:
```
data_root/
├── images/              # RGB images
├── depth/               # Depth maps
├── pose/                # OpenPose keypoints
├── segmentation/        # Segmentation masks
├── train_metadata.json  # Training metadata
└── val_metadata.json    # Validation metadata
```

To create dummy metadata for testing:
```python
from data.dataset import create_dummy_metadata
create_dummy_metadata('path/to/data_root', num_samples=100)
```

## Training

```bash
python train.py \
  --data_root path/to/dataset \
  --image_size 512 \
  --batch_size 4 \
  --num_epochs 200 \
  --learning_rate 8e-6 \
  --checkpoint_dir checkpoints
```

### Training Arguments
- `--data_root`: Path to dataset root directory
- `--image_size`: Image resolution (default: 512)
- `--batch_size`: Batch size for training (default: 4)
- `--num_epochs`: Number of training epochs (default: 200)
- `--learning_rate`: Initial learning rate (default: 8e-6)
- `--diffusion_steps`: Number of diffusion steps (default: 1000)
- `--use_wandb`: Enable Weights & Biases logging

## Inference

Generate fashion images with trained model:

```bash
python inference.py \
  --checkpoint_path checkpoints/best_model.pth \
  --image_path input/model.jpg \
  --text "elegant red evening gown with embroidered details" \
  --output_path output/generated.png
```

### Inference Arguments
- `--checkpoint_path`: Path to trained model checkpoint
- `--image_path`: Input model image
- `--text`: Text description of desired garment
- `--depth_path`: (Optional) Path to depth map
- `--pose_path`: (Optional) Path to pose keypoints
- `--seg_path`: (Optional) Path to segmentation mask
- `--num_steps`: Number of diffusion steps (default: 40)
- `--guidance_scale`: Classifier-free guidance scale (default: 8.5)


