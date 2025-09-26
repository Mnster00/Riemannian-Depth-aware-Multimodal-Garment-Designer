"""
Fashion Multimodal Dataset with depth, pose, segmentation, and text
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import json
import os
from typing import Dict, Optional
import torchvision.transforms as transforms


class FashionMultimodalDataset(Dataset):
    """Dataset for multimodal fashion synthesis"""
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        image_size: int = 512,
        max_text_length: int = 77
    ):
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.max_text_length = max_text_length
        
        # Load metadata
        metadata_path = os.path.join(data_root, f'{split}_metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Define transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.depth_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        
        # Simple vocabulary for text encoding (in practice would use CLIP tokenizer)
        self.vocab = self._build_vocab()
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get multimodal sample
        Returns:
            Dict containing image, text, pose, depth, segmentation tensors
        """
        sample_info = self.metadata[idx]
        
        # Load image
        image_path = os.path.join(self.data_root, 'images', sample_info['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.image_transform(image)
        
        # Load depth map
        depth_path = os.path.join(self.data_root, 'depth', sample_info['depth'])
        depth = Image.open(depth_path).convert('L')
        depth = self.depth_transform(depth)
        
        # Load pose (simplified - in practice would load OpenPose keypoints)
        pose_path = os.path.join(self.data_root, 'pose', sample_info['pose'])
        pose = self._load_pose(pose_path)
        
        # Load segmentation
        seg_path = os.path.join(self.data_root, 'segmentation', sample_info['segmentation'])
        segmentation = self._load_segmentation(seg_path)
        
        # Process text
        text = self._encode_text(sample_info['text'])
        
        return {
            'image': image,
            'text': text,
            'pose': pose,
            'depth': depth,
            'segmentation': segmentation
        }
    
    def _load_pose(self, pose_path: str) -> torch.Tensor:
        """Load pose keypoints as 18-channel tensor"""
        # Simplified version - in practice would parse OpenPose JSON
        if os.path.exists(pose_path):
            pose_data = np.load(pose_path)
            pose_tensor = torch.from_numpy(pose_data).float()
            # Ensure correct shape [18, H, W]
            if pose_tensor.dim() == 2:
                pose_tensor = pose_tensor.unsqueeze(0).repeat(18, 1, 1)
        else:
            # Return random pose if file not found (for demo)
            pose_tensor = torch.randn(18, self.image_size, self.image_size) * 0.1
        
        return pose_tensor
    
    def _load_segmentation(self, seg_path: str) -> torch.Tensor:
        """Load segmentation mask as multi-channel tensor"""
        if os.path.exists(seg_path):
            seg = Image.open(seg_path)
            seg = np.array(seg.resize((self.image_size, self.image_size), Image.NEAREST))
            
            # Convert to one-hot encoding (20 classes)
            num_classes = 20
            seg_tensor = torch.zeros(num_classes, self.image_size, self.image_size)
            for c in range(num_classes):
                seg_tensor[c] = torch.from_numpy((seg == c).astype(np.float32))
        else:
            # Return random segmentation if file not found (for demo)
            seg_tensor = torch.zeros(20, self.image_size, self.image_size)
            seg_tensor[0] = 1.0  # Background class
        
        return seg_tensor
    
    def _encode_text(self, text: str) -> torch.Tensor:
        """Simple text encoding (in practice would use CLIP tokenizer)"""
        words = text.lower().split()[:self.max_text_length]
        
        # Convert words to indices
        indices = []
        for word in words:
            if word in self.vocab:
                indices.append(self.vocab[word])
            else:
                indices.append(self.vocab['<unk>'])  # Unknown token
        
        # Pad to max length
        while len(indices) < self.max_text_length:
            indices.append(self.vocab['<pad>'])
        
        return torch.tensor(indices[:self.max_text_length], dtype=torch.long)
    
    def _build_vocab(self) -> Dict[str, int]:
        """Build simple vocabulary from metadata"""
        vocab = {'<pad>': 0, '<unk>': 1}
        idx = 2
        
        # Collect all unique words
        all_words = set()
        for sample in self.metadata:
            words = sample.get('text', '').lower().split()
            all_words.update(words)
        
        # Assign indices
        for word in sorted(all_words):
            vocab[word] = idx
            idx += 1
        
        return vocab


def create_dummy_metadata(data_root: str, num_samples: int = 100):
    """Create dummy metadata for testing"""
    os.makedirs(data_root, exist_ok=True)
    
    # Sample fashion descriptions
    descriptions = [
        "elegant red evening gown with embroidered details",
        "casual denim jacket with white t-shirt",
        "formal black suit with white shirt and tie",
        "floral summer dress with cinched waist",
        "athletic wear with blue shorts and tank top",
        "vintage leather jacket with ripped jeans",
        "bohemian maxi dress with paisley print",
        "minimalist white blouse with black trousers",
        "cozy knit sweater with plaid skirt",
        "streetwear hoodie with cargo pants"
    ]
    
    metadata = []
    for i in range(num_samples):
        metadata.append({
            'image': f'img_{i:05d}.jpg',
            'depth': f'depth_{i:05d}.png',
            'pose': f'pose_{i:05d}.npy',
            'segmentation': f'seg_{i:05d}.png',
            'text': descriptions[i % len(descriptions)]
        })
    
    # Save metadata
    for split in ['train', 'val', 'test']:
        split_metadata = metadata if split == 'train' else metadata[:20]
        with open(os.path.join(data_root, f'{split}_metadata.json'), 'w') as f:
            json.dump(split_metadata, f, indent=2)
    
    # Create subdirectories
    for subdir in ['images', 'depth', 'pose', 'segmentation']:
        os.makedirs(os.path.join(data_root, subdir), exist_ok=True)
    
    print(f"Created dummy metadata at {data_root}")
