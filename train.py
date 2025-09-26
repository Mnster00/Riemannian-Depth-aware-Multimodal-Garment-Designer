"""
Training script for RDMGD model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import os
import wandb

from models.rdmgd import RDMGD
from data.dataset import FashionMultimodalDataset
from utils.diffusion import DiffusionScheduler


def train_epoch(model, dataloader, optimizer, scheduler, diffusion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        image = batch['image'].to(device)
        text = batch['text'].to(device)
        pose = batch['pose'].to(device)
        depth = batch['depth'].to(device)
        segmentation = batch['segmentation'].to(device)
        
        # Sample timestep and noise
        B = image.shape[0]
        t = torch.randint(0, diffusion.num_steps, (B,), device=device)
        noise = torch.randn_like(image)
        
        # Forward pass
        pred = model(image, text, pose, depth, segmentation, t[0], noise)
        
        # Compute loss
        losses = model.compute_loss(pred, image, depth)
        loss = losses['total']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Update progress bar
        total_loss += loss.item()
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.6f}"
        })
    
    return total_loss / len(dataloader)


def validate(model, dataloader, diffusion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            image = batch['image'].to(device)
            text = batch['text'].to(device)
            pose = batch['pose'].to(device)
            depth = batch['depth'].to(device)
            segmentation = batch['segmentation'].to(device)
            
            # Sample timestep
            B = image.shape[0]
            t = torch.randint(0, diffusion.num_steps, (B,), device=device)
            noise = torch.randn_like(image)
            
            # Forward pass
            pred = model(image, text, pose, depth, segmentation, t[0], noise)
            
            # Compute loss
            losses = model.compute_loss(pred, image, depth)
            total_loss += losses['total'].item()
    
    return total_loss / len(dataloader)


def main(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(project="rdmgd", config=args)
    
    # Create datasets
    train_dataset = FashionMultimodalDataset(
        data_root=args.data_root,
        split='train',
        image_size=args.image_size
    )
    val_dataset = FashionMultimodalDataset(
        data_root=args.data_root,
        split='val',
        image_size=args.image_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize model
    model = RDMGD(
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        num_modalities=5
    ).to(device)
    
    # Initialize diffusion scheduler
    diffusion = DiffusionScheduler(
        num_steps=args.diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end
    )
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs * len(train_loader),
        eta_min=1e-6
    )
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, diffusion, device, epoch
        )
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        if epoch % args.val_freq == 0:
            val_loss = validate(model, val_loader, diffusion, device)
            print(f"Val Loss: {val_loss:.4f}")
            
            # Log to wandb
            if args.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'lr': scheduler.get_last_lr()[0]
                })
            
            # Save checkpoint if best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'args': args
                }
                torch.save(
                    checkpoint,
                    os.path.join(args.checkpoint_dir, 'best_model.pth')
                )
                print(f"Saved best model with val_loss: {val_loss:.4f}")
        
        # Regular checkpoint
        if epoch % args.save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'args': args
            }
            torch.save(
                checkpoint,
                os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            )
    
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RDMGD model')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to dataset root')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Image size for training')
    
    # Model arguments
    parser.add_argument('--latent_dim', type=int, default=512,
                        help='Latent dimension')
    parser.add_argument('--diffusion_steps', type=int, default=1000,
                        help='Number of diffusion steps')
    parser.add_argument('--beta_start', type=float, default=0.0001,
                        help='Start beta for diffusion')
    parser.add_argument('--beta_end', type=float, default=0.02,
                        help='End beta for diffusion')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=8e-6,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--val_freq', type=int, default=5,
                        help='Validate every N epochs')
    
    # Other arguments
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    
    args = parser.parse_args()
    main(args)
