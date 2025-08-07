import os
import time
import argparse
from tqdm import tqdm
import numpy as np
import imageio # type: ignore
import glob
import cv2 # type: ignore

import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
from torch.autograd import Variable # type: ignore
import torchvision.transforms as transforms # type: ignore
from torchvision.models import vgg16 # type: ignore

from Network import DecScaleClampedIllumEdgeGuidedNetworkBatchNorm
from Utils import mor_utils

# Set device
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('[*] GPU Device selected as default execution device.')
    else:
        device = torch.device('cpu')
        print('[X] WARN: No GPU Devices found on the system! Using the CPU.')
    return device

class PIENetDataset(Dataset):
    """Dataset class for PIE-Net training"""
    
    def __init__(self, data_root, split='train', img_size=256):
        self.data_root = data_root
        self.img_size = img_size
        self.split = split
        
        # Get list of image files
        self.image_files = self._get_image_files()
        
        # Initialize edge detector
        self.edge_detector = cv2.Canny
        
    def _get_image_files(self):
        """Get list of image files based on dataset structure"""
        # Adjust this based on your dataset structure
        image_dir = os.path.join(self.data_root, self.split, 'images')
        image_files = glob.glob(os.path.join(image_dir, '*.png')) + \
                     glob.glob(os.path.join(image_dir, '*.jpg'))
        return sorted(image_files)
    
    def _load_image(self, path):
        """Load and preprocess image"""
        img = imageio.imread(path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        
        # Handle grayscale images
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        
        # Convert to CHW format
        img = img.transpose((2, 0, 1))
        return img
    
    def _compute_edges(self, image):
        """Compute edges using Canny edge detector"""
        # Convert to grayscale for edge detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        # Apply Canny edge detection
        edges = self.edge_detector(gray, 50, 150)
        edges = edges.astype(np.float32) / 255.0
        
        # Convert to 3-channel
        edges = np.stack([edges, edges, edges], axis=0)
        return edges
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load RGB image
        img_path = self.image_files[idx]
        rgb_img = self._load_image(img_path)
        
        # Load ground truth albedo and shading (adjust paths based on your dataset)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Assuming dataset structure: images/, albedo/, shading/
        albedo_path = img_path.replace('/images/', '/albedo/')
        shading_path = img_path.replace('/images/', '/shading/')
        
        try:
            albedo_gt = self._load_image(albedo_path)
            shading_gt = self._load_image(shading_path)
            
            # For shading, convert to single channel
            if shading_gt.shape[0] == 3:
                shading_gt = np.mean(shading_gt, axis=0, keepdims=True)
        except:
            # If ground truth not available, create dummy data
            print(f"Warning: Ground truth not found for {img_path}")
            albedo_gt = rgb_img.copy()
            shading_gt = np.ones((1, self.img_size, self.img_size), dtype=np.float32)
        
        # Compute edges from ground truth
        albedo_edges = self._compute_edges(albedo_gt.transpose(1, 2, 0))
        shading_edges = self._compute_edges(shading_gt.squeeze())
        
        return {
            'rgb': torch.from_numpy(rgb_img).float(),
            'albedo_gt': torch.from_numpy(albedo_gt).float(),
            'shading_gt': torch.from_numpy(shading_gt).float(),
            'albedo_edges_gt': torch.from_numpy(albedo_edges).float(),
            'shading_edges_gt': torch.from_numpy(shading_edges).float(),
            'filename': base_name
        }

class PIENetLoss(nn.Module):
    """Combined loss function for PIE-Net"""
    
    def __init__(self, device, lambda_u=0.5, lambda_d=0.4, lambda_e=0.4, lambda_p=0.05,
                 lambda_smse=0.95, lambda_mse=0.05):
        super(PIENetLoss, self).__init__()
        self.device = device
        self.lambda_u = lambda_u
        self.lambda_d = lambda_d
        self.lambda_e = lambda_e
        self.lambda_p = lambda_p
        self.lambda_smse = lambda_smse
        self.lambda_mse = lambda_mse
        
        # Initialize VGG for perceptual loss
        self.vgg = vgg16(pretrained=True).features[:16].to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def scale_invariant_mse(self, pred, target):
        """Scale-invariant MSE loss"""
        # Flatten tensors
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # Compute scale-invariant loss
        diff = pred_flat - target_flat
        diff_mean = torch.mean(diff, dim=1, keepdim=True)
        
        loss = torch.mean(diff**2, dim=1) - torch.mean(diff_mean**2)
        return torch.mean(loss)
    
    def combined_mse_loss(self, pred, target):
        """Combination of scale-invariant MSE and standard MSE"""
        smse = self.scale_invariant_mse(pred, target)
        mse = self.mse_loss(pred, target)
        return self.lambda_smse * smse + self.lambda_mse * mse
    
    def perceptual_loss(self, pred, target):
        """Perceptual loss using VGG features"""
        # Ensure 3-channel input
        if pred.size(1) == 1:
            pred = pred.repeat(1, 3, 1, 1)
        if target.size(1) == 1:
            target = target.repeat(1, 3, 1, 1)
            
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        return self.l1_loss(pred_features, target_features)
    
    def dssim_loss(self, pred, target):
        """Structural dissimilarity loss"""
        return 1 - self.ssim(pred, target)
    
    def ssim(self, pred, target, window_size=11):
        """Compute SSIM"""
        mu1 = F.avg_pool2d(pred, window_size, 1, window_size//2)
        mu2 = F.avg_pool2d(target, window_size, 1, window_size//2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(pred * pred, window_size, 1, window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(target * target, window_size, 1, window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(pred * target, window_size, 1, window_size//2) - mu1_mu2
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return ssim_map.mean()
    
    def forward(self, predictions, targets):
        """Compute total loss"""
        total_loss = 0.0
        loss_dict = {}
        
        # Edge losses (Le)
        edge_loss = 0.0
        
        # Reflectance edges at multiple scales
        if 'reflec_edge' in predictions:
            edge_loss += self.combined_mse_loss(predictions['reflec_edge'], 
                                              targets['albedo_edges_gt'])
        if 'reflec_edge_64' in predictions:
            target_64 = F.interpolate(targets['albedo_edges_gt'], size=(64, 64))
            edge_loss += self.combined_mse_loss(predictions['reflec_edge_64'], target_64)
        if 'reflec_edge_128' in predictions:
            target_128 = F.interpolate(targets['albedo_edges_gt'], size=(128, 128))
            edge_loss += self.combined_mse_loss(predictions['reflec_edge_128'], target_128)
        
        # Shading edges at multiple scales
        if 'illum_edge' in predictions:
            edge_loss += self.combined_mse_loss(predictions['illum_edge'], 
                                              targets['shading_edges_gt'])
        if 'illum_edge_64' in predictions:
            target_64 = F.interpolate(targets['shading_edges_gt'], size=(64, 64))
            edge_loss += self.combined_mse_loss(predictions['illum_edge_64'], target_64)
        if 'illum_edge_128' in predictions:
            target_128 = F.interpolate(targets['shading_edges_gt'], size=(128, 128))
            edge_loss += self.combined_mse_loss(predictions['illum_edge_128'], target_128)
        
        loss_dict['edge_loss'] = edge_loss
        total_loss += self.lambda_e * edge_loss
        
        # Unrefined losses (Lu)
        unrefined_loss = 0.0
        if 'unrefined_reflec' in predictions:
            unrefined_loss += self.combined_mse_loss(predictions['unrefined_reflec'], 
                                                   targets['albedo_gt'])
        if 'unrefined_shd' in predictions:
            unrefined_loss += self.combined_mse_loss(predictions['unrefined_shd'], 
                                                   targets['shading_gt'])
        
        loss_dict['unrefined_loss'] = unrefined_loss
        total_loss += self.lambda_u * unrefined_loss
        
        # Refined losses (Lr)
        refined_loss = 0.0
        if 'reflectance' in predictions:
            refined_loss += self.combined_mse_loss(predictions['reflectance'], 
                                                 targets['albedo_gt'])
        if 'shading' in predictions:
            refined_loss += self.combined_mse_loss(predictions['shading'], 
                                                 targets['shading_gt'])
        
        loss_dict['refined_loss'] = refined_loss
        total_loss += refined_loss
        
        # Reconstruction loss (Lrec)
        if 'recon' in predictions:
            recon_loss = self.combined_mse_loss(predictions['recon'], targets['rgb'])
            loss_dict['recon_loss'] = recon_loss
            total_loss += recon_loss
        
        # DSSIM loss
        dssim_loss = 0.0
        if 'reflectance' in predictions:
            dssim_loss += self.dssim_loss(predictions['reflectance'], targets['albedo_gt'])
        if 'shading' in predictions:
            dssim_loss += self.dssim_loss(predictions['shading'], targets['shading_gt'])
        
        loss_dict['dssim_loss'] = dssim_loss
        total_loss += self.lambda_d * dssim_loss
        
        # Perceptual loss (only for reflectance)
        perceptual_loss = 0.0
        if 'reflectance' in predictions:
            perceptual_loss = self.perceptual_loss(predictions['reflectance'], 
                                                 targets['albedo_gt'])
        
        loss_dict['perceptual_loss'] = perceptual_loss
        total_loss += self.lambda_p * perceptual_loss
        
        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_losses = {}
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        rgb = batch['rgb'].to(device)
        targets = {
            'albedo_gt': batch['albedo_gt'].to(device),
            'shading_gt': batch['shading_gt'].to(device),
            'albedo_edges_gt': batch['albedo_edges_gt'].to(device),
            'shading_edges_gt': batch['shading_edges_gt'].to(device),
            'rgb': rgb
        }
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(rgb)
        
        # Compute loss
        loss, loss_dict = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update running statistics
        running_loss += loss.item()
        for key, value in loss_dict.items():
            if key not in running_losses:
                running_losses[key] = 0.0
            running_losses[key] += value.item() if hasattr(value, 'item') else value
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Compute average losses
    avg_loss = running_loss / len(dataloader)
    avg_losses = {key: value / len(dataloader) for key, value in running_losses.items()}
    
    return avg_loss, avg_losses

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    running_losses = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            # Move data to device
            rgb = batch['rgb'].to(device)
            targets = {
                'albedo_gt': batch['albedo_gt'].to(device),
                'shading_gt': batch['shading_gt'].to(device),
                'albedo_edges_gt': batch['albedo_edges_gt'].to(device),
                'shading_edges_gt': batch['shading_edges_gt'].to(device),
                'rgb': rgb
            }
            
            # Forward pass
            predictions = model(rgb)
            
            # Compute loss
            loss, loss_dict = criterion(predictions, targets)
            
            # Update running statistics
            running_loss += loss.item()
            for key, value in loss_dict.items():
                if key not in running_losses:
                    running_losses[key] = 0.0
                running_losses[key] += value.item() if hasattr(value, 'item') else value
    
    # Compute average losses
    avg_loss = running_loss / len(dataloader)
    avg_losses = {key: value / len(dataloader) for key, value in running_losses.items()}
    
    return avg_loss, avg_losses

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='PIE-Net Training')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of the dataset')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Frequency of saving checkpoints (epochs)')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Get device
    device = get_device()
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = PIENetDataset(args.data_root, split='train')
    val_dataset = PIENetDataset(args.data_root, split='val')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    print("Initializing model...")
    model = DecScaleClampedIllumEdgeGuidedNetworkBatchNorm().to(device)
    
    # Create loss function
    criterion = PIENetLoss(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Create utils
    utils = mor_utils(device)
    
    # Resume training if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming training from {args.resume}")
        model, optimizer, start_epoch = utils.loadModels(model, args.resume, optimizer, Test=False)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_loss, train_losses = train_epoch(model, train_loader, criterion, optimizer, device, epoch+1)
        
        # Validate
        val_loss, val_losses = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Print statistics
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train Losses: {train_losses}")
        print(f"Val Losses: {val_losses}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or val_loss < best_val_loss:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.t7')
            utils.saveModels(model, optimizer, epoch+1, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = os.path.join(args.save_dir, 'best_model.t7')
                utils.saveModels(model, optimizer, epoch+1, best_checkpoint_path)
                print(f"Best model saved: {best_checkpoint_path}")
    
    print("Training completed!")

if __name__ == '__main__':
    main()
