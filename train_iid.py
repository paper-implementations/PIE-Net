import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
from tqdm import tqdm
import time
from pathlib import Path

from unified_dataset import UnifiedIIDDataset, create_iid_data_loaders
from Network import DecScaleClampedIllumEdgeGuidedNetworkBatchNorm


class IIDLoss(nn.Module):
    """Loss functions for Intrinsic Image Decomposition"""
    
    def __init__(self, lambda_recon=1.0, lambda_albedo=1.0, lambda_shading=1.0, lambda_smoothness=0.1):
        super(IIDLoss, self).__init__()
        self.lambda_recon = lambda_recon
        self.lambda_albedo = lambda_albedo
        self.lambda_shading = lambda_shading
        self.lambda_smoothness = lambda_smoothness
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Dict with 'reflectance', 'shading', 'recon'
            targets: Dict with 'albedo', 'shading', 'rgb'
        """
        total_loss = 0.0
        loss_dict = {}
        
        # Reconstruction loss (I = R * S)
        if 'recon' in predictions and 'rgb' in targets:
            recon_loss = self.l1_loss(predictions['recon'], targets['rgb'])
            total_loss += self.lambda_recon * recon_loss
            loss_dict['recon_loss'] = recon_loss.item()
        
        # Albedo/Reflectance loss
        if 'reflectance' in predictions and 'albedo' in targets:
            albedo_loss = self.l1_loss(predictions['reflectance'], targets['albedo'])
            total_loss += self.lambda_albedo * albedo_loss
            loss_dict['albedo_loss'] = albedo_loss.item()
        
        # Shading loss
        if 'shading' in predictions and 'shading' in targets:
            shading_loss = self.l1_loss(predictions['shading'], targets['shading'])
            total_loss += self.lambda_shading * shading_loss
            loss_dict['shading_loss'] = shading_loss.item()
        
        # Smoothness loss (encourage smooth albedo and shading)
        if 'reflectance' in predictions:
            albedo_smooth_loss = self._smoothness_loss(predictions['reflectance'])
            total_loss += self.lambda_smoothness * albedo_smooth_loss
            loss_dict['albedo_smooth_loss'] = albedo_smooth_loss.item()
        
        if 'shading' in predictions:
            shading_smooth_loss = self._smoothness_loss(predictions['shading'])
            total_loss += self.lambda_smoothness * shading_smooth_loss
            loss_dict['shading_smooth_loss'] = shading_smooth_loss.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def _smoothness_loss(self, x):
        """Compute smoothness loss using gradients"""
        # Compute gradients
        grad_x = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        grad_y = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        
        return torch.mean(grad_x) + torch.mean(grad_y)


class IIDTrainer:
    """Trainer class for Intrinsic Image Decomposition"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = DecScaleClampedIllumEdgeGuidedNetworkBatchNorm().to(self.device)
        
        # Create loss function
        self.criterion = IIDLoss(
            lambda_recon=config.get('lambda_recon', 1.0),
            lambda_albedo=config.get('lambda_albedo', 1.0),
            lambda_shading=config.get('lambda_shading', 1.0),
            lambda_smoothness=config.get('lambda_smoothness', 0.1)
        )
        
        # Create optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Create scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.get('lr_step_size', 50),
            gamma=config.get('lr_gamma', 0.5)
        )
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = create_iid_data_loaders(config)
        
        # Create tensorboard writer
        self.writer = SummaryWriter(config.get('log_dir', 'logs/iid_training'))
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # Create output directory
        self.output_dir = Path(config.get('output_dir', 'outputs/iid_model'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load checkpoint if exists
        if config.get('resume_from'):
            self.load_checkpoint(config['resume_from'])
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            rgb = batch['rgb'].to(self.device)
            
            # Prepare targets
            targets = {'rgb': rgb}
            if 'albedo' in batch:
                targets['albedo'] = batch['albedo'].to(self.device)
            if 'shading' in batch:
                targets['shading'] = batch['shading'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(rgb)
            
            # Compute loss
            loss, loss_dict = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update progress bar
            epoch_losses.append(loss_dict)
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{loss_dict.get('recon_loss', 0):.4f}",
                'albedo': f"{loss_dict.get('albedo_loss', 0):.4f}"
            })
            
            # Log to tensorboard
            if batch_idx % self.config.get('log_interval', 100) == 0:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'train/{key}', value, global_step)
        
        # Compute average losses
        avg_losses = {}
        for key in epoch_losses[0].keys():
            avg_losses[key] = np.mean([loss_dict[key] for loss_dict in epoch_losses])
        
        return avg_losses
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        epoch_losses = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move data to device
                rgb = batch['rgb'].to(self.device)
                
                # Prepare targets
                targets = {'rgb': rgb}
                if 'albedo' in batch:
                    targets['albedo'] = batch['albedo'].to(self.device)
                if 'shading' in batch:
                    targets['shading'] = batch['shading'].to(self.device)
                
                # Forward pass
                predictions = self.model(rgb)
                
                # Compute loss
                loss, loss_dict = self.criterion(predictions, targets)
                epoch_losses.append(loss_dict)
        
        # Compute average losses
        avg_losses = {}
        for key in epoch_losses[0].keys():
            avg_losses[key] = np.mean([loss_dict[key] for loss_dict in epoch_losses])
        
        return avg_losses
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        checkpoint_path = self.output_dir / filename
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch}")
    
    def train(self, num_epochs):
        """Main training loop"""
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_losses = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step()
            
            # Log epoch results
            print(f"\nEpoch {epoch}:")
            print(f"Train Loss: {train_losses['total_loss']:.4f}")
            print(f"Val Loss: {val_losses['total_loss']:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Log to tensorboard
            for key, value in train_losses.items():
                self.writer.add_scalar(f'epoch/train_{key}', value, epoch)
            for key, value in val_losses.items():
                self.writer.add_scalar(f'epoch/val_{key}', value, epoch)
            self.writer.add_scalar('epoch/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save best model
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                self.save_checkpoint('best_model.pth')
            
            # Save checkpoint every few epochs
            if epoch % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
        
        # Save final model
        self.save_checkpoint('final_model.pth')
        print("Training completed!")
    
    def test(self):
        """Test the model"""
        print("Running test...")
        
        # Load best model
        best_model_path = self.output_dir / 'best_model.pth'
        if best_model_path.exists():
            self.load_checkpoint(str(best_model_path))
        
        self.model.eval()
        test_losses = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Testing'):
                # Move data to device
                rgb = batch['rgb'].to(self.device)
                
                # Prepare targets
                targets = {'rgb': rgb}
                if 'albedo' in batch:
                    targets['albedo'] = batch['albedo'].to(self.device)
                if 'shading' in batch:
                    targets['shading'] = batch['shading'].to(self.device)
                
                # Forward pass
                predictions = self.model(rgb)
                
                # Compute loss
                loss, loss_dict = self.criterion(predictions, targets)
                test_losses.append(loss_dict)
        
        # Compute average losses
        avg_losses = {}
        for key in test_losses[0].keys():
            avg_losses[key] = np.mean([loss_dict[key] for loss_dict in test_losses])
        
        print("\nTest Results:")
        for key, value in avg_losses.items():
            print(f"{key}: {value:.4f}")
        
        return avg_losses


def main():
    parser = argparse.ArgumentParser(description='Train IID model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--test', action='store_true', help='Run test only')
    
    args = parser.parse_args()
    
    # Default configuration
    config = {
        'datasets': {
            'midintrinsics': {
                'path': 'datasets/MIDIntrinsics',
                'enabled': True
            },
            'mit_intrinsic': {
                'path': 'datasets/MIT-intrinsic',
                'enabled': True
            },
            'mpi_sintel': {
                'path': 'datasets/MPI_Sintel',
                'enabled': True
            },
            'ned': {
                'path': 'datasets/NED',
                'enabled': True
            }
        },
        'image_size': 256,
        'batch_size': 8,
        'num_workers': 4,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'lr_step_size': 50,
        'lr_gamma': 0.5,
        'lambda_recon': 1.0,
        'lambda_albedo': 1.0,
        'lambda_shading': 1.0,
        'lambda_smoothness': 0.1,
        'log_dir': 'logs/iid_training',
        'output_dir': 'outputs/iid_model',
        'log_interval': 100,
        'save_interval': 10,
        'max_samples_per_dataset': None  # Set to a number for debugging
    }
    
    if args.resume:
        config['resume_from'] = args.resume
    
    # Create trainer
    trainer = IIDTrainer(config)
    
    if args.test:
        trainer.test()
    else:
        trainer.train(args.epochs)


if __name__ == '__main__':
    main()
