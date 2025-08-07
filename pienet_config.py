# config.py - Configuration file for PIE-Net training

class TrainingConfig:
    """Configuration class for PIE-Net training parameters"""
    
    # Dataset settings
    DATA_ROOT = '/path/to/your/dataset'
    IMAGE_SIZE = 256
    
    # Training hyperparameters
    BATCH_SIZE = 8
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Loss function weights (from paper)
    LAMBDA_U = 0.5      # Unrefined loss weight
    LAMBDA_D = 0.4      # DSSIM loss weight  
    LAMBDA_E = 0.4      # Edge loss weight
    LAMBDA_P = 0.05     # Perceptual loss weight
    LAMBDA_SMSE = 0.95  # Scale-invariant MSE weight
    LAMBDA_MSE = 0.05   # Standard MSE weight
    
    # Training settings
    SAVE_DIR = 'checkpoints'
    SAVE_FREQ = 10  # Save every N epochs
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Learning rate schedule
    LR_STEP_SIZE = 30
    LR_GAMMA = 0.5

# data_preparation.py - Script to prepare dataset for training
import os
import shutil
import cv2 # type: ignore
import numpy as np
import imageio # type: ignore
from tqdm import tqdm

def prepare_dataset_structure(source_dir, target_dir):
    """
    Prepare dataset with proper directory structure:
    
    target_dir/
    ├── train/
    │   ├── images/
    │   ├── albedo/
    │   └── shading/
    └── val/
        ├── images/
        ├── albedo/
        └── shading/
    """
    
    # Create directory structure
    for split in ['train', 'val']:
        for subdir in ['images', 'albedo', 'shading']:
            os.makedirs(os.path.join(target_dir, split, subdir), exist_ok=True)
    
    print(f"Created dataset structure at {target_dir}")

def generate_synthetic_data(target_dir, num_samples=1000):
    """
    Generate synthetic data for testing the training pipeline
    This creates simple synthetic albedo/shading pairs
    """
    
    for split in ['train', 'val']:
        split_samples = num_samples if split == 'train' else num_samples // 5
        
        print(f"Generating {split_samples} synthetic samples for {split}...")
        
        for i in tqdm(range(split_samples)):
            # Generate random albedo (reflectance)
            albedo = np.random.rand(256, 256, 3) * 0.8 + 0.1
            
            # Generate smooth shading
            x = np.linspace(-1, 1, 256)
            y = np.linspace(-1, 1, 256)
            X, Y = np.meshgrid(x, y)
            shading = 0.5 + 0.4 * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
            shading = np.clip(shading, 0.1, 1.0)
            shading = shading[:, :, np.newaxis]
            
            # Generate input image (albedo * shading)
            input_img = albedo * shading
            input_img = np.clip(input_img, 0, 1)
            
            # Save images
            base_name = f"synthetic_{i:06d}"
            
            imageio.imwrite(os.path.join(target_dir, split, 'images', f'{base_name}.png'), 
                          (input_img * 255).astype(np.uint8))
            imageio.imwrite(os.path.join(target_dir, split, 'albedo', f'{base_name}.png'), 
                          (albedo * 255).astype(np.uint8))
            imageio.imwrite(os.path.join(target_dir, split, 'shading', f'{base_name}.png'), 
                          (shading.squeeze() * 255).astype(np.uint8))

def convert_dataset_to_pienet_format(source_dir, target_dir, dataset_type='ned'):
    """
    Convert existing datasets to PIE-Net format
    
    Args:
        source_dir: Path to original dataset
        target_dir: Path where converted dataset will be saved
        dataset_type: Type of dataset ('ned', 'mit', 'sintel', 'iiw')
    """
    
    if dataset_type.lower() == 'ned':
        convert_ned_dataset(source_dir, target_dir)
    elif dataset_type.lower() == 'mit':
        convert_mit_dataset(source_dir, target_dir)
    elif dataset_type.lower() == 'sintel':
        convert_sintel_dataset(source_dir, target_dir)
    else:
        print(f"Dataset type {dataset_type} not supported yet")

def convert_ned_dataset(source_dir, target_dir):
    """Convert NED dataset to training format"""
    
    prepare_dataset_structure(source_dir, target_dir)
    
    # NED dataset structure (adjust based on actual NED dataset)
    splits = ['train', 'test']  # Adjust splits as needed
    
    for split in splits:
        target_split = 'train' if split == 'train' else 'val'
        
        source_split_dir = os.path.join(source_dir, split)
        if not os.path.exists(source_split_dir):
            continue
            
        # Process images
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(glob.glob(os.path.join(source_split_dir, 'images', ext)))
        
        print(f"Processing {len(image_files)} images for {split} split...")
        
        for img_path in tqdm(image_files):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # Copy RGB image
            rgb_img = imageio.imread(img_path)
            rgb_img = cv2.resize(rgb_img, (256, 256))
            imageio.imwrite(os.path.join(target_dir, target_split, 'images', f'{base_name}.png'), rgb_img)
            
            # Copy albedo if exists
            albedo_path = os.path.join(source_split_dir, 'albedo', f'{base_name}.png')
            if os.path.exists(albedo_path):
                albedo_img = imageio.imread(albedo_path)
                albedo_img = cv2.resize(albedo_img, (256, 256))
                imageio.imwrite(os.path.join(target_dir, target_split, 'albedo', f'{base_name}.png'), albedo_img)
            
            # Copy shading if exists
            shading_path = os.path.join(source_split_dir, 'shading', f'{base_name}.png')
            if os.path.exists(shading_path):
                shading_img = imageio.imread(shading_path)
                shading_img = cv2.resize(shading_img, (256, 256))
                imageio.imwrite(os.path.join(target_dir, target_split, 'shading', f'{base_name}.png'), shading_img)

# training_utils.py - Additional utilities for training
import matplotlib.pyplot as plt # type: ignore
import json

def plot_training_curves(loss_history, save_path='training_curves.png'):
    """Plot training and validation loss curves"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('PIE-Net Training Curves')
    
    # Total loss
    axes[0, 0].plot(loss_history['train_total'], label='Train')
    axes[0, 0].plot(loss_history['val_total'], label='Validation')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Edge loss
    axes[0, 1].plot(loss_history['train_edge'], label='Train')
    axes[0, 1].plot(loss_history['val_edge'], label='Validation')
    axes[0, 1].set_title('Edge Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Refined loss
    axes[0, 2].plot(loss_history['train_refined'], label='Train')
    axes[0, 2].plot(loss_history['val_refined'], label='Validation')
    axes[0, 2].set_title('Refined Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Unrefined loss
    axes[1, 0].plot(loss_history['train_unrefined'], label='Train')
    axes[1, 0].plot(loss_history['val_unrefined'], label='Validation')
    axes[1, 0].set_title('Unrefined Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # DSSIM loss
    axes[1, 1].plot(loss_history['train_dssim'], label='Train')
    axes[1, 1].plot(loss_history['val_dssim'], label='Validation')
    axes[1, 1].set_title('DSSIM Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Perceptual loss
    axes[1, 2].plot(loss_history['train_perceptual'], label='Train')
    axes[1, 2].plot(loss_history['val_perceptual'], label='Validation')
    axes[1, 2].set_title('Perceptual Loss')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_training_config(config, save_path):
    """Save training configuration to JSON file"""
    config_dict = {
        'data_root': config.DATA_ROOT,
        'image_size': config.IMAGE_SIZE,
        'batch_size': config.BATCH_SIZE,
        'num_epochs': config.NUM_EPOCHS,
        'learning_rate': config.LEARNING_RATE,
        'weight_decay': config.WEIGHT_DECAY,
        'lambda_u': config.LAMBDA_U,
        'lambda_d': config.LAMBDA_D,
        'lambda_e': config.LAMBDA_E,
        'lambda_p': config.LAMBDA_P,
        'lambda_smse': config.LAMBDA_SMSE,
        'lambda_mse': config.LAMBDA_MSE,
    }
    
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=4)

# Example usage and training script with config
"""
Usage Example:

1. Prepare your dataset:
   python data_preparation.py --source_dir /path/to/original/dataset --target_dir /path/to/training/dataset --dataset_type ned

2. Generate synthetic data for testing:
   python data_preparation.py --generate_synthetic --target_dir /path/to/synthetic/dataset --num_samples 1000

3. Train the model:
   python train.py --data_root /path/to/training/dataset --batch_size 8 --num_epochs 100 --save_dir checkpoints

4. Resume training from checkpoint:
   python train.py --data_root /path/to/training/dataset --resume checkpoints/checkpoint_epoch_50.t7
"""

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='PIE-Net Data Preparation')
    parser.add_argument('--source_dir', type=str, help='Source dataset directory')
    parser.add_argument('--target_dir', type=str, required=True, help='Target dataset directory')
    parser.add_argument('--dataset_type', type=str, default='ned', choices=['ned', 'mit', 'sintel', 'iiw'])
    parser.add_argument('--generate_synthetic', action='store_true', help='Generate synthetic data')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of synthetic samples')
    
    args = parser.parse_args()
    
    if args.generate_synthetic:
        prepare_dataset_structure(None, args.target_dir)
        generate_synthetic_data(args.target_dir, args.num_samples)
        print(f"Generated synthetic dataset at {args.target_dir}")
    elif args.source_dir:
        convert_dataset_to_pienet_format(args.source_dir, args.target_dir, args.dataset_type)
        print(f"Converted {args.dataset_type} dataset from {args.source_dir} to {args.target_dir}")
    else:
        print("Please provide either --source_dir for conversion or --generate_synthetic for synthetic data")
