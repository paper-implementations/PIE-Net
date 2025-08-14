import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import imageio
import numpy as np
import cv2
from PIL import Image


class PIENetDataset(Dataset):
    """Dataset class for PIE-Net training"""
    
    def __init__(self, data_root, split='train', image_size=256, transform=None):
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.transform = transform
        
        # Get paths
        self.images_path = os.path.join(data_root, split, 'images')
        self.albedo_path = os.path.join(data_root, split, 'albedo')
        self.shading_path = os.path.join(data_root, split, 'shading')
        
        # Get all image files
        self.image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.image_files.extend(glob.glob(os.path.join(self.images_path, ext)))
        
        self.image_files.sort()
        
        print(f"Found {len(self.image_files)} images in {split} split")
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.images_path}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Get base filename
        img_path = self.image_files[idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Load RGB image
        rgb_img = self._load_image(img_path)
        
        # Load ground truth if available (for supervised training)
        albedo_file = os.path.join(self.albedo_path, f"{base_name}.png")
        shading_file = os.path.join(self.shading_path, f"{base_name}.png")
        
        sample = {'rgb': rgb_img, 'filename': base_name}
        
        if os.path.exists(albedo_file):
            albedo_img = self._load_image(albedo_file)
            sample['albedo'] = albedo_img
        
        if os.path.exists(shading_file):
            shading_img = self._load_image(shading_file, is_shading=True)
            sample['shading'] = shading_img
        
        return sample
    
    def _load_image(self, img_path, is_shading=False):
        """Load and preprocess image"""
        try:
            img = imageio.imread(img_path)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a dummy image if loading fails
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Handle different image types
        if len(img.shape) == 2:  # Grayscale
            if is_shading:
                img = img  # Keep as grayscale for shading
            else:
                img = np.stack([img, img, img], axis=-1)  # Convert to RGB
        elif len(img.shape) == 3:
            if img.shape[2] == 4:  # RGBA
                img = img[:, :, :3]  # Remove alpha channel
        
        # Resize
        img = cv2.resize(img, (self.image_size, self.image_size))
        
        # Normalise to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Handle NaN values
        img[np.isnan(img)] = 0
        
        # Convert to tensor format (C, H, W)
        if is_shading and len(img.shape) == 2:
            img = torch.from_numpy(img).unsqueeze(0)  # Add channel dimension
        else:
            img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
        
        return img


class PIENetTestDataset(Dataset):
    """Dataset for testing/evaluation (no ground truth needed)"""
    
    def __init__(self, data_path, image_size=256):
        self.data_path = data_path
        self.image_size = image_size
        
        # Get all image files
        self.image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.image_files.extend(glob.glob(os.path.join(data_path, ext)))
        
        self.image_files.sort()
        print(f"Found {len(self.image_files)} test images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Load image
        img = imageio.imread(img_path)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0
        img[np.isnan(img)] = 0
        img = torch.from_numpy(img).permute(2, 0, 1)
        
        return {'rgb': img, 'filename': base_name}


def create_data_loaders(config):
    """Create training and validation data loaders"""
    
    # Create datasets
    train_dataset = PIENetDataset(
        data_root=config.DATA_ROOT,
        split='train',
        image_size=config.IMAGE_SIZE
    )
    
    val_dataset = PIENetDataset(
        data_root=config.DATA_ROOT,
        split='val',
        image_size=config.IMAGE_SIZE
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=False
    )
    
    return train_loader, val_loader


def create_test_loader(data_path, batch_size=1, num_workers=2):
    """Create test data loader"""
    test_dataset = PIENetTestDataset(data_path)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return test_loader