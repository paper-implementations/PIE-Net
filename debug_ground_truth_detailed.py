#!/usr/bin/env python3
"""
Detailed debug script to test the exact ground truth loading logic
"""

import os
import torch
import numpy as np
import cv2
import imageio
from pathlib import Path

def test_exact_ground_truth_loading():
    """Test the exact ground truth loading logic from the dataset"""
    print("="*50)
    print("Testing Exact Ground Truth Loading Logic")
    print("="*50)
    
    # Test the exact logic from the dataset
    def _load_ground_truth_exact(gt_path, gt_type, image_size=256):
        """Exact copy of the _load_ground_truth method from unified_dataset.py"""
        try:
            if gt_path.endswith('.exr'):
                # Handle EXR files (MIDIntrinsics albedo)
                try:
                    import OpenEXR
                    import Imath
                    
                    exr_file = OpenEXR.InputFile(gt_path)
                    dw = exr_file.header()['dataWindow']
                    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
                    
                    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
                    channels = exr_file.channels("RGB", FLOAT)
                    
                    img = np.zeros((size[1], size[0], 3), dtype=np.float32)
                    for i, channel in enumerate(['R', 'G', 'B']):
                        img[:, :, i] = np.frombuffer(channels[i], dtype=np.float32).reshape(size[1], size[0])
                    
                    # EXR files are already in float format, no need to divide by 255
                    # Just clip to reasonable range and normalize
                    img = np.clip(img, 0, 2.0)  # Clip to reasonable range
                    
                except ImportError:
                    print(f"OpenEXR not available, skipping {gt_path}")
                    return None
                
            else:
                # Handle standard image files
                img = imageio.imread(gt_path)
                
                if len(img.shape) == 2:  # Grayscale
                    if gt_type == 'shading':
                        img = img  # Keep as grayscale for shading
                    else:
                        img = np.stack([img, img, img], axis=-1)
                elif len(img.shape) == 3 and img.shape[2] == 4:
                    img = img[:, :, :3]
                
                # Normalize standard image files to [0, 1]
                img = img.astype(np.float32) / 255.0
            
            # Resize
            img = cv2.resize(img, (image_size, image_size))
            
            # Handle NaN values
            img[np.isnan(img)] = 0
            
            # Convert to tensor
            if gt_type == 'shading' and len(img.shape) == 2:
                img = torch.from_numpy(img).unsqueeze(0)  # Single channel for shading
            else:
                img = torch.from_numpy(img).permute(2, 0, 1)  # Multi-channel
            
            return img
            
        except Exception as e:
            print(f"Error loading ground truth {gt_path}: {e}")
            if gt_type == 'shading':
                return torch.zeros(1, image_size, image_size)
            else:
                return torch.zeros(3, image_size, image_size)
    
    # Test MIDIntrinsics albedo
    print("\nTesting MIDIntrinsics albedo loading:")
    albedo_path = "datasets/MIDIntrinsics/train_albedo/elm_revis_living28/albedo.exr"
    print(f"Path: {albedo_path}")
    print(f"Exists: {os.path.exists(albedo_path)}")
    
    albedo_tensor = _load_ground_truth_exact(albedo_path, 'albedo')
    if albedo_tensor is not None:
        print(f"✅ Albedo loaded successfully")
        print(f"   Shape: {albedo_tensor.shape}")
        print(f"   Min: {albedo_tensor.min():.4f}, Max: {albedo_tensor.max():.4f}")
        print(f"   Mean: {albedo_tensor.mean():.4f}")
    else:
        print("❌ Albedo loading failed")
    
    # Test MIT-intrinsic reflectance
    print("\nTesting MIT-intrinsic reflectance loading:")
    reflectance_path = "datasets/MIT-intrinsic/data/apple/reflectance.png"
    print(f"Path: {reflectance_path}")
    print(f"Exists: {os.path.exists(reflectance_path)}")
    
    reflectance_tensor = _load_ground_truth_exact(reflectance_path, 'albedo')
    if reflectance_tensor is not None:
        print(f"✅ Reflectance loaded successfully")
        print(f"   Shape: {reflectance_tensor.shape}")
        print(f"   Min: {reflectance_tensor.min():.4f}, Max: {reflectance_tensor.max():.4f}")
        print(f"   Mean: {reflectance_tensor.mean():.4f}")
    else:
        print("❌ Reflectance loading failed")
    
    # Test MIT-intrinsic shading
    print("\nTesting MIT-intrinsic shading loading:")
    shading_path = "datasets/MIT-intrinsic/data/apple/shading.png"
    print(f"Path: {shading_path}")
    print(f"Exists: {os.path.exists(shading_path)}")
    
    shading_tensor = _load_ground_truth_exact(shading_path, 'shading')
    if shading_tensor is not None:
        print(f"✅ Shading loaded successfully")
        print(f"   Shape: {shading_tensor.shape}")
        print(f"   Min: {shading_tensor.min():.4f}, Max: {shading_tensor.max():.4f}")
        print(f"   Mean: {shading_tensor.mean():.4f}")
    else:
        print("❌ Shading loading failed")

def test_dataset_getitem():
    """Test the exact __getitem__ method"""
    print("\n" + "="*50)
    print("Testing Dataset __getitem__ Method")
    print("="*50)
    
    from unified_dataset import UnifiedIIDDataset
    
    config = {
        'datasets': {
            'midintrinsics': {'path': 'datasets/MIDIntrinsics', 'enabled': False},  # Disable to test others
            'mit_intrinsic': {'path': 'datasets/MIT-intrinsic', 'enabled': True},
            'mpi_sintel': {'path': 'datasets/MPI_Sintel', 'enabled': True},
            'ned': {'path': 'datasets/NED', 'enabled': True}
        }
    }
    
    dataset = UnifiedIIDDataset(
        datasets_config=config['datasets'],
        split='train',
        image_size=256,
        max_samples_per_dataset=3
    )
    
    # Test a few samples with detailed logging
    for i in range(min(5, len(dataset))):
        print(f"\n--- Sample {i} ---")
        sample = dataset[i]
        
        print(f"Dataset: {sample['dataset']}")
        print(f"Scene: {sample['scene']}")
        print(f"RGB shape: {sample['rgb'].shape}")
        print(f"RGB range: [{sample['rgb'].min():.4f}, {sample['rgb'].max():.4f}]")
        print(f"Albedo shape: {sample['albedo'].shape}")
        print(f"Albedo range: [{sample['albedo'].min():.4f}, {sample['albedo'].max():.4f}]")
        print(f"Shading shape: {sample['shading'].shape}")
        print(f"Shading range: [{sample['shading'].min():.4f}, {sample['shading'].max():.4f}]")
        
        # Check if albedo is all zeros
        if sample['albedo'].max() < 0.01:
            print("⚠️  WARNING: Albedo is essentially black!")
        
        # Check if shading is all zeros
        if sample['shading'].max() < 0.01:
            print("⚠️  WARNING: Shading is essentially black!")
        
        # Check if shading is too high
        if sample['shading'].max() > 10.0:
            print("⚠️  WARNING: Shading values are too high!")

def main():
    print("🔍 Detailed Ground Truth Debugging")
    
    test_exact_ground_truth_loading()
    test_dataset_getitem()

if __name__ == "__main__":
    main()
