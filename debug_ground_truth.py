#!/usr/bin/env python3
"""
Debug script to test ground truth loading
"""

import os
import torch
import numpy as np
import cv2
import imageio
from pathlib import Path

# Test OpenEXR import
try:
    import OpenEXR
    import Imath
    print("✅ OpenEXR imported successfully")
except ImportError as e:
    print(f"❌ OpenEXR import failed: {e}")

def test_midintrinsics_ground_truth():
    """Test MIDIntrinsics ground truth loading"""
    print("\n" + "="*50)
    print("Testing MIDIntrinsics Ground Truth")
    print("="*50)
    
    # Test a specific albedo file
    albedo_path = "datasets/MIDIntrinsics/train_albedo/joy_bedroom2/albedo.exr"
    
    if os.path.exists(albedo_path):
        print(f"✅ Albedo file exists: {albedo_path}")
        
        try:
            import OpenEXR
            import Imath
            
            exr_file = OpenEXR.InputFile(albedo_path)
            dw = exr_file.header()['dataWindow']
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
            
            FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
            channels = exr_file.channels("RGB", FLOAT)
            
            img = np.zeros((size[1], size[0], 3), dtype=np.float32)
            for i, channel in enumerate(['R', 'G', 'B']):
                img[:, :, i] = np.frombuffer(channels[i], dtype=np.float32).reshape(size[1], size[0])
            
            print(f"✅ EXR loaded successfully")
            print(f"   Shape: {img.shape}")
            print(f"   Min: {img.min():.4f}, Max: {img.max():.4f}")
            print(f"   Mean: {img.mean():.4f}")
            
            # Resize and convert to tensor
            img = cv2.resize(img, (256, 256))
            img = torch.from_numpy(img).permute(2, 0, 1)
            
            print(f"✅ Tensor created")
            print(f"   Tensor shape: {img.shape}")
            print(f"   Tensor min: {img.min():.4f}, max: {img.max():.4f}")
            
        except Exception as e:
            print(f"❌ Error loading EXR: {e}")
    else:
        print(f"❌ Albedo file not found: {albedo_path}")

def test_mit_intrinsic_ground_truth():
    """Test MIT-intrinsic ground truth loading"""
    print("\n" + "="*50)
    print("Testing MIT-Intrinsic Ground Truth")
    print("="*50)
    
    # Test reflectance and shading files
    reflectance_path = "datasets/MIT-intrinsic/data/apple/reflectance.png"
    shading_path = "datasets/MIT-intrinsic/data/apple/shading.png"
    
    # Test reflectance
    if os.path.exists(reflectance_path):
        print(f"✅ Reflectance file exists: {reflectance_path}")
        
        try:
            img = imageio.imread(reflectance_path)
            print(f"✅ Reflectance loaded successfully")
            print(f"   Shape: {img.shape}")
            print(f"   Min: {img.min()}, Max: {img.max()}")
            print(f"   Mean: {img.mean():.2f}")
            
            # Resize and normalize
            img = cv2.resize(img, (256, 256))
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)
            
            print(f"✅ Reflectance tensor created")
            print(f"   Tensor shape: {img.shape}")
            print(f"   Tensor min: {img.min():.4f}, max: {img.max():.4f}")
            
        except Exception as e:
            print(f"❌ Error loading reflectance: {e}")
    else:
        print(f"❌ Reflectance file not found: {reflectance_path}")
    
    # Test shading
    if os.path.exists(shading_path):
        print(f"✅ Shading file exists: {shading_path}")
        
        try:
            img = imageio.imread(shading_path)
            print(f"✅ Shading loaded successfully")
            print(f"   Shape: {img.shape}")
            print(f"   Min: {img.min()}, Max: {img.max()}")
            print(f"   Mean: {img.mean():.2f}")
            
            # Resize and normalize
            img = cv2.resize(img, (256, 256))
            img = img.astype(np.float32) / 255.0
            
            # Handle grayscale
            if len(img.shape) == 2:
                img = torch.from_numpy(img).unsqueeze(0)
            else:
                img = torch.from_numpy(img).permute(2, 0, 1)
            
            print(f"✅ Shading tensor created")
            print(f"   Tensor shape: {img.shape}")
            print(f"   Tensor min: {img.min():.4f}, max: {img.max():.4f}")
            
        except Exception as e:
            print(f"❌ Error loading shading: {e}")
    else:
        print(f"❌ Shading file not found: {shading_path}")

def test_mpi_sintel_ground_truth():
    """Test MPI_Sintel ground truth loading"""
    print("\n" + "="*50)
    print("Testing MPI_Sintel Ground Truth")
    print("="*50)
    
    # Test albedo file
    albedo_path = "datasets/MPI_Sintel/training/albedo/alley_1/frame_0001.png"
    
    if os.path.exists(albedo_path):
        print(f"✅ Albedo file exists: {albedo_path}")
        
        try:
            img = imageio.imread(albedo_path)
            print(f"✅ Albedo loaded successfully")
            print(f"   Shape: {img.shape}")
            print(f"   Min: {img.min()}, Max: {img.max()}")
            print(f"   Mean: {img.mean():.2f}")
            
            # Resize and normalize
            img = cv2.resize(img, (256, 256))
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)
            
            print(f"✅ Albedo tensor created")
            print(f"   Tensor shape: {img.shape}")
            print(f"   Tensor min: {img.min():.4f}, max: {img.max():.4f}")
            
        except Exception as e:
            print(f"❌ Error loading albedo: {e}")
    else:
        print(f"❌ Albedo file not found: {albedo_path}")

def test_dataset_sample():
    """Test a single sample from the dataset"""
    print("\n" + "="*50)
    print("Testing Dataset Sample")
    print("="*50)
    
    from unified_dataset import UnifiedIIDDataset
    
    # Create a small test dataset
    config = {
        'datasets': {
            'midintrinsics': {'path': 'datasets/MIDIntrinsics', 'enabled': True},
            'mit_intrinsic': {'path': 'datasets/MIT-intrinsic', 'enabled': True},
            'mpi_sintel': {'path': 'datasets/MPI_Sintel', 'enabled': True},
            'ned': {'path': 'datasets/NED', 'enabled': True}
        }
    }
    
    dataset = UnifiedIIDDataset(
        datasets_config=config['datasets'],
        split='train',
        image_size=256,
        max_samples_per_dataset=1
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test a few samples
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Dataset: {sample['dataset']}")
        print(f"  Scene: {sample['scene']}")
        print(f"  RGB shape: {sample['rgb'].shape}")
        print(f"  RGB range: [{sample['rgb'].min():.4f}, {sample['rgb'].max():.4f}]")
        print(f"  Albedo shape: {sample['albedo'].shape}")
        print(f"  Albedo range: [{sample['albedo'].min():.4f}, {sample['albedo'].max():.4f}]")
        print(f"  Shading shape: {sample['shading'].shape}")
        print(f"  Shading range: [{sample['shading'].min():.4f}, {sample['shading'].max():.4f}]")

def main():
    print("🔍 Debugging Ground Truth Loading")
    print("="*50)
    
    # Test each dataset's ground truth loading
    test_midintrinsics_ground_truth()
    test_mit_intrinsic_ground_truth()
    test_mpi_sintel_ground_truth()
    test_dataset_sample()

if __name__ == "__main__":
    main()
