#!/usr/bin/env python3
"""
Dataset converter for PIE-Net training
Converts popular intrinsic image datasets to the expected format
"""

import os
import shutil
import glob
import cv2
import numpy as np
import imageio
from tqdm import tqdm
import argparse


def create_directory_structure(output_dir):
    """Create the standard PIE-Net directory structure"""
    dirs_to_create = [
        'train/images', 'train/albedo', 'train/shading',
        'val/images', 'val/albedo', 'val/shading'
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(os.path.join(output_dir, dir_path), exist_ok=True)
    
    print(f"✅ Created directory structure in {output_dir}")


def convert_mit_intrinsic(source_dir, output_dir, train_split=0.8):
    """
    Convert MIT Intrinsic dataset
    Expected structure:
    mit_dataset/
    ├── data/
    │   ├── apple/
    │   │   ├── apple-1/
    │   │   │   ├── apple-1.png
    │   │   │   ├── apple-1-albedo.png
    │   │   │   └── apple-1-shading.png
    """
    print("Converting MIT Intrinsic dataset...")
    create_directory_structure(output_dir)
    
    # Find all object directories
    data_dir = os.path.join(source_dir, 'data')
    if not os.path.exists(data_dir):
        data_dir = source_dir
    
    all_files = []
    
    # Search for image files recursively
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')) and not any(x in file for x in ['albedo', 'shading']):
                base_name = os.path.splitext(file)[0]
                img_path = os.path.join(root, file)
                albedo_path = os.path.join(root, f"{base_name}-albedo.png")
                shading_path = os.path.join(root, f"{base_name}-shading.png")
                
                if os.path.exists(albedo_path) and os.path.exists(shading_path):
                    all_files.append((img_path, albedo_path, shading_path, base_name))
    
    print(f"Found {len(all_files)} complete triplets")
    
    # Split into train/val
    np.random.shuffle(all_files)
    split_idx = int(len(all_files) * train_split)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    # Process files
    def process_split(file_list, split_name):
        for i, (img_path, albedo_path, shading_path, base_name) in enumerate(tqdm(file_list, desc=f"Processing {split_name}")):
            new_name = f"{split_name}_{i:06d}"
            
            # Copy and resize images
            img = imageio.imread(img_path)
            albedo = imageio.imread(albedo_path)
            shading = imageio.imread(shading_path)
            
            # Resize to 256x256
            img = cv2.resize(img, (256, 256))
            albedo = cv2.resize(albedo, (256, 256))
            shading = cv2.resize(shading, (256, 256))
            
            # Save
            imageio.imwrite(os.path.join(output_dir, split_name, 'images', f'{new_name}.png'), img)
            imageio.imwrite(os.path.join(output_dir, split_name, 'albedo', f'{new_name}.png'), albedo)
            imageio.imwrite(os.path.join(output_dir, split_name, 'shading', f'{new_name}.png'), shading)
    
    process_split(train_files, 'train')
    process_split(val_files, 'val')
    
    print(f"✅ MIT dataset converted: {len(train_files)} train, {len(val_files)} val")


def convert_iiw_dataset(source_dir, output_dir):
    """
    Convert Intrinsic Images in the Wild (IIW) dataset
    Note: IIW doesn't have dense ground truth, only sparse comparisons
    This creates a structure for self-supervised training
    """
    print("Converting IIW dataset (images only - no dense GT)...")
    create_directory_structure(output_dir)
    
    # Find all images
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(source_dir, '**', ext), recursive=True))
    
    print(f"Found {len(image_files)} images")
    
    # Split into train/val (80/20)
    np.random.shuffle(image_files)
    split_idx = int(len(image_files) * 0.8)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    def process_split(file_list, split_name):
        for i, img_path in enumerate(tqdm(file_list, desc=f"Processing {split_name}")):
            new_name = f"{split_name}_{i:06d}"
            
            # Load and resize image
            img = imageio.imread(img_path)
            if len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
                img = img[:, :, :3]  # Remove alpha
            
            img = cv2.resize(img, (256, 256))
            
            # Save only in images directory (no ground truth)
            imageio.imwrite(os.path.join(output_dir, split_name, 'images', f'{new_name}.png'), img)
    
    process_split(train_files, 'train')
    process_split(val_files, 'val')
    
    print(f"✅ IIW dataset converted: {len(train_files)} train, {len(val_files)} val (self-supervised)")


def convert_sintel_dataset(source_dir, output_dir):
    """
    Convert MPI Sintel dataset (if available)
    Expected to have albedo and shading folders
    """
    print("Converting Sintel dataset...")
    create_directory_structure(output_dir)
    
    # Look for standard Sintel structure
    final_dir = os.path.join(source_dir, 'training', 'final')
    albedo_dir = os.path.join(source_dir, 'training', 'albedo')
    shading_dir = os.path.join(source_dir, 'training', 'shading')
    
    if not all(os.path.exists(d) for d in [final_dir, albedo_dir, shading_dir]):
        print("❌ Sintel dataset structure not found. Expected:")
        print("  training/final/    - RGB images")
        print("  training/albedo/   - Albedo ground truth")
        print("  training/shading/  - Shading ground truth")
        return
    
    # Process each sequence
    sequences = os.listdir(final_dir)
    all_triplets = []
    
    for seq in sequences:
        seq_final = os.path.join(final_dir, seq)
        seq_albedo = os.path.join(albedo_dir, seq)
        seq_shading = os.path.join(shading_dir, seq)
        
        if not all(os.path.exists(d) for d in [seq_final, seq_albedo, seq_shading]):
            continue
        
        # Get frame files
        frames = sorted([f for f in os.listdir(seq_final) if f.endswith('.png')])
        
        for frame in frames:
            img_path = os.path.join(seq_final, frame)
            albedo_path = os.path.join(seq_albedo, frame)
            shading_path = os.path.join(seq_shading, frame)
            
            if all(os.path.exists(p) for p in [img_path, albedo_path, shading_path]):
                all_triplets.append((img_path, albedo_path, shading_path, f"{seq}_{frame}"))
    
    print(f"Found {len(all_triplets)} complete triplets")
    
    # Split and process (similar to MIT)
    np.random.shuffle(all_triplets)
    split_idx = int(len(all_triplets) * 0.8)
    train_files = all_triplets[:split_idx]
    val_files = all_triplets[split_idx:]
    
    def process_split(file_list, split_name):
        for i, (img_path, albedo_path, shading_path, orig_name) in enumerate(tqdm(file_list, desc=f"Processing {split_name}")):
            new_name = f"{split_name}_{i:06d}"
            
            # Load and process images
            img = imageio.imread(img_path)
            albedo = imageio.imread(albedo_path)
            shading = imageio.imread(shading_path)
            
            # Resize
            img = cv2.resize(img, (256, 256))
            albedo = cv2.resize(albedo, (256, 256))
            shading = cv2.resize(shading, (256, 256))
            
            # Save
            imageio.imwrite(os.path.join(output_dir, split_name, 'images', f'{new_name}.png'), img)
            imageio.imwrite(os.path.join(output_dir, split_name, 'albedo', f'{new_name}.png'), albedo)
            imageio.imwrite(os.path.join(output_dir, split_name, 'shading', f'{new_name}.png'), shading)
    
    process_split(train_files, 'train')
    process_split(val_files, 'val')
    
    print(f"✅ Sintel dataset converted: {len(train_files)} train, {len(val_files)} val")


def create_synthetic_dataset(output_dir, num_samples=5000):
    """Create synthetic dataset for testing"""
    print(f"Creating synthetic dataset with {num_samples} samples...")
    create_directory_structure(output_dir)
    
    def generate_samples(split_name, num_split_samples):
        for i in tqdm(range(num_split_samples), desc=f"Generating {split_name}"):
            # Generate random albedo (reflectance)
            albedo = np.random.rand(256, 256, 3) * 0.7 + 0.2  # Range 0.2-0.9
            
            # Generate smooth shading using perlin-like noise
            x = np.linspace(0, 4*np.pi, 256)
            y = np.linspace(0, 4*np.pi, 256)
            X, Y = np.meshgrid(x, y)
            
            # Combine multiple frequency components for realistic shading
            shading = (0.5 + 0.3 * np.sin(X) * np.cos(Y) + 
                      0.1 * np.sin(2*X) * np.cos(2*Y) +
                      0.05 * np.random.randn(256, 256))
            shading = np.clip(shading, 0.1, 1.0)
            
            # Add some directional lighting
            light_direction = np.random.uniform(0, 2*np.pi)
            directional = 0.8 + 0.2 * np.cos(X * np.cos(light_direction) + Y * np.sin(light_direction))
            shading = shading * directional
            shading = np.clip(shading, 0.05, 1.0)
            
            # Generate input image (albedo * shading)
            input_img = albedo * shading[:, :, np.newaxis]
            input_img = np.clip(input_img, 0, 1)
            
            # Convert to uint8
            input_img = (input_img * 255).astype(np.uint8)
            albedo = (albedo * 255).astype(np.uint8)
            shading = (shading * 255).astype(np.uint8)
            
            # Save
            sample_name = f"{split_name}_{i:06d}"
            imageio.imwrite(os.path.join(output_dir, split_name, 'images', f'{sample_name}.png'), input_img)
            imageio.imwrite(os.path.join(output_dir, split_name, 'albedo', f'{sample_name}.png'), albedo)
            imageio.imwrite(os.path.join(output_dir, split_name, 'shading', f'{sample_name}.png'), shading)
    
    # Generate splits
    train_samples = int(num_samples * 0.8)
    val_samples = num_samples - train_samples
    
    generate_samples('train', train_samples)
    generate_samples('val', val_samples)
    
    print(f"✅ Synthetic dataset created: {train_samples} train, {val_samples} val")


def main():
    parser = argparse.ArgumentParser(description='Convert datasets to PIE-Net format')
    parser.add_argument('--dataset_type', type=str, required=True,
                       choices=['mit', 'iiw', 'sintel', 'synthetic'],
                       help='Type of dataset to convert')
    parser.add_argument('--source_dir', type=str, 
                       help='Source dataset directory (not needed for synthetic)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for converted dataset')
    parser.add_argument('--num_samples', type=int, default=5000,
                       help='Number of synthetic samples to generate')
    
    args = parser.parse_args()
    
    if args.dataset_type == 'synthetic':
        create_synthetic_dataset(args.output_dir, args.num_samples)
    elif args.dataset_type == 'mit':
        if not args.source_dir:
            print("❌ --source_dir required for MIT dataset")
            return
        convert_mit_intrinsic(args.source_dir, args.output_dir)
    elif args.dataset_type == 'iiw':
        if not args.source_dir:
            print("❌ --source_dir required for IIW dataset")
            return
        convert_iiw_dataset(args.source_dir, args.output_dir)
    elif args.dataset_type == 'sintel':
        if not args.source_dir:
            print("❌ --source_dir required for Sintel dataset")
            return
        convert_sintel_dataset(args.source_dir, args.output_dir)
    
    print(f"\n✅ Dataset conversion completed!")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🔧 Ready for PIE-Net training!")


if __name__ == '__main__':
    main()