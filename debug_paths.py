#!/usr/bin/env python3
"""
Debug script to check path matching in dataset loading
"""

import os
import glob
from unified_dataset import UnifiedIIDDataset

def debug_midintrinsics_paths():
    """Debug MIDIntrinsics path matching"""
    print("="*50)
    print("Debugging MIDIntrinsics Paths")
    print("="*50)
    
    base_path = "datasets/MIDIntrinsics"
    albedo_path = os.path.join(base_path, 'train_albedo')
    multi_illum_path = os.path.join(base_path, 'multi_illumination_train_mip2_jpg')
    
    print(f"Albedo path: {albedo_path}")
    print(f"Multi-illum path: {multi_illum_path}")
    
    # Get scene directories
    albedo_scenes = [d for d in os.listdir(albedo_path) if os.path.isdir(os.path.join(albedo_path, d))]
    print(f"Number of albedo scenes: {len(albedo_scenes)}")
    
    # Check first few scenes
    for i, scene in enumerate(albedo_scenes[:5]):
        albedo_file = os.path.join(albedo_path, scene, 'albedo.exr')
        multi_illum_dir = os.path.join(multi_illum_path, scene)
        
        print(f"\nScene {i}: {scene}")
        print(f"  Albedo file: {albedo_file}")
        print(f"  Albedo exists: {os.path.exists(albedo_file)}")
        print(f"  Multi-illum dir: {multi_illum_dir}")
        print(f"  Multi-illum exists: {os.path.exists(multi_illum_dir)}")
        
        if os.path.exists(multi_illum_dir):
            illum_files = glob.glob(os.path.join(multi_illum_dir, 'dir_*.jpg'))
            print(f"  Number of illum files: {len(illum_files)}")
            if illum_files:
                print(f"  First illum file: {illum_files[0]}")

def debug_mit_intrinsic_paths():
    """Debug MIT-intrinsic path matching"""
    print("\n" + "="*50)
    print("Debugging MIT-Intrinsic Paths")
    print("="*50)
    
    base_path = "datasets/MIT-intrinsic"
    data_path = os.path.join(base_path, 'data')
    
    print(f"Data path: {data_path}")
    
    # Get object directories
    objects = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    print(f"Number of objects: {len(objects)}")
    
    # Check first few objects
    for i, obj in enumerate(objects[:3]):
        obj_path = os.path.join(data_path, obj)
        reflectance_file = os.path.join(obj_path, 'reflectance.png')
        shading_file = os.path.join(obj_path, 'shading.png')
        
        print(f"\nObject {i}: {obj}")
        print(f"  Object path: {obj_path}")
        print(f"  Reflectance file: {reflectance_file}")
        print(f"  Reflectance exists: {os.path.exists(reflectance_file)}")
        print(f"  Shading file: {shading_file}")
        print(f"  Shading exists: {os.path.exists(shading_file)}")
        
        illum_files = glob.glob(os.path.join(obj_path, 'light*.png'))
        print(f"  Number of illum files: {len(illum_files)}")
        if illum_files:
            print(f"  First illum file: {illum_files[0]}")

def debug_dataset_sample_paths():
    """Debug actual paths used in dataset samples"""
    print("\n" + "="*50)
    print("Debugging Dataset Sample Paths")
    print("="*50)
    
    from unified_dataset import UnifiedIIDDataset
    
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
        max_samples_per_dataset=3
    )
    
    # Check the internal samples list
    print(f"Total samples: {len(dataset.samples)}")
    
    for i, sample in enumerate(dataset.samples[:5]):
        print(f"\nSample {i}:")
        print(f"  Dataset: {sample['dataset']}")
        print(f"  Scene: {sample['scene']}")
        print(f"  RGB path: {sample['rgb_path']}")
        print(f"  RGB exists: {os.path.exists(sample['rgb_path'])}")
        print(f"  Albedo path: {sample['albedo_path']}")
        print(f"  Albedo exists: {os.path.exists(sample['albedo_path']) if sample['albedo_path'] else False}")
        print(f"  Shading path: {sample['shading_path']}")
        print(f"  Shading exists: {os.path.exists(sample['shading_path']) if sample['shading_path'] else False}")

def main():
    print("🔍 Debugging Path Matching")
    
    debug_midintrinsics_paths()
    debug_mit_intrinsic_paths()
    debug_dataset_sample_paths()

if __name__ == "__main__":
    main()
