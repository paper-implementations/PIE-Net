#!/usr/bin/env python3
"""
Debug script to check albedo loading
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from unified_dataset import UnifiedIIDDataset, custom_collate_fn


def debug_albedo_loading():
    """Debug albedo loading specifically"""
    
    print("🔍 Debugging Albedo Loading")
    print("=" * 50)
    
    # Test configuration
    test_config = {
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
        'derive_shading': True,
        'max_samples_per_dataset': 2
    }
    
    # Test each dataset individually
    for dataset_name in ['midintrinsics', 'mit_intrinsic', 'mpi_sintel']:
        print(f"\n📊 Testing {dataset_name.upper()}")
        print("-" * 30)
        
        # Create config for single dataset
        single_dataset_config = test_config.copy()
        for key in single_dataset_config['datasets']:
            single_dataset_config['datasets'][key]['enabled'] = (key == dataset_name)
        
        try:
            # Create dataset
            dataset = UnifiedIIDDataset(
                datasets_config=single_dataset_config['datasets'],
                split='train',
                image_size=test_config['image_size'],
                max_samples_per_dataset=test_config['max_samples_per_dataset'],
                derive_shading=test_config['derive_shading']
            )
            
            if len(dataset) > 0:
                # Test first sample
                sample = dataset[0]
                
                print(f"Sample keys: {list(sample.keys())}")
                print(f"RGB shape: {sample['rgb'].shape}")
                print(f"RGB mean: {sample['rgb'].mean():.4f}")
                print(f"RGB std: {sample['rgb'].std():.4f}")
                print(f"RGB min: {sample['rgb'].min():.4f}")
                print(f"RGB max: {sample['rgb'].max():.4f}")
                
                if 'albedo' in sample:
                    albedo = sample['albedo']
                    print(f"Albedo shape: {albedo.shape}")
                    print(f"Albedo mean: {albedo.mean():.4f}")
                    print(f"Albedo std: {albedo.std():.4f}")
                    print(f"Albedo min: {albedo.min():.4f}")
                    print(f"Albedo max: {albedo.max():.4f}")
                    
                    # Check if albedo is all zeros
                    non_zero_albedo = (albedo > 0.01).sum().item()
                    total_albedo = albedo.numel()
                    print(f"Albedo non-zero pixels: {non_zero_albedo}/{total_albedo} ({100*non_zero_albedo/total_albedo:.2f}%)")
                    
                    # Check if albedo is all ones
                    all_ones = (albedo > 0.99).sum().item()
                    print(f"Albedo all-ones pixels: {all_ones}/{total_albedo} ({100*all_ones/total_albedo:.2f}%)")
                    
                if 'shading' in sample:
                    shading = sample['shading']
                    print(f"Shading shape: {shading.shape}")
                    print(f"Shading mean: {shading.mean():.4f}")
                    print(f"Shading std: {shading.std():.4f}")
                    print(f"Shading min: {shading.min():.4f}")
                    print(f"Shading max: {shading.max():.4f}")
                
                # Create a simple visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # RGB
                rgb = sample['rgb'].permute(1, 2, 0).cpu().numpy()
                rgb = np.clip(rgb, 0, 1)
                axes[0].imshow(rgb)
                axes[0].set_title(f"RGB - {dataset_name}")
                axes[0].axis('off')
                
                # Albedo
                if 'albedo' in sample:
                    albedo_viz = sample['albedo'].permute(1, 2, 0).cpu().numpy()
                    albedo_viz = np.clip(albedo_viz, 0, 1)
                    axes[1].imshow(albedo_viz)
                    axes[1].set_title(f"Albedo - mean: {albedo.mean():.3f}")
                    axes[1].axis('off')
                else:
                    axes[1].text(0.5, 0.5, "No Albedo", ha='center', va='center')
                    axes[1].set_title("No Albedo")
                    axes[1].axis('off')
                
                # Shading
                if 'shading' in sample:
                    shading_viz = sample['shading'].squeeze(0).cpu().numpy()
                    shading_viz = np.clip(shading_viz, 0, 1)
                    axes[2].imshow(shading_viz, cmap='gray')
                    axes[2].set_title(f"Shading - mean: {shading.mean():.3f}")
                    axes[2].axis('off')
                else:
                    axes[2].text(0.5, 0.5, "No Shading", ha='center', va='center')
                    axes[2].set_title("No Shading")
                    axes[2].axis('off')
                
                plt.tight_layout()
                
                # Save visualization
                output_dir = Path('test_outputs')
                output_dir.mkdir(exist_ok=True)
                save_path = output_dir / f"debug_{dataset_name}_albedo.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Debug visualization saved to {save_path}")
                plt.close()
                
            else:
                print("No samples found")
                
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


def debug_batch_albedo():
    """Debug albedo in batch loading"""
    
    print(f"\n🔍 Debugging Batch Albedo Loading")
    print("=" * 50)
    
    # Test configuration
    test_config = {
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
        'derive_shading': True,
        'max_samples_per_dataset': 2
    }
    
    try:
        # Create dataset
        dataset = UnifiedIIDDataset(
            datasets_config=test_config['datasets'],
            split='train',
            image_size=test_config['image_size'],
            max_samples_per_dataset=test_config['max_samples_per_dataset'],
            derive_shading=test_config['derive_shading']
        )
        
        print(f"Dataset loaded: {len(dataset)} samples")
        
        # Create data loader
        from torch.utils.data import DataLoader
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            collate_fn=custom_collate_fn
        )
        
        # Test batch loading
        for batch in loader:
            print(f"\nBatch keys: {list(batch.keys())}")
            print(f"Batch RGB shape: {batch['rgb'].shape}")
            print(f"Batch RGB mean: {batch['rgb'].mean():.4f}")
            print(f"Batch RGB std: {batch['rgb'].std():.4f}")
            
            if 'albedo' in batch:
                albedo = batch['albedo']
                print(f"Batch Albedo shape: {albedo.shape}")
                print(f"Batch Albedo mean: {albedo.mean():.4f}")
                print(f"Batch Albedo std: {albedo.std():.4f}")
                print(f"Batch Albedo min: {albedo.min():.4f}")
                print(f"Batch Albedo max: {albedo.max():.4f}")
                
                # Check if albedo is all zeros
                non_zero_albedo = (albedo > 0.01).sum().item()
                total_albedo = albedo.numel()
                print(f"Batch Albedo non-zero pixels: {non_zero_albedo}/{total_albedo} ({100*non_zero_albedo/total_albedo:.2f}%)")
                
                # Check if albedo is all ones
                all_ones = (albedo > 0.99).sum().item()
                print(f"Batch Albedo all-ones pixels: {all_ones}/{total_albedo} ({100*all_ones/total_albedo:.2f}%)")
            
            if 'shading' in batch:
                shading = batch['shading']
                print(f"Batch Shading shape: {shading.shape}")
                print(f"Batch Shading mean: {shading.mean():.4f}")
                print(f"Batch Shading std: {shading.std():.4f}")
            
            print(f"Datasets in batch: {batch['dataset']}")
            
            # Create batch visualization
            fig, axes = plt.subplots(4, 3, figsize=(15, 20))
            
            for i in range(4):
                # RGB
                rgb = batch['rgb'][i].permute(1, 2, 0).cpu().numpy()
                rgb = np.clip(rgb, 0, 1)
                axes[i, 0].imshow(rgb)
                axes[i, 0].set_title(f"RGB - {batch['dataset'][i]}")
                axes[i, 0].axis('off')
                
                # Albedo
                if 'albedo' in batch:
                    albedo_viz = batch['albedo'][i].permute(1, 2, 0).cpu().numpy()
                    albedo_viz = np.clip(albedo_viz, 0, 1)
                    axes[i, 1].imshow(albedo_viz)
                    axes[i, 1].set_title(f"Albedo - mean: {batch['albedo'][i].mean():.3f}")
                    axes[i, 1].axis('off')
                else:
                    axes[i, 1].text(0.5, 0.5, "No Albedo", ha='center', va='center')
                    axes[i, 1].set_title("No Albedo")
                    axes[i, 1].axis('off')
                
                # Shading
                if 'shading' in batch:
                    shading_viz = batch['shading'][i].squeeze(0).cpu().numpy()
                    shading_viz = np.clip(shading_viz, 0, 1)
                    axes[i, 2].imshow(shading_viz, cmap='gray')
                    axes[i, 2].set_title(f"Shading - mean: {batch['shading'][i].mean():.3f}")
                    axes[i, 2].axis('off')
                else:
                    axes[i, 2].text(0.5, 0.5, "No Shading", ha='center', va='center')
                    axes[i, 2].set_title("No Shading")
                    axes[i, 2].axis('off')
            
            plt.tight_layout()
            
            # Save visualization
            output_dir = Path('test_outputs')
            output_dir.mkdir(exist_ok=True)
            save_path = output_dir / "debug_batch_albedo.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Batch debug visualization saved to {save_path}")
            plt.close()
            
            break
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_albedo_loading()
    debug_batch_albedo()
    print("\n✅ Debug completed! Check test_outputs/ for debug visualizations.")
