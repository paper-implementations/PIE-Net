#!/usr/bin/env python3
"""
Test script specifically for NED dataset albedo derivation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from unified_dataset import UnifiedIIDDataset, custom_collate_fn


def test_ned_albedo():
    """Test NED dataset albedo derivation specifically"""
    
    print("🔍 Testing NED Dataset Albedo Derivation")
    print("=" * 50)
    
    # Test configuration with only NED enabled
    test_config = {
        'datasets': {
            'midintrinsics': {
                'path': 'datasets/MIDIntrinsics',
                'enabled': False
            },
            'mit_intrinsic': {
                'path': 'datasets/MIT-intrinsic',
                'enabled': False
            },
            'mpi_sintel': {
                'path': 'datasets/MPI_Sintel',
                'enabled': False
            },
            'ned': {
                'path': 'datasets/NED',
                'enabled': True
            }
        },
        'image_size': 256,
        'derive_shading': True,
        'max_samples_per_dataset': 3
    }
    
    try:
        # Create dataset with only NED
        dataset = UnifiedIIDDataset(
            datasets_config=test_config['datasets'],
            split='train',
            image_size=test_config['image_size'],
            max_samples_per_dataset=test_config['max_samples_per_dataset'],
            derive_shading=test_config['derive_shading']
        )
        
        print(f"NED dataset loaded: {len(dataset)} samples")
        
        if len(dataset) > 0:
            # Test first sample
            sample = dataset[0]
            
            print(f"\nSample keys: {list(sample.keys())}")
            print(f"Dataset: {sample['dataset']}")
            print(f"Scene: {sample['scene']}")
            print(f"Filename: {sample['filename']}")
            
            print(f"\nRGB shape: {sample['rgb'].shape}")
            print(f"RGB mean: {sample['rgb'].mean():.4f}")
            print(f"RGB std: {sample['rgb'].std():.4f}")
            print(f"RGB min: {sample['rgb'].min():.4f}")
            print(f"RGB max: {sample['rgb'].max():.4f}")
            
            if 'albedo' in sample:
                albedo = sample['albedo']
                print(f"\nAlbedo shape: {albedo.shape}")
                print(f"Albedo mean: {albedo.mean():.4f}")
                print(f"Albedo std: {albedo.std():.4f}")
                print(f"Albedo min: {albedo.min():.4f}")
                print(f"Albedo max: {albedo.max():.4f}")
                
                # Check if albedo is meaningful
                non_zero_albedo = (albedo > 0.01).sum().item()
                total_albedo = albedo.numel()
                print(f"Albedo non-zero pixels: {non_zero_albedo}/{total_albedo} ({100*non_zero_albedo/total_albedo:.2f}%)")
                
                # Check if albedo is all zeros
                all_zeros = (albedo < 0.01).sum().item()
                print(f"Albedo all-zeros pixels: {all_zeros}/{total_albedo} ({100*all_zeros/total_albedo:.2f}%)")
            
            if 'shading' in sample:
                shading = sample['shading']
                print(f"\nShading shape: {shading.shape}")
                print(f"Shading mean: {shading.mean():.4f}")
                print(f"Shading std: {shading.std():.4f}")
                print(f"Shading min: {shading.min():.4f}")
                print(f"Shading max: {shading.max():.4f}")
                
                # Check if shading is meaningful
                non_zero_shading = (shading > 0.01).sum().item()
                total_shading = shading.numel()
                print(f"Shading non-zero pixels: {non_zero_shading}/{total_shading} ({100*non_zero_shading/total_shading:.2f}%)")
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # RGB
            rgb = sample['rgb'].permute(1, 2, 0).cpu().numpy()
            rgb = np.clip(rgb, 0, 1)
            axes[0].imshow(rgb)
            axes[0].set_title(f"RGB - NED\nmean: {sample['rgb'].mean():.3f}")
            axes[0].axis('off')
            
            # Albedo (derived)
            if 'albedo' in sample:
                albedo_viz = sample['albedo'].permute(1, 2, 0).cpu().numpy()
                albedo_viz = np.clip(albedo_viz, 0, 1)
                axes[1].imshow(albedo_viz)
                axes[1].set_title(f"Albedo (Derived)\nmean: {albedo.mean():.3f}")
                axes[1].axis('off')
            else:
                axes[1].text(0.5, 0.5, "No Albedo", ha='center', va='center')
                axes[1].set_title("No Albedo")
                axes[1].axis('off')
            
            # Shading (derived)
            if 'shading' in sample:
                shading_viz = sample['shading'].squeeze(0).cpu().numpy()
                shading_viz = np.clip(shading_viz, 0, 1)
                axes[2].imshow(shading_viz, cmap='gray')
                axes[2].set_title(f"Shading (Derived)\nmean: {shading.mean():.3f}")
                axes[2].axis('off')
            else:
                axes[2].text(0.5, 0.5, "No Shading", ha='center', va='center')
                axes[2].set_title("No Shading")
                axes[2].axis('off')
            
            plt.tight_layout()
            
            # Save visualization
            output_dir = Path('test_outputs')
            output_dir.mkdir(exist_ok=True)
            save_path = output_dir / "ned_albedo_test.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nNED visualization saved to {save_path}")
            plt.close()
            
            # Test batch loading
            from torch.utils.data import DataLoader
            loader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=True,
                num_workers=0,
                collate_fn=custom_collate_fn
            )
            
            for batch in loader:
                print(f"\nBatch loaded successfully!")
                print(f"Batch keys: {list(batch.keys())}")
                print(f"Batch RGB shape: {batch['rgb'].shape}")
                print(f"Batch Albedo shape: {batch['albedo'].shape}")
                print(f"Batch Shading shape: {batch['shading'].shape}")
                print(f"Datasets in batch: {batch['dataset']}")
                
                # Check batch statistics
                albedo_batch = batch['albedo']
                print(f"Batch Albedo mean: {albedo_batch.mean():.4f}")
                print(f"Batch Albedo std: {albedo_batch.std():.4f}")
                print(f"Batch Albedo non-zero: {(albedo_batch > 0.01).sum().item()}/{albedo_batch.numel()} ({100*(albedo_batch > 0.01).sum().item()/albedo_batch.numel():.2f}%)")
                
                shading_batch = batch['shading']
                print(f"Batch Shading mean: {shading_batch.mean():.4f}")
                print(f"Batch Shading std: {shading_batch.std():.4f}")
                print(f"Batch Shading non-zero: {(shading_batch > 0.01).sum().item()}/{shading_batch.numel()} ({100*(shading_batch > 0.01).sum().item()/shading_batch.numel():.2f}%)")
                
                break
                
        else:
            print("No NED samples found")
            
    except Exception as e:
        print(f"Error testing NED dataset: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_ned_albedo()
    print("\n✅ NED albedo test completed!")
