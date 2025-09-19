#!/usr/bin/env python3
"""
Test script for the unified IID dataset pipeline
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from .unified_dataset import UnifiedIIDDataset, create_iid_data_loaders, custom_collate_fn


def visualize_batch(batch, save_path=None):
    """Visualize a batch of images"""
    batch_size = batch['rgb'].shape[0]
    
    # Create subplots
    fig, axes = plt.subplots(batch_size, 4, figsize=(16, 4 * batch_size))
    if batch_size == 1:
        axes = axes.unsqueeze(0)
    
    for i in range(batch_size):
        # RGB image
        rgb = batch['rgb'][i].permute(1, 2, 0).cpu().numpy()
        rgb = np.clip(rgb, 0, 1)
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"RGB - {batch['dataset'][i]}")
        axes[i, 0].axis('off')
        
        # Albedo/Reflectance
        if 'albedo' in batch:
            albedo = batch['albedo'][i]
            if albedo.shape[0] == 1:  # Single channel
                albedo = albedo.repeat(3, 1, 1)
            albedo = albedo.permute(1, 2, 0).cpu().numpy()
            albedo = np.clip(albedo, 0, 1)
            axes[i, 1].imshow(albedo)
            axes[i, 1].set_title("Albedo/Reflectance")
            axes[i, 1].axis('off')
        else:
            axes[i, 1].text(0.5, 0.5, "No Albedo", ha='center', va='center')
            axes[i, 1].set_title("No Albedo")
            axes[i, 1].axis('off')
        
        # Shading
        if 'shading' in batch:
            shading = batch['shading'][i]
            if shading.shape[0] == 1:  # Single channel
                shading = shading.repeat(3, 1, 1)
            shading = shading.permute(1, 2, 0).cpu().numpy()
            shading = np.clip(shading, 0, 1)
            axes[i, 2].imshow(shading, cmap='gray')
            
            # Add title indicating if shading was derived
            dataset_name = batch['dataset'][i]
            if dataset_name in ['midintrinsics', 'mpi_sintel']:
                axes[i, 2].set_title("Shading (Derived)")
            else:
                axes[i, 2].set_title("Shading (GT)")
            axes[i, 2].axis('off')
        else:
            axes[i, 2].text(0.5, 0.5, "No Shading", ha='center', va='center')
            axes[i, 2].set_title("No Shading")
            axes[i, 2].axis('off')
        
        # Scene info
        axes[i, 3].text(0.1, 0.8, f"Dataset: {batch['dataset'][i]}", fontsize=12)
        axes[i, 3].text(0.1, 0.6, f"Scene: {batch['scene'][i]}", fontsize=12)
        axes[i, 3].text(0.1, 0.4, f"File: {batch['filename'][i]}", fontsize=10)
        
        # Add shading info
        if 'shading' in batch:
            shading = batch['shading'][i]
            shading_mean = shading.mean().item()
            shading_std = shading.std().item()
            axes[i, 3].text(0.1, 0.2, f"Shading mean: {shading_mean:.3f}", fontsize=10)
            axes[i, 3].text(0.1, 0.0, f"Shading std: {shading_std:.3f}", fontsize=10)
        
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def test_shading_derivation():
    """Test shading derivation specifically"""
    print(f"\n{'='*50}")
    print("Testing Shading Derivation")
    print(f"{'='*50}")
    
    # Test configuration with shading derivation enabled
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
        'batch_size': 4,
        'num_workers': 0,
        'derive_shading': True,
        'max_samples_per_dataset': 2
    }
    
    try:
        # Create dataset with shading derivation
        dataset = UnifiedIIDDataset(
            datasets_config=test_config['datasets'],
            split='train',
            image_size=test_config['image_size'],
            max_samples_per_dataset=test_config['max_samples_per_dataset'],
            derive_shading=True
        )
        
        print(f"Dataset loaded with shading derivation enabled")
        print(f"Total samples: {len(dataset)}")
        
        # Create data loader
        from torch.utils.data import DataLoader
        loader = DataLoader(
            dataset,
            batch_size=test_config['batch_size'],
            shuffle=True,
            num_workers=0,
            collate_fn=custom_collate_fn
        )
        
        # Test batch loading and analyze shading
        for batch in loader:
            print(f"\nBatch loaded successfully!")
            print(f"Batch keys: {list(batch.keys())}")
            print(f"Batch RGB shape: {batch['rgb'].shape}")
            print(f"Batch Albedo shape: {batch['albedo'].shape}")
            print(f"Batch Shading shape: {batch['shading'].shape}")
            
            # Analyze shading statistics
            shading = batch['shading']
            print(f"\nShading Statistics:")
            print(f"  Mean: {shading.mean():.4f}")
            print(f"  Std: {shading.std():.4f}")
            print(f"  Min: {shading.min():.4f}")
            print(f"  Max: {shading.max():.4f}")
            
            # Check for non-zero shading (should not be all black)
            non_zero_pixels = (shading > 0.01).sum().item()
            total_pixels = shading.numel()
            print(f"  Non-zero pixels: {non_zero_pixels}/{total_pixels} ({100*non_zero_pixels/total_pixels:.2f}%)")
            
            # Show dataset distribution
            datasets = batch['dataset']
            print(f"\nDatasets in batch: {datasets}")
            
            # Visualize batch
            output_dir = Path('test_outputs')
            output_dir.mkdir(exist_ok=True)
            save_path = output_dir / "shading_derivation_test.png"
            visualize_batch(batch, save_path)
            break
            
    except Exception as e:
        print(f"Error testing shading derivation: {e}")
        import traceback
        traceback.print_exc()


def test_dataset_loading():
    """Test dataset loading for each dataset individually"""
    
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
        'batch_size': 6,
        'num_workers': 0,  # Use 0 for debugging
        'derive_shading': True,  # Enable shading derivation
        'max_samples_per_dataset': 5  # Limit for testing
    }
    
    print("Testing dataset loading...")
    
    # Test each dataset individually
    for dataset_name in ['midintrinsics', 'mit_intrinsic', 'mpi_sintel', 'ned']:
        print(f"\n{'='*50}")
        print(f"Testing {dataset_name.upper()} dataset")
        print(f"{'='*50}")
        
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
            
            print(f"Dataset loaded successfully!")
            print(f"Number of samples: {len(dataset)}")
            
            if len(dataset) > 0:
                # Test loading a sample
                sample = dataset[0]
                print(f"Sample keys: {list(sample.keys())}")
                print(f"RGB shape: {sample['rgb'].shape}")
                if 'albedo' in sample:
                    print(f"Albedo shape: {sample['albedo'].shape}")
                if 'shading' in sample:
                    print(f"Shading shape: {sample['shading'].shape}")
                    # Check if shading is not all zeros
                    shading = sample['shading']
                    non_zero = (shading > 0.01).sum().item()
                    total = shading.numel()
                    print(f"Shading non-zero pixels: {non_zero}/{total} ({100*non_zero/total:.2f}%)")
                
                # Create data loader with custom collate function
                from torch.utils.data import DataLoader
                loader = DataLoader(
                    dataset,
                    batch_size=test_config['batch_size'],
                    shuffle=True,
                    num_workers=0,
                    collate_fn=custom_collate_fn
                )
                
                # Test batch loading
                for batch in loader:
                    print(f"Batch loaded successfully!")
                    print(f"Batch keys: {list(batch.keys())}")
                    print(f"Batch RGB shape: {batch['rgb'].shape}")
                    
                    # Check shading in batch
                    if 'shading' in batch:
                        shading = batch['shading']
                        print(f"Batch shading mean: {shading.mean():.4f}")
                        print(f"Batch shading std: {shading.std():.4f}")
                        non_zero = (shading > 0.01).sum().item()
                        total = shading.numel()
                        print(f"Batch shading non-zero: {non_zero}/{total} ({100*non_zero/total:.2f}%)")
                    
                    # Visualize batch
                    output_dir = Path('test_outputs')
                    output_dir.mkdir(exist_ok=True)
                    save_path = output_dir / f"{dataset_name}_batch.png"
                    visualize_batch(batch, save_path)
                    break
            else:
                print("No samples found in dataset")
                
        except Exception as e:
            print(f"Error loading {dataset_name} dataset: {e}")
            import traceback
            traceback.print_exc()


def test_unified_loading():
    """Test unified dataset loading with all datasets"""
    
    print(f"\n{'='*50}")
    print("Testing UNIFIED dataset loading")
    print(f"{'='*50}")
    
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
        'batch_size': 16,
        'num_workers': 0,
        'derive_shading': True,  # Enable shading derivation
        'max_samples_per_dataset': 3  # Limit for testing
    }
    
    try:
        # Create unified dataset
        dataset = UnifiedIIDDataset(
            datasets_config=test_config['datasets'],
            split='train',
            image_size=test_config['image_size'],
            max_samples_per_dataset=test_config['max_samples_per_dataset'],
            derive_shading=test_config['derive_shading']
        )
        
        print(f"Unified dataset loaded successfully!")
        print(f"Total samples: {len(dataset)}")
        
        # Show dataset distribution
        unique_datasets, counts = np.unique(dataset.dataset_names, return_counts=True)
        print("Dataset distribution:")
        for dataset_name, count in zip(unique_datasets, counts):
            print(f"  {dataset_name}: {count} samples")
        
        if len(dataset) > 0:
            # Create data loader with custom collate function
            from torch.utils.data import DataLoader
            loader = DataLoader(
                dataset,
                batch_size=test_config['batch_size'],
                shuffle=True,
                num_workers=0,
                collate_fn=custom_collate_fn
            )
            
            # Test batch loading
            for batch in loader:
                print(f"Unified batch loaded successfully!")
                print(f"Batch keys: {list(batch.keys())}")
                print(f"Batch RGB shape: {batch['rgb'].shape}")
                print(f"Datasets in batch: {batch['dataset']}")
                
                # Check shading statistics
                if 'shading' in batch:
                    shading = batch['shading']
                    print(f"Shading mean: {shading.mean():.4f}")
                    print(f"Shading std: {shading.std():.4f}")
                    non_zero = (shading > 0.01).sum().item()
                    total = shading.numel()
                    print(f"Shading non-zero pixels: {non_zero}/{total} ({100*non_zero/total:.2f}%)")
                
                # Visualize batch
                output_dir = Path('test_outputs')
                output_dir.mkdir(exist_ok=True)
                save_path = output_dir / "unified_batch.png"
                visualize_batch(batch, save_path)
                break
        else:
            print("No samples found in unified dataset")
            
    except Exception as e:
        print(f"Error loading unified dataset: {e}")
        import traceback
        traceback.print_exc()


def test_data_loaders():
    """Test the data loader factory"""
    
    print(f"\n{'='*50}")
    print("Testing Data Loader Factory")
    print(f"{'='*50}")
    
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
        'batch_size': 2,
        'num_workers': 0,
        'derive_shading': True,  # Enable shading derivation
        'max_samples_per_dataset': 2
    }
    
    try:
        # Create data loaders
        train_loader, val_loader, test_loader = create_iid_data_loaders(test_config)
        
        print(f"Data loaders created successfully!")
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        
        # Test train loader
        print("\nTesting train loader...")
        for batch in train_loader:
            print(f"Train batch loaded: {batch['rgb'].shape}")
            if 'shading' in batch:
                shading = batch['shading']
                print(f"Train shading mean: {shading.mean():.4f}")
            break
        
        # Test val loader
        print("\nTesting val loader...")
        for batch in val_loader:
            print(f"Val batch loaded: {batch['rgb'].shape}")
            if 'shading' in batch:
                shading = batch['shading']
                print(f"Val shading mean: {shading.mean():.4f}")
            break
        
        # Test test loader
        print("\nTesting test loader...")
        for batch in test_loader:
            print(f"Test batch loaded: {batch['rgb'].shape}")
            if 'shading' in batch:
                shading = batch['shading']
                print(f"Test shading mean: {shading.mean():.4f}")
            break
            
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main test function"""
    print("Starting dataset pipeline tests...")
    
    # Create output directory
    output_dir = Path('test_outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Test shading derivation specifically
    test_shading_derivation()
    
    # Test individual datasets
    test_dataset_loading()
    
    # Test unified dataset
    test_unified_loading()
    
    # Test data loaders
    test_data_loaders()
    
    print(f"\n{'='*50}")
    print("All tests completed!")
    print(f"Check 'test_outputs' directory for visualizations")
    print(f"Look for 'shading_derivation_test.png' to see derived shading")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
