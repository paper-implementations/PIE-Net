#!/usr/bin/env python3
"""
Quick start script for the IID dataset pipeline
Demonstrates how to use the unified dataset with the DecScaleClampedIllumEdgeGuidedNetworkBatchNorm
"""

import torch
from src import *
from train_iid import IIDLoss, IIDTrainer


def main():
    print("🚀 Quick Start: IID Dataset Pipeline")
    print("=" * 50)
    
    # Configuration for quick testing
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
        'batch_size': 4,  # Smaller batch size for testing
        'num_workers': 2,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'lr_step_size': 10,
        'lr_gamma': 0.5,
        'lambda_recon': 1.0,
        'lambda_albedo': 1.0,
        'lambda_shading': 1.0,
        'lambda_smoothness': 0.1,
        'log_dir': 'logs/iid_quick_start',
        'output_dir': 'outputs/iid_quick_start',
        'log_interval': 10,
        'save_interval': 5,
        'max_samples_per_dataset': 50  # Limit for quick testing
    }
    
    print("📊 Creating data loaders...")
    
    # Create data loaders
    train_loader, val_loader, test_loader = unified_dataset.create_iid_data_loaders(config)
    
    print(f"✅ Training samples: {len(train_loader.dataset)}")
    print(f"✅ Validation samples: {len(val_loader.dataset)}")
    print(f"✅ Test samples: {len(test_loader.dataset)}")
    
    # Test a batch
    print("\n🔍 Testing batch loading...")
    for batch in train_loader:
        print(f"   Batch keys: {list(batch.keys())}")
        print(f"   RGB shape: {batch['rgb'].shape}")
        print(f"   Albedo shape: {batch['albedo'].shape}")
        print(f"   Shading shape: {batch['shading'].shape}")
        print(f"   Datasets in batch: {batch['dataset']}")
        break
    
    print("\n🧠 Creating model...")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Network.DecScaleClampedIllumEdgeGuidedNetworkBatchNorm().to(device)
    
    print(f"✅ Model created on {device}")
    print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test model forward pass
    print("\n🔍 Testing model forward pass...")
    model.eval()
    with torch.no_grad():
        test_input = batch['rgb'].to(device)
        output = model(test_input)
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output keys: {list(output.keys())}")
        print(f"   Reflectance shape: {output['reflectance'].shape}")
        print(f"   Shading shape: {output['shading'].shape}")
        print(f"   Reconstruction shape: {output['recon'].shape}")
    
    print("\n🎯 Creating loss function...")
    
    # Create loss function
    criterion = IIDLoss(
        lambda_recon=config['lambda_recon'],
        lambda_albedo=config['lambda_albedo'],
        lambda_shading=config['lambda_shading'],
        lambda_smoothness=config['lambda_smoothness']
    )
    
    print("✅ Loss function created")
    
    # Test loss computation
    print("\n🔍 Testing loss computation...")
    targets = {
        'rgb': batch['rgb'].to(device),
        'albedo': batch['albedo'].to(device),
        'shading': batch['shading'].to(device)
    }
    
    loss, loss_dict = criterion(output, targets)
    print(f"   Total loss: {loss.item():.4f}")
    for key, value in loss_dict.items():
        print(f"   {key}: {value:.4f}")
    
    print("\n🎉 Dataset pipeline is working correctly!")
    print("\n📝 Next steps:")
    print("   1. Run full training: python train_iid.py --epochs 100")
    print("   2. Test the pipeline: python test_dataset.py")
    print("   3. Run inference: python inference.py --model outputs/iid_model/best_model.pth --input your_image.jpg")
    
    print("\n🔧 Configuration options:")
    print("   - Adjust batch_size for your GPU memory")
    print("   - Modify max_samples_per_dataset for faster testing")
    print("   - Enable/disable datasets in the config")
    print("   - Adjust loss weights (lambda_*) for different training objectives")


if __name__ == "__main__":
    main()
