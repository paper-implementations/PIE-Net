# PIE-Net Training Setup

This repository contains a complete training implementation for the PIE-Net (Photometric Invariant Edge Guided Network) for Intrinsic Image Decomposition.

## Files Overview

- `train.py` - Main training script with full implementation
- `config.py` - Configuration parameters and data preparation utilities
- `Network.py` - PIE-Net architecture (provided)
- `Utils.py` - Utility functions (provided)
- `Eval.py` - Evaluation script (provided)

## Key Features Implemented

### Loss Functions (Following the Paper)
- **Edge Losses (Le)**: Multi-scale supervision for reflectance and shading edges
- **Unrefined Losses (Lu)**: Loss for globally consistent unrefined outputs  
- **Refined Losses (Lr)**: Loss for final refined intrinsic decomposition
- **Reconstruction Loss (Lrec)**: Ensures albedo × shading = input image
- **DSSIM Loss**: Structural dissimilarity for better structural preservation
- **Perceptual Loss**: VGG16-based perceptual loss for realistic reflectance

### Training Strategy
- **Multi-scale supervision**: 64×64, 128×128, and 256×256 resolutions
- **Scale-invariant MSE**: Combined with standard MSE as mentioned in paper
- **Cross Color Ratios (CCR)**: Illumination-invariant edge computation
- **Hierarchical approach**: Global + local refinement as described in paper
- **Attention mechanisms**: Spatial attention for focusing on hard negatives

## Setup Instructions

### 1. Environment Setup
```bash
pip install torch torchvision opencv-python imageio tqdm numpy matplotlib
```

### 2. Dataset Preparation

#### Option A: Use Real Dataset (e.g., NED, MIT Intrinsics)
```bash
# Convert existing dataset to training format
python config.py --source_dir /path/to/original/dataset \
                 --target_dir /path/to/training/dataset \
                 --dataset_type ned
```

#### Option B: Generate Synthetic Data for Testing
```bash
# Generate synthetic dataset for testing the pipeline
python config.py --generate_synthetic \
                 --target_dir /path/to/synthetic/dataset \
                 --num_samples 1000
```

### 3. Expected Dataset Structure
```
dataset/
├── train/
│   ├── images/     # Input RGB images
│   ├── albedo/     # Ground truth reflectance
│   └── shading/    # Ground truth shading
└── val/
    ├── images/
    ├── albedo/
    └── shading/
```

## Training

### Basic Training
```bash
python train.py --data_root /path/to/dataset \
                --batch_size 8 \
                --num_epochs 100 \
                --learning_rate 1e-4 \
                --save_dir checkpoints
```

### Resume Training
```bash
python train.py --data_root /path/to/dataset \
                --resume checkpoints/checkpoint_epoch_50.t7
```

### Advanced Training Options
```bash
python train.py --data_root /path/to/dataset \
                --batch_size 16 \
                --num_epochs 200 \
                --learning_rate 5e-5 \
                --save_dir experiments/pienet_v1 \
                --save_freq 5
```

## Training Parameters (Following Paper)

The implementation uses the exact hyperparameters from the PIE-Net paper:

- **λu = 0.5**: Unrefined loss weight
- **λd = 0.4**: DSSIM loss weight
- **λe = 0.4**: Edge loss weight  
- **λp = 0.05**: Perceptual loss weight
- **Scale-invariant MSE weight = 0.95**
- **Standard MSE weight = 0.05**

## Expected Training Behavior

### Phase 1: Global Consistency (Early epochs)
- Network learns basic image decomposition
- Edge guidance helps separate strong illumination from reflectance
- Unrefined outputs become globally consistent

### Phase 2: Local Refinement (Later epochs)
- Refinement module eliminates hard negative illumination transitions
- Attention mechanisms focus on problematic regions
- Final outputs achieve both global and local consistency

### Loss Progression
- **Edge Loss**: Should decrease as network learns to predict edges
- **Unrefined Loss**: Decreases as global decomposition improves
- **Refined Loss**: Further decreases as local refinement improves
- **Reconstruction Loss**: Ensures physical consistency (albedo × shading = input)

## Evaluation

After training, use the provided evaluation script:

```bash
python Eval.py
# Modify the modelSaveLoc in Eval.py to point to your trained model
```

## Hardware Requirements

### Minimum Requirements
- GPU: 8GB VRAM (GTX 1070 / RTX 2070 or better)
- RAM: 16GB system memory
- Storage: 50GB for dataset + checkpoints

### Recommended Requirements  
- GPU: 16GB+ VRAM (RTX 3080 / A100 or better)
- RAM: 32GB+ system memory
- Storage: 100GB+ SSD storage

## Training Time Estimates

- **Batch size 8, 100 epochs**: ~24-48 hours on RTX 3080
- **Batch size 16, 100 epochs**: ~12-24 hours on A100
- **Convergence**: Typically achieved within 50-80 epochs

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **Dataset loading errors**: Check dataset structure and file paths
3. **Loss not decreasing**: Check learning rate, try starting with synthetic data
4. **Poor edge detection**: Ensure ground truth edges are properly computed

### Debugging Tips

1. Start with synthetic data to verify pipeline works
2. Monitor individual loss components during training
3. Visualize intermediate outputs (edges, unrefined results)
4. Use smaller image size (128×128) for faster iteration during debugging

## Citation

If you use this training code, please cite the original PIE-Net paper:

```bibtex
@inproceedings{das2022pienet,
  title={PIE-Net: Photometric Invariant Edge Guided Network for Intrinsic Image Decomposition},
  author={Das, Partha and Karaoglu, Sezer and Gevers, Theo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19790--19799},
  year={2022}
}
```

## Additional Notes

- The training implementation closely follows the original paper methodology
- Cross Color Ratios (CCR) are computed automatically during training
- Multi-scale edge supervision is implemented as described
- The hierarchical global→local refinement approach is preserved
- All loss weights match those reported in the paper

This implementation should reproduce results similar to those reported in the original PIE-Net paper when trained on appropriate datasets like NED, MIT Intrinsics, or MPI Sintel.