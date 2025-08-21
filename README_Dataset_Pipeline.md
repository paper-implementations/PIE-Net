# Unified Dataset Pipeline for Intrinsic Image Decomposition (IID)

This repository provides a unified dataset pipeline for training and testing intrinsic image decomposition models using multiple datasets. The pipeline supports four major IID datasets and is designed to work with the `DecScaleClampedIllumEdgeGuidedNetworkBatchNorm` network.

## Supported Datasets

### 1. MIDIntrinsics Dataset
- **Path**: `datasets/MIDIntrinsics/`
- **Structure**: 
  - `train_albedo/` - Ground truth albedo maps (.exr format)
  - `test_albedo/` - Test albedo maps
  - `multi_illumination_train_mip2_jpg/` - Multi-illumination images
  - `multi_illumination_test_mip2_jpg/` - Test multi-illumination images
- **Ground Truth**: Albedo maps (reflectance)
- **Input**: Multi-illumination images under different lighting conditions

### 2. MIT-Intrinsic Dataset
- **Path**: `datasets/MIT-intrinsic/`
- **Structure**: `data/` containing object directories with:
  - `light*.png` - Images under different lighting
  - `reflectance.png` - Ground truth reflectance
  - `shading.png` - Ground truth shading
- **Ground Truth**: Both reflectance and shading
- **Input**: Images under controlled lighting conditions

### 3. MPI Sintel Dataset
- **Path**: `datasets/MPI_Sintel/`
- **Structure**:
  - `training/final/` - Final rendered images
  - `training/albedo/` - Ground truth albedo maps
  - `test/final/` and `test/albedo/` for testing
- **Ground Truth**: Albedo maps
- **Input**: Rendered images from animated sequences

### 4. NED (Natural Environment Dataset)
- **Path**: `datasets/NED/`
- **Structure**: Lighting condition subdirectories:
  - `clear/`, `cloudy/`, `overcast/`, `sunset/`, `twilight/`
- **Format**: `.mat` files (MATLAB format)
- **Ground Truth**: None (used for unsupervised training)
- **Input**: Natural environment images

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your datasets are organized according to the structure described above.

## Usage

### 1. Testing the Dataset Pipeline

First, test that the dataset pipeline works correctly:

```bash
python test_dataset.py
```

This will:
- Test each dataset individually
- Test the unified dataset loading
- Generate visualizations in the `test_outputs/` directory
- Verify that all data can be loaded properly

### 2. Training the IID Model

#### Basic Training
```bash
python train_iid.py --epochs 100
```

#### Training with Custom Configuration
```bash
python train_iid.py --config config.yaml --epochs 200
```

#### Resuming from Checkpoint
```bash
python train_iid.py --resume outputs/iid_model/checkpoint_epoch_50.pth --epochs 100
```

#### Testing Only
```bash
python train_iid.py --test
```

### 3. Configuration

The training can be configured using the `config.yaml` file or by modifying the default configuration in `train_iid.py`.

Key configuration options:

```yaml
# Dataset Configuration
datasets:
  midintrinsics:
    path: "datasets/MIDIntrinsics"
    enabled: true
  mit_intrinsic:
    path: "datasets/MIT-intrinsic"
    enabled: true
  mpi_sintel:
    path: "datasets/MPI_Sintel"
    enabled: true
  ned:
    path: "datasets/NED"
    enabled: true

# Training Configuration
training:
  image_size: 256
  batch_size: 8
  num_workers: 4
  learning_rate: 1e-4
  weight_decay: 1e-4
  lr_step_size: 50
  lr_gamma: 0.5
  num_epochs: 100

# Loss Configuration
loss:
  lambda_recon: 1.0      # Reconstruction loss weight
  lambda_albedo: 1.0     # Albedo loss weight
  lambda_shading: 1.0    # Shading loss weight
  lambda_smoothness: 0.1 # Smoothness loss weight
```

## Dataset Pipeline Features

### 1. Unified Dataset Class (`UnifiedIIDDataset`)

The main dataset class that handles all four datasets:

```python
from unified_dataset import UnifiedIIDDataset

# Create dataset
dataset = UnifiedIIDDataset(
    datasets_config=config['datasets'],
    split='train',
    image_size=256,
    max_samples_per_dataset=None
)

# Get a sample
sample = dataset[0]
print(sample.keys())  # ['rgb', 'dataset', 'scene', 'filename', 'albedo', 'shading']
```

### 2. Data Loader Factory (`IIDDataLoader`)

Convenient factory for creating train/val/test data loaders:

```python
from unified_dataset import create_iid_data_loaders

# Create all data loaders
train_loader, val_loader, test_loader = create_iid_data_loaders(config)
```

### 3. Loss Functions (`IIDLoss`)

Comprehensive loss function for IID training:

- **Reconstruction Loss**: Ensures I = R × S
- **Albedo Loss**: Supervised loss for reflectance
- **Shading Loss**: Supervised loss for shading
- **Smoothness Loss**: Encourages smooth albedo and shading

### 4. Training Pipeline (`IIDTrainer`)

Complete training pipeline with:

- Model training and validation
- Checkpoint saving/loading
- TensorBoard logging
- Learning rate scheduling
- Best model selection

## Output Structure

The training pipeline creates the following output structure:

```
outputs/iid_model/
├── best_model.pth          # Best model based on validation loss
├── final_model.pth         # Final model after training
├── checkpoint_epoch_X.pth  # Periodic checkpoints
└── logs/                   # TensorBoard logs
```

## Monitoring Training

Use TensorBoard to monitor training progress:

```bash
tensorboard --logdir logs/iid_training
```

Available metrics:
- Training and validation losses
- Individual loss components (reconstruction, albedo, shading, smoothness)
- Learning rate progression

## Dataset-Specific Notes

### MIDIntrinsics
- Uses EXR format for albedo maps (requires OpenEXR)
- Multiple illumination images per scene
- No explicit shading ground truth

### MIT-Intrinsic
- Complete ground truth (both reflectance and shading)
- Controlled lighting conditions
- Good for supervised training

### MPI Sintel
- Synthetic dataset with albedo ground truth
- No explicit shading ground truth
- Useful for testing generalization

### NED
- Natural environment images
- No ground truth available
- Useful for unsupervised or self-supervised training

## Troubleshooting

### Common Issues

1. **OpenEXR not found**: Install OpenEXR for MIDIntrinsics dataset
   ```bash
   pip install OpenEXR
   ```

2. **Memory issues**: Reduce batch size or image size in config

3. **Dataset not found**: Check dataset paths in configuration

4. **CUDA out of memory**: Reduce batch size or use gradient accumulation

### Debug Mode

For debugging, limit samples per dataset:

```python
config['max_samples_per_dataset'] = 10  # Only load 10 samples per dataset
```

## Citation

If you use this dataset pipeline in your research, please cite the original datasets:

- MIDIntrinsics: [Paper reference]
- MIT-Intrinsic: [Paper reference]  
- MPI Sintel: [Paper reference]
- NED: [Paper reference]

## License

This code is provided for research purposes. Please check the licenses of the individual datasets for commercial use.
