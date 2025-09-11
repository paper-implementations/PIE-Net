import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import imageio
import numpy as np
import cv2
from PIL import Image
import json
import scipy.io as sio
import random
from pathlib import Path


def custom_collate_fn(batch):
    """
    Custom collate function to handle samples with different keys
    """
    # Find all unique keys across all samples
    all_keys = set()
    for sample in batch:
        all_keys.update(sample.keys())
    
    # Create a batch dictionary
    batch_dict = {}
    
    for key in all_keys:
        values = []
        for sample in batch:
            if key in sample:
                values.append(sample[key])
            else:
                # Create a default value based on the key type
                if key in ['rgb', 'albedo']:
                    # Create a zero tensor for missing RGB/albedo
                    values.append(torch.zeros(3, 256, 256))
                elif key == 'shading':
                    # Create a zero tensor for missing shading
                    values.append(torch.zeros(1, 256, 256))
                elif key in ['dataset', 'scene', 'filename']:
                    # Create empty string for missing metadata
                    values.append('')
                elif key == 'metadata':
                    # Create None for missing metadata
                    values.append(None)
                else:
                    # For other keys, try to infer the type
                    if len(values) > 0:
                        # Use the same type as existing values
                        if isinstance(values[0], torch.Tensor):
                            values.append(torch.zeros_like(values[0]))
                        elif isinstance(values[0], str):
                            values.append('')
                        else:
                            values.append(None)
                    else:
                        values.append(None)
        
        # Stack tensors or create lists for other types
        if len(values) > 0 and isinstance(values[0], torch.Tensor):
            batch_dict[key] = torch.stack(values)
        else:
            batch_dict[key] = values
    
    return batch_dict


def derive_albedo_from_rgb(rgb, method='rgb_proxy'):
    """
    Derive albedo from RGB image when ground truth is not available
    
    Args:
        rgb: RGB image tensor (C, H, W) in range [0, 1]
        method: Method to use ('rgb_proxy', 'smooth_rgb', 'edge_preserving')
    
    Returns:
        albedo: Derived albedo tensor (C, H, W) in range [0, 1]
    """
    # Ensure input is tensor
    if not isinstance(rgb, torch.Tensor):
        rgb = torch.from_numpy(rgb)
    
    if method == 'rgb_proxy':
        # Simple approach: use RGB as albedo proxy
        # This works well for natural environment images
        albedo = rgb.clone()
        
    elif method == 'smooth_rgb':
        # Apply slight smoothing to reduce noise
        # Convert to numpy for OpenCV operations
        rgb_np = rgb.permute(1, 2, 0).cpu().numpy()
        
        # Apply bilateral filter to preserve edges while smoothing
        albedo_np = cv2.bilateralFilter(rgb_np, 9, 75, 75)
        
        # Convert back to tensor
        albedo = torch.from_numpy(albedo_np).permute(2, 0, 1)
        
    elif method == 'edge_preserving':
        # More sophisticated edge-preserving smoothing
        rgb_np = rgb.permute(1, 2, 0).cpu().numpy()
        
        # Apply guided filter for edge-preserving smoothing
        albedo_np = cv2.ximgproc.guidedFilter(rgb_np, rgb_np, 2, 0.1)
        
        # Convert back to tensor
        albedo = torch.from_numpy(albedo_np).permute(2, 0, 1)
        
    else:
        # Default to RGB proxy
        albedo = rgb.clone()
    
    # Ensure values are in [0, 1] range
    albedo = torch.clamp(albedo, 0, 1)
    
    return albedo


def derive_shading_from_rgb_albedo(rgb, albedo, epsilon=1e-8):
    """
    Derive shading from RGB and albedo using the relationship: I = R * S
    Therefore: S = I / R
    
    Args:
        rgb: RGB image tensor (C, H, W) in range [0, 1]
        albedo: Albedo/reflectance tensor (C, H, W) in range [0, 1]
        epsilon: Small value to prevent division by zero
    
    Returns:
        shading: Derived shading tensor (1, H, W) in range [0, 1]
    """
    # Ensure inputs are tensors
    if not isinstance(rgb, torch.Tensor):
        rgb = torch.from_numpy(rgb)
    if not isinstance(albedo, torch.Tensor):
        albedo = torch.from_numpy(albedo)
    
    # Convert RGB to grayscale using standard weights
    if rgb.shape[0] == 3:  # RGB
        rgb_gray = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    else:
        rgb_gray = rgb.squeeze(0)
    
    # Convert albedo to grayscale
    if albedo.shape[0] == 3:  # RGB albedo
        albedo_gray = 0.299 * albedo[0] + 0.587 * albedo[1] + 0.114 * albedo[2]
    else:
        albedo_gray = albedo.squeeze(0)
    
    # Add epsilon to albedo to prevent division by zero
    albedo_safe = albedo_gray + epsilon
    
    # Derive shading: S = I / R
    shading = rgb_gray / albedo_safe
    
    # Handle cases where albedo is very small (dark regions)
    # In such cases, we can't reliably derive shading, so use a fallback
    # Use the RGB intensity as a proxy for shading
    low_albedo_mask = albedo_gray < 0.1
    if low_albedo_mask.sum() > 0:
        # For low albedo regions, use RGB intensity as shading proxy
        shading[low_albedo_mask] = rgb_gray[low_albedo_mask]
    
    # Normalize shading to [0, 1] range
    shading_min = shading.min()
    shading_max = shading.max()
    
    if shading_max > shading_min:
        shading = (shading - shading_min) / (shading_max - shading_min)
    else:
        # If all values are the same, create a reasonable shading
        shading = torch.ones_like(shading) * 0.5
    
    # Add channel dimension
    shading = shading.unsqueeze(0)  # (1, H, W)
    
    return shading


class UnifiedIIDDataset(Dataset):
    """
    Unified dataset class for Intrinsic Image Decomposition (IID) training
    Supports multiple datasets: MIDIntrinsics, MIT-intrinsic, MPI_Sintel, NED
    """
    
    def __init__(self, 
                 datasets_config,
                 split='train', 
                 image_size=256, 
                 transform=None,
                 max_samples_per_dataset=None,
                 derive_shading=True):
        """
        Args:
            datasets_config: Dict with dataset paths and configurations
            split: 'train', 'val', or 'test'
            image_size: Target image size
            transform: Optional transforms
            max_samples_per_dataset: Limit samples per dataset (for debugging)
            derive_shading: Whether to derive shading from RGB and albedo when not available
        """
        self.datasets_config = datasets_config
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.max_samples_per_dataset = max_samples_per_dataset
        self.derive_shading = derive_shading
        
        # Initialize dataset samples
        self.samples = []
        self.dataset_names = []
        
        # Load samples from each dataset
        self._load_midintrinsics_samples()
        self._load_mit_intrinsic_samples()
        self._load_mpi_sintel_samples()
        self._load_ned_samples()
        
        print(f"Loaded {len(self.samples)} total samples for {split} split")
        print(f"Dataset distribution: {dict(zip(*np.unique(self.dataset_names, return_counts=True)))}")
    
    def _load_midintrinsics_samples(self):
        """Load MIDIntrinsics dataset samples"""
        if 'midintrinsics' not in self.datasets_config:
            return
            
        config = self.datasets_config['midintrinsics']
        
        # Check if dataset is enabled
        if not config.get('enabled', True):
            return
            
        base_path = config['path']
        
        if self.split == 'train':
            albedo_path = os.path.join(base_path, 'train_albedo')
            multi_illum_path = os.path.join(base_path, 'multi_illumination_train_mip2_jpg')
        else:
            albedo_path = os.path.join(base_path, 'test_albedo')
            multi_illum_path = os.path.join(base_path, 'multi_illumination_test_mip2_jpg')
        
        if not os.path.exists(albedo_path) or not os.path.exists(multi_illum_path):
            print(f"MIDIntrinsics {self.split} paths not found")
            return
        
        # Get scene directories
        albedo_scenes = [d for d in os.listdir(albedo_path) if os.path.isdir(os.path.join(albedo_path, d))]
        
        for scene in albedo_scenes:
            albedo_file = os.path.join(albedo_path, scene, 'albedo.exr')
            multi_illum_dir = os.path.join(multi_illum_path, scene)
            
            if not os.path.exists(albedo_file) or not os.path.exists(multi_illum_dir):
                continue
            
            # Get illumination images
            illum_files = glob.glob(os.path.join(multi_illum_dir, 'dir_*.jpg'))
            
            # Limit samples if specified
            if self.max_samples_per_dataset:
                illum_files = illum_files[:self.max_samples_per_dataset]
            
            for illum_file in illum_files:
                sample = {
                    'dataset': 'midintrinsics',
                    'scene': scene,
                    'rgb_path': illum_file,
                    'albedo_path': albedo_file,
                    'shading_path': None,  # MIDIntrinsics doesn't have explicit shading
                    'meta_path': os.path.join(multi_illum_dir, 'meta.json')
                }
                self.samples.append(sample)
                self.dataset_names.append('midintrinsics')
    
    def _load_mit_intrinsic_samples(self):
        """Load MIT-intrinsic dataset samples"""
        if 'mit_intrinsic' not in self.datasets_config:
            return
            
        config = self.datasets_config['mit_intrinsic']
        
        # Check if dataset is enabled
        if not config.get('enabled', True):
            return
            
        base_path = config['path']
        data_path = os.path.join(base_path, 'data')
        
        if not os.path.exists(data_path):
            print(f"MIT-intrinsic data path not found: {data_path}")
            return
        
        # Get object directories
        objects = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        
        for obj in objects:
            obj_path = os.path.join(data_path, obj)
            
            # Check for required files
            reflectance_file = os.path.join(obj_path, 'reflectance.png')
            shading_file = os.path.join(obj_path, 'shading.png')
            
            if not os.path.exists(reflectance_file) or not os.path.exists(shading_file):
                continue
            
            # Get illumination images (light*.png files)
            illum_files = glob.glob(os.path.join(obj_path, 'light*.png'))
            
            # Limit samples if specified
            if self.max_samples_per_dataset:
                illum_files = illum_files[:self.max_samples_per_dataset]
            
            for illum_file in illum_files:
                sample = {
                    'dataset': 'mit_intrinsic',
                    'scene': obj,
                    'rgb_path': illum_file,
                    'albedo_path': reflectance_file,  # MIT uses 'reflectance' instead of 'albedo'
                    'shading_path': shading_file,
                    'meta_path': None
                }
                self.samples.append(sample)
                self.dataset_names.append('mit_intrinsic')
    
    def _load_mpi_sintel_samples(self):
        """Load MPI_Sintel dataset samples"""
        if 'mpi_sintel' not in self.datasets_config:
            return
            
        config = self.datasets_config['mpi_sintel']
        
        # Check if dataset is enabled
        if not config.get('enabled', True):
            return
            
        base_path = config['path']
        
        if self.split == 'train':
            final_path = os.path.join(base_path, 'training', 'final')
            albedo_path = os.path.join(base_path, 'training', 'albedo')
        else:
            final_path = os.path.join(base_path, 'test', 'final')
            albedo_path = os.path.join(base_path, 'test', 'albedo')
        
        if not os.path.exists(final_path) or not os.path.exists(albedo_path):
            print(f"MPI_Sintel {self.split} paths not found")
            return
        
        # Get sequence directories
        sequences = [d for d in os.listdir(final_path) if os.path.isdir(os.path.join(final_path, d))]
        
        for seq in sequences:
            final_seq_path = os.path.join(final_path, seq)
            albedo_seq_path = os.path.join(albedo_path, seq)
            
            if not os.path.exists(albedo_seq_path):
                continue
            
            # Get frame files
            final_files = sorted(glob.glob(os.path.join(final_seq_path, 'frame_*.png')))
            albedo_files = sorted(glob.glob(os.path.join(albedo_seq_path, 'frame_*.png')))
            
            # Match frames
            for final_file, albedo_file in zip(final_files, albedo_files):
                sample = {
                    'dataset': 'mpi_sintel',
                    'scene': seq,
                    'rgb_path': final_file,
                    'albedo_path': albedo_file,
                    'shading_path': None,  # MPI_Sintel doesn't have explicit shading
                    'meta_path': None
                }
                self.samples.append(sample)
                self.dataset_names.append('mpi_sintel')
                
                # Limit samples if specified
                if self.max_samples_per_dataset and len([s for s in self.samples if s['dataset'] == 'mpi_sintel']) >= self.max_samples_per_dataset:
                    break
    
    def _load_ned_samples(self):
        """Load NED (Natural Environment Dataset) samples"""
        if 'ned' not in self.datasets_config:
            return
            
        config = self.datasets_config['ned']
        
        # Check if dataset is enabled
        if not config.get('enabled', True):
            return
            
        base_path = config['path']
        
        # NED has different lighting conditions as subdirectories
        lighting_conditions = ['clear', 'cloudy', 'overcast', 'sunset', 'twilight']
        
        for condition in lighting_conditions:
            condition_path = os.path.join(base_path, condition)
            if not os.path.exists(condition_path):
                continue
            
            # Get .mat files
            mat_files = glob.glob(os.path.join(condition_path, '*.mat'))
            
            # Limit samples if specified
            if self.max_samples_per_dataset:
                mat_files = mat_files[:self.max_samples_per_dataset]
            
            for mat_file in mat_files:
                sample = {
                    'dataset': 'ned',
                    'scene': f"{condition}_{os.path.splitext(os.path.basename(mat_file))[0]}",
                    'rgb_path': mat_file,  # NED stores data in .mat format
                    'albedo_path': None,  # NED doesn't have explicit albedo
                    'shading_path': None,  # NED doesn't have explicit shading
                    'meta_path': None,
                    'lighting_condition': condition
                }
                self.samples.append(sample)
                self.dataset_names.append('ned')
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        dataset_name = sample['dataset']
        
        # Load RGB image
        rgb_img = self._load_rgb_image(sample['rgb_path'], dataset_name)
        
        # Load ground truth if available
        albedo_img = None
        shading_img = None
        
        if sample['albedo_path'] and os.path.exists(sample['albedo_path']):
            albedo_img = self._load_ground_truth(sample['albedo_path'], 'albedo')
        elif sample['dataset'] == 'ned':
            # For NED dataset, derive albedo from RGB
            albedo_img = derive_albedo_from_rgb(rgb_img, method='rgb_proxy')
        
        if sample['shading_path'] and os.path.exists(sample['shading_path']):
            shading_img = self._load_ground_truth(sample['shading_path'], 'shading')
        
        # Derive shading from RGB and albedo if not available and derivation is enabled
        if shading_img is None and albedo_img is not None and self.derive_shading:
            try:
                shading_img = derive_shading_from_rgb_albedo(rgb_img, albedo_img)
            except Exception as e:
                print(f"Error deriving shading for {sample['rgb_path']}: {e}")
                shading_img = torch.zeros(1, self.image_size, self.image_size)
        
        # Create output sample with consistent structure
        output_sample = {
            'rgb': rgb_img,
            'dataset': dataset_name,
            'scene': sample['scene'],
            'filename': os.path.splitext(os.path.basename(sample['rgb_path']))[0]
        }
        
        # Add ground truth if available
        if albedo_img is not None:
            output_sample['albedo'] = albedo_img
        else:
            # Create a dummy albedo tensor for consistency
            output_sample['albedo'] = torch.zeros(3, self.image_size, self.image_size)
        
        if shading_img is not None:
            output_sample['shading'] = shading_img
        else:
            # Create a dummy shading tensor for consistency
            output_sample['shading'] = torch.zeros(1, self.image_size, self.image_size)
        
        # Add metadata if available
        if sample['meta_path'] and os.path.exists(sample['meta_path']):
            try:
                with open(sample['meta_path'], 'r') as f:
                    output_sample['metadata'] = json.load(f)
            except:
                output_sample['metadata'] = None
        else:
            output_sample['metadata'] = None
        
        # Apply transforms if specified
        if self.transform:
            output_sample = self.transform(output_sample)
        
        return output_sample
    
    def _load_rgb_image(self, img_path, dataset_name):
        """Load RGB image based on dataset format"""
        if dataset_name == 'ned':
            # NED uses .mat files
            return self._load_ned_mat_image(img_path)
        else:
            # Other datasets use image files
            return self._load_standard_image(img_path)
    
    def _load_ned_mat_image(self, mat_path):
        """Load image from NED .mat file"""
        try:
            mat_data = sio.loadmat(mat_path)
            # NED .mat files typically contain 'image' or 'img' field
            if 'image' in mat_data:
                img = mat_data['image']
            elif 'img' in mat_data:
                img = mat_data['img']
            else:
                # Try to find any 3D array
                for key in mat_data.keys():
                    if isinstance(mat_data[key], np.ndarray) and mat_data[key].ndim == 3:
                        img = mat_data[key]
                        break
                else:
                    raise ValueError(f"No image data found in {mat_path}")
            
            # Ensure proper format
            if img.dtype != np.float32:
                img = img.astype(np.float32)
            
            # Normalize to [0, 1] if needed
            if img.max() > 1.0:
                img = img / 255.0
            
            # Resize
            img = cv2.resize(img, (self.image_size, self.image_size))
            
            # Convert to tensor (C, H, W)
            img = torch.from_numpy(img).permute(2, 0, 1)
            
            return img
            
        except Exception as e:
            print(f"Error loading NED mat file {mat_path}: {e}")
            # Return dummy image
            return torch.zeros(3, self.image_size, self.image_size)
    
    def _load_standard_image(self, img_path):
        """Load standard image file (PNG, JPG, etc.)"""
        try:
            img = imageio.imread(img_path)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros(3, self.image_size, self.image_size)
        
        # Handle different image types
        if len(img.shape) == 2:  # Grayscale
            img = np.stack([img, img, img], axis=-1)
        elif len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
            img = img[:, :, :3]
        
        # Resize
        img = cv2.resize(img, (self.image_size, self.image_size))
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Handle NaN values
        img[np.isnan(img)] = 0
        
        # Convert to tensor (C, H, W)
        img = torch.from_numpy(img).permute(2, 0, 1)
        
        return img
    
    def _load_ground_truth(self, gt_path, gt_type):
        """Load ground truth image (albedo or shading)"""
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
            img = cv2.resize(img, (self.image_size, self.image_size))
            
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
                return torch.zeros(1, self.image_size, self.image_size)
            else:
                return torch.zeros(3, self.image_size, self.image_size)


class IIDDataLoader:
    """Data loader factory for IID datasets"""
    
    def __init__(self, config):
        self.config = config
    
    def create_train_loader(self):
        """Create training data loader"""
        train_dataset = UnifiedIIDDataset(
            datasets_config=self.config['datasets'],
            split='train',
            image_size=self.config.get('image_size', 256),
            max_samples_per_dataset=self.config.get('max_samples_per_dataset', None),
            derive_shading=self.config.get('derive_shading', True)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 8),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            drop_last=True,
            collate_fn=custom_collate_fn
        )
        
        return train_loader
    
    def create_val_loader(self):
        """Create validation data loader"""
        val_dataset = UnifiedIIDDataset(
            datasets_config=self.config['datasets'],
            split='val',
            image_size=self.config.get('image_size', 256),
            max_samples_per_dataset=self.config.get('max_samples_per_dataset', None),
            derive_shading=self.config.get('derive_shading', True)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 8),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            drop_last=False,
            collate_fn=custom_collate_fn
        )
        
        return val_loader
    
    def create_test_loader(self):
        """Create test data loader"""
        test_dataset = UnifiedIIDDataset(
            datasets_config=self.config['datasets'],
            split='test',
            image_size=self.config.get('image_size', 256),
            max_samples_per_dataset=self.config.get('max_samples_per_dataset', None),
            derive_shading=self.config.get('derive_shading', True)
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.get('batch_size', 1),
            shuffle=False,
            num_workers=self.config.get('num_workers', 2),
            pin_memory=True,
            drop_last=False,
            collate_fn=custom_collate_fn
        )
        
        return test_loader


# Example configuration
EXAMPLE_CONFIG = {
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
    'batch_size': 8,
    'num_workers': 4,
    'derive_shading': True,  # Enable shading derivation
    'max_samples_per_dataset': None  # Set to a number for debugging
}


def create_iid_data_loaders(config):
    """Convenience function to create all data loaders"""
    loader_factory = IIDDataLoader(config)
    
    train_loader = loader_factory.create_train_loader()
    val_loader = loader_factory.create_val_loader()
    test_loader = loader_factory.create_test_loader()
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    config = EXAMPLE_CONFIG.copy()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_iid_data_loaders(config)
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Test a batch
    for batch in train_loader:
        print(f"Batch keys: {batch.keys()}")
        print(f"RGB shape: {batch['rgb'].shape}")
        if 'albedo' in batch:
            print(f"Albedo shape: {batch['albedo'].shape}")
        if 'shading' in batch:
            print(f"Shading shape: {batch['shading'].shape}")
        break
