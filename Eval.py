import os
import time
import json
import argparse
from tqdm import tqdm
import numpy as np
import imageio
import glob
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Network import DecScaleClampedIllumEdgeGuidedNetworkBatchNorm
from Utils import mor_utils
from dataset import PIENetTestDataset, PIENetDataset
from losses import SSIMLoss, ScaleInvariantMSELoss

torch.backends.cudnn.benchmark = True


class IntrinsicEvaluator:
    """Comprehensive evaluator for intrinsic image decomposition"""
    
    def __init__(self, device):
        self.device = device
        self.ssim_loss = SSIMLoss()
        self.scale_invariant_mse = ScaleInvariantMSELoss()
        
        # Initialize metric storage
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reset all metrics for new evaluation"""
        self.metrics = {
            'reflectance_mse': [],
            'reflectance_lmse': [],  # Local/Scale-invariant MSE
            'reflectance_ssim': [],
            'reflectance_dssim': [],
            'shading_mse': [],
            'shading_lmse': [],
            'shading_ssim': [],
            'shading_dssim': [],
            'reconstruction_mse': [],
            'reconstruction_ssim': [],
            'inference_times': [],
        }
    
    def compute_mse(self, pred, target):
        """Compute Mean Squared Error"""
        return F.mse_loss(pred, target).item()
    
    def compute_lmse(self, pred, target):
        """Compute Local Mean Squared Error (Scale-Invariant MSE)"""
        return self.scale_invariant_mse(pred, target).item()
    
    def compute_ssim(self, pred, target):
        """Compute Structural Similarity Index"""
        # SSIM returns dissimilarity, so we convert to similarity
        dssim = self.ssim_loss(pred, target).item()
        ssim = 1.0 - dssim
        return ssim, dssim
    
    def compute_whdr(self, pred_reflectance, comparisons):
        """
        Compute Weighted Human Disagreement Rate (WHDR)
        Used for IIW dataset evaluation
        
        Args:
            pred_reflectance: Predicted reflectance image
            comparisons: List of human comparisons [(point1, point2, darker_point), ...]
        """
        if not comparisons:
            return None
            
        total_weight = 0
        weighted_error = 0
        
        for comp in comparisons:
            point1, point2, darker_point, weight = comp
            
            # Get reflectance values at comparison points
            r1 = pred_reflectance[point1[1], point1[0]].mean().item()
            r2 = pred_reflectance[point2[1], point2[0]].mean().item()
            
            # Determine if prediction agrees with human judgment
            pred_darker = 1 if r1 < r2 else 2
            human_darker = darker_point
            
            # Add to error if prediction disagrees
            if pred_darker != human_darker:
                weighted_error += weight
            total_weight += weight
        
        return weighted_error / total_weight if total_weight > 0 else 0
    
    def evaluate_sample(self, outputs, targets, input_img, inference_time):
        """Evaluate a single sample"""
        sample_metrics = {}
        
        # Store inference time
        self.metrics['inference_times'].append(inference_time)
        
        # Reconstruction metrics (always available)
        recon_mse = self.compute_mse(outputs['recon'], input_img)
        recon_ssim, _ = self.compute_ssim(outputs['recon'], input_img)
        
        self.metrics['reconstruction_mse'].append(recon_mse)
        self.metrics['reconstruction_ssim'].append(recon_ssim)
        
        sample_metrics['reconstruction_mse'] = recon_mse
        sample_metrics['reconstruction_ssim'] = recon_ssim
        
        # Reflectance metrics (if ground truth available)
        if 'albedo' in targets:
            gt_albedo = targets['albedo']
            pred_albedo = outputs['reflectance']
            
            # Ensure same dimensions
            if gt_albedo.shape != pred_albedo.shape:
                gt_albedo = F.interpolate(gt_albedo, size=pred_albedo.shape[-2:], 
                                        mode='bilinear', align_corners=False)
            
            ref_mse = self.compute_mse(pred_albedo, gt_albedo)
            ref_lmse = self.compute_lmse(pred_albedo, gt_albedo)
            ref_ssim, ref_dssim = self.compute_ssim(pred_albedo, gt_albedo)
            
            self.metrics['reflectance_mse'].append(ref_mse)
            self.metrics['reflectance_lmse'].append(ref_lmse)
            self.metrics['reflectance_ssim'].append(ref_ssim)
            self.metrics['reflectance_dssim'].append(ref_dssim)
            
            sample_metrics.update({
                'reflectance_mse': ref_mse,
                'reflectance_lmse': ref_lmse,
                'reflectance_ssim': ref_ssim,
                'reflectance_dssim': ref_dssim,
            })
        
        # Shading metrics (if ground truth available)
        if 'shading' in targets:
            gt_shading = targets['shading']
            pred_shading = outputs['shading']
            
            # Handle dimension mismatch
            if len(gt_shading.shape) == 3 and gt_shading.shape[0] == 1:
                gt_shading = gt_shading.unsqueeze(0)
            if len(pred_shading.shape) == 3:
                pred_shading = pred_shading.unsqueeze(0)
            
            # Ensure same dimensions
            if gt_shading.shape != pred_shading.shape:
                gt_shading = F.interpolate(gt_shading, size=pred_shading.shape[-2:], 
                                         mode='bilinear', align_corners=False)
            
            shd_mse = self.compute_mse(pred_shading, gt_shading)
            shd_lmse = self.compute_lmse(pred_shading, gt_shading)
            shd_ssim, shd_dssim = self.compute_ssim(pred_shading, gt_shading)
            
            self.metrics['shading_mse'].append(shd_mse)
            self.metrics['shading_lmse'].append(shd_lmse)
            self.metrics['shading_ssim'].append(shd_ssim)
            self.metrics['shading_dssim'].append(shd_dssim)
            
            sample_metrics.update({
                'shading_mse': shd_mse,
                'shading_lmse': shd_lmse,
                'shading_ssim': shd_ssim,
                'shading_dssim': shd_dssim,
            })
        
        return sample_metrics
    
    def get_summary_metrics(self):
        """Get summary statistics for all metrics"""
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if values:  # Only if we have data
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        return summary
    
    def save_results(self, save_path, sample_results=None):
        """Save evaluation results to JSON"""
        results = {
            'summary_metrics': self.get_summary_metrics(),
            'raw_metrics': self.metrics
        }
        
        if sample_results:
            results['per_sample_metrics'] = sample_results
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"[*] Evaluation results saved to: {save_path}")
    
    def print_summary(self):
        """Print summary of evaluation metrics"""
        summary = self.get_summary_metrics()
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        # Performance metrics
        if 'inference_times' in summary:
            print(f"Inference Time: {summary['inference_times']['mean']:.4f}s ± {summary['inference_times']['std']:.4f}s")
            print(f"FPS: {1.0/summary['inference_times']['mean']:.2f}")
        
        # Reconstruction metrics (always available)
        print("\nRECONSTRUCTION METRICS:")
        if 'reconstruction_mse' in summary:
            print(f"  MSE: {summary['reconstruction_mse']['mean']:.6f} ± {summary['reconstruction_mse']['std']:.6f}")
        if 'reconstruction_ssim' in summary:
            print(f"  SSIM: {summary['reconstruction_ssim']['mean']:.4f} ± {summary['reconstruction_ssim']['std']:.4f}")
        
        # Reflectance metrics
        if 'reflectance_mse' in summary:
            print("\nREFLECTANCE METRICS:")
            print(f"  MSE: {summary['reflectance_mse']['mean']:.6f} ± {summary['reflectance_mse']['std']:.6f}")
            print(f"  LMSE: {summary['reflectance_lmse']['mean']:.6f} ± {summary['reflectance_lmse']['std']:.6f}")
            print(f"  SSIM: {summary['reflectance_ssim']['mean']:.4f} ± {summary['reflectance_ssim']['std']:.4f}")
            print(f"  DSSIM: {summary['reflectance_dssim']['mean']:.4f} ± {summary['reflectance_dssim']['std']:.4f}")
        
        # Shading metrics
        if 'shading_mse' in summary:
            print("\nSHADING METRICS:")
            print(f"  MSE: {summary['shading_mse']['mean']:.6f} ± {summary['shading_mse']['std']:.6f}")
            print(f"  LMSE: {summary['shading_lmse']['mean']:.6f} ± {summary['shading_lmse']['std']:.6f}")
            print(f"  SSIM: {summary['shading_ssim']['mean']:.4f} ± {summary['shading_ssim']['std']:.4f}")
            print(f"  DSSIM: {summary['shading_dssim']['mean']:.4f} ± {summary['shading_dssim']['std']:.4f}")
        
        print("="*60)


def setup_device(device_arg='auto'):
    """Setup device for evaluation"""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f'[*] GPU Device selected: {torch.cuda.get_device_name()}')
        else:
            device = torch.device('cpu')
            print('[X] WARN: No GPU found, using CPU')
    else:
        device = torch.device(device_arg)
        print(f'[*] Device selected: {device}')
    
    return device


def load_model(model_path, device):
    """Load trained PIE-Net model"""
    print(f'[*] Loading model from: {model_path}')
    
    # Create model
    model = DecScaleClampedIllumEdgeGuidedNetworkBatchNorm().to(device)
    
    # Load weights
    utils = mor_utils(device)
    model, _, _ = utils.loadModels(model, model_path)
    model.eval()
    
    print('[*] Model loaded successfully')
    return model


def save_qualitative_results(outputs, targets, input_img, filename, save_dir):
    """Save qualitative results for visual inspection"""
    # Convert tensors to numpy arrays for saving
    def tensor_to_numpy(tensor):
        if tensor.dim() == 4:  # Batch dimension
            tensor = tensor[0]
        tensor = tensor.cpu().detach().clone()
        tensor = torch.clamp(tensor, 0, 1)
        
        if tensor.dim() == 3 and tensor.size(0) == 3:  # RGB
            tensor = tensor.permute(1, 2, 0)  # CHW -> HWC
        elif tensor.dim() == 3 and tensor.size(0) == 1:  # Grayscale
            tensor = tensor.squeeze(0)  # Remove channel dimension
        
        return (tensor.numpy() * 255).astype(np.uint8)
    
    # Save individual outputs
    input_np = tensor_to_numpy(input_img)
    reflectance_np = tensor_to_numpy(outputs['reflectance'])
    shading_np = tensor_to_numpy(outputs['shading'])
    recon_np = tensor_to_numpy(outputs['recon'])
    
    imageio.imwrite(os.path.join(save_dir, f'{filename}_input.png'), input_np)
    imageio.imwrite(os.path.join(save_dir, f'{filename}_reflectance.png'), reflectance_np)
    imageio.imwrite(os.path.join(save_dir, f'{filename}_shading.png'), shading_np)
    imageio.imwrite(os.path.join(save_dir, f'{filename}_reconstruction.png'), recon_np)
    
    # Create comparison montage if ground truth available
    if 'albedo' in targets and 'shading' in targets:
        gt_albedo_np = tensor_to_numpy(targets['albedo'])
        gt_shading_np = tensor_to_numpy(targets['shading'])
        
        # Create montage: [Input, GT_Albedo, GT_Shading], [Recon, Pred_Albedo, Pred_Shading]
        if len(gt_shading_np.shape) == 2:
            gt_shading_np = np.stack([gt_shading_np] * 3, axis=-1)  # Convert to RGB
        if len(shading_np.shape) == 2:
            shading_np = np.stack([shading_np] * 3, axis=-1)  # Convert to RGB
        
        row1 = np.concatenate([input_np, gt_albedo_np, gt_shading_np], axis=1)
        row2 = np.concatenate([recon_np, reflectance_np, shading_np], axis=1)
        montage = np.concatenate([row1, row2], axis=0)
        
        imageio.imwrite(os.path.join(save_dir, f'{filename}_comparison.png'), montage)


def evaluate_model(model, data_loader, evaluator, save_dir, save_qualitative=True):
    """Evaluate model on dataset"""
    model.eval()
    sample_results = {}
    
    print(f"[*] Starting evaluation on {len(data_loader.dataset)} samples...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
            # Move data to device
            rgb = batch['rgb'].to(evaluator.device)
            filename = batch['filename'][0] if isinstance(batch['filename'], list) else batch['filename']
            
            # Prepare targets
            targets = {}
            if 'albedo' in batch:
                targets['albedo'] = batch['albedo'].to(evaluator.device)
            if 'shading' in batch:
                targets['shading'] = batch['shading'].to(evaluator.device)
            
            # Forward pass with timing
            start_time = time.time()
            outputs = model(rgb)
            inference_time = time.time() - start_time
            
            # Evaluate metrics
            sample_metrics = evaluator.evaluate_sample(outputs, targets, rgb, inference_time)
            sample_results[filename] = sample_metrics
            
            # Save qualitative results
            if save_qualitative:
                save_qualitative_results(outputs, targets, rgb, filename, save_dir)
    
    return sample_results


def main():
    parser = argparse.ArgumentParser(description='PIE-Net Enhanced Evaluation')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to test data directory')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cuda/cpu)')
    parser.add_argument('--has_gt', action='store_true',
                       help='Set if dataset has ground truth albedo/shading')
    parser.add_argument('--save_qualitative', action='store_true', default=True,
                       help='Save qualitative results (images)')
    
    args = parser.parse_args()
    
    # Setup
    device = setup_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Create dataset and dataloader
    if args.has_gt:
        # Use dataset with ground truth
        from dataset import PIENetDataset
        dataset = PIENetDataset(args.data_path, split='val', image_size=256)
    else:
        # Use test dataset (no ground truth)
        dataset = PIENetTestDataset(args.data_path, image_size=256)
    
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Create evaluator
    evaluator = IntrinsicEvaluator(device)
    
    # Run evaluation
    sample_results = evaluate_model(model, data_loader, evaluator, args.output_dir, args.save_qualitative)
    
    # Print and save results
    evaluator.print_summary()
    evaluator.save_results(os.path.join(args.output_dir, 'evaluation_metrics.json'), sample_results)
    
    print(f"\n[*] Evaluation completed!")
    print(f"[*] Results saved in: {args.output_dir}")
    if args.save_qualitative:
        print(f"[*] Qualitative results (images) saved in: {args.output_dir}")


if __name__ == '__main__':
    main()