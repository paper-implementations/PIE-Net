#!/usr/bin/env python3
"""
Inference script for the trained IID model
"""

import os
import torch
import numpy as np
import cv2
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

from Network import DecScaleClampedIllumEdgeGuidedNetworkBatchNorm


class IIDInference:
    """Inference class for Intrinsic Image Decomposition"""
    
    def __init__(self, model_path, device=None):
        """
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Running inference on {self.device}")
    
    def _load_model(self, model_path):
        """Load the trained model"""
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model
        model = DecScaleClampedIllumEdgeGuidedNetworkBatchNorm().to(self.device)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def preprocess_image(self, image_path, target_size=256):
        """Preprocess input image"""
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path
        
        # Resize
        image = cv2.resize(image, (target_size, target_size))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image
    
    def postprocess_output(self, output_dict):
        """Postprocess model output"""
        processed_output = {}
        
        for key, tensor in output_dict.items():
            if isinstance(tensor, torch.Tensor):
                # Move to CPU and convert to numpy
                tensor = tensor.detach().cpu().squeeze(0)
                
                if tensor.dim() == 3:  # (C, H, W)
                    # Convert to (H, W, C) for visualization
                    tensor = tensor.permute(1, 2, 0).numpy()
                    # Clip to [0, 1]
                    tensor = np.clip(tensor, 0, 1)
                else:  # (H, W) for single channel
                    tensor = tensor.numpy()
                    tensor = np.clip(tensor, 0, 1)
                
                processed_output[key] = tensor
        
        return processed_output
    
    def predict(self, image_path, target_size=256):
        """
        Run inference on a single image
        
        Args:
            image_path: Path to input image or numpy array
            target_size: Target size for processing
            
        Returns:
            Dictionary containing predictions
        """
        # Preprocess image
        input_tensor = self.preprocess_image(image_path, target_size).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output_dict = self.model(input_tensor)
        
        # Postprocess output
        processed_output = self.postprocess_output(output_dict)
        
        return processed_output
    
    def predict_batch(self, image_paths, target_size=256):
        """
        Run inference on multiple images
        
        Args:
            image_paths: List of image paths
            target_size: Target size for processing
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for image_path in image_paths:
            result = self.predict(image_path, target_size)
            results.append(result)
        
        return results
    
    def visualize_results(self, input_image, predictions, save_path=None):
        """
        Visualize inference results
        
        Args:
            input_image: Original input image
            predictions: Model predictions
            save_path: Path to save visualization
        """
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Input image
        if isinstance(input_image, str):
            input_img = cv2.imread(input_image)
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        else:
            input_img = input_image
        
        axes[0, 0].imshow(input_img)
        axes[0, 0].set_title('Input Image')
        axes[0, 0].axis('off')
        
        # Reflectance/Albedo
        if 'reflectance' in predictions:
            axes[0, 1].imshow(predictions['reflectance'])
            axes[0, 1].set_title('Reflectance/Albedo')
            axes[0, 1].axis('off')
        
        # Shading
        if 'shading' in predictions:
            if len(predictions['shading'].shape) == 3:
                axes[0, 2].imshow(predictions['shading'])
            else:
                axes[0, 2].imshow(predictions['shading'], cmap='gray')
            axes[0, 2].set_title('Shading')
            axes[0, 2].axis('off')
        
        # Reconstruction
        if 'recon' in predictions:
            axes[1, 0].imshow(predictions['recon'])
            axes[1, 0].set_title('Reconstruction (R × S)')
            axes[1, 0].axis('off')
        
        # Reflectance edges
        if 'reflec_edge' in predictions:
            axes[1, 1].imshow(predictions['reflec_edge'], cmap='gray')
            axes[1, 1].set_title('Reflectance Edges')
            axes[1, 1].axis('off')
        
        # Illumination edges
        if 'illum_edge' in predictions:
            axes[1, 2].imshow(predictions['illum_edge'], cmap='gray')
            axes[1, 2].set_title('Illumination Edges')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_predictions(self, predictions, output_dir, base_name):
        """
        Save individual prediction components
        
        Args:
            predictions: Model predictions
            output_dir: Output directory
            base_name: Base name for saved files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for key, tensor in predictions.items():
            if isinstance(tensor, np.ndarray):
                # Convert to uint8 for saving
                if tensor.max() <= 1.0:
                    tensor = (tensor * 255).astype(np.uint8)
                
                # Save as image
                if len(tensor.shape) == 3:
                    # RGB image
                    img = Image.fromarray(tensor)
                else:
                    # Grayscale image
                    img = Image.fromarray(tensor, mode='L')
                
                save_path = output_dir / f"{base_name}_{key}.png"
                img.save(save_path)
                print(f"Saved {key} to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='IID Model Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output', type=str, default='inference_outputs', help='Output directory')
    parser.add_argument('--size', type=int, default=256, help='Input image size')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--save_components', action='store_true', help='Save individual components')
    
    args = parser.parse_args()
    
    # Create inference object
    inference = IIDInference(args.model)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image
        print(f"Processing single image: {input_path}")
        
        # Run inference
        predictions = inference.predict(str(input_path), args.size)
        
        # Save results
        base_name = input_path.stem
        
        if args.visualize:
            viz_path = output_dir / f"{base_name}_visualization.png"
            inference.visualize_results(str(input_path), predictions, viz_path)
        
        if args.save_components:
            inference.save_predictions(predictions, output_dir, base_name)
    
    elif input_path.is_dir():
        # Directory of images
        print(f"Processing directory: {input_path}")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        print(f"Found {len(image_files)} images")
        
        for image_file in image_files:
            print(f"Processing {image_file.name}...")
            
            # Run inference
            predictions = inference.predict(str(image_file), args.size)
            
            # Save results
            base_name = image_file.stem
            
            if args.visualize:
                viz_path = output_dir / f"{base_name}_visualization.png"
                inference.visualize_results(str(image_file), predictions, viz_path)
            
            if args.save_components:
                inference.save_predictions(predictions, output_dir, base_name)
    
    else:
        print(f"Input path {args.input} does not exist")
        return
    
    print(f"Inference completed! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
