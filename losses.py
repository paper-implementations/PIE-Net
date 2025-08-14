import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""
    
    def __init__(self, layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1']):
        super(PerceptualLoss, self).__init__()
        
        # Load pre-trained VGG16
        vgg = models.vgg16(pretrained=True).features
        self.layers = layers
        self.vgg_layers = nn.ModuleDict()
        
        layer_names = [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
        ]
        
        # Extract specific layers
        current_layer = 0
        for i, layer in enumerate(vgg):
            name = layer_names[i]
            self.vgg_layers[name] = layer
            if name in self.layers:
                current_layer += 1
            if current_layer >= len(self.layers):
                break
        
        # Freeze VGG parameters
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
    
    def forward(self, pred, target):
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)
        
        loss = 0
        for layer in self.layers:
            loss += F.mse_loss(pred_features[layer], target_features[layer])
        
        return loss / len(self.layers)
    
    def extract_features(self, x):
        features = {}
        for name, layer in self.vgg_layers.items():
            x = layer(x)
            if name in self.layers:
                features[name] = x
        return features


class SSIMLoss(nn.Module):
    """Structural Similarity Index Loss"""
    
    def __init__(self, window_size=11, channel=3):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self._create_window(window_size, channel)
    
    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    
    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, pred, target):
        if pred.device != self.window.device:
            self.window = self.window.to(pred.device)
        
        mu1 = F.conv2d(pred, self.window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(target, self.window, padding=self.window_size//2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred * pred, self.window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, self.window, padding=self.window_size//2, groups=self.channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim_map.mean()


class ScaleInvariantMSELoss(nn.Module):
    """Scale-Invariant Mean Squared Error Loss"""
    
    def __init__(self):
        super(ScaleInvariantMSELoss, self).__init__()
    
    def forward(self, pred, target):
        # Flatten tensors
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # Compute optimal scaling factor
        numerator = torch.sum(pred_flat * target_flat, dim=1, keepdim=True)
        denominator = torch.sum(pred_flat * pred_flat, dim=1, keepdim=True) + 1e-8
        scale = numerator / denominator
        
        # Scale prediction
        scaled_pred = scale * pred_flat
        
        # Compute MSE
        mse = torch.mean((scaled_pred - target_flat)**2, dim=1)
        return torch.mean(mse)


class EdgeLoss(nn.Module):
    """Edge-aware loss for reflectance and shading"""
    
    def __init__(self):
        super(EdgeLoss, self).__init__()
        
        # Sobel edge detection kernels
        sobel_x = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, pred_edge, target_edge):
        """
        pred_edge: predicted edge map
        target_edge: target edge map (can be derived from ground truth or input image)
        """
        return F.mse_loss(pred_edge, target_edge)
    
    def compute_edges(self, img):
        """Compute edge maps using Sobel operator"""
        if img.size(1) == 3:  # RGB image
            img_gray = 0.299 * img[:, 0:1, :, :] + 0.587 * img[:, 1:2, :, :] + 0.114 * img[:, 2:3, :, :]
        else:
            img_gray = img
        
        edge_x = F.conv2d(img_gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(img_gray, self.sobel_y, padding=1)
        edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)
        
        return edge_magnitude


class PIENetLoss(nn.Module):
    """Combined loss function for PIE-Net"""
    
    def __init__(self, config):
        super(PIENetLoss, self).__init__()
        
        # Loss weights from config
        self.lambda_u = config.LAMBDA_U
        self.lambda_d = config.LAMBDA_D
        self.lambda_e = config.LAMBDA_E
        self.lambda_p = config.LAMBDA_P
        self.lambda_smse = config.LAMBDA_SMSE
        self.lambda_mse = config.LAMBDA_MSE
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.scale_invariant_mse = ScaleInvariantMSELoss()
        self.ssim_loss = SSIMLoss()
        self.perceptual_loss = PerceptualLoss()
        self.edge_loss = EdgeLoss()
    
    def forward(self, outputs, targets, input_img):
        """
        Compute combined loss for PIE-Net
        
        Args:
            outputs: Dictionary containing network outputs
            targets: Dictionary containing ground truth (if available)
            input_img: Original input image
        """
        losses = {}
        total_loss = 0
        
        # Reconstruction loss (most important for self-supervised learning)
        pred_recon = outputs['recon']  # reflectance * shading
        recon_loss = self.mse_loss(pred_recon, input_img)
        losses['reconstruction'] = recon_loss
        total_loss += recon_loss
        
        # Reflectance and shading losses (if ground truth available)
        if 'albedo' in targets and 'shading' in targets:
            # Reflectance loss
            reflec_loss_mse = self.lambda_mse * self.mse_loss(outputs['reflectance'], targets['albedo'])
            reflec_loss_smse = self.lambda_smse * self.scale_invariant_mse(outputs['reflectance'], targets['albedo'])
            reflec_loss = reflec_loss_mse + reflec_loss_smse
            losses['reflectance'] = reflec_loss
            total_loss += reflec_loss
            
            # Shading loss
            if len(targets['shading'].shape) == 3:  # Add channel dimension if needed
                targets['shading'] = targets['shading'].unsqueeze(1)
            
            shad_loss_mse = self.lambda_mse * self.mse_loss(outputs['shading'], targets['shading'])
            shad_loss_smse = self.lambda_smse * self.scale_invariant_mse(outputs['shading'], targets['shading'])
            shad_loss = shad_loss_mse + shad_loss_smse
            losses['shading'] = shad_loss
            total_loss += shad_loss
        
        # Unrefined losses
        if 'unrefined_reflec' in outputs and 'unrefined_shd' in outputs:
            unrefined_recon = outputs['unrefined_reflec'] * outputs['unrefined_shd']
            unrefined_loss = self.lambda_u * self.mse_loss(unrefined_recon, input_img)
            losses['unrefined'] = unrefined_loss
            total_loss += unrefined_loss
        
        # DSSIM loss (structural similarity)
        if 'reflectance' in outputs:
            dssim_loss = self.lambda_d * self.ssim_loss(outputs['reflectance'], input_img)
            losses['dssim'] = dssim_loss
            total_loss += dssim_loss
        
        # Edge losses
        if 'reflec_edge' in outputs:
            input_edges = self.edge_loss.compute_edges(input_img)
            edge_loss_reflec = self.lambda_e * self.edge_loss(outputs['reflec_edge'], input_edges)
            losses['edge_reflec'] = edge_loss_reflec
            total_loss += edge_loss_reflec
        
        if 'illum_edge' in outputs:
            edge_loss_illum = self.lambda_e * self.edge_loss(
                torch.mean(outputs['illum_edge'], dim=1, keepdim=True),
                self.edge_loss.compute_edges(input_img)
            )
            losses['edge_illum'] = edge_loss_illum
            total_loss += edge_loss_illum
        
        # Perceptual loss
        if 'reflectance' in outputs:
            if outputs['reflectance'].size(1) == 3:  # Ensure 3 channels for VGG
                perceptual_loss = self.lambda_p * self.perceptual_loss(outputs['reflectance'], input_img)
                losses['perceptual'] = perceptual_loss
                total_loss += perceptual_loss
        
        # Smoothness constraints
        reflec_smooth = self._smoothness_loss(outputs['reflectance'])
        shading_smooth = self._smoothness_loss(outputs['shading'])
        smooth_loss = 0.01 * (reflec_smooth + shading_smooth)
        losses['smoothness'] = smooth_loss
        total_loss += smooth_loss
        
        losses['total'] = total_loss
        return losses
    
    def _smoothness_loss(self, img):
        """Compute smoothness loss (total variation)"""
        if img.size(1) == 1:  # Single channel
            img = img.squeeze(1)
            dy = torch.abs(img[:, 1:, :] - img[:, :-1, :])
            dx = torch.abs(img[:, :, 1:] - img[:, :, :-1])
        else:  # Multi-channel
            dy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
            dx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
        
        return torch.mean(dx) + torch.mean(dy)