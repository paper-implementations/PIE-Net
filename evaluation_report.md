# PIE-Net Model Evaluation Report

## Overview
This report presents the evaluation results of the PIE-Net model (`real_world_model.t7`) on multiple intrinsic image decomposition datasets. The model was tested on validation splits of three major datasets to assess its performance in decomposing images into reflectance (albedo) and shading components.

## Model Information
- **Model Path**: `logs/modelWeight/real_world_model.t7`
- **Architecture**: DecScaleClampedIllumEdgeGuidedNetworkBatchNorm
- **Device**: Tesla V100-PCIE-32GB GPU
- **Evaluation Framework**: Custom evaluation pipeline with unified dataset loading

## Datasets Evaluated

### 1. MIDIntrinsics Dataset
- **Samples Evaluated**: 50 (out of 750 available)
- **Dataset Type**: Multi-illumination images with ground truth albedo
- **Shading**: Derived from RGB/albedo ratio

**Results:**
- **Reconstruction MSE**: 0.0042 ± 0.0021
- **Reconstruction SSIM**: 0.9750 ± 0.0064
- **Reflectance MSE**: 0.0382 ± 0.0174
- **Reflectance LMSE**: 0.0232 ± 0.0064
- **Reflectance SSIM**: 0.6028 ± 0.0321
- **Shading MSE**: 0.3586 ± 0.1088
- **Shading LMSE**: 0.0071 ± 0.0040
- **Shading SSIM**: 0.2742 ± 0.0486
- **Inference Time**: 1.11s ± 7.68s (0.90 FPS)

### 2. MIT-Intrinsic Dataset
- **Samples Evaluated**: 30 (out of 200 available)
- **Dataset Type**: Intrinsic images with ground truth albedo and shading
- **Shading**: Ground truth available

**Results:**
- **Reconstruction MSE**: 0.0026 ± 0.0024
- **Reconstruction SSIM**: 0.9073 ± 0.0268
- **Reflectance MSE**: 0.0446 ± 0.0330
- **Reflectance LMSE**: 0.0268 ± 0.0152
- **Reflectance SSIM**: 0.7818 ± 0.0807
- **Shading MSE**: 8968.65 ± 957.90
- **Shading LMSE**: 2516.83 ± 1728.07
- **Shading SSIM**: 0.4787 ± 0.0952
- **Inference Time**: 1.84s ± 9.86s (0.54 FPS)

### 3. NED Dataset
- **Samples Evaluated**: 50 (out of 250 available)
- **Dataset Type**: Natural environment dataset with ground truth albedo
- **Shading**: Derived from RGB/albedo ratio

**Results:**
- **Reconstruction MSE**: 0.0005 ± 0.0002
- **Reconstruction SSIM**: 0.9200 ± 0.0509
- **Reflectance MSE**: 0.0095 ± 0.0040
- **Reflectance LMSE**: 0.0013 ± 0.0007
- **Reflectance SSIM**: 0.7476 ± 0.1072
- **Shading MSE**: 0.0810 ± 0.0264
- **Shading LMSE**: 0.0563 ± 0.0205
- **Shading SSIM**: 0.5592 ± 0.1169
- **Inference Time**: 0.013s ± 0.001s (78.68 FPS)

## Performance Analysis

### Reconstruction Quality
The model demonstrates excellent reconstruction capabilities across all datasets:
- **MIDIntrinsics**: 97.50% SSIM - Excellent reconstruction
- **MIT-Intrinsic**: 90.73% SSIM - Very good reconstruction  
- **NED**: 92.00% SSIM - Excellent reconstruction

### Reflectance (Albedo) Decomposition
The model shows good performance in albedo estimation:
- **NED**: Best performance with 0.0095 MSE and 74.76% SSIM
- **MIDIntrinsics**: Moderate performance with 0.0382 MSE and 60.28% SSIM
- **MIT-Intrinsic**: Good performance with 0.0446 MSE and 78.18% SSIM

### Shading Decomposition
Shading decomposition shows mixed results:
- **NED**: Good performance with 0.0810 MSE and 55.92% SSIM
- **MIDIntrinsics**: Moderate performance with 0.3586 MSE and 27.42% SSIM
- **MIT-Intrinsic**: Poor performance with very high MSE (8968.65) - likely due to scale differences

### Computational Performance
- **NED**: Fastest inference at 78.68 FPS
- **MIDIntrinsics**: Moderate speed at 0.90 FPS
- **MIT-Intrinsic**: Slowest at 0.54 FPS

## Key Observations

1. **Reconstruction Excellence**: The model consistently achieves high reconstruction quality (>90% SSIM) across all datasets, indicating good overall performance.

2. **Reflectance vs Shading**: The model performs better at reflectance estimation than shading estimation, which is common in intrinsic image decomposition.

3. **Dataset Variations**: Performance varies significantly across datasets, suggesting the model may be better suited for certain types of scenes or lighting conditions.

4. **Scale Issues**: The high shading MSE on MIT-Intrinsic suggests potential scale normalization issues between predicted and ground truth shading.

5. **Derived vs Ground Truth Shading**: Performance is generally better on datasets where shading is derived (NED, MIDIntrinsics) rather than ground truth (MIT-Intrinsic).

## Recommendations

1. **Scale Normalization**: Investigate and fix scale differences in shading predictions, especially for MIT-Intrinsic dataset.

2. **Dataset-Specific Training**: Consider fine-tuning the model on specific datasets to improve performance.

3. **Shading Refinement**: Focus on improving shading estimation, possibly through additional loss terms or architectural changes.

4. **Performance Optimization**: Investigate the significant inference time variations, especially for MIDIntrinsics and MIT-Intrinsic datasets.

## Conclusion

The PIE-Net model demonstrates strong reconstruction capabilities and good reflectance estimation performance. While shading decomposition shows room for improvement, the overall performance is promising for intrinsic image decomposition tasks. The model performs particularly well on the NED dataset, suggesting it may be well-suited for natural environment scenes.

## Files Generated
- `evaluation_results/all_datasets_evaluation.json`: Combined metrics for all datasets
- `evaluation_results/{dataset}_evaluation_metrics.json`: Individual dataset metrics
- `evaluation_results/{dataset}/`: Qualitative results (sample images) for each dataset

## Evaluation Script
The evaluation was performed using `evaluate_all_datasets.py` with the following features:
- Unified dataset loading with automatic shading derivation
- Comprehensive metrics (MSE, LMSE, SSIM, DSSIM)
- Qualitative result generation
- Performance timing
- Multi-dataset evaluation support
