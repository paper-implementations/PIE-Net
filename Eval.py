import os
import time

from tqdm import tqdm
import numpy as np
import imageio
import glob
import cv2

import torch

from Network import DecScaleClampedIllumEdgeGuidedNetworkBatchNorm
from Utils import mor_utils


torch.backends.cudnn.benchmark = True

cudaDevice = ''

if len(cudaDevice) < 1:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('[*] GPU Device selected as default execution device.')
    else:
        device = torch.device('cpu')
        print('[X] WARN: No GPU Devices found on the system! Using the CPU. '
              'Execution maybe slow!')
else:
    device = torch.device('cuda:%s' % cudaDevice)
    print('[*] GPU Device %s selected as default execution device.' %
          cudaDevice)

visuals = 'test_outputs/'
os.makedirs(visuals, exist_ok=True)

modelSaveLoc = './logs/modelWeight/real_world_model.t7'

# data_root = '/home/pdas/Experiments/TrimBot/test_around_garden/uvc_camera_cam_0/'
# data_root = './datasets/intrinsic-data.tar/MIT-intrinsic/'  # Root directory (doesn't contain direct images)

# Check for available test data and use the first valid location
potential_paths = [
    # './test_outputs/',  # Use existing output images for testing
    # './datasets/intrinsic-data.tar/MIT-intrinsic/data/apple/',
    # './datasets/intrinsic-data.tar/MIT-intrinsic/data/box/',
    # './datasets/Test/',
    # './'
]

data_root = './datasets/Test/'  # Default
for path in potential_paths:
    if os.path.exists(path):
        test_files = glob.glob(path + '*.png')
        if test_files:
            data_root = path
            print(f"[*] Using data from: {data_root} ({len(test_files)} PNG files)")
            break

query_fmt = 'png'

batch_size = 1
nthreads = 4
if batch_size < nthreads:
    nthreads = batch_size

done = u'\u2713'

print('[I] STATUS: Create utils instances...', end='')
support = mor_utils(device)
print(done)

print('[I] STATUS: Load Network and transfer to device...', end='')
net = DecScaleClampedIllumEdgeGuidedNetworkBatchNorm().to(device)
net, _, _ = support.loadModels(net, modelSaveLoc)
print('[I] STATUS: Network loaded and transferred to device.', end='')
net.to(device)
print(done)

def readFile(name):
    im = imageio.imread(name)
    rgb = im.astype(np.float32)
    rgb[np.isnan(rgb)] = 0
    rgb = cv2.resize(rgb, (256, 256))
    rgb = rgb / 255

    rgb = rgb.transpose((2, 0, 1))
    return rgb

def save_image_fixed(filepath, image_tensor):
    """Fixed version of image saving that handles single-channel images"""
    # Convert tensor to numpy
    pred = image_tensor.cpu().detach().clone().numpy()
    pred[pred < 0] = 0
    pred = (pred / pred.max()) * 255
    pred = pred.transpose((1, 2, 0))  # CHW -> HWC
    pred = pred.astype(np.uint8)
    
    # Handle single-channel images by converting to grayscale or RGB
    if pred.shape[-1] == 1:
        # Squeeze to 2D for grayscale
        pred = pred.squeeze(-1)
    
    # Save the image
    imageio.imwrite(filepath, pred)

def Eval(net):
    net.eval()

    files = glob.glob(data_root + '*.%s' % query_fmt)
    print('Found %d files at query location' % len(files))
    
    if len(files) == 0:
        print("❌ No files found! Current data_root:", data_root)
        print("Available directories:")
        if os.path.exists('./datasets/'):
            for item in os.listdir('./datasets/'):
                if os.path.isdir(os.path.join('./datasets/', item)):
                    print(f"  - ./datasets/{item}/")
        return

    for data in tqdm(files):

        data = data.split('/')[-1].split('.')[0]
        img = readFile(data_root + data + '.%s' % query_fmt)
        rgb = torch.from_numpy(img).float().to(device)
        rgb = rgb.unsqueeze(0)
        [b, c, w, h] = rgb.shape

        net_time = time.time()
        with torch.no_grad():
            pred = net(rgb)

        net_timed = time.time() - net_time

        for j in range(b):
            # Save each output separately with proper handling
            save_image_fixed(f'{visuals}/{data}_input.png', rgb[j])
            save_image_fixed(f'{visuals}/{data}_reflectance.png', pred['reflectance'][j])
            save_image_fixed(f'{visuals}/{data}_shading.png', pred['shading'][j])
            
        print(f"✅ Processed: {data} (Time: {net_timed:.3f}s)")


print('[*] Beginning Testing:')
print('\tVisuals Dumped at: ', visuals)

# Check if we have any files to process
test_files = glob.glob(data_root + '*.png')
if len(test_files) == 0:
    print(f"[!] No PNG files found in {data_root}")
    print("[*] Creating a test image from existing test_outputs if available...")
    
    # Check if test_outputs has any images we can use
    existing_outputs = glob.glob('./test_outputs/*.png')
    if existing_outputs:
        print(f"[*] Found {len(existing_outputs)} existing output images")
        # Copy one to our test directory for demonstration
        os.makedirs(data_root, exist_ok=True)
        import shutil
        test_file = existing_outputs[0]
        new_test_file = os.path.join(data_root, 'test_image.png')
        shutil.copy2(test_file, new_test_file)
        print(f"[*] Copied {test_file} to {new_test_file} for testing")
    else:
        print("[!] No test images available. Please add some PNG images to test with.")
        print("    You can:")
        print("    1. Add PNG images to ./datasets/Test/")
        print("    2. Use images from the MIT intrinsic dataset")
        print("    3. Download test data using the download script")

Eval(net)
