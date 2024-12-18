"""
Testing script for DenoiseNet on SIDD dataset
"""
import numpy as np
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import yaml
from basicsr.models.archs.denoisenet_arch import DenoiseNet
import scipy.io as sio

parser = argparse.ArgumentParser(description='Image Denoising using DenoiseNet')
parser.add_argument('--input_dir', default='./Datasets/test/SIDD/', type=str)
parser.add_argument('--result_dir', default='./results/Denoising/SIDD/', type=str)
parser.add_argument('--weights', default='./pretrained_models/denoisenet.pth', type=str)
args = parser.parse_args()

def main():
    try:
        # Load configuration
        yaml_file = 'Options/DenoiseNet.yml'
        with open(yaml_file, mode='r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # Check CUDA availability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        model = DenoiseNet(**config['network_g'])
        if not os.path.exists(args.weights):
            raise FileNotFoundError(f"Weights file not found: {args.weights}")
            
        model.load_state_dict(torch.load(args.weights, map_location=device)['params'])
        model = nn.DataParallel(model).to(device)
        model.eval()

        # Process test data
        filepath = os.path.join(args.input_dir, 'ValidationNoisyBlocksSrgb.mat')
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Input file not found: {filepath}")
            
        img = sio.loadmat(filepath)
        Inoisy = np.float32(img['ValidationNoisyBlocksSrgb']) / 255.

        # Run inference
        restored = np.zeros_like(Inoisy)
        with torch.no_grad():
            for i in tqdm(range(40)):
                for k in range(32):
                    try:
                        noisy_patch = torch.from_numpy(Inoisy[i,k,:,:,:]).unsqueeze(0).to(device)
                        restored_patch = model(noisy_patch)
                        restored[i,k,:,:,:] = restored_patch.cpu().numpy()
                        torch.cuda.empty_cache()  # Clear GPU memory after each batch
                    except Exception as e:
                        print(f"Error processing patch {i},{k}: {e}")
                        continue

        # Create result directory if it doesn't exist
        os.makedirs(args.result_dir, exist_ok=True)
        
        # Save results
        try:
            sio.savemat(os.path.join(args.result_dir, 'results.mat'), 
                      {"Idenoised": restored})
        except Exception as e:
            print(f"Error saving results: {e}")

    except Exception as e:
        print(f"Error in testing: {e}")

if __name__ == '__main__':
    main()
