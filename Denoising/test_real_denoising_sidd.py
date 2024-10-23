"""
Testing script for DenoiseNet on SIDD dataset
"""
import numpy as np
import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from basicsr.models.archs.denoisenet_arch import DenoiseNet
import scipy.io as sio

parser = argparse.ArgumentParser(description='Image Denoising using DenoiseNet')
parser.add_argument('--input_dir', default='./Datasets/test/SIDD/', type=str)
parser.add_argument('--result_dir', default='./results/Denoising/SIDD/', type=str)
parser.add_argument('--weights', default='./pretrained_models/denoisenet.pth', type=str)
args = parser.parse_args()

def main():
    # Load configuration
    yaml_file = 'Options/DenoiseNet.yml'
    with open(yaml_file, mode='r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Create model
    model = DenoiseNet(**config['network_g'])
    model.load_state_dict(torch.load(args.weights)['params'])
    model = nn.DataParallel(model).cuda()
    model.eval()

    # Process test data
    filepath = os.path.join(args.input_dir, 'ValidationNoisyBlocksSrgb.mat')
    img = sio.loadmat(filepath)
    Inoisy = np.float32(img['ValidationNoisyBlocksSrgb']) / 255.

    # Run inference
    restored = np.zeros_like(Inoisy)
    with torch.no_grad():
        for i in tqdm(range(40)):
            for k in range(32):
                noisy_patch = torch.from_numpy(Inoisy[i,k,:,:,:]).unsqueeze(0).cuda()
                restored_patch = model(noisy_patch)
                restored[i,k,:,:,:] = restored_patch.cpu().numpy()

    # Save results
    sio.savemat(os.path.join(args.result_dir, 'results.mat'), 
                {"Idenoised": restored})

if __name__ == '__main__':
    main()
