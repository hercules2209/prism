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
from basicsr.metrics import calculate_psnr, calculate_ssim
import scipy.io as sio

parser = argparse.ArgumentParser(description='Image Denoising using DenoiseNet')
parser.add_argument('--input_dir', default='./Datasets/test/SIDD/', type=str)
parser.add_argument('--result_dir', default='./results/Denoising/SIDD/', type=str)
parser.add_argument('--weights', default='./pretrained_models/denoisenet.pth', type=str)
args = parser.parse_args()

def main():
    try:
        # Load configuration
        yaml_file = os.path.join(os.path.dirname(__file__), 'Options/DenoiseNet.yml')
        with open(yaml_file, mode='r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        model_config = config['network_g']
        if 'type' in model_config:
            del model_config['type']
        model = DenoiseNet(**model_config)
        model.load_state_dict(torch.load(args.weights, map_location=device)['params'])
        model = nn.DataParallel(model).to(device)
        model.eval()
        
        # Process test data
        noisy_path = os.path.join(args.input_dir, 'ValidationNoisyBlocksSrgb.mat')
        gt_path = os.path.join(args.input_dir, 'ValidationGtBlocksSrgb.mat')
        
        if not os.path.exists(noisy_path) or not os.path.exists(gt_path):
            raise FileNotFoundError("Input or GT file not found")
            
        noisy_data = sio.loadmat(noisy_path)
        gt_data = sio.loadmat(gt_path)
        
        Inoisy = np.float32(noisy_data['ValidationNoisyBlocksSrgb']) / 255.
        Igt = np.float32(gt_data['ValidationGtBlocksSrgb']) / 255.

        # Run inference
        restored = np.zeros_like(Inoisy)
        with torch.no_grad():
            for i in tqdm(range(40)):
                for k in range(32):
                    try:
                        noisy_patch = Inoisy[i,k,:,:,:]
                        noisy_patch = np.transpose(noisy_patch, (2, 0, 1))
                        noisy_patch = torch.from_numpy(noisy_patch).unsqueeze(0).to(device)
                        restored_patch = model(noisy_patch)
                        
                        restored_patch = restored_patch.squeeze(0).cpu().numpy()
                        restored_patch = np.transpose(restored_patch, (1, 2, 0))
                        restored[i,k,:,:,:] = restored_patch
                        
                        # Calculate metrics for each patch
                        gt_patch = Igt[i,k,:,:,:]
                        psnr = calculate_psnr(
                            restored_patch * 255, 
                            gt_patch * 255,
                            crop_border=0,
                            test_y_channel=False
                        )
                        ssim = calculate_ssim(
                            restored_patch * 255,
                            gt_patch * 255,
                            crop_border=0,
                            test_y_channel=False
                        )
                        
                        torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"Error processing patch {i},{k}: {e}")
                        continue
        # Calculate average metrics
        avg_psnr = 0
        avg_ssim = 0
        count = 0
        
        for i in range(40):
            for k in range(32):
                restored_patch = restored[i,k,:,:,:]
                gt_patch = Igt[i,k,:,:,:]
                
                psnr = calculate_psnr(
                    restored_patch * 255,
                    gt_patch * 255,
                    crop_border=0,
                    test_y_channel=False
                )
                ssim = calculate_ssim(
                    restored_patch * 255,
                    gt_patch * 255,
                    crop_border=0,
                    test_y_channel=False
                )
                
                if not np.isinf(psnr):
                    avg_psnr += psnr
                    avg_ssim += ssim
                    count += 1
        
        avg_psnr = avg_psnr / count
        avg_ssim = avg_ssim / count
        
        print(f'\nAverage PSNR: {avg_psnr:.4f}')
        print(f'Average SSIM: {avg_ssim:.4f}')

        # Save results and metrics
        os.makedirs(args.result_dir, exist_ok=True)
        results_dict = {
            "Idenoised": restored,
            "PSNR": avg_psnr,
            "SSIM": avg_ssim
        }
        sio.savemat(os.path.join(args.result_dir, 'results.mat'), results_dict)

    except Exception as e:
        print(f"Error in testing: {e}")

if __name__ == '__main__':
    main()