import os
import argparse
import torch
import torch.nn as nn
import cv2
import numpy as np
from basicsr.models.archs.denoisenet_arch import DenoiseNet

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DenoiseNet([64, 128, 256, 512])
    model.load_state_dict(torch.load(model_path, map_location=device)['params'])
    model = nn.DataParallel(model).to(device)
    model.eval()
    return model

def split_image_into_patches(image, patch_size=256):
    patches = []
    h, w, _ = image.shape
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[i:i+patch_size, j+j+patch_size]
            patches.append(patch)
    return patches, h, w

def combine_patches_to_image(patches, image_height, image_width, patch_size=256):
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    patch_idx = 0
    for i in range(0, image_height, patch_size):
        for j in range(0, image_width, patch_size):
            image[i:i+patch_size, j+j+patch_size] = patches[patch_idx]
            patch_idx += 1
    return image

def denoise_image(model, image, patch_size=256):
    patches, h, w = split_image_into_patches(image, patch_size)
    denoised_patches = []
    for patch in patches:
        patch = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
        with torch.no_grad():
            denoised_patch = model(patch).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        denoised_patches.append(denoised_patch.astype(np.uint8))
    denoised_image = combine_patches_to_image(denoised_patches, h, w, patch_size)
    return denoised_image

def main():
    parser = argparse.ArgumentParser(description='Denoise images using a pretrained model.')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or folder.')
    parser.add_argument('--output', type=str, required=True, help='Path to save denoised image or folder.')
    parser.add_argument('--model', type=str, default='Denoising/pretrained_models/model.pth', help='Path to the pretrained model.')
    args = parser.parse_args()

    model = load_model(args.model)

    if os.path.isfile(args.input):
        image = cv2.imread(args.input)
        denoised_image = denoise_image(model, image)
        cv2.imwrite(args.output, denoised_image)
    elif os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        for filename in os.listdir(args.input):
            input_path = os.path.join(args.input, filename)
            output_path = os.path.join(args.output, filename)
            image = cv2.imread(input_path)
            denoised_image = denoise_image(model, image)
            cv2.imwrite(output_path, denoised_image)
    else:
        raise ValueError("Input path must be a file or directory")

if __name__ == '__main__':
    main()
