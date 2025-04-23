import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from basicsr.models.archs.kbnet_s_arch import KBNet_s
from basicsr.models.archs.modified_kbnet_s_arch import KBNet

def load_model(model_path, model_type='KBNet_s'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_type == 'KBNet_s':
        model = KBNet_s()
    elif model_type == 'KBNet_s_ver1':
        model = KBNet()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    model.load_state_dict(torch.load(model_path, map_location=device)['params'])
    model = model.to(device)
    model.eval()
    return model

def split_image_into_patches(image, patch_size=256):
    patches = []
    _, h, w = image.size()
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[:, i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    return patches

def combine_patches_to_image(patches, image_size, patch_size=256):
    _, h, w = image_size
    image = torch.zeros((3, h, w))
    patch_idx = 0
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            image[:, i:i+patch_size, j:j+patch_size] = patches[patch_idx]
            patch_idx += 1
    return image

def process_image(model, image, patch_size=256):
    patches = split_image_into_patches(image, patch_size)
    processed_patches = []
    for patch in patches:
        patch = patch.unsqueeze(0).to(next(model.parameters()).device)
        with torch.no_grad():
            processed_patch = model(patch)
        processed_patches.append(processed_patch.squeeze(0).cpu())
    processed_image = combine_patches_to_image(processed_patches, image.size(), patch_size)
    return processed_image

def main():
    parser = argparse.ArgumentParser(description="Image Denoising Inference Script")
    parser.add_argument('--input', type=str, required=True, help="Path to input image or folder")
    parser.add_argument('--output', type=str, required=True, help="Path to save output image or folder")
    parser.add_argument('--model', type=str, default='Denoising/pretrained_models/model.pth', help="Path to model file")
    parser.add_argument('--model_type', type=str, default='KBNet_s', help="Type of model architecture")
    args = parser.parse_args()

    model = load_model(args.model, args.model_type)

    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        for filename in os.listdir(args.input):
            input_path = os.path.join(args.input, filename)
            output_path = os.path.join(args.output, filename)
            image = Image.open(input_path).convert('RGB')
            image = transforms.ToTensor()(image)
            processed_image = process_image(model, image)
            processed_image = transforms.ToPILImage()(processed_image)
            processed_image.save(output_path)
    else:
        image = Image.open(args.input).convert('RGB')
        image = transforms.ToTensor()(image)
        processed_image = process_image(model, image)
        processed_image = transforms.ToPILImage()(processed_image)
        processed_image.save(args.output)

if __name__ == "__main__":
    main()
