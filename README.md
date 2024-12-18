# prism
This repo hosts the code for Image restoration using transformers sammsung prism project.
# DenoiseNet: Deep Learning for Image Denoising

A deep learning framework for image denoising using progressive learning on the SIDD dataset.

## Project Structure
```
.
├── basicsr/                  # Core framework
│   ├── data/                # Data loading and processing
│   ├── metrics/             # Evaluation metrics (PSNR, SSIM, etc.)
│   ├── models/              # Model architectures and training
│   │   └── archs/
│   │       └── denoisenet_arch.py  # Our model implementation
│   └── utils/               # Utility functions
└── Denoising/              # Task-specific code
    ├── Datasets/           # Dataset storage
    ├── Options/            # Training configurations
    └── pretrained_models/  # Saved models
```

## Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Dataset Preparation
1. Download SIDD dataset and generate patches:
```bash
cd Denoising
python download_data.py --data train-test --dataset SIDD --noise real
python generate_patches_sidd.py
mkdir Datasets/val
mv Datasets/Downloads/SIDD_val/SIDD Datasets/val
cd ..
```

## Training

### 1. Configure training parameters in `Denoising/Options/RealDenoising_DenoiseNet.yml`

### 2. Start training:
```bash
python basicsr/train.py -opt Denoising/Options/RealDenoising_DenoiseNet.yml
```
After Training is done, the model will be saved in `experiments/RealDenoising_DenoiseNet/models/`
__Move Weights to `Denoising/pretrained_models/` before testing__
```bash
cp experiments/RealDenoising_DenoiseNet/models/net_g_latest.pth Denoising/pretrained_models/model.pth
```

Training features:
- Progressive patch sizes: 128→512
- Progressive batch sizes: 16→4
- Cosine annealing learning rate


Configurable parameters in `RealDenoising_DenoiseNet.yml`:
Model architecture:
- network_g:
  channels: [64, 128, 256, 512] # Length/size of array determines number of encoder/decoder blocks -1 (4-1 = 3 encoder/decoder blocks here) and each value is the number of channels for each block
Hardware-dependent:
- `num_gpu`: Number of available GPUs (e.g., 1 for single A6000)
- `num_worker_per_gpu`: CPU threads for data loading (typically 4-8)
- `batch_size_per_gpu`: Memory-dependent (16 for A6000, adjust lower for smaller GPUs)
- `pin_memory`: True for CUDA prefetcher
- `prefetch_mode`: 'cuda' for GPU training

Training strategy:
- `total_iter`: Total training iterations (e.g., 300k for full training)
- `mini_batch_sizes`: Progressive reduction (e.g., [16,12,8,4])
- `gt_sizes`: Progressive patch sizes (e.g., [128,256,384,512])
- `val_freq`: Validation frequency
- `save_checkpoint_freq`: Checkpoint saving frequency

Optimizer settings:
- Learning rate
- Weight decay
- Scheduler periods
- Warmup iterations

Memory usage scales with:
- Batch size
- Patch size
- Number of workers
- Model channels

## Testing

Evaluate on SIDD validation set:
input: `Denoising/Datasets/test/SIDD/`
Contains:
- ValidationNoisyBlocksSrgb.mat (Noisy images)
- ValidationGtBlocksSrgb.mat (Ground truth)
output: `results/Denoising/SIDD/`

```bash
mkdir results/Denoising/SIDD
python Denoising/test_real_denoising_sidd.py \
    --input_dir Denoising/Datasets/test/SIDD/ \
    --result_dir results/Denoising/SIDD/ \
    --weights Denoising/pretrained_models/model.pth
```


### Configuration
Training parameters in `RealDenoising_DenoiseNet.yml`:
- Network architecture
- Training strategy
- Loss functions
- Data augmentation
- Validation settings

### Metrics
Available evaluation metrics:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity)


## Monitoring

Training progress is logged to:
- TensorBoard: `tb_logger/`
- Log files: `experiments/`
- Checkpoints: `experiments/training_states/`

View training progress:
```bash
tensorboard --logdir tb_logger
```

