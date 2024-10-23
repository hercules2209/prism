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
# Install dependencies
pip install torch torchvision
pip install basicsr
pip install matlab  # For evaluation
pip install tensorboard  # For logging
```

### Dataset Preparation
1. Download SIDD dataset:
```bash
python Denoising/download_data.py --data train --noise real
```

2. Generate training patches:
```bash
python Denoising/generate_patches_sidd.py
```

## Training

1. Configure training parameters in `Denoising/Options/RealDenoising_DenoiseNet.yml`

2. Start training:
```bash
./train.sh Denoising/Options/RealDenoising_DenoiseNet.yml
```

Training features:
- Progressive patch sizes: 128→384
- Progressive batch sizes: 8→1
- Cosine annealing learning rate
- Multi-GPU support
- Automatic resumption

## Testing

Evaluate on SIDD validation set:
```bash
python Denoising/test_real_denoising_sidd.py \
    --weights pretrained_models/model.pth \
    --input_dir Datasets/test/SIDD/
```

## Key Components

### Core Files
- `denoisenet_arch.py`: Model architecture
- `train.py`: Training loop
- `test.py`: Evaluation code
- `paired_image_dataset.py`: Data loading
- `losses.py`: PSNR loss implementation

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
- FID (Fréchet Inception Distance)
- NIQE (Natural Image Quality Evaluator)

## Monitoring

Training progress is logged to:
- TensorBoard: `tb_logger/`
- Log files: `experiments/`
- Checkpoints: `experiments/training_states/`

View training progress:
```bash
tensorboard --logdir tb_logger
```

