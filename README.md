# MeanFlow-TSE: One-Step Generative Target Speaker Extraction with Mean Flow

## Overview

MeanFlowTSE combines AD-FlowTSE and MeanFlow/AlphaFlow training objectives for effective one-step target speaker extraction. Experiments on Libri2Mix dataset show that MeanFlowTSE achieve the SOTA performance in SI-SDR, PESQ, and ESTOI compared to the previous generative (diffusion / flow-matching) TSE models.

## Installation

### Requirements

Note that this is the setup that worked for the author but you might need to adjust it based on your hardware configurations.
```bash
# MeanFlowTSE Requirements
# Python 3.9+

# PyTorch and related (CUDA 11.8)
--extra-index-url https://download.pytorch.org/whl/cu118
torch==2.0.1
torchvision==0.15.2
torchaudio==2.0.2

# Core scientific packages
numpy==1.24.3
scipy==1.10.1
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
h5py==3.9.0
tqdm

# Audio processing
librosa==0.10.0.post2
soundfile==0.12.1
pydub==0.25.1
resampy==0.4.2

# PyTorch Lightning
pytorch-lightning==2.0.6
tensorboard
wandb

# Asteroid (audio source separation toolkit)
asteroid==0.6.0
asteroid-filterbanks

# Audio processing utilities
julius
torch-optimizer>=0.0.1a12,<0.2.0

# Flow matching
flow-matching

# Utilities
cached-property

# Speech metrics
pystoi==0.3.3

# Model components
einops==0.6.1
timm==0.9.5

# Data processing
openpyxl

# Configuration management
omegaconf==2.3.0
hydra-core==1.3.2
pyyaml

# Optional: Transformer models
transformers==4.30.2
accelerate==0.20.3

# Additional metrics (optional)
# Note: Install these separately if needed for evaluation
# speechmos  # For DNSMOS scores (optional, may have conflicts)
# wespeakerruntime  # For speaker similarity (optional)
#
# To install DNSMOS and speaker similarity:
#   pip install speechmos
#   pip install wespeakerruntime
```

### Dataset Preparation

For data preparation of Libri2Mix dataset, We followed the the official data-preparation pipeline from https://github.com/BUTSpeechFIT/speakerbeam.

## Training

### Basic Training

```bash
python train_meanflow.py --config config/config_MeanFlowTSE_clean.yaml
```

### Configuration

Key configuration parameters in the YAML file:

```yaml
# Alpha scheduling (epochs converted to iterations internally)
meanflow:
  flow_ratio: 0.5                    # Ratio of rectified flow vs alpha-flow
  alpha_schedule_start_epoch: 0      # Start transition at epoch 0
  alpha_schedule_end_epoch: 2000     # Finish transition at epoch 2000
  alpha_gamma: 25.0                  # Temperature for sigmoid schedule
  alpha_min: 0.005                   # Minimum alpha value

# Model architecture
model:
  input_dim: 512
  output_dim: 512
  hidden_size: 1024
  depth: 16
  num_heads: 16

# Training settings
train:
  batch_size: 32
  num_epochs: 2000
  accumulation_steps: 2
  gradient_clip_val: 0.5
  precision: "bf16-mixed"

# Multi-GPU training
ddp:
  use_ddp: true
  num_gpus: 4
  strategy: 'ddp'
```

The script automatically detects available GPUs and uses DDP if configured.

## Evaluation

### Test with Predicted Mixing Ratio

```bash
python eval_steps.py \
    --config config/config_MeanFlowTSE_clean.yaml \
    --t_predicter ECAPAMLP \
    --num_steps 1
```

### Test with Different Sampling Steps

```bash
# 1-step inference
python eval_steps.py --config config/config_MeanFlowTSE_clean.yaml --t_predicter ECAPAMLP --num_steps 1

# 5-step inference
python eval_steps.py --config config/config_MeanFlowTSE_clean.yaml --t_predicter ECAPAMLP --num_steps 5

# 10-step inference
python eval_steps.py --config config/config_MeanFlowTSE_clean.yaml --t_predicter ECAPAMLP --num_steps 10
```

### Mixing Ratio Prediction Options

- `GT`: Ground truth mixing ratio (oracle)
- `ECAPAMLP`: Learned predictor using ECAPA-TDNN + MLP
- `RAND`: Random mixing ratio ∈ [0, 1]

### Metrics

The evaluation computes:
- **SI-SDR** (Scale-Invariant Signal-to-Distortion Ratio)
- **PESQ** (Perceptual Evaluation of Speech Quality)
- **eSTOI** (Extended Short-Time Objective Intelligibility)

Results are saved to CSV files with per-sample metrics and summary statistics.

### Additional Metrics: DNSMOS and Speaker Similarity

For more comprehensive evaluation, you can calculate additional perceptual quality metrics using the `calculate_dnsmos_wespeaker.py` script:

#### What it does:

1. **DNSMOS Scores**: Uses the SpeechMOS library to compute Deep Noise Suppression Mean Opinion Score (DNSMOS), which provides:
   - `dnsmos_overall`: Overall perceptual quality
   - `dnsmos_sig`: Speech signal quality (distortion level)
   - `dnsmos_bak`: Background noise quality
   - `dnsmos_p808`: ITU-T P.808 MOS prediction

2. **Speaker Similarity**: Uses WeSpeaker to compute cosine similarity between speaker embeddings of:
   - Generated estimation audio
   - Ground truth source audio
   
   This measures how well the extracted speech preserves the target speaker's voice characteristics.

#### Usage:

```bash
# Calculate DNSMOS and merge with existing metrics
python calculate_dnsmos_wespeaker.py \
    --results_dir test_results_meanflow/clean_ECAPAMLP_steps1 \
    --existing_metrics test_results_meanflow/clean_ECAPAMLP_steps1/metrics_results.csv \
    --output_csv test_results_meanflow/clean_ECAPAMLP_steps1/metrics_complete.csv
```

## Model Checkpoints

Checkpoints are saved in the configured directory:

```
exp/MeanFlowTSE_clean/checkpoints/
├── best.ckpt          # Best model based on validation loss
├── last.ckpt          # Latest checkpoint
└── epoch_XXXX.ckpt    # Periodic checkpoints (if enabled)
```

### Loading Checkpoints

```python
from train_meanflow import LightningModule

# Load trained model
model = LightningModule.load_from_checkpoint(
    'exp/MeanFlowTSE_clean/checkpoints/best.ckpt',
    config=config
)
model.eval()
```

## Project Structure

```
MeanFlowTSE/
├── meanflow.py              # MeanFlowTSE class with alpha scheduling
├── train_meanflow.py        # Training script with PyTorch Lightning
├── eval_steps.py            # Evaluation script with multi-step sampling
├── config/
│   └── config_*.yaml        # Configuration files
├── models/
│   └── udit_meanflow/       # UDiT model implementation
├── data/
│   └── datasets.py          # Dataset loaders for Libri2Mix
└── utils/
    └── transforms.py        # STFT/iSTFT utilities
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{meanflowtse2025,
  title={MeanFlow-TSE: One-Step Generative Target Speaker Extraction with Mean Flow},
  author={Riki Shimizu, Xilin Jiang, Nima Mesgarani},
  journal={arXiv preprint},
  year={2025}
}
```

## Acknowledgments

This implementation builds upon:
- [SpeakerBeam](https://github.com/speechLabBcCuny/SpeakerBeam) - LibriMix Data Preparation Pipeline
- [AD-FlowTSE](https://github.com/aleXiehta/AD-FlowTSE) - Backgone code/implementation

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub or contact rs4613@columbia.edu
