# MeanFlow-TSE: One-Step Generative Target Speaker Extraction with Mean Flow

Official implementation of **MeanFlowTSE**, a target speaker extraction system using flow matching with curriculum learning through alpha scheduling.

## Overview

MeanFlowTSE combines AD-FlowTSE and MeanFlow/AlphaFlow training objectives for effective one-step target speaker extraction. Experiments on Libri2Mix dataset show that MeanFlowTSE achieve the SOTA performance in SI-SDR, PESQ, and ESTOI compared to the previous generative (diffusion / flow-matching) TSE models.

## Installation

### Requirements

```bash
# Python 3.8+
pip install torch torchvision torchaudio
pip install pytorch-lightning
pip install asteroid
pip install einops
pip install pystoi
pip install pandas tqdm pyyaml
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
