"""
Evaluation script for MeanFlowTSE on Libri2Mix.
"""

import argparse
import os
import random
import torch
import torchaudio
import pytorch_lightning as pl
import pandas as pd
from train_meanflow import LightningModule as MeanFlowLightningModule, parse_config as parse_meanflow_config
from models.t_predicter import TPredicter
from data.datasets import LibriMixInformed
from torch.utils.data import Dataset, DataLoader
from asteroid.metrics import get_metrics
from tqdm import tqdm
from utils.transforms import istft_torch
from pystoi import stoi as calculate_stoi

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation script for MeanFlowTSE')
    parser.add_argument('--config', default=None, help='Path to the config file.')
    parser.add_argument('--t_predicter', type=str, choices=['ECAPAMLP', 'GT', 'RAND', 'ONE', 'ZERO', 'HALF'], default='GT', help='Type of t_predicter to use.')
    parser.add_argument('--num_steps', type=int, default=1, help='Number of Euler steps for sampling (default: 1)')
    args = parser.parse_args()
    return args

def scale_audio(audio):
    """Scale audio to [-1, 1] range."""
    max_val = torch.max(torch.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val
    return audio

def calculate_metrics(mixture, reference, estimation):
    """Calculate SI-SDR, PESQ, and eSTOI metrics."""
    # Calculate SI-SDR and PESQ using asteroid
    metrics = get_metrics(
        mixture.detach().cpu().numpy(), 
        reference.detach().cpu().numpy(), 
        estimation.detach().cpu().numpy(), 
        sample_rate=16000, 
        metrics_list=["si_sdr", "pesq"], 
        ignore_metrics_errors=True
    )
    
    # Calculate eSTOI separately using pystoi
    try:
        estoi_score = calculate_stoi(
            reference.detach().cpu().numpy().squeeze(),
            estimation.detach().cpu().numpy().squeeze(),
            fs_sig=16000,
            extended=True
        )
        metrics['estoi'] = estoi_score
    except Exception as e:
        print(f"Error calculating eSTOI: {e}")
        metrics['estoi'] = None
    
    return metrics

def pad_and_reshape(tensor, multiple):
    """
    Pad the tensor along the last dimension to make its length a multiple of `multiple`
    and reshape it into (n*k, d, multiple) using torch.chunk.
    
    Args:
        tensor: Input tensor of shape (n, d, l)
        multiple: The multiple to pad the length to
    
    Returns:
        reshaped_tensor: Reshaped tensor of shape (n*k, d, multiple)
        original_length: Original length of the last dimension before padding
    """
    n, d, l = tensor.shape
    padding_length = (multiple - (l % multiple)) % multiple
    padded_tensor = torch.nn.functional.pad(tensor, (0, padding_length))
    reshaped_tensor = torch.cat(torch.chunk(padded_tensor, padded_tensor.shape[-1] // multiple, dim=-1), dim=0)
    return reshaped_tensor, l

def reshape_and_remove_padding(tensor, original_length):
    """
    Reshape the tensor back to its original shape and remove the padding.
    
    Args:
        tensor: Input tensor of shape (n*k, d, multiple)
        original_length: Original length of the last dimension before padding
    
    Returns:
        original_tensor: Tensor reshaped back to (n, d, original_length)
    """
    n_k, d, multiple = tensor.shape
    n = original_length // multiple + (1 if original_length % multiple != 0 else 0)
    reshaped_tensor = torch.cat(torch.chunk(tensor, n, dim=0), dim=-1)
    original_tensor = reshaped_tensor[:, :, :original_length]
    return original_tensor

def sample_euler_multistep(model, mixture_spec, enrollment_spec, alpha, num_steps=1):
    """
    Multi-step Euler sampling from alpha to 1.
    
    Args:
        model: The neural network model
        mixture_spec: Mixture spectrogram at alpha (B, C, T)
        enrollment_spec: Enrollment spectrogram (B, C, T_enroll)
        alpha: Mixing ratio for each sample (B,) or scalar
        num_steps: Number of Euler steps to take
    
    Returns:
        source_hat_spec: Predicted source spectrogram at t=1 (B, C, T)
    """
    batch_size = mixture_spec.size(0)
    device = mixture_spec.device
    
    # Ensure alpha is properly shaped: (B,)
    if not torch.is_tensor(alpha):
        alpha = torch.tensor([alpha], device=device)
    if alpha.ndim == 0:
        alpha = alpha.unsqueeze(0)
    if alpha.shape[0] == 1 and batch_size > 1:
        alpha = alpha.repeat(batch_size)
    
    # Initialize at mixture position
    z = mixture_spec
    
    # Create time grid from alpha to 1.0
    alpha_mean = alpha.mean().item()
    t_grid = torch.linspace(alpha_mean, 1.0, num_steps + 1, device=device)
    
    # Perform Euler steps
    for step in range(num_steps):
        # Current time for each sample
        if step == 0:
            # First step: use individual alpha values
            t_current = alpha.clone()  # (B,)
        else:
            # Subsequent steps: use the grid value for all samples
            t_current = torch.full((batch_size,), t_grid[step].item(), device=device)
        
        # Target time for this step
        t_target = torch.full((batch_size,), t_grid[step + 1].item(), device=device)
        
        # Time step size
        dt = t_target - t_current  # (B,)
        
        # Get velocity at current position
        velocity = model(
            z,
            t_current,
            t_target,
            enrollment_spec
        )
        
        # Euler step: z_{t+dt} = z_t + dt * v(z_t, t, r)
        dt_expanded = dt.view(batch_size, 1, 1)
        z = z + dt_expanded * velocity
    
    return z

def generate_samples_from_testset(config_path, lightning_module_class, output_dir, predicter_type, num_steps=1, error_range=[-0.0, 0.0], save_audio=False):
    """Generate samples from test set and evaluate metrics."""
    config = parse_meanflow_config(config_path)
    model_name = 'MeanFlowTSE'
    pl.seed_everything(config['seed'])
    checkpoint_path = config['eval']['checkpoint']
    print(f'Loading model from {checkpoint_path}')
    model = lightning_module_class.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()
    model = model.cuda() if torch.cuda.is_available() else model

    if predicter_type == "ECAPAMLP":
        t_predicter_checkpoint = config['eval']['t_predicter']
        print(f'Loading t_predicter model from {t_predicter_checkpoint}')
        t_predicter = TPredicter(**config['t_predicter'])
        t_predicter.eval()
        t_predicter = t_predicter.cuda() if torch.cuda.is_available() else t_predicter
        config['eval']['t_predicter'] = t_predicter

    test_dataset = LibriMixInformed(
        csv_dir=config['dataset']['test_dir'],
        librimix_meta_dir=config['dataset']['librimix_meta_dir'],
        task=config['dataset']['task'],
        sample_rate=config['dataset']['sample_rate'],
        n_src=config['dataset']['n_src'],
        n_fft=config['dataset']['n_fft'],
        hop_length=config['dataset']['hop_length'],
        win_length=config['dataset']['win_length'],
        segment=None,
        segment_aux=3,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['train']['num_workers'],
        pin_memory=True
    )
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    print(f"Evaluating with {num_steps} Euler step(s)")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc=f"Generating {model_name} samples")):
            mixture = batch['mixture_spec'].cuda() if torch.cuda.is_available() else batch['mixture_spec']
            enrollment = batch['enroll_spec'].cuda() if torch.cuda.is_available() else batch['enroll_spec']

            if predicter_type == "GT":
                alpha = batch['mixing_ratio'].cuda() if torch.cuda.is_available() else batch['mixing_ratio']
            elif predicter_type == "RAND":
                alpha = torch.rand(1).cuda() if torch.cuda.is_available() else torch.rand(1)
            elif predicter_type == "ONE":
                alpha = torch.tensor([1.0]).cuda() if torch.cuda.is_available() else torch.tensor([1.0])
            elif predicter_type == "ZERO":
                alpha = torch.tensor([0.0]).cuda() if torch.cuda.is_available() else torch.tensor([0.0])
            elif predicter_type == "HALF":
                alpha = torch.tensor([0.5]).cuda() if torch.cuda.is_available() else torch.tensor([0.5])
            else:
                mixture_wav = batch['mixture'].cuda() if torch.cuda.is_available() else batch['mixture']
                enrollment_wav = batch['enroll'].cuda() if torch.cuda.is_available() else batch['enroll']
                alpha_true = batch['mixing_ratio'].cuda() if torch.cuda.is_available() else batch['mixing_ratio']
                alpha = t_predicter(mixture_wav, enrollment_wav, aug=False)

            multiple = config['dataset']['sample_rate'] * 3 // config['dataset']['hop_length'] + 1  # 3 seconds
            mixture, original_length = pad_and_reshape(mixture, multiple)
            
            # Get batch size after padding/reshaping
            batch_size = mixture.shape[0]
            
            # Multi-step Euler sampling from mixture (at alpha) to clean source (at t=1)
            source_hat_spec = sample_euler_multistep(
                model=model.model,
                mixture_spec=mixture.float(),
                enrollment_spec=enrollment.repeat(batch_size, 1, 1),
                alpha=alpha,
                num_steps=num_steps
            )
            
            source_hat_spec = reshape_and_remove_padding(source_hat_spec, original_length)
            source_hat = istft_torch(
                source_hat_spec, 
                n_fft=config['dataset']['n_fft'], 
                hop_length=config['dataset']['hop_length'], 
                win_length=config['dataset']['win_length'],
                length=batch['source'].shape[-1]
            )
            source_hat = scale_audio(source_hat.cpu())
            
            if save_audio:
                output_path = os.path.join(
                    output_dir,
                    batch['utt_id'][0],
                    batch['mixture_filename'][0].replace('.wav', ''),
                    'estimation.wav',
                )
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                torchaudio.save(output_path, source_hat, config['dataset']['sample_rate'])

                source_path = os.path.join(
                    output_dir,
                    batch['utt_id'][0],
                    batch['mixture_filename'][0].replace('.wav', ''),
                    'source.wav',
                )
                enroll_path = os.path.join(
                    output_dir,
                    batch['utt_id'][0],
                    batch['mixture_filename'][0].replace('.wav', ''),
                    'enroll.wav',
                )
                background_path = os.path.join(
                    output_dir,
                    batch['utt_id'][0],
                    batch['mixture_filename'][0].replace('.wav', ''),
                    'background.wav',
                )
                mixture_path = os.path.join(
                    output_dir,
                    batch['utt_id'][0],
                    batch['mixture_filename'][0].replace('.wav', ''),
                    'mixture.wav',
                )
                os.makedirs(os.path.dirname(source_path), exist_ok=True)
                os.makedirs(os.path.dirname(enroll_path), exist_ok=True)
                os.makedirs(os.path.dirname(background_path), exist_ok=True)
                os.makedirs(os.path.dirname(mixture_path), exist_ok=True)
                
                torchaudio.save(source_path, batch['source'], config['dataset']['sample_rate'])
                torchaudio.save(enroll_path, batch['enroll'], config['dataset']['sample_rate'])
                torchaudio.save(background_path, batch['background'], config['dataset']['sample_rate'])
                torchaudio.save(mixture_path, batch['mixture'], config['dataset']['sample_rate'])
            
            all_output_metrics = calculate_metrics(batch['mixture_rescaled'], batch['source_rescaled'], source_hat)
            
            if predicter_type == "ECAPAMLP":
                results.append({
                    'filename': batch['mixture_filename'][0],
                    **all_output_metrics,
                    'alpha': alpha_true.item(),
                    'alpha_hat': alpha.item(),
                })
            else:
                results.append({
                    'filename': batch['mixture_filename'][0],
                    **all_output_metrics,
                    'alpha': alpha.item(),
                    'alpha_hat': None,
                })
    
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(output_dir, 'metrics_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f'Saved metrics results to {results_csv_path}')
    
    # Print summary statistics
    print(f"\n{'='*50}")
    print(f"Evaluation Results Summary ({model_name}, {num_steps} step(s))")
    print(f"{'='*50}")
    for metric in ['si_sdr', 'pesq', 'estoi']:
        if metric in results_df.columns:
            print(f"{metric.upper():8s}: {results_df[metric].mean():.4f} ± {results_df[metric].std():.4f}")
    print(f"{'='*50}\n")
    
    return results_df

def main():
    args = parse_args()
    config_path = args.config
    error = 0.0

    if "noisy" in config_path:
        task = "noisy"
    else:
        task = "clean"
    output_dir = f'test_results_meanflow/{task}_{args.t_predicter}_steps{args.num_steps}'

    print(f'Generating samples using MeanFlowTSE model with {args.num_steps} Euler step(s)...')
    generate_samples_from_testset(
        config_path,
        MeanFlowLightningModule,
        output_dir,
        predicter_type=args.t_predicter,
        num_steps=args.num_steps,
        error_range=[-error, error],
        save_audio=False
    )

if __name__ == '__main__':
    main()