import argparse
import os
import time
import torch
import torchaudio
import pytorch_lightning as pl
import pandas as pd
import psutil
import numpy as np
from train_meanflow import LightningModule as MeanFlowLightningModule, parse_config as parse_meanflow_config, MeanFlowModelWrapper
from models.t_predicter import TPredicter
from data.datasets import LibriMixInformed
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from flow_matching.solver.ode_solver import ODESolver
from utils.transforms import istft_torch

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation script for MeanFlow')
    parser.add_argument('--config', default=None, help='Path to the config file.')
    parser.add_argument('--t_predicter', type=str, choices=['ECAPAMLP', 'GT', 'RAND', 'ONE', 'ZERO', 'HALF'], default='GT', help='Type of t_predicter to use.')
    parser.add_argument('--n_samples', type=int, default=None, help='Number of samples to evaluate. If None, evaluates all samples.')
    parser.add_argument('--warmup_samples', type=int, default=0, help='Number of initial samples to exclude from results (warmup).')
    args = parser.parse_args()
    return args

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    cpu_mem_mb = mem_info.rss / 1024 / 1024
    
    gpu_mem_mb = 0
    if torch.cuda.is_available():
        gpu_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    return cpu_mem_mb, gpu_mem_mb

def extract_3sec_segment(spec, sample_rate, hop_length):
    """
    Extract exactly 3 seconds from the spectrogram.
    
    Args:
        spec: Input spectrogram of shape (batch, freq, time)
        sample_rate: Audio sample rate
        hop_length: STFT hop length
    
    Returns:
        3-second segment of the spectrogram
    """
    # Calculate number of frames for 3 seconds
    target_frames = sample_rate * 3 // hop_length + 1
    
    # Get the actual number of frames
    actual_frames = spec.shape[-1]
    
    if actual_frames >= target_frames:
        # Take first 3 seconds
        return spec[:, :, :target_frames]
    else:
        # If shorter than 3 seconds, pad with zeros
        padding = target_frames - actual_frames
        return torch.nn.functional.pad(spec, (0, padding))

def generate_samples_from_testset(config_path, lightning_module_class, output_dir, predicter_type, n_samples=None, warmup_samples=0):
    config = parse_meanflow_config(config_path)
    model_name = 'MeanFlow'
    pl.seed_everything(config['seed'])
    checkpoint_path = config['eval']['checkpoint']
    print(f'Loading model from {checkpoint_path}')
    model = lightning_module_class.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()
    model = model.cuda() if torch.cuda.is_available() else model

    # Load t_predicter if needed (outside the loop)
    t_predicter = None
    if predicter_type == "ECAPAMLP":
        t_predicter_checkpoint = config['eval']['t_predicter']
        print(f'Loading t_predicter model from {t_predicter_checkpoint}')
        t_predicter = TPredicter(**config['t_predicter'])
        t_predicter.eval()
        t_predicter = t_predicter.cuda() if torch.cuda.is_available() else t_predicter

    # Wrap the model for ODESolver compatibility (outside the loop)
    wrapped_model = MeanFlowModelWrapper(model.model)
    solver = ODESolver(velocity_model=wrapped_model)

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
    
    # Lists to track RTF and memory (excluding warmup)
    rtf_list = []
    cpu_mem_list = []
    gpu_mem_list = []
    
    # Warmup tracking
    warmup_rtf_list = []
    warmup_cpu_mem_list = []
    warmup_gpu_mem_list = []
    
    # Determine number of samples to process (including warmup)
    total_samples_to_process = n_samples if n_samples is not None else len(test_loader)
    total_samples_with_warmup = total_samples_to_process + warmup_samples
    
    print(f"Warmup samples: {warmup_samples}")
    print(f"Evaluation samples: {total_samples_to_process}")
    print(f"Total samples to process: {total_samples_with_warmup}\n")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc=f"Generating {model_name} samples", total=total_samples_with_warmup)):
            if i >= total_samples_with_warmup:
                break
            
            is_warmup = i < warmup_samples
            
            # Extract 3-second segments
            mixture = batch['mixture_spec'].cuda() if torch.cuda.is_available() else batch['mixture_spec']
            enrollment = batch['enroll_spec'].cuda() if torch.cuda.is_available() else batch['enroll_spec']
            
            mixture = extract_3sec_segment(mixture, config['dataset']['sample_rate'], config['dataset']['hop_length'])
            
            # Reset memory stats at the start of each batch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            
            # Start timing - right before inference
            start_time = time.time()
            
            # Get alpha value (included in timing for ECAPAMLP)
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
            else:  # ECAPAMLP
                mixture_wav = batch['mixture'].cuda() if torch.cuda.is_available() else batch['mixture']
                enrollment_wav = batch['enroll'].cuda() if torch.cuda.is_available() else batch['enroll']
                alpha_true = batch['mixing_ratio'].cuda() if torch.cuda.is_available() else batch['mixing_ratio']
                alpha = t_predicter(mixture_wav, enrollment_wav, aug=False)
            
            # Time grid: from alpha (mixture position) to 1.0 (clean source)
            alpha_grid = torch.tensor([alpha.item(), 1.0], device=mixture.device)
            
            # Model inference
            source_hat_spec = solver.sample(
                time_grid=alpha_grid,
                x_init=mixture.float(),
                method=config['solver']['method'],
                step_size=config['solver']['test_step_size'],
                enrollment=enrollment,
                r=torch.full((1,), 1.0, device=mixture.device),  # Target is clean source at t=1
            )
            
            # End timing and synchronize GPU
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            
            # Calculate RTF (Real-Time Factor)
            # RTF = processing_time / audio_duration
            audio_duration = 3.0  # Fixed 3 seconds
            processing_time = end_time - start_time
            rtf = processing_time / audio_duration
            
            # Get memory usage
            cpu_mem, gpu_mem = get_memory_usage()
            
            # Store in appropriate list (warmup or evaluation)
            if is_warmup:
                warmup_rtf_list.append(rtf)
                warmup_cpu_mem_list.append(cpu_mem)
                warmup_gpu_mem_list.append(gpu_mem)
            else:
                rtf_list.append(rtf)
                cpu_mem_list.append(cpu_mem)
                gpu_mem_list.append(gpu_mem)
            
                # Store results only for non-warmup samples
                if predicter_type == "ECAPAMLP":
                    results.append({
                        'filename': batch['mixture_filename'][0],
                        'alpha': alpha_true.item(),
                        'alpha_hat': alpha.item(),
                        'rtf': rtf,
                        'cpu_memory_mb': cpu_mem,
                        'gpu_memory_mb': gpu_mem,
                        'audio_duration_sec': audio_duration,
                        'processing_time_sec': processing_time,
                    })
                else:
                    results.append({
                        'filename': batch['mixture_filename'][0],
                        'alpha': alpha.item(),
                        'alpha_hat': None,
                        'rtf': rtf,
                        'cpu_memory_mb': cpu_mem,
                        'gpu_memory_mb': gpu_mem,
                        'audio_duration_sec': audio_duration,
                        'processing_time_sec': processing_time,
                    })
    
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(output_dir, 'performance_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f'Saved performance results to {results_csv_path}')
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"Performance Evaluation Summary ({model_name})")
    print(f"{'='*60}")
    
    # Show warmup statistics if applicable
    if warmup_samples > 0 and len(warmup_rtf_list) > 0:
        print(f"\nWarmup Statistics ({warmup_samples} samples - EXCLUDED from results):")
        print(f"  RTF: {np.mean(warmup_rtf_list):.6f} ± {np.std(warmup_rtf_list):.6f}")
        print(f"  Processing time: {np.mean([w * 3.0 for w in warmup_rtf_list]):.6f}s ± {np.std([w * 3.0 for w in warmup_rtf_list]):.6f}s")
        print(f"\n{'-'*60}")
    
    # RTF statistics (evaluation samples only)
    print(f"\nRTF Statistics (evaluation samples only):")
    print(f"  {'Mean':12s}: {np.mean(rtf_list):.6f} ± {np.std(rtf_list):.6f}")
    print(f"  {'Median':12s}: {np.median(rtf_list):.6f}")
    print(f"  {'Min':12s}: {np.min(rtf_list):.6f}")
    print(f"  {'Max':12s}: {np.max(rtf_list):.6f}")
    print(f"  {'Real-time':12s}: {'Yes (RTF < 1.0)' if np.mean(rtf_list) < 1.0 else 'No (RTF >= 1.0)'}")
    
    print(f"\n{'-'*60}")
    
    # Memory statistics (evaluation samples only)
    print(f"\nMemory Usage (evaluation samples only):")
    print(f"  CPU Memory:")
    print(f"    {'Mean':12s}: {np.mean(cpu_mem_list):.2f} MB ± {np.std(cpu_mem_list):.2f} MB")
    print(f"    {'Max':12s}: {np.max(cpu_mem_list):.2f} MB")
    
    if torch.cuda.is_available():
        print(f"  GPU Memory:")
        print(f"    {'Mean':12s}: {np.mean(gpu_mem_list):.2f} MB ± {np.std(gpu_mem_list):.2f} MB")
        print(f"    {'Max':12s}: {np.max(gpu_mem_list):.2f} MB")
    
    print(f"\n{'-'*60}")
    
    # Processing statistics
    print(f"\nProcessing Statistics:")
    print(f"  {'Warmup samples':20s}: {warmup_samples}")
    print(f"  {'Evaluation samples':20s}: {len(rtf_list)}")
    print(f"  {'Total processed':20s}: {warmup_samples + len(rtf_list)}")
    print(f"  {'Audio duration':20s}: {audio_duration:.2f} seconds (per sample)")
    print(f"  {'Avg processing time':20s}: {results_df['processing_time_sec'].mean():.6f} seconds")
    print(f"  {'Min processing time':20s}: {results_df['processing_time_sec'].min():.6f} seconds")
    print(f"  {'Max processing time':20s}: {results_df['processing_time_sec'].max():.6f} seconds")
    
    print(f"{'='*60}\n")
    
    return results_df

def main():
    args = parse_args()
    config_path = args.config

    if "noisy" in config_path:
        task = "noisy"
    else:
        task = "clean"
    output_dir = f'test_results_meanflow/{task}_{args.t_predicter}'

    print(f'Generating samples using MeanFlow model...')
    if args.n_samples is not None:
        print(f'Evaluating on {args.n_samples} samples (after {args.warmup_samples} warmup samples)')
    if args.warmup_samples > 0:
        print(f'First {args.warmup_samples} samples will be excluded from results (warmup)')
    
    generate_samples_from_testset(
        config_path,
        MeanFlowLightningModule,
        output_dir,
        predicter_type=args.t_predicter,
        n_samples=args.n_samples,
        warmup_samples=args.warmup_samples
    )

if __name__ == '__main__':
    main()
