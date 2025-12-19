"""
Training script for MeanFlowTSE with alpha-flow scheduling and periodic checkpointing.
"""
import argparse
import os
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    Callback
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from asteroid.engine.optimizers import make_optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LambdaLR, SequentialLR

from meanflow import MeanFlowTSE
from utils import neg_sisdr_loss_wrapper
from data.datasets import get_dataloaders
from models.udit_meanflow.udit_meanflow import UDiT
from utils.transforms import istft_torch
from flow_matching.path import CondOTProbPath
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.utils import ModelWrapper


def parse_args():
    parser = argparse.ArgumentParser(description='Training script for MeanFlowTSE')
    parser.add_argument('--config', default='config/config.yaml', help='Path to the config file.')
    args = parser.parse_args()
    return args


def parse_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class MeanFlowModelWrapper(ModelWrapper):
    """
    Wrapper to make UDiT compatible with ODESolver.
    ODESolver expects a model that takes (x, t, **extras).
    UDiT expects (x, t, r, enrollment).
    """
    def __init__(self, model):
        super().__init__(model)
        self.model = model
        
    def forward(self, x, t, **extras):
        """
        Args:
            x: (B, C, T) tensor
            t: (B,) or scalar tensor of current timestep
            **extras: Must contain 'r' and 'enrollment'
        """
        r = extras.get('r', t)  # If r not provided, use t (rectified flow mode)
        enrollment = extras['enrollment']
        return self.model(x, t, r, enrollment)


class MetricPrinterCallback(Callback):
    """Custom callback to print metrics at the end of each epoch."""
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Print training metrics at the end of each training epoch."""
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        
        # Collect metrics to print
        train_loss = metrics.get('train_loss', None)
        train_mse = metrics.get('train_mse', None)
        train_alpha = metrics.get('train_alpha_epoch', None)
        lr = None
        
        # Get learning rate
        if trainer.optimizers:
            lr = trainer.optimizers[0].param_groups[0]['lr']
        
        # Build print string
        print_str = f"Epoch {epoch}"
        if train_loss is not None:
            print_str += f" | Train Loss: {train_loss:.4f}"
        if train_mse is not None:
            print_str += f" | Train MSE: {train_mse:.4f}"
        if train_alpha is not None:
            print_str += f" | Alpha: {train_alpha:.4f}"
        if lr is not None:
            print_str += f" | LR: {lr:.6f}"
        
        print(print_str)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Print validation metrics at the end of each validation epoch."""
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        
        val_loss = metrics.get('val_loss', None)
        
        if val_loss is not None:
            print(f"Epoch {epoch} | Val Loss: {val_loss:.4f}")


class LightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = UDiT(
            **config['model']
        )
        self.config = config
        self.save_hyperparameters(config)
        self.neg_si_sdr = neg_sisdr_loss_wrapper
        
        # Initialize MeanFlowTSE with alpha scheduling
        meanflow_config = config.get('meanflow', {})
        
        self.meanflow = MeanFlowTSE(
            flow_ratio=meanflow_config.get('flow_ratio', 0.50),
            use_enrollment=True,
            data_dim='1d',
            alpha_schedule_start=meanflow_config.get('alpha_schedule_start_epoch', 0),
            alpha_schedule_end=meanflow_config.get('alpha_schedule_end_epoch', None),
            alpha_gamma=meanflow_config.get('alpha_gamma', 25.0),
            alpha_min=meanflow_config.get('alpha_min', 5e-3),
            gamma=config.get('loss', {}).get('gamma', 0.0),
        )
        
        # Track global step for alpha scheduling
        self.steps_per_epoch = None

    def on_train_start(self):
        """Called when training starts - compute steps per epoch and adjust alpha schedule."""
        if self.steps_per_epoch is None:
            # Get dataloader to compute steps per epoch
            train_dataloader = self.trainer.train_dataloader
            self.steps_per_epoch = len(train_dataloader)
            
            # Convert epoch-based schedule to iteration-based schedule
            alpha_start_epoch = self.config.get('meanflow', {}).get('alpha_schedule_start_epoch', 0)
            alpha_end_epoch = self.config.get('meanflow', {}).get('alpha_schedule_end_epoch', None)
            
            self.meanflow.alpha_schedule_start = alpha_start_epoch * self.steps_per_epoch
            
            if alpha_end_epoch is not None:
                self.meanflow.alpha_schedule_end = alpha_end_epoch * self.steps_per_epoch
            
            print(f"Alpha schedule: iterations {self.meanflow.alpha_schedule_start} to {self.meanflow.alpha_schedule_end}")
            print(f"Steps per epoch: {self.steps_per_epoch}")

    def forward(self, x, t, r, enrollment):
        return self.model(x, t, r, enrollment)

    def training_step(self, batch, batch_idx):
        """
        Training step using MeanFlowTSE with alpha scheduling.
        
        Convention:
        - t=0: background/mixture (noisy)
        - t=1: source (clean target)
        
        The model works directly with unnormalized spectrograms.
        """
        # Get current global step for alpha scheduling
        current_iteration = self.current_epoch * self.steps_per_epoch + batch_idx

        # Extract data
        background = batch['background_rescaled_spec']  # t=0 (noisy mixture)
        source = batch['source_rescaled_spec']  # t=1 (clean target)
        enrollment = batch['enroll_spec']  # Reference
        batch_size = source.size(0)
        
        # Compute loss with alpha scheduling
        loss, mse_val, alpha, raw_alpha = self.meanflow.loss(
            model=self.model,
            source=source,
            background=background,
            enrollment=enrollment,
            iteration=current_iteration
        )

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
        self.log('train_mse', mse_val, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
        self.log('train_alpha', alpha, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
        self.log('train_alpha_raw', raw_alpha, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step - generate samples and compute SI-SDR.
        Uses ODESolver for inference.

        Convention:
        - t=0: background/mixture (noisy) - starting point
        - t=1: source (clean) - target
        - mixing_ratio: actual ratio used by LibriMix to create the mixture
        """

        enrollment = batch['enroll_spec']  # Unnormalized
        mixture = batch['mixture_spec']  # The actual mixture spectrogram
        mixing_ratio = batch['mixing_ratio']  # The actual ratio used by LibriMix
        if mixing_ratio.ndim > 1:
            mixing_ratio = mixing_ratio.squeeze()  # Remove extra dimensions if present
        
        batch_size = mixture.size(0)
        
        with torch.no_grad():
            # Create model wrapper for ODESolver
            wrapped_model = MeanFlowModelWrapper(self.model)
            solver = ODESolver(velocity_model=wrapped_model)
            
            # Sample from mixture (at mixing_ratio) to clean source (at t=1)
            # Time grid: [mixing_ratio_mean, 1.0] (from current position to clean target)
            mixing_ratio_mean = mixing_ratio.mean().item()
            time_grid = torch.tensor([mixing_ratio_mean, 1.0], device=mixture.device)
            
            # Sample using ODESolver
            source_hat_spec = solver.sample(
                x_init=mixture,
                time_grid=time_grid,
                method=self.config['solver']['method'],
                step_size=self.config['solver']['step_size'],
                enrollment=enrollment,
                r=time_grid[-1].expand(batch_size),  # r is the target timestep (t=1)
            )
            
            # Convert spectrogram to waveform using iSTFT
            source_hat = istft_torch(
                source_hat_spec,
                n_fft=self.config['dataset']['n_fft'],
                hop_length=self.config['dataset']['hop_length'],
                win_length=self.config['dataset']['win_length'],
                length=batch['source'].shape[-1],
            )
            
            # Compute SI-SDR loss (negative because we want to minimize)
            loss = self.neg_si_sdr(source_hat, batch['source'])
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
        
        return loss


    def configure_optimizers(self):
        optimizer = make_optimizer(self.model.parameters(), **self.config['optim'])
        
        if self.config['scheduler']['type'] == 'ReduceLROnPlateau':
            scheduler = {
                'scheduler': ReduceLROnPlateau(
                    optimizer=optimizer,
                    factor=self.config['scheduler']['lr_reduce_factor'],
                    patience=self.config['scheduler']['lr_reduce_patience']
                ),
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
                'strict': True,
            }
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        
        elif self.config['scheduler']['type'] == 'CosineAnnealingLR':
            def warmup_lambda(epoch):
                if epoch < self.config['scheduler']['warmup_epochs']:
                    return epoch / self.config['scheduler']['warmup_epochs']
                return 1.0

            warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.config['scheduler']['t_max'],
                eta_min=self.config['scheduler']['eta_min']
            )

            scheduler = {
                'scheduler': SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[self.config['scheduler']['warmup_epochs']]
                ),
                'interval': 'epoch',
                'frequency': 1
            }

            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        else:
            return optimizer


class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.rank = int(os.environ.get('RANK', 0))

    def setup(self, stage=None):
        self.train_loader, self.val_loader = get_dataloaders(
            self.config,
            is_ddp=False,
            world_size=self.world_size,
            rank=self.rank
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


def main():
    args = parse_args()
    config = parse_config(args.config)
    pl.seed_everything(config['seed'])
    torch.set_float32_matmul_precision('medium')

    os.makedirs(config['train']['log_dir'], exist_ok=True)
    os.makedirs(config['checkpoint']['dir'], exist_ok=True)

    if config['checkpoint']['load_weights_only'] and config['checkpoint']['resume']:
        model = LightningModule.load_from_checkpoint(
            config['checkpoint']['resume'],
            config=config,
            strict=True
        )
    else:
        model = LightningModule(config)

    data_module = DataModule(config)

    callbacks = []

    # Add metric printer callback
    metric_printer = MetricPrinterCallback()
    callbacks.append(metric_printer)

    if config['early_stopping']['enabled']:
        early_stopping_callback = EarlyStopping(
            monitor=config['early_stopping']['monitor'],
            patience=config['early_stopping']['patience'],
            verbose=config['early_stopping']['verbose'],
            mode=config['early_stopping']['mode'],
            min_delta=config['early_stopping']['delta'],
        )
        callbacks.append(early_stopping_callback)

    # Main checkpoint callback for best/last
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['checkpoint']['dir'],
        filename=config['checkpoint']['ckpt_name'],
        save_top_k=config['checkpoint']['save_best'],
        save_last=config['checkpoint']['save_last'],
        verbose=config['checkpoint']['verbose'],
        monitor=config['checkpoint']['monitor'],
        mode=config['checkpoint']['mode'],
    )
    callbacks.append(checkpoint_callback)

    # Periodic checkpoint callback (saves every N epochs)
    periodic_config = config['checkpoint'].get('periodic', {})
    if periodic_config.get('enabled', False):
        save_every_n_epochs = periodic_config.get('save_every_n_epochs', 100)
        periodic_checkpoint_callback = ModelCheckpoint(
            dirpath=config['checkpoint']['dir'],
            filename='epoch_{epoch:04d}',
            save_top_k=-1,  # Save all periodic checkpoints
            every_n_epochs=save_every_n_epochs,
            verbose=periodic_config.get('verbose', True),
            save_on_train_epoch_end=True,
        )
        callbacks.append(periodic_checkpoint_callback)
        print(f"Periodic checkpointing enabled: saving every {save_every_n_epochs} epochs")

    if config['train']['log_lr']:
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)

    tb_logger = TensorBoardLogger(
        save_dir=config['train']['log_dir'],
        name='lightning_logs',
        version='0',
    )

    ddp_config = config.get('ddp', {})
    use_ddp = ddp_config.get('use_ddp', False)
    num_nodes = ddp_config.get('num_nodes', 1)
    num_gpus = ddp_config.get('num_gpus', torch.cuda.device_count())
    strategy = DDPStrategy(find_unused_parameters=False)

    if use_ddp and num_gpus > 1:
        trainer = pl.Trainer(
            max_epochs=config['train']['num_epochs'],
            accelerator='gpu',
            devices=num_gpus,
            num_nodes=num_nodes,
            strategy=strategy,
            accumulate_grad_batches=config['train']['accumulation_steps'],
            callbacks=callbacks,
            default_root_dir=config['train']['log_dir'],
            logger=tb_logger,
            log_every_n_steps=config['train']['log_interval'],
            precision=config['train']['precision'],
            detect_anomaly=config['train']['detect_anomaly'],
            gradient_clip_val=config['train']['gradient_clip_val'],
            limit_train_batches=config['train']['limit_train_batches'],
            limit_val_batches=config['train']['limit_val_batches'],
            enable_progress_bar=False,
        )
    else:
        trainer = pl.Trainer(
            max_epochs=config['train']['num_epochs'],
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            accumulate_grad_batches=config['train']['accumulation_steps'],
            callbacks=callbacks,
            default_root_dir=config['train']['log_dir'],
            logger=tb_logger,
            log_every_n_steps=config['train']['log_interval'],
            precision=config['train']['precision'],
            gradient_clip_val=config['train']['gradient_clip_val'],
            detect_anomaly=config['train']['detect_anomaly'],
            limit_train_batches=config['train']['limit_train_batches'],
            limit_val_batches=config['train']['limit_val_batches'],
            enable_progress_bar=False
        )

    ckpt_path = config['checkpoint']['resume'] if config['checkpoint']['resume'] else None

    if config['checkpoint']['load_weights_only']:
        trainer.fit(model, datamodule=data_module)
    else:
        trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()