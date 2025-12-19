import torch
import torch.nn.functional as F
from einops import rearrange
import numpy as np


def stopgrad(x):
    return x.detach()

def adaptive_l2_loss(error, gamma=0.0, c=1e-3, alpha=None):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    
    Args:
        error: Error tensor (B, C, T) or (B, C, H, W)
        gamma: Exponent parameter (default: 0.0)
        c: Small constant for numerical stability (default: 1e-3)
        alpha: Optional alpha value for alpha-dependent weighting (default: None)
               When provided, uses w = alpha / (delta_sq + c)^p instead of w = 1 / (delta_sq + c)^p
    """
    if error.ndim == 3:
        delta_sq = torch.mean(error ** 2, dim=(1, 2), keepdim=False)
    elif error.ndim == 4:
        delta_sq = torch.mean(error ** 2, dim=(1, 2, 3), keepdim=False)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {error.ndim}D")
    
    p = 1.0 - gamma
    
    # Apply alpha-dependent weighting if alpha is provided
    if alpha is not None:
        # For alpha-Flow: w = alpha / (delta_sq + c)^p
        w = alpha**p / (delta_sq + c).pow(p)
    else:
        # Standard weighting: w = 1 / (delta_sq + c)^p
        w = 1.0 / (delta_sq + c).pow(p)
    
    loss = delta_sq  # ||Δ||^2
    return (stopgrad(w) * loss).mean()


class MeanFlowTSE:
    """
    MeanFlowTSE: Target speaker extraction with flow matching objectives.
    Includes alpha scheduling for curriculum learning from trajectory flow matching
    to consistency training.
    
    Convention:
    - t=0: background/mixture (noisy starting point)
    - t=1: source (clean target)
    - r ≥ t always (r is the target timestep, moving forward in time)
    """
    
    def __init__(
        self,
        flow_ratio=0.50,
        use_enrollment=True,
        data_dim='1d',
        alpha_schedule_start=0,  # Iteration to start transitioning from alpha=1
        alpha_schedule_end=None,  # Iteration to finish transitioning to alpha=0
        alpha_gamma=25.0,  # Temperature parameter for sigmoid schedule
        alpha_min=5e-3,  # Minimum alpha value (clamping)
        gamma=0.0,  # Gamma parameter for adaptive loss weighting
    ):
        super().__init__()
        self.use_enrollment = use_enrollment
        self.data_dim = data_dim
        self.flow_ratio = flow_ratio
        self.gamma = gamma
        
        # Alpha scheduling parameters
        self.alpha_schedule_start = alpha_schedule_start
        self.alpha_schedule_end = alpha_schedule_end
        self.alpha_gamma = alpha_gamma
        self.alpha_min = alpha_min
        
        # Track current training iteration
        self.current_iteration = 0

    def get_alpha(self, iteration=None, return_raw=False):
        """
        Compute alpha value based on the current training iteration using sigmoid schedule.
        
        Args:
            iteration: Current training iteration (uses self.current_iteration if None)
            return_raw: If True, return (clipped_alpha, raw_alpha_before_clipping)
        
        Returns:
            alpha: Value in (0, 1], with 1 = trajectory flow matching, 0 = consistency training
            If return_raw=True: tuple of (clipped_alpha, raw_alpha)
        """
        if iteration is None:
            iteration = self.current_iteration
        
        # If no schedule defined, use alpha=1 (pure trajectory flow matching mode)
        if self.alpha_schedule_end is None:
            return (1.0, 1.0) if return_raw else 1.0
        
        # Before schedule starts, use alpha=1 (trajectory flow matching pretraining)
        if iteration < self.alpha_schedule_start:
            return (1.0, 1.0) if return_raw else 1.0
        
        # After schedule ends, use alpha≈0 (consistency training fine-tuning)
        if iteration >= self.alpha_schedule_end:
            # Compute raw alpha even after schedule ends
            k_s = self.alpha_schedule_start
            k_e = self.alpha_schedule_end
            scale = 1.0 / (k_e - k_s)
            offset = -(k_s + k_e) / 2.0 / (k_e - k_s)
            x = (scale * iteration + offset) * self.alpha_gamma
            raw_alpha = 1.0 - torch.sigmoid(torch.tensor(x)).item()
            return (self.alpha_min, raw_alpha) if return_raw else self.alpha_min
        
        # During transition: sigmoid schedule
        k_s = self.alpha_schedule_start
        k_e = self.alpha_schedule_end
        scale = 1.0 / (k_e - k_s)
        offset = -(k_s + k_e) / 2.0 / (k_e - k_s)
        
        # alpha = 1 - sigmoid((scale * k + offset) * gamma)
        x = (scale * iteration + offset) * self.alpha_gamma
        raw_alpha = 1.0 - torch.sigmoid(torch.tensor(x)).item()
        
        # Apply clamping
        if raw_alpha > (1.0 - self.alpha_min):
            clipped_alpha = 1.0
        elif raw_alpha < self.alpha_min:
            clipped_alpha = self.alpha_min
        else:
            clipped_alpha = raw_alpha
        
        return (clipped_alpha, raw_alpha) if return_raw else clipped_alpha

    def loss_rectified_flow(self, model, source, background, enrollment, alpha=None):
        """
        Rectified flow loss with alpha-dependent weighting.
        Uses sigmoid(randn) for t sampling and sets r = t.
        
        Flow direction: background (t=0) -> source (t=1)
        
        Args:
            model: The neural network model
            source: Clean source (t=1)
            background: Background/noise (t=0)
            enrollment: Enrollment/reference signals
            alpha: Optional alpha value for weighting. If None, uses 1.0 (standard weighting).
        """
        batch_size = source.shape[0]
        device = source.device

        # Sample time using sigmoid of normal
        nt = torch.randn(batch_size, device=device)
        t = torch.sigmoid(nt)
        
        # For rectified flow, r = t
        r = t.clone()

        # Reshape time variables
        if self.data_dim == '1d':
            t_ = rearrange(t, "b -> b 1 1")
        else:
            t_ = rearrange(t, "b -> b 1 1 1")

        # Interpolate: z = (1-t) * background + t * source
        z = (1 - t_) * background + t_ * source
        
        # Target velocity: from background to source
        v = source - background

        # Model prediction
        u = model(z, t, r, enrollment=enrollment)

        # Compute loss with alpha-dependent adaptive weighting
        error = u - v
        loss_alpha = alpha if alpha is not None else 1.0
        loss = adaptive_l2_loss(error, gamma=self.gamma, alpha=loss_alpha)
        mse_val = (error ** 2).mean()
        
        return loss, mse_val

    def loss_alpha_flow(self, model, source, background, enrollment, alpha):
        """
        Alpha-Flow loss for alpha ∈ (0, 1).
        Interpolates between trajectory flow matching and consistency training.
        
        Args:
            model: The neural network model
            source: Clean source (t=1)
            background: Background/noise (t=0)
            enrollment: Enrollment/reference signals
            alpha: Consistency step ratio in (0, 1]
        """
        batch_size = source.shape[0]
        device = source.device

        # Sample t and r from logistic distribution
        mu, sigma = -0.4, 1.0
        normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
        samples = 1 / (1 + np.exp(-normal_samples))  # sigmoid
        
        # Ensure r >= t (r is the target, ahead in flow)
        t_np = np.minimum(samples[:, 0], samples[:, 1])
        r_np = np.maximum(samples[:, 0], samples[:, 1])
        
        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        
        # Sample intermediate point s using alpha
        s = alpha * r + (1 - alpha) * t

        # Reshape time variables
        if self.data_dim == '1d':
            t_ = rearrange(t, "b -> b 1 1")
            s_ = rearrange(s, "b -> b 1 1")
            r_ = rearrange(r, "b -> b 1 1")
        else:
            t_ = rearrange(t, "b -> b 1 1 1")
            s_ = rearrange(s, "b -> b 1 1 1")
            r_ = rearrange(r, "b -> b 1 1 1")

        # Get point at time t
        x_t = (1 - t_) * background + t_ * source
        
        # Predict velocity from t to s
        u2 = model(x_t, t, s, enrollment=enrollment)
        
        # Move to intermediate point s
        x_s = x_t + (s_ - t_) * u2
        
        # Predict velocity from s to r
        u1 = model(x_s, s, r, enrollment=enrollment)
        
        # Predict direct velocity from t to r
        u_tr = model(x_t, t, r, enrollment=enrollment)
        
        # Compute interpolated target
        lambda_val = (s - t) / (r - t + 1e-8)
        if self.data_dim == '1d':
            lambda_ = rearrange(lambda_val, "b -> b 1 1")
        else:
            lambda_ = rearrange(lambda_val, "b -> b 1 1 1")
        
        target_u = (1 - lambda_) * u1 + lambda_ * u2
        
        # Compute loss
        error = u_tr - stopgrad(target_u)
        loss = adaptive_l2_loss(error, gamma=self.gamma, alpha=alpha)
        mse_val = (error ** 2).mean()
        
        return loss, mse_val

    def loss_alpha_flow_alpha1(self, model, source, background, enrollment, alpha):
        """
        Special case of Alpha-Flow when alpha=1 (reduces to trajectory flow matching).
        
        Args:
            model: The neural network model
            source: Clean source (t=1)
            background: Background/noise (t=0)
            enrollment: Enrollment/reference signals
            alpha: Consistency step ratio (should be 1.0)
        """
        batch_size = source.shape[0]
        device = source.device

        # Sample t and r from logistic distribution
        mu, sigma = -0.4, 1.0
        normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
        samples = 1 / (1 + np.exp(-normal_samples))  # sigmoid
        
        # Ensure r >= t
        t_np = np.minimum(samples[:, 0], samples[:, 1])
        r_np = np.maximum(samples[:, 0], samples[:, 1])
        
        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        
        # When alpha=1, s=r (direct jump)
        s = r.clone()

        # Reshape time variables
        if self.data_dim == '1d':
            t_ = rearrange(t, "b -> b 1 1")
            r_ = rearrange(r, "b -> b 1 1")
        else:
            t_ = rearrange(t, "b -> b 1 1 1")
            r_ = rearrange(r, "b -> b 1 1 1")

        # Get point at time t
        x_t = (1 - t_) * background + t_ * source
        
        # Predict direct velocity from t to r
        u_tr = model(x_t, t, r, enrollment=enrollment)
        
        # Ground truth velocity
        v_true = (r_ - t_) * (source - background)
        
        # Compute loss
        error = u_tr - v_true
        loss = adaptive_l2_loss(error, gamma=self.gamma, alpha=alpha)
        mse_val = (error ** 2).mean()
        
        return loss, mse_val
    
    def loss(self, model, source, background, enrollment, iteration=None):
        """
        Combined loss function with alpha scheduling for curriculum learning.
        
        Training phases:
        1. Trajectory flow matching pretraining (alpha=1)
        2. Alpha-Flow transition (alpha ∈ (0,1))
        3. Consistency training fine-tuning (alpha→0)
        
        Convention:
        - background: at t=0 (starting point, noisy mixture)
        - source: at t=1 (target, clean)
        - r ≥ t always (r is the target timestep)
        
        Args:
            model: The neural network model
            source: Clean source (t=1) - unnormalized spectrograms (B, C, T)
            background: Background/noise (t=0) - unnormalized spectrograms (B, C, T)
            enrollment: Enrollment/reference signals (B, C, T_enroll)
            iteration: Current training iteration (optional, uses self.current_iteration if None)
        
        Returns:
            loss: Scalar loss value
            mse_val: MSE value for logging
            alpha: Current alpha value (for logging)
            raw_alpha: Raw alpha value before clipping (for logging)
        """
        # Update iteration counter
        if iteration is not None:
            self.current_iteration = iteration
        
        # Get current alpha value based on schedule
        alpha, raw_alpha = self.get_alpha(return_raw=True)
        
        # Select loss function based on alpha and flow_ratio
        if alpha >= (1.0 - self.alpha_min):
            # Phase 1: Trajectory flow matching (alpha = 1)
            if torch.rand(1).item() < self.flow_ratio:
                loss, mse_val = self.loss_rectified_flow(model, source, background, enrollment, alpha=1.0)
            else:
                loss, mse_val = self.loss_alpha_flow_alpha1(model, source, background, enrollment, alpha=1.0)
        elif alpha <= self.alpha_min:
            # Phase 3: Consistency training (alpha ≈ 0)
            if torch.rand(1).item() < self.flow_ratio:
                loss, mse_val = self.loss_rectified_flow(model, source, background, enrollment, alpha=self.alpha_min)
            else:
                loss, mse_val = self.loss_alpha_flow(model, source, background, enrollment, alpha=self.alpha_min)
        else:
            # Phase 2: Alpha-Flow transition (0 < alpha < 1)
            if torch.rand(1).item() < self.flow_ratio:
                loss, mse_val = self.loss_rectified_flow(model, source, background, enrollment, alpha=alpha)
            else:
                loss, mse_val = self.loss_alpha_flow(model, source, background, enrollment, alpha=alpha)
        
        return loss, mse_val, alpha, raw_alpha

