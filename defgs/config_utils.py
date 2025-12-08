from dataclasses import dataclass

@dataclass
class TrainingConfig:
    iterations: int = 1000
    percent_dense: float = 0.01
    
    # Learning rate parameters
    position_lr_init: float = 1e-2
    position_lr_final: float = 1e-4
    position_lr_delay_mult: float = 0.1
    position_lr_max_steps: int = 1000
    deform_lr_max_steps: int = 1000
    feature_lr: float = 1e-2
    opacity_lr: float = 1e-2
    scaling_lr: float = 1e-2
    rotation_lr: float = 1e-2
    
    # Loss parameters
    lambda_dssim: float = 0.2
    
    # Densification parameters
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_grad_threshold: float = 0.0007