"""
Configuration system for F1 lap time prediction models.

Provides default configs for different model architectures and training setups.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    name: str = "seq2seq"
    input_size: int = 35  # From dataloaders (13 numeric + 2 tire deg + 4 cat + 12 bool + 4 compound)
    output_size: int = 1  # Predict single lap time
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    encoder: str = 'gru'
    decoder_type: str = 'gru'
    # Size of the strategy-known extra features fed to the decoder at each step
    # (TyreLife, fuel_proxy, stint_lap, 4x compound one-hot). 0 for phase-1
    # stint-based training (decoder sees only prev_laptime); 7 for phase-2
    # autoregressive. Set here so the value round-trips through config.json.
    decoder_extra_features_size: int = 0
    # Decoder output head: MLP (Linear->GELU->Linear) when True, single Linear
    # otherwise. True for new models; False reproduces legacy checkpoints.
    use_mlp_decoder_head: bool = True
    # MLP head shape when use_mlp_decoder_head is True. Defaults preserve the
    # legacy head (Linear(hidden, 64) -> GELU -> Linear(64, output), no dropout)
    # so existing checkpoints reload bit-for-bit. E13 sweeps these via CLI.
    decoder_head_hidden: int = 64
    decoder_head_layers: int = 2
    decoder_head_dropout: float = 0.0
    # Skip/residual from decoder input (prev_laptime + strategy extras) to the
    # head input. 'none' = recurrent path only (legacy); 'additive' adds a small
    # MLP residual; 'film' modulates the decoder output with per-step
    # gamma/beta. Both variants are zero-initialised so untrained runs match
    # 'none' bit-for-bit. Designed to widen the freeze-extras MAE gap (E14).
    decoder_skip_type: str = 'none'
    # Decoder hidden-state init (B3). When False: legacy projection of the last
    # encoder output. When True: concat(last_output, mean_pool_last_k) -> Linear.
    # encoder_pool_k is clamped to the actual encoder sequence length at runtime.
    # Default False so existing checkpoints reload bit-for-bit.
    encoder_pool: bool = False
    encoder_pool_k: int = 5
    embedding_dims: Optional[Dict[str, int]] = None
    vocab_sizes: Optional[Dict[str, int]] = None
    
    def __post_init__(self):
        if self.embedding_dims is None:
            self.embedding_dims = {
                'Driver': 32,
                'Team': 16,
                'Circuit': 16,
                'Year': 8,
            }

        # Attempt to infer vocab sizes from cleaned data vocabs (data/vocabs/*.json).
        # This prevents embedding index out-of-bounds when new categories (e.g. 2018) are added.
        if self.vocab_sizes is None:
            try:
                vocabs_dir = Path("data") / "vocabs"
                inferred = {}
                for key in ['Driver', 'Team', 'Circuit', 'Year']:
                    fp = vocabs_dir / f"{key}.json"
                    if fp.exists():
                        with open(fp, 'r', encoding='utf-8') as f:
                            mapping = json.load(f)
                        # mapping values are indices; vocab_size = max_index + 1
                        max_index = max(mapping.values()) if mapping else 0
                        inferred[key] = int(max_index) + 1

                # Fallback to sensible defaults for any missing entries
                defaults = {
                    'Driver': 76,
                    'Team': 18,
                    'Circuit': 35,
                    'Year': 8,
                }
                # Merge inferred with defaults
                self.vocab_sizes = {k: inferred.get(k, defaults[k]) for k in defaults}
            except Exception:
                # If anything fails, use hard-coded defaults
                self.vocab_sizes = {
                    'Driver': 76,
                    'Team': 18,
                    'Circuit': 35,
                    'Year': 8,
                }


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    weight_decay: float = 1e-5
    gradient_clip: Optional[float] = 1.0
    accumulation_steps: int = 1
    use_mixed_precision: bool = True  # Enable FP16 training on CUDA

    # Linear LR scaling with batch size
    lr_scale_with_batch: bool = True
    lr_scale_reference_batch: int = 32
    
    # Learning rate scheduling
    scheduler_type: str = "cosine"  # "cosine_with_warmup", "cosine", "linear"
    warm_up_epochs: int = 5
    
    # Teacher forcing schedule (E3 best-stability config: hold_then_decay, hold=20, end=0.3)
    teacher_forcing_start: float = 1.0  # Start with full teacher forcing
    teacher_forcing_end: float = 0.3    # End value after decay
    teacher_forcing_decay: str = "hold_then_decay"  # "linear", "exponential", "hold_then_decay", "constant"
    # Hold then decay settings: number of epochs to keep `teacher_forcing_start` before decaying
    teacher_forcing_hold_epochs: int = 20

    # Multi-step autoregressive training
    multistep_horizon: int = 1  # Max prediction horizon (1 = single-step, default)
    multistep_curriculum: str = 'none'  # 'none' (always max H), 'linear' (H grows over epochs)
    multistep_curriculum_start_epoch: int = 0  # Epoch to start increasing H
    multistep_curriculum_end_epoch: int = -1  # Epoch to reach max H (-1 = num_epochs)

    # Scheduled sampling — encoder context noise injection (exposure bias correction)
    scheduled_sampling_enabled: bool = False
    scheduled_sampling_start_epoch: int = 10      # Epoch to begin injecting noise
    scheduled_sampling_end_epoch: int = -1        # Epoch to reach max noise (-1 = num_epochs)
    scheduled_sampling_max_prob: float = 0.5      # Max probability of corrupting each context lap's LapTime
    scheduled_sampling_noise_std: float = 0.02    # Noise std in normalized space (~0.02 ~ 300ms)
    
    # Early stopping
    early_stopping_patience: int = 20
    early_stopping_min_epochs: int = 0  # Grace period: don't start patience counter before this epoch
    validation_freq: int = 1  # Validate every N epochs
    early_stopping_use_ema: bool = False
    early_stopping_ema_alpha: float = 0.3

    # Multi-task loss weights
    pit_loss_weight: float = 1e-3
    compound_loss_weight: float = 0.01
    # Dynamic auxiliary loss balancing (automatic scaling of pit/compound losses)
    dynamic_aux_balance: bool = True
    dynamic_aux_ema_alpha: float = 0.05
    dynamic_aux_min_scale: float = 0.001
    dynamic_aux_max_scale: float = 20.0
    
    # Data
    train_years: list = None
    val_years: list = None
    test_years: list = None
    
    def __post_init__(self):
        if self.train_years is None:
            self.train_years = [2018, 2019, 2020, 2021, 2022, 2023]
        if self.val_years is None:
            self.val_years = [2024]
        if self.test_years is None:
            self.test_years = [2025]

        # Validate no overlap between train/val/test splits
        train_set = set(self.train_years)
        val_set = set(self.val_years)
        test_set = set(self.test_years)
        if train_set & val_set:
            raise ValueError(f"Train and val years overlap: {train_set & val_set}")
        if train_set & test_set:
            raise ValueError(f"Train and test years overlap: {train_set & test_set}")
        if val_set & test_set:
            raise ValueError(f"Val and test years overlap: {val_set & test_set}")


@dataclass
class DataConfig:
    """Data loading configuration."""
    window_size: int = 10  # For stint dataloader
    context_window: int = 5  # For autoregressive dataloader
    augment_prob: float = 0.3
    normalize: bool = True
    scaler_type: str = "standard"
    num_workers: int = -1  # -1 = auto-detect
    shuffle_train: bool = True
    shuffle_val: bool = False
    shuffle_test: bool = False

    def __post_init__(self):
        if self.num_workers == -1:
            import os
            import sys
            if sys.platform == 'win32':
                # Windows uses spawn (not fork): each worker pickles the entire
                # dataset including precomputed tensors, which is very slow for
                # large in-memory caches.  Keep 0 (main-thread) by default.
                self.num_workers = 0
            else:
                cpu_count = os.cpu_count() or 1
                self.num_workers = min(cpu_count, 8)


@dataclass
class Config:
    """Master configuration combining all components."""
    model: ModelConfig = None
    training: TrainingConfig = None
    data: DataConfig = None
    device: str = "cuda"
    seed: int = 42
    output_dir: str = "./results"
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'device': self.device,
            'seed': self.seed,
            'output_dir': self.output_dir,
        }
    
    def save(self, path: Path):
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Config saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'Config':
        """Load config from JSON file."""
        path = Path(path)
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        config = cls(
            model=ModelConfig(**data.get('model', {})),
            training=TrainingConfig(**data.get('training', {})),
            data=DataConfig(**data.get('data', {})),
            device=data.get('device', 'cuda'),
            seed=data.get('seed', 42),
            output_dir=data.get('output_dir', './results'),
        )
        
        logger.info(f"Config loaded from {path}")
        return config
    
    def __str__(self) -> str:
        """String representation."""
        lines = ["Configuration:"]
        
        lines.append("\nModel:")
        for key, val in asdict(self.model).items():
            lines.append(f"  {key}: {val}")
        
        lines.append("\nTraining:")
        for key, val in asdict(self.training).items():
            if key not in ['train_years', 'val_years', 'test_years']:
                lines.append(f"  {key}: {val}")
            else:
                lines.append(f"  {key}: {val}")
        
        lines.append("\nData:")
        for key, val in asdict(self.data).items():
            lines.append(f"  {key}: {val}")
        
        lines.append(f"\nDevice: {self.device}")
        lines.append(f"Seed: {self.seed}")
        lines.append(f"Output: {self.output_dir}")
        
        return "\n".join(lines)


# Preset configurations
def get_phase1_config() -> Config:
    """
    Phase 1: Pure teacher forcing, stint-based sequences.
    
    Returns
    -------
    Config
        Configuration for Phase 1 training
    """
    model_config = ModelConfig(
        name="seq2seq",
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
    )
    
    training_config = TrainingConfig(
        batch_size=32,
        learning_rate=1e-3,
        num_epochs=100,
        teacher_forcing_start=1.0,
        teacher_forcing_end=1.0,  # Keep at 100% since it's pure teacher forcing
        train_years=[2018, 2019, 2020, 2021, 2022, 2023],
        val_years=[2024],
        test_years=[2025],
    )
    
    data_config = DataConfig(
        window_size=20,
        augment_prob=0.3,
    )
    
    return Config(
        model=model_config,
        training=training_config,
        data=data_config,
        device="cuda",
        output_dir="./results/phase1",
    )


def get_phase2_config() -> Config:
    """
    Phase 2: Full-race sequences with auxiliary heads.
    
    Returns
    -------
    Config
        Configuration for Phase 2 training
    """
    model_config = ModelConfig(
        name="seq2seq",
        hidden_size=256,
        num_layers=3,
        dropout=0.3,
        decoder_extra_features_size=11,
    )
    
    training_config = TrainingConfig(
        batch_size=32,
        learning_rate=5e-4,
        num_epochs=100,
        weight_decay=5e-5,
        teacher_forcing_start=1.0,
        teacher_forcing_end=0.3,
        teacher_forcing_decay="linear",
        early_stopping_patience=30,
        early_stopping_min_epochs=30,
        early_stopping_use_ema=True,
        early_stopping_ema_alpha=0.25,
        pit_loss_weight=1e-3,
        compound_loss_weight=0.01,
        train_years=[2018, 2019, 2020, 2021, 2022, 2023],
        val_years=[2024],
        test_years=[2025],
    )
    
    data_config = DataConfig(
        window_size=50,
        context_window=10,
        augment_prob=0.20,
    )
    
    return Config(
        model=model_config,
        training=training_config,
        data=data_config,
        device="cuda",
        output_dir="./results/phase2",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    config = get_phase1_config()
    print(config)
    
    # Save config
    config.save(Path("./config_phase1.json"))
    
    # Load config
    loaded_config = Config.load(Path("./config_phase1.json"))
    print("\nLoaded config:")
    print(loaded_config)
