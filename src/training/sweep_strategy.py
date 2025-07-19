from typing import Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field,field_validator,model_validator
import wandb
from enum import Enum

class SweepMethod(str, Enum):
    BAYES = "bayes"
    GRID = "grid"
    RANDOM = "random"

class MetricGoal(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"

class ParameterDistribution(str, Enum):
    UNIFORM = "uniform"
    LOG_UNIFORM = "log_uniform"
    CATEGORICAL = "categorical"

class SweepParameter(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None
    distribution: ParameterDistribution = ParameterDistribution.UNIFORM
    values: Optional[list] = None

class SweepConfig(BaseModel):
    method: SweepMethod = Field(
        default=SweepMethod.BAYES,
        description="Search strategy"
    )
    metric: Dict[str, Any] = Field(
        default={"name": "eval_loss", "goal": MetricGoal.MINIMIZE},
        description="Metric to optimize"
    )
    parameters: Dict[str, Any] = Field(
        ...,
        description="Parameters to sweep over"
    )
    early_terminate: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Early stopping configuration"
    )

    @field_validator('parameters')
    def validate_parameters(cls, v):
        required_params = {
            'learning_rate',
            'lora_r',
            'lora_alpha'
        }
        if not required_params.issubset(v.keys()):
            missing = required_params - set(v.keys())
            raise ValueError(f"Missing required sweep parameters: {missing}")
        return v

class SweepStrategy:
    def __init__(self, config: Dict[str, Any]):
        self.config = SweepConfig(**config)

    def get_sweep_mapping(self) -> Dict[str, Tuple[Optional[str], str]]:
        """Returns parameter to config path mapping"""
        return {
            'learning_rate': ('training', 'learning_rate'),
            'lora_r': ('lora', 'r'),
            'lora_alpha': ('lora', 'lora_alpha'),
            'gradient_accumulation_steps': ('training', 'gradient_accumulation_steps'),
            'batch_size': ('training', 'per_device_train_batch_size'),
            'weight_decay': ('training', 'weight_decay'),
            'warmup_steps': ('training', 'warmup_steps'),
            'max_seq_length': (None, 'max_seq_length'),
        }

    def apply_to_config(self, experiment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply sweep parameters to experiment config"""
        if not wandb.config:
            return experiment_config

        config = experiment_config.copy()
        mapping = self.get_sweep_mapping()

        for param_name, (section, key) in mapping.items():
            if hasattr(wandb.config, param_name):
                value = getattr(wandb.config, param_name)
                
                if section is None:
                    config[key] = value
                else:
                    config.setdefault(section, {})
                    config[section][key] = value

        return config

    def initialize_sweep(self, project: str) -> str:
        """Launch a new sweep and return its ID"""
        return wandb.sweep(
            self.config.dict(),
            project=project
        )