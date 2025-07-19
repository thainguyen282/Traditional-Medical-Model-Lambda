from typing import Dict, Any, Optional, List
from pydantic import BaseModel, validator, Field, root_validator,field_validator,model_validator
import yaml
from pathlib import Path
import warnings

class DatasetConfig(BaseModel):
    """Configuration for a single dataset"""
    tag: str = Field(..., min_length=1)
    path: str
    train_samples: int = Field(gt=0)
    weight: float = Field(gt=0, le=1)
    
    @field_validator('tag')
    def validate_tag(cls, v):
        if not v.isidentifier():
            raise ValueError(f"Dataset tag '{v}' must be a valid Python identifier")
        return v

class TrainingConfig(BaseModel):
    """Training hyperparameters"""
    learning_rate: float = Field(gt=0, le=1)
    per_device_train_batch_size: int = Field(ge=1, le=64)
    gradient_accumulation_steps: int = Field(ge=1, le=128)
    num_train_epochs: int = Field(ge=1, le=100)
    weight_decay: float = Field(ge=0, le=1)
    warmup_steps: int = Field(ge=0)

class LoRAConfig(BaseModel):
    """LoRA-specific parameters"""
    r: int = Field(ge=1, le=256)
    lora_alpha: int = Field(ge=1, le=512)
    lora_dropout: float = Field(ge=0, le=0.5)

class ExperimentConfig(BaseModel):
    """Complete experiment configuration"""
    name: str = Field(..., min_length=3, max_length=50)
    model: str
    max_seq_length: int = Field(ge=128, le=8192)
    output_dir: str = "./outputs"
    dataset: List[DatasetConfig]
    training: TrainingConfig
    lora: LoRAConfig
    monitoring: Dict[str, Any] = {}
    
    @field_validator('name')
    def validate_name(cls, v):
        if not v.replace('-', '').isalnum():
            raise ValueError("Experiment name must be alphanumeric with optional hyphens")
        return v
    


class ConfigManager:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
    
    
    def load(self) -> Dict[str, Any]:
        """Load and preprocess raw YAML"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        config.setdefault('defaults', {})
        config.setdefault('experiments', [])
        return config
    
    def validate(self, raw_config: Dict[str, Any]) -> ExperimentConfig:
        """Validate and normalize configuration"""
        try:
            merged_config = {
                **raw_config.get('defaults', {}),
                **raw_config  # Experiment-specific overrides
            }
            print("merged config",merged_config)
            return ExperimentConfig(**merged_config)
        except Exception as e:
            raise ValueError(f"Invalid configuration: {str(e)}") from e

    def get_validated_config(self) -> ExperimentConfig:
        """Full pipeline: load -> validate"""
        return self.validate(self._load_raw_config())