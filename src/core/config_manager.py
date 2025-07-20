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
    def __init__(self, ):
        # self.config_path = Path(config_path)
        pass
    
    
    def load(self,config_path,override_args=None) -> Dict[str, Any]:
        """Load and preprocess raw YAML"""
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        merge_config=merge_configs(config,override_args)
        self.config=merge_config
        return merge_config
    
    def validate(self, raw_config: Dict[str, Any]) -> ExperimentConfig:
        """Validate and normalize configuration"""
        try:
            merged_config = {
                **raw_config.get('defaults', {}),
                **raw_config  
            }
            print("merged config",merged_config)
            return ExperimentConfig(**merged_config)
        except Exception as e:
            raise ValueError(f"Invalid configuration: {str(e)}") from e

    def get_validated_config(self) -> ExperimentConfig:
        """Full pipeline: load -> validate"""
        return self.validate(self._load_raw_config())


def merge_configs(base_config: dict, override_args: dict) -> dict:
    """
    Merges dotted argument overrides into a nested config structure.
    
    Args:
        base_config: Your loaded base configuration (nested dict)
        override_args: Dictionary of arguments from argparse (dotted keys)
        
    Returns:
        Merged configuration dictionary
    """
    config = base_config.copy()
    
    for dotted_key, value in override_args.items():
        if value is None:  
            continue
            
        keys = dotted_key.split('.')
        current_level = config
        
        for key in keys[:-1]:
            if key not in current_level:
                current_level[key] = {}
            current_level = current_level[key]
        
        current_level[keys[-1]] = value
    
    return config