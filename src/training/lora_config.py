from typing import Dict, Any, Literal
from pydantic import BaseModel, Field, validator
import warnings

class LoRAConfig(BaseModel):
    """Validated configuration for LoRA (Low-Rank Adaptation)"""
    r: int = Field(
        default=32,
        ge=1,
        le=256,
        description="LoRA rank (intrinsic dimension)"
    )
    lora_alpha: int = Field(
        default=32,
        ge=1,
        le=1024,
        description="Scaling factor for LoRA weights"
    )
    lora_dropout: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Dropout probability for LoRA layers"
    )
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        description="List of module names to apply LoRA to"
    )
    bias: Literal["none", "all", "lora_only"] = Field(
        default="none",
        description="Bias training type"
    )
    use_rslora: bool = Field(
        default=False,
        description="Whether to use Rank-Stabilized LoRA"
    )

    @validator('target_modules')
    def validate_target_modules(cls, v):
        if not v:
            raise ValueError("At least one target module must be specified")
        return v

    @validator('lora_alpha')
    def validate_alpha_ratio(cls, v, values):
        if 'r' in values and v < values['r']:
            warnings.warn(
                f"lora_alpha ({v}) is smaller than r ({values['r']}). "
                "This may lead to unstable training.",
                UserWarning
            )
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Return the complete configuration as a dictionary"""
        return self.dict()

    def to_peft_config(self) -> Dict[str, Any]:
        """Convert to PEFT library compatible format"""
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "use_rslora": self.use_rslora
        }

class LoraConfigBuilder:
    def __init__(self, config: Dict[str, Any]):
        self.config = LoRAConfig(**config)
    def get_config_dict(self):
        return self.config.to_dict()
        
