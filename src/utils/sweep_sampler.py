import numpy as np
from scipy.stats import loguniform

class ParameterSampler:
    
    @staticmethod
    def sample(param_config: dict):
        param_type = param_config.get("distribution", "categorical")
        
        if param_type == "log_uniform":
            return loguniform.rvs(
                param_config["min"],
                param_config["max"]
            )
        elif param_type == "categorical":
            values = param_config["values"]
            probs = param_config.get("probabilities", None)
            return np.random.choice(values, p=probs)
        else:
            raise ValueError(f"Unknown distribution: {param_type}")

    @staticmethod
    def validate(param_config: dict):
        if "distribution" in param_config:
            if param_config["distribution"] == "log_uniform":
                assert "min" in param_config, "Missing min bound"
                assert "max" in param_config, "Missing max bound"
                assert 0 < param_config["min"] < param_config["max"], "Invalid range"