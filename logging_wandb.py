from transformers import TrainerCallback
import wandb
import torch
import psutil
import time
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

class CustomWandbCallback(TrainerCallback):
    """
    Enhanced W&B callback combining training metrics, system monitoring, and model analytics
    """
    
    def __init__(self, log_system_metrics: bool = True, log_model_metrics: bool = True):
        super().__init__()
        self.start_time = time.time()
        self.total_samples = 0
        self.step_count = 0
        self.log_system_metrics = log_system_metrics
        self.log_model_metrics = log_model_metrics
        
        # Track peak metrics
        self.peak_gpu_memory = 0
        self.peak_cpu_percent = 0
        
        # Performance tracking
        self.step_times = []
        self.last_step_time = time.time()
        
        # GPU availability check
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            torch.cuda.reset_peak_memory_stats()
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize training metrics"""
        self.start_time = time.time()
        self.total_samples = 0
        
        initial_metrics = {
            
        }
        
        if self.log_system_metrics:
            initial_metrics.update(self._get_system_metrics())
        
        wandb.log(initial_metrics, step=0)
    
    def on_step_end(self, args, state, control, **kwargs):
        """Log per-step metrics including samples processed and performance"""
        current_time = time.time()
        step_time = current_time - self.last_step_time
        self.step_times.append(step_time)
        self.last_step_time = current_time
        
        # Calculate samples processed
        step_samples = args.per_device_train_batch_size * args.gradient_accumulation_steps
        self.total_samples += step_samples
        
        # Calculate throughput
        samples_per_second = step_samples / step_time if step_time > 0 else 0
        avg_step_time = sum(self.step_times[-10:]) / min(len(self.step_times), 10)  # Last 10 steps
        
        step_metrics = {
            "samples_processed": self.total_samples,
            "samples_per_second": samples_per_second,
            "avg_step_time_sec": avg_step_time,
            "current_step_time_sec": step_time,
        }
        
        # GPU memory tracking
        if self.cuda_available:
            current_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            
            self.peak_gpu_memory = max(self.peak_gpu_memory, peak_memory)
            
            step_metrics.update({
                "gpu_memory_current_mb": current_memory,
                "gpu_memory_peak_mb": peak_memory,
                "gpu_memory_peak_session_mb": self.peak_gpu_memory,
                "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
            })
        
        wandb.log(step_metrics, step=state.global_step)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Enhanced logging with system metrics and model analytics"""
        if logs is None:
            logs = {}
        
        # Add timestamp
        logs["timestamp"] = time.time()
        logs["training_time_elapsed"] = time.time() - self.start_time
        
        # System metrics
        if self.log_system_metrics:
            system_metrics = self._get_system_metrics()
            logs.update(system_metrics)
        
        # Model metrics
        if self.log_model_metrics:
            model_metrics = self._get_model_metrics(kwargs.get('model'))
            logs.update(model_metrics)
        
        # Training efficiency metrics
        if self.total_samples > 0:
            elapsed_time = time.time() - self.start_time
            logs["overall_samples_per_second"] = self.total_samples / elapsed_time
            logs["estimated_time_remaining"] = self._estimate_time_remaining(state, args)
        
        # Log to W&B
        wandb.log(logs, step=state.global_step)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Log epoch-level metrics"""
        epoch_metrics = {
            "epoch_completed": state.epoch,
            "samples_processed_epoch": self.total_samples,
            "training_time_epoch": time.time() - self.start_time,
        }
        
        if self.cuda_available:
            epoch_metrics["gpu_memory_peak_epoch_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
            # Reset peak memory stats for next epoch
            torch.cuda.reset_peak_memory_stats()
        
        wandb.log(epoch_metrics, step=state.global_step)
    
    def on_train_end(self, args, state, control, **kwargs):
        """Log final training metrics and summary"""
        total_training_time = time.time() - self.start_time
        
        final_metrics = {
            "total_training_time_sec": total_training_time,
            "total_training_time_hours": total_training_time / 3600,
            "total_samples_processed": self.total_samples,
            "average_samples_per_second": self.total_samples / total_training_time,
            "average_training_time_per_sample": total_training_time / self.total_samples,
            "total_steps": state.global_step,
            "average_step_time": sum(self.step_times) / len(self.step_times) if self.step_times else 0,
        }
        
        if self.cuda_available:
            final_metrics.update({
                "peak_gpu_memory_mb": self.peak_gpu_memory,
                "final_gpu_memory_mb": torch.cuda.memory_allocated() / (1024 ** 2),
            })
        
        wandb.log(final_metrics, step=state.global_step)
        
        # Log summary table
        self._log_training_summary(final_metrics)
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Log evaluation start"""
        eval_metrics = {
            "evaluation_started": True,
            "eval_step": state.global_step,
        }
        wandb.log(eval_metrics, step=state.global_step)
    
    def _get_system_metrics(self) -> Dict[str, float]:
        """Get comprehensive system resource metrics"""
        metrics = {}
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            metrics['cpu_percent'] = cpu_percent
            self.peak_cpu_percent = max(self.peak_cpu_percent, cpu_percent)
            metrics['cpu_peak_percent'] = self.peak_cpu_percent
            
            # Memory usage
            memory = psutil.virtual_memory()
            metrics['memory_percent'] = memory.percent
            metrics['memory_available_gb'] = memory.available / (1024**3)
            metrics['memory_used_gb'] = memory.used / (1024**3)
            metrics['memory_total_gb'] = memory.total / (1024**3)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            metrics['disk_percent'] = (disk.used / disk.total) * 100
            metrics['disk_available_gb'] = disk.free / (1024**3)
            
            # GPU metrics (if available)
            if self.cuda_available:
                metrics['gpu_count'] = torch.cuda.device_count()
                
                # Try to get detailed GPU info
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Primary GPU
                        metrics['gpu_utilization'] = gpu.load * 100
                        metrics['gpu_memory_utilization'] = gpu.memoryUtil * 100
                        metrics['gpu_temperature'] = gpu.temperature
                        metrics['gpu_power_draw'] = getattr(gpu, 'powerDraw', 0)
                except ImportError:
                    logger.warning("GPUtil not available for detailed GPU metrics")
                
        except Exception as e:
            logger.warning(f"Error collecting system metrics: {e}")
        
        return metrics
    
    def _get_model_metrics(self, model) -> Dict[str, Any]:
        """Get comprehensive model-specific metrics"""
        metrics = {}
        
        if model is not None:
            try:
                # Parameter counts
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                metrics['total_parameters'] = total_params
                metrics['trainable_parameters'] = trainable_params
                metrics['trainable_percentage'] = (trainable_params / total_params) * 100 if total_params > 0 else 0
                metrics['frozen_parameters'] = total_params - trainable_params
                
                # Gradient norms
                total_norm = 0
                param_count = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        param_count += 1
                
                if param_count > 0:
                    metrics['gradient_norm'] = total_norm ** (1. / 2)
                    metrics['gradient_norm_avg'] = (total_norm ** (1. / 2)) / param_count
                
                # Model size estimation
                param_size = total_params * 4  # Assuming float32
                metrics['model_size_mb'] = param_size / (1024**2)
                
                # LoRA specific metrics (if applicable)
                if hasattr(model, 'peft_config'):
                    lora_params = sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n)
                    metrics['lora_parameters'] = lora_params
                    metrics['lora_percentage'] = (lora_params / total_params) * 100 if total_params > 0 else 0
                
            except Exception as e:
                logger.warning(f"Error collecting model metrics: {e}")
        
        return metrics
    
    def _estimate_time_remaining(self, state, args) -> float:
        """Estimate remaining training time"""
        if len(self.step_times) == 0:
            return 0
        
        avg_step_time = sum(self.step_times[-10:]) / min(len(self.step_times), 10)
        remaining_steps = (args.num_train_epochs * state.max_steps) - state.global_step
        return remaining_steps * avg_step_time
    
    def _log_training_summary(self, metrics: Dict[str, Any]):
        """Log a comprehensive training summary table"""
        summary_data = [
            ["Metric", "Value"],
            ["Total Training Time", f"{metrics['total_training_time_hours']:.2f} hours"],
            ["Total Samples Processed", f"{metrics['total_samples_processed']:,}"],
            ["Average Samples/Second", f"{metrics['average_samples_per_second']:.2f}"],
            ["Total Steps", f"{metrics['total_steps']:,}"],
            ["Average Step Time", f"{metrics['average_step_time']:.4f} seconds"],
        ]
        
        if self.cuda_available:
            summary_data.extend([
                ["Peak GPU Memory", f"{metrics['peak_gpu_memory_mb']:.2f} MB"],
                ["Final GPU Memory", f"{metrics['final_gpu_memory_mb']:.2f} MB"],
            ])
        
        # Log as W&B table
        wandb.log({"training_summary": wandb.Table(data=summary_data, columns=["Metric", "Value"])})


# Example usage with enhanced configuration
def create_enhanced_callback(
    log_system_metrics: bool = True,
    log_model_metrics: bool = True,
) -> CustomWandbCallback:
    """Factory function to create configured callback"""
    return CustomWandbCallback(
        log_system_metrics=log_system_metrics,
        log_model_metrics=log_model_metrics,
    )