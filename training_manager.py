"""
Multi-Agent Group Relative Policy Optimization (MAGRPO) Training Manager

This module implements the training component of the dual-LLM CORY+MAGRPO system.
It provides independent, continuous training for both Traffic LLM and Regional LLM
using group-relative policy optimization on dedicated GPUs.

Key Features:
- Independent training process with multiprocessing communication
- Dual replay buffers for Traffic LLM (group size: 8) and Regional LLM (group size: 12)
- PEFT/LoRA integration for parameter-efficient fine-tuning
- Group Relative Policy Optimization (GRPO) with relative reward calculation
- Hot-reload mechanism for model weight synchronization
- Comprehensive logging for training monitoring
"""

import os
import sys
import time
import signal
import multiprocessing as mp
import threading
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field
import json
import torch
import numpy as np
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
import logging
from datetime import datetime
import requests
import shutil

# Enable vLLM runtime LoRA updating for hot-reload functionality
os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"


@dataclass
class TrainingConfig:
    """Configuration for MAGRPO training."""
    
    # Model Configuration
    model_path: str = "/data/zhouyuping/Qwen/"
    traffic_gpu: str = "cuda:2"  # GPU for Traffic LLM training
    regional_gpu: str = "cuda:3"  # GPU for Regional LLM training
    
    # GRPO Configuration - Reduced for memory optimization
    traffic_group_size: int = 4  # Group size for Traffic LLM (reduced from 8)
    regional_group_size: int = 6  # Group size for Regional LLM (reduced from 12)
    
    # LoRA Configuration - Optimized for memory
    lora_r: int = 8  # Reduced from 16 for memory optimization
    lora_alpha: int = 16  # Reduced proportionally
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    
    # Training Configuration
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 8  # Increased to reduce memory per step
    
    # Checkpoint Configuration
    save_steps: int = 100  # Save adapters every N training steps
    max_checkpoints: int = 5  # Keep only the last N checkpoints
    
    # Logging Configuration
    log_steps: int = 10
    log_dir: str = "logs/training"
    
    # Hot-reload Configuration
    vllm_inference_urls: List[str] = field(default_factory=lambda: ["http://localhost:8000"])  # vLLM inference server URLs
    adapter_sync_dir: str = "lora_adapters"  # Directory to store synced adapters
    enable_hot_reload: bool = True  # Enable hot-reload mechanism


class ReplayBuffer:
    """Replay buffer for storing and grouping training samples."""
    
    def __init__(self, name: str, group_size: int, max_size: int = 10000):
        self.name = name
        self.group_size = group_size
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        
        # Statistics
        self.total_samples_received = 0
        self.total_groups_processed = 0
        
    def add_sample(self, sample: Dict[str, Any]):
        """Add a training sample to the buffer."""
        with self.lock:
            self.buffer.append(sample)
            self.total_samples_received += 1
            
    def can_form_group(self) -> bool:
        """Check if buffer has enough samples to form a training group."""
        with self.lock:
            return len(self.buffer) >= self.group_size
            
    def get_training_group(self) -> Optional[List[Dict[str, Any]]]:
        """Extract a training group from the buffer."""
        with self.lock:
            if len(self.buffer) < self.group_size:
                return None
            
            # Extract samples for group formation
            group = []
            for _ in range(self.group_size):
                group.append(self.buffer.popleft())
            
            self.total_groups_processed += 1
            return group
    
    def size(self) -> int:
        """Get current buffer size."""
        with self.lock:
            return len(self.buffer)
    
    def get_stats(self) -> Dict[str, int]:
        """Get buffer statistics."""
        with self.lock:
            return {
                'name': self.name,
                'current_size': len(self.buffer),
                'total_samples_received': self.total_samples_received,
                'total_groups_processed': self.total_groups_processed
            }


class MAGRPOTrainer:
    """MAGRPO trainer for a single LLM (Traffic or Regional)."""
    
    def __init__(self, name: str, config: TrainingConfig, gpu_device: str, task_type: str):
        self.name = name
        self.config = config
        self.device = torch.device(gpu_device)
        self.task_type = task_type  # "traffic" or "regional"
        
        # Training statistics
        self.training_steps = 0
        self.total_loss = 0.0
        self.last_save_step = 0
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Initialize model and training components
        self._initialize_model()
        self._initialize_optimizer()
        
        self.logger.info(f"MAGRPO_TRAINER_INIT: {self.name} trainer initialized on {gpu_device}")
        
    def _setup_logger(self) -> logging.Logger:
        """Setup dedicated logger for this trainer."""
        logger = logging.getLogger(f"magrpo_trainer_{self.name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Create handler
            os.makedirs(self.config.log_dir, exist_ok=True)
            log_file = os.path.join(self.config.log_dir, f"{self.name}_training.log")
            handler = logging.FileHandler(log_file)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # Also log to console
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
        
    def _initialize_model(self):
        """Initialize model with PEFT/LoRA configuration."""
        try:
            self.logger.info(f"MAGRPO_MODEL_INIT: Loading base model from {self.config.model_path}")
            
            # Load base model with memory optimizations
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.float16,
                device_map=None,  # We'll move to device manually
                trust_remote_code=True,
                low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
                use_cache=False  # Disable KV cache for training to save memory
            )
            
            # Disable cache for training to save memory
            self.base_model.config.use_cache = False
            
            # Configure LoRA
            task_type = TaskType.CAUSAL_LM
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.lora_target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=task_type
            )
            
            # Apply PEFT with memory optimization
            self.model = get_peft_model(self.base_model, lora_config, low_cpu_mem_usage=True)
            
            # Use to_empty() instead of to() for meta tensors created by low_cpu_mem_usage=True
            # This moves the model to the target device while leaving parameters uninitialized
            self.model.to_empty(device=self.device)
            
            # Reinitialize LoRA adapter parameters after moving to device
            for name, param in self.model.named_parameters():
                if param.requires_grad and 'lora_' in name:
                    # Reinitialize LoRA parameters with proper initialization
                    if 'lora_A' in name:
                        # LoRA A matrix - use kaiming uniform initialization
                        torch.nn.init.kaiming_uniform_(param, a=5**0.5)
                    elif 'lora_B' in name:
                        # LoRA B matrix - zero initialization
                        torch.nn.init.zeros_(param)
                    else:
                        # Other LoRA parameters - normal initialization
                        torch.nn.init.normal_(param, std=0.02)
            
            # Enable gradient checkpointing for memory efficiency
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            
            # Print trainable parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            
            self.logger.info(f"MAGRPO_PEFT: {self.name} -> Trainable: {trainable_params:,} / Total: {total_params:,} "
                           f"({trainable_params/total_params*100:.2f}%)")
            
        except Exception as e:
            self.logger.error(f"MAGRPO_MODEL_ERROR: Failed to initialize model: {e}")
            raise
            
    def _initialize_optimizer(self):
        """Initialize optimizer and scheduler."""
        try:
            # Only optimize trainable parameters
            optimizer_grouped_parameters = [
                {"params": [p for p in self.model.parameters() if p.requires_grad]}
            ]
            
            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
            # Note: We'll create scheduler when we know total training steps
            self.scheduler = None
            
            # Initialize gradient scaler for automatic mixed precision
            self.scaler = torch.cuda.amp.GradScaler()
            
            self.logger.info(f"MAGRPO_OPTIMIZER: {self.name} optimizer initialized (lr={self.config.learning_rate}) with AMP scaler")
            
        except Exception as e:
            self.logger.error(f"MAGRPO_OPTIMIZER_ERROR: {e}")
            raise
    
    def train_step(self, training_group: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Execute one MAGRPO training step on a group of samples.
        
        Args:
            training_group: List of training samples forming a group
            
        Returns:
            Training metrics dictionary
        """
        try:
            self.model.train()
            
            # Calculate relative rewards within the group
            group_rewards = self._calculate_relative_rewards(training_group)
            
            # Prepare training inputs
            batch_inputs = self._prepare_training_inputs(training_group, group_rewards)
            
            if not batch_inputs:
                self.logger.warning(f"MAGRPO_TRAIN_STEP: No valid inputs for {self.name}")
                return {'loss': 0.0, 'relative_reward_mean': 0.0, 'relative_reward_std': 0.0}
            
            # Forward pass with gradient accumulation
            total_loss = 0.0
            accumulation_steps = self.config.gradient_accumulation_steps
            
            for i, (input_ids, attention_mask, labels, weight) in enumerate(batch_inputs):
                with torch.cuda.amp.autocast():  # Use automatic mixed precision
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss * weight  # Weight by relative reward
                    # Scale loss for gradient accumulation
                    loss = loss / accumulation_steps
                
                total_loss += loss.detach()
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Clear intermediate variables to save memory
                del outputs, loss
                
                if i % accumulation_steps == 0 or i == len(batch_inputs) - 1:
                    # Unscale gradients for gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    
                    # Optimizer step with scaler
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if self.scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # Clear GPU cache
                    torch.cuda.empty_cache()
            
            # Average loss across batch (already scaled)
            avg_loss = total_loss * accumulation_steps
            
            # Update statistics
            self.training_steps += 1
            step_loss = avg_loss.item()
            self.total_loss += step_loss
            
            # Calculate reward statistics
            reward_values = [r for _, _, _, r in batch_inputs]
            reward_mean = np.mean(reward_values) if reward_values else 0.0
            reward_std = np.std(reward_values) if reward_values else 0.0
            
            # Log training step
            if self.training_steps % self.config.log_steps == 0:
                avg_loss_since_start = self.total_loss / self.training_steps
                lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate
                
                self.logger.info(f"MAGRPO_STEP: {self.name} Step {self.training_steps} -> "
                               f"Loss: {step_loss:.4f}, Avg: {avg_loss_since_start:.4f}, "
                               f"LR: {lr:.2e}, Reward μ: {reward_mean:.3f}, σ: {reward_std:.3f}")
            
            # Save checkpoint periodically
            if self.training_steps % self.config.save_steps == 0:
                self._save_checkpoint()
            
            return {
                'loss': step_loss,
                'relative_reward_mean': reward_mean,
                'relative_reward_std': reward_std,
                'learning_rate': self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate
            }
            
        except Exception as e:
            self.logger.error(f"MAGRPO_TRAIN_ERROR: {self.name} training step failed: {e}")
            return {'loss': 0.0, 'relative_reward_mean': 0.0, 'relative_reward_std': 0.0}
    
    def _calculate_relative_rewards(self, training_group: List[Dict[str, Any]]) -> List[float]:
        """Calculate relative rewards within a group (core of GRPO)."""
        try:
            # Extract rewards for this LLM type
            if self.task_type == "traffic":
                rewards = [sample['rewards']['traffic_llm']['total_reward'] for sample in training_group]
            else:  # regional
                rewards = [sample['rewards']['regional_llm']['total_reward'] for sample in training_group]
            
            # Calculate baseline (group mean)
            baseline_reward = np.mean(rewards)
            
            # Calculate relative rewards
            relative_rewards = [reward - baseline_reward for reward in rewards]
            
            self.logger.debug(f"MAGRPO_RELATIVE_REWARDS: {self.name} -> "
                            f"Baseline: {baseline_reward:.3f}, "
                            f"Relative range: [{min(relative_rewards):.3f}, {max(relative_rewards):.3f}]")
            
            return relative_rewards
            
        except Exception as e:
            self.logger.error(f"MAGRPO_RELATIVE_REWARD_ERROR: {e}")
            return [0.0] * len(training_group)
    
    def _prepare_training_inputs(self, training_group: List[Dict[str, Any]], 
                               relative_rewards: List[float]) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]]:
        """
        Prepare training inputs from group samples.
        
        Returns:
            List of (input_ids, attention_mask, labels, weight) tuples
        """
        try:
            batch_inputs = []
            
            for sample, rel_reward in zip(training_group, relative_rewards):
                # Create training text based on LLM type
                if self.task_type == "traffic":
                    training_text = self._create_traffic_training_text(sample)
                else:
                    training_text = self._create_regional_training_text(sample)
                
                if not training_text:
                    continue
                
                # Tokenize
                encoding = self.tokenizer(
                    training_text,
                    truncation=True,
                    max_length=512,  # Reasonable length for training
                    padding="max_length",
                    return_tensors="pt"
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # For causal language modeling, labels = input_ids
                labels = input_ids.clone()
                
                # Convert relative reward to weight (ensure positive)
                weight = max(0.1, 1.0 + rel_reward)  # Minimum weight of 0.1
                
                batch_inputs.append((input_ids, attention_mask, labels, weight))
            
            return batch_inputs
            
        except Exception as e:
            self.logger.error(f"MAGRPO_INPUT_PREP_ERROR: {e}")
            return []
    
    def _create_traffic_training_text(self, sample: Dict[str, Any]) -> str:
        """Create training text for Traffic LLM (Pioneer role)."""
        try:
            state_context = sample.get('state_context', {})
            pioneer_decision = sample.get('pioneer_decision', {})
            macro_route = sample.get('macro_route', [])
            
            # Create structured training example
            training_text = f"TRAFFIC_LLM_TRAINING:\n"
            training_text += f"MACRO_ROUTE_DECISION:\n"
            training_text += f"Start Region: {sample.get('vehicle_region_start', 'Unknown')}\n"
            training_text += f"Destination Region: {sample.get('vehicle_region_dest', 'Unknown')}\n"
            
            # Add state context
            if state_context:
                regional_congestion = state_context.get('regional_congestion', {})
                training_text += f"Regional Congestion: {regional_congestion}\n"
            
            # Add decision
            training_text += f"Selected Route: {macro_route}\n"
            training_text += f"Decision Reasoning: {pioneer_decision.get('reasoning', 'N/A')}\n"
            training_text += f"Travel Time: {sample.get('travel_time', 0):.1f}s\n"
            
            return training_text
            
        except Exception as e:
            self.logger.error(f"TRAFFIC_TEXT_ERROR: {e}")
            return ""
    
    def _create_regional_training_text(self, sample: Dict[str, Any]) -> str:
        """Create training text for Regional LLM (Observer role)."""
        try:
            observer_feedback = sample.get('observer_feedback', {})
            regional_route = sample.get('regional_route', [])
            target_region = sample.get('target_region', -1)
            
            # Create structured training example
            training_text = f"REGIONAL_LLM_TRAINING:\n"
            training_text += f"REGIONAL_ROUTE_DECISION:\n"
            training_text += f"Target Region: {target_region}\n"
            training_text += f"Regional Route: {regional_route}\n"
            
            # Add observer feedback
            if observer_feedback:
                training_text += f"Feasibility Score: {observer_feedback.get('feasibility_score', 0.5):.3f}\n"
                training_text += f"Efficiency Score: {observer_feedback.get('efficiency_score', 0.5):.3f}\n"
                training_text += f"Fairness Score: {observer_feedback.get('fairness_score', 0.5):.3f}\n"
                training_text += f"Observer Reasoning: {observer_feedback.get('observer_reasoning', 'N/A')}\n"
            
            training_text += f"Travel Time: {sample.get('travel_time', 0):.1f}s\n"
            
            return training_text
            
        except Exception as e:
            self.logger.error(f"REGIONAL_TEXT_ERROR: {e}")
            return ""
    
    def _save_checkpoint(self):
        """Save LoRA adapter checkpoint and trigger hot-reload."""
        try:
            checkpoint_dir = os.path.join(self.config.log_dir, f"{self.name}_checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            step_dir = os.path.join(checkpoint_dir, f"step_{self.training_steps}")
            self.model.save_pretrained(step_dir)
            
            # Verify step_dir was created successfully
            if os.path.exists(step_dir):
                # Create a symlink to the latest checkpoint
                latest_dir = os.path.join(checkpoint_dir, "latest")
                if os.path.exists(latest_dir):
                    if os.path.islink(latest_dir):
                        os.unlink(latest_dir)
                    else:
                        import shutil
                        shutil.rmtree(latest_dir)
                
                # Create relative symlink for better portability
                relative_step_dir = f"step_{self.training_steps}"
                os.symlink(relative_step_dir, latest_dir)
                
                self.logger.info(f"MAGRPO_CHECKPOINT: {self.name} saved at step {self.training_steps}")
                
                # Trigger hot-reload mechanism
                if self.config.enable_hot_reload:
                    self._trigger_hot_reload(step_dir)  # Use step_dir directly
                
                # Clean up old checkpoints
                self._cleanup_old_checkpoints(checkpoint_dir)
            else:
                self.logger.error(f"MAGRPO_CHECKPOINT_ERROR: Failed to create checkpoint directory {step_dir}")
            
        except Exception as e:
            self.logger.error(f"MAGRPO_CHECKPOINT_ERROR: {e}")
    
    def _trigger_hot_reload(self, adapter_path: str):
        """Trigger hot-reload of LoRA adapter to vLLM inference servers."""
        try:
            adapter_name = f"{self.name.lower()}_adapter_step_{self.training_steps}"
            
            # Prepare adapter for sync
            sync_adapter_path = self._prepare_adapter_for_sync(adapter_path, adapter_name)
            
            # Check if any vLLM servers are actually available and support LoRA
            successful_loads = 0
            total_servers = len(self.config.vllm_inference_urls)
            
            # Hot-reload to all inference servers
            for vllm_url in self.config.vllm_inference_urls:
                success = self._reload_adapter_to_vllm(vllm_url, adapter_name, sync_adapter_path)
                if success:
                    self.logger.info(f"HOT_RELOAD_SUCCESS: {adapter_name} loaded to {vllm_url}")
                    successful_loads += 1
                else:
                    self.logger.debug(f"HOT_RELOAD_SKIP: {adapter_name} not loaded to {vllm_url}")
            
            if successful_loads == 0:
                self.logger.warning(f"HOT_RELOAD_NO_SERVERS: No vLLM servers support LoRA loading. "
                                  f"Adapters saved locally at {sync_adapter_path}")
            else:
                self.logger.info(f"HOT_RELOAD_SUMMARY: {successful_loads}/{total_servers} servers updated")
                    
        except Exception as e:
            self.logger.error(f"HOT_RELOAD_ERROR: {e}")
    
    def _prepare_adapter_for_sync(self, adapter_path: str, adapter_name: str) -> str:
        """Prepare adapter for synchronization to vLLM servers."""
        try:
            # Create sync directory if it doesn't exist
            sync_base_dir = os.path.join(self.config.log_dir, self.config.adapter_sync_dir)
            os.makedirs(sync_base_dir, exist_ok=True)
            
            # Copy adapter to sync directory with versioned name
            sync_adapter_path = os.path.join(sync_base_dir, adapter_name)
            
            # Remove existing adapter if present
            if os.path.exists(sync_adapter_path):
                shutil.rmtree(sync_adapter_path)
            
            # Copy the adapter
            shutil.copytree(adapter_path, sync_adapter_path)
            
            self.logger.info(f"ADAPTER_SYNC_PREP: {adapter_name} prepared at {sync_adapter_path}")
            return sync_adapter_path
            
        except Exception as e:
            self.logger.error(f"ADAPTER_SYNC_PREP_ERROR: {e}")
            return adapter_path  # Fallback to original path
    
    def _check_vllm_lora_support(self, vllm_url: str) -> bool:
        """Check if vLLM server supports dynamic LoRA loading."""
        try:
            # Try to get server info or available endpoints
            response = requests.get(f"{vllm_url}/v1/models", timeout=5)
            if response.status_code != 200:
                return False
            
            # Check if this looks like a vLLM server with LoRA support
            # We can try a test call to see what endpoints are available
            test_response = requests.post(
                f"{vllm_url}/v1/load_lora_adapter",
                json={"lora_name": "test", "lora_path": "/nonexistent"},
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            # If we get a proper error about the path not existing, the endpoint is supported
            # If we get a 404 or "wrong endpoint" error, it's not supported
            if test_response.status_code == 404:
                return False
            
            response_text = test_response.text.lower()
            if "请求有误" in response_text or "wrong request" in response_text or "not found" in response_text:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _reload_adapter_to_vllm(self, vllm_url: str, adapter_name: str, adapter_path: str) -> bool:
        """Reload LoRA adapter to vLLM inference server via API."""
        try:
            # First check if the server supports LoRA loading
            if not self._check_vllm_lora_support(vllm_url):
                self.logger.warning(f"VLLM_LORA_UNSUPPORTED: {vllm_url} does not support dynamic LoRA loading")
                return False
            
            # First, try to unload existing adapter with the same name (if any)
            self._unload_adapter_from_vllm(vllm_url, adapter_name)
            
            # Load new adapter
            load_url = f"{vllm_url}/v1/load_lora_adapter"
            load_payload = {
                "lora_name": adapter_name,
                "lora_path": adapter_path
            }
            
            response = requests.post(
                load_url,
                json=load_payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                self.logger.info(f"VLLM_LOAD_SUCCESS: {adapter_name} -> {vllm_url}")
                return True
            else:
                response_text = response.text
                if "请求有误" in response_text or "wrong request" in response_text.lower():
                    self.logger.warning(f"VLLM_LORA_UNSUPPORTED: {vllm_url} does not support LoRA loading API")
                    return False
                else:
                    self.logger.error(f"VLLM_LOAD_ERROR: {adapter_name} -> {vllm_url} "
                                    f"(Status: {response.status_code}, Response: {response_text})")
                    return False
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"VLLM_LOAD_NETWORK_ERROR: {adapter_name} -> {vllm_url} ({e})")
            return False
        except Exception as e:
            self.logger.error(f"VLLM_LOAD_UNKNOWN_ERROR: {adapter_name} -> {vllm_url} ({e})")
            return False
    
    def _unload_adapter_from_vllm(self, vllm_url: str, adapter_name: str):
        """Unload existing adapter from vLLM inference server."""
        try:
            unload_url = f"{vllm_url}/v1/unload_lora_adapter"
            unload_payload = {
                "lora_name": adapter_name
            }
            
            response = requests.post(
                unload_url,
                json=unload_payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.debug(f"VLLM_UNLOAD_SUCCESS: {adapter_name} unloaded from {vllm_url}")
            else:
                # It's OK if unload fails (adapter might not exist)
                self.logger.debug(f"VLLM_UNLOAD_INFO: {adapter_name} unload from {vllm_url} "
                                f"(Status: {response.status_code})")
                
        except Exception as e:
            # It's OK if unload fails
            self.logger.debug(f"VLLM_UNLOAD_ERROR: {adapter_name} unload from {vllm_url} ({e})")
    
    def _cleanup_old_checkpoints(self, checkpoint_dir: str):
        """Clean up old checkpoints to save disk space."""
        try:
            checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("step_")]
            checkpoints.sort(key=lambda x: int(x.split("_")[1]))
            
            while len(checkpoints) > self.config.max_checkpoints:
                old_checkpoint = checkpoints.pop(0)
                old_path = os.path.join(checkpoint_dir, old_checkpoint)
                if os.path.exists(old_path):
                    import shutil
                    shutil.rmtree(old_path)
                    self.logger.info(f"MAGRPO_CLEANUP: Removed old checkpoint {old_checkpoint}")
                    
        except Exception as e:
            self.logger.error(f"MAGRPO_CLEANUP_ERROR: {e}")


class TrainingManager:
    """
    Main training manager for MAGRPO dual-LLM system.
    
    Manages independent training processes for Traffic LLM and Regional LLM,
    with multiprocessing communication from the main simulation process.
    """
    
    def __init__(self, config: TrainingConfig, training_queue: mp.Queue):
        self.config = config
        self.training_queue = training_queue
        self.running = True
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Initialize replay buffers
        self.traffic_buffer = ReplayBuffer("Traffic", config.traffic_group_size)
        self.regional_buffer = ReplayBuffer("Regional", config.regional_group_size)
        
        # Initialize trainers
        self.traffic_trainer = MAGRPOTrainer("Traffic", config, config.traffic_gpu, "traffic")
        self.regional_trainer = MAGRPOTrainer("Regional", config, config.regional_gpu, "regional")
        
        # Statistics
        self.start_time = time.time()
        self.total_samples_processed = 0
        self.training_stats = {
            'traffic': {'total_steps': 0, 'total_loss': 0.0, 'last_loss': 0.0},
            'regional': {'total_steps': 0, 'total_loss': 0.0, 'last_loss': 0.0}
        }
        
        self.logger.info("TRAINING_MANAGER_INIT: MAGRPO Training Manager initialized")
        self.logger.info(f"TRAINING_CONFIG: Traffic group size: {config.traffic_group_size}, Regional group size: {config.regional_group_size}")
        self.logger.info(f"TRAINING_GPUS: Traffic GPU: {config.traffic_gpu}, Regional GPU: {config.regional_gpu}")
        self.logger.info(f"TRAINING_QUEUE: Queue available: {self.training_queue is not None}")
        
    def _setup_logger(self) -> logging.Logger:
        """Setup main training manager logger."""
        logger = logging.getLogger("training_manager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Create handler
            os.makedirs(self.config.log_dir, exist_ok=True)
            log_file = os.path.join(self.config.log_dir, "training_manager.log")
            handler = logging.FileHandler(log_file)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def run(self):
        """Main training loop."""
        try:
            self.logger.info("TRAINING_MANAGER_START: Starting main training loop")
            last_heartbeat_time = time.time()
            heartbeat_interval = 60.0  # Log heartbeat every 60 seconds
            
            while self.running:
                # Process incoming training data
                self._process_incoming_data()
                
                # Train Traffic LLM if buffer is ready
                if self.traffic_buffer.can_form_group():
                    training_group = self.traffic_buffer.get_training_group()
                    if training_group:
                        metrics = self.traffic_trainer.train_step(training_group)
                        self._update_training_stats('traffic', metrics)
                
                # Train Regional LLM if buffer is ready
                if self.regional_buffer.can_form_group():
                    training_group = self.regional_buffer.get_training_group()
                    if training_group:
                        metrics = self.regional_trainer.train_step(training_group)
                        self._update_training_stats('regional', metrics)
                
                # Periodic status logging
                if self.total_samples_processed > 0 and self.total_samples_processed % 100 == 0:
                    self._log_training_status()
                
                # Heartbeat logging for debugging
                current_time = time.time()
                if current_time - last_heartbeat_time >= heartbeat_interval:
                    runtime = current_time - self.start_time
                    self.logger.info(f"TRAINING_HEARTBEAT: Runtime: {runtime:.1f}s, "
                                   f"Samples received: {self.total_samples_processed}, "
                                   f"Traffic buffer: {self.traffic_buffer.size()}, "
                                   f"Regional buffer: {self.regional_buffer.size()}")
                    last_heartbeat_time = current_time
                
                # Short sleep to prevent CPU spinning
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            self.logger.info("TRAINING_MANAGER_INTERRUPT: Received interrupt signal")
        except Exception as e:
            self.logger.error(f"TRAINING_MANAGER_ERROR: {e}")
        finally:
            self._shutdown()
    
    def _process_incoming_data(self):
        """Process incoming training data from the queue."""
        try:
            data_received_this_cycle = 0
            while True:
                try:
                    # Non-blocking get with timeout
                    sample = self.training_queue.get(timeout=0.1)
                    self.total_samples_processed += 1
                    data_received_this_cycle += 1
                    
                    # Add to both buffers (each LLM gets trained on all samples)
                    self.traffic_buffer.add_sample(sample)
                    self.regional_buffer.add_sample(sample)
                    
                    self.logger.info(f"TRAINING_DATA_RECEIVED: Sample {self.total_samples_processed} from vehicle {sample.get('vehicle_id', 'unknown')} -> "
                                   f"Travel time: {sample.get('travel_time', 0):.1f}s, "
                                   f"Traffic buffer: {self.traffic_buffer.size()}, "
                                   f"Regional buffer: {self.regional_buffer.size()}")
                    
                except:
                    # No more data available, break inner loop
                    break
            
            # Log if we processed data this cycle
            if data_received_this_cycle > 0:
                self.logger.info(f"TRAINING_DATA_CYCLE: Processed {data_received_this_cycle} samples this cycle")
                    
        except Exception as e:
            self.logger.error(f"TRAINING_DATA_PROCESSING_ERROR: {e}")
    
    def _update_training_stats(self, trainer_type: str, metrics: Dict[str, float]):
        """Update training statistics."""
        stats = self.training_stats[trainer_type]
        stats['total_steps'] += 1
        stats['total_loss'] += metrics.get('loss', 0.0)
        stats['last_loss'] = metrics.get('loss', 0.0)
    
    def _log_training_status(self):
        """Log comprehensive training status."""
        try:
            runtime = time.time() - self.start_time
            
            # Buffer status
            traffic_stats = self.traffic_buffer.get_stats()
            regional_stats = self.regional_buffer.get_stats()
            
            # Training progress
            traffic_progress = self.training_stats['traffic']
            regional_progress = self.training_stats['regional']
            
            self.logger.info(f"TRAINING_STATUS: Runtime: {runtime:.1f}s, Samples: {self.total_samples_processed}")
            self.logger.info(f"  Traffic -> Buffer: {traffic_stats['current_size']}/{self.config.traffic_group_size}, "
                           f"Groups: {traffic_stats['total_groups_processed']}, "
                           f"Steps: {traffic_progress['total_steps']}, "
                           f"Loss: {traffic_progress['last_loss']:.4f}")
            self.logger.info(f"  Regional -> Buffer: {regional_stats['current_size']}/{self.config.regional_group_size}, "
                           f"Groups: {regional_stats['total_groups_processed']}, "
                           f"Steps: {regional_progress['total_steps']}, "
                           f"Loss: {regional_progress['last_loss']:.4f}")
            
        except Exception as e:
            self.logger.error(f"TRAINING_STATUS_ERROR: {e}")
    
    def _shutdown(self):
        """Graceful shutdown of training manager."""
        try:
            self.logger.info("TRAINING_MANAGER_SHUTDOWN: Graceful shutdown initiated")
            self.running = False
            
            # Save final checkpoints
            self.traffic_trainer._save_checkpoint()
            self.regional_trainer._save_checkpoint()
            
            # Log final statistics
            self._log_training_status()
            
            self.logger.info("TRAINING_MANAGER_SHUTDOWN: Shutdown completed")
            
        except Exception as e:
            self.logger.error(f"TRAINING_SHUTDOWN_ERROR: {e}")


def run_training_manager(config_dict: Dict[str, Any], training_queue: mp.Queue):
    """
    Entry point for training manager process.
    
    This function is called from the main process to start the training manager
    in a separate process.
    """
    try:
        # Convert config dict back to TrainingConfig object
        config = TrainingConfig(**config_dict)
        
        # Initialize and run training manager
        manager = TrainingManager(config, training_queue)
        
        # Handle signals for graceful shutdown
        def signal_handler(signum, frame):
            manager.running = False
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Run main training loop
        manager.run()
        
    except Exception as e:
        print(f"TRAINING_MANAGER_PROCESS_ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Test mode - can be used for debugging
    print("MAGRPO Training Manager - Test Mode")
    
    # Create test configuration
    config = TrainingConfig()
    
    # Create dummy queue for testing
    test_queue = mp.Queue()
    
    # Add some test data
    test_sample = {
        'vehicle_id': 'test_001',
        'travel_time': 580.0,
        'rewards': {
            'traffic_llm': {'total_reward': 0.8},
            'regional_llm': {'total_reward': 0.75}
        },
        'macro_route': [1, 3, 5],
        'state_context': {'regional_congestion': {1: 0.5, 3: 0.7, 5: 0.3}},
        'pioneer_decision': {'reasoning': 'Test reasoning'},
        'observer_feedback': {'feasibility_score': 0.8, 'efficiency_score': 0.75, 'fairness_score': 0.9}
    }
    
    test_queue.put(test_sample)
    
    # Run training manager
    run_training_manager(config.__dict__, test_queue)