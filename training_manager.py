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

# Enhanced visualization imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Optional W&B import (graceful fallback if not available)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Enable vLLM runtime LoRA updating for hot-reload functionality
os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"


@dataclass
class TrainingConfig:
    """Configuration for MAGRPO training with Progressive Mixed Training support."""
    
    # Model Configuration
    model_path: str = "/data/zhouyuping/Qwen/"
    
    def __post_init__(self):
        # 从环境变量读取训练GPU配置
        training_gpus = os.getenv("TRAINING_CUDA_VISIBLE_DEVICES", "2,3").split(",")
        if len(training_gpus) >= 2:
            self.traffic_gpu = f"cuda:{training_gpus[0]}"
            self.regional_gpu = f"cuda:{training_gpus[1]}"
        else:
            # 默认配置
            self.traffic_gpu = "cuda:2"
            self.regional_gpu = "cuda:3"
    
    traffic_gpu: str = "cuda:2"  # GPU for Traffic LLM training
    regional_gpu: str = "cuda:3"  # GPU for Regional LLM training
    
    # GRPO Configuration - Further optimized for stability
    traffic_group_size: int = 4  # Group size for Traffic LLM (reduced from 8)
    regional_group_size: int = 6  # Group size for Regional LLM (reduced from 12)
    
    # LoRA Configuration - Optimized for memory
    lora_r: int = 8  # Reduced from 16 for memory optimization
    lora_alpha: int = 16  # Reduced proportionally
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    
    # Training Configuration
    learning_rate: float = 1e-4
    warmup_steps: int = 15
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 8  # Increased to reduce memory per step
    
    # Progressive Mixed Training Configuration
    enable_progressive_training: bool = True  # Enable progressive mixed training
    offline_pretraining_steps: int = 10  # Steps for initial offline pretraining
    online_steps_per_reinforcement: int = 10  # Online steps before triggering reinforcement
    reinforcement_batch_multiplier: int = 3  # Multiplier for reinforcement batch size
    progressive_learning_rate_decay: float = 0.95  # LR decay during progressive phases
    historical_data_dir: str = "logs/training/historical_data"  # Directory for historical data
    
    # Training Mode Configuration
    training_mode: str = "progressive"  # "online_only", "offline_only", "progressive"
    warmup_phase_enabled: bool = True  # Enable offline warmup phase
    online_phase_enabled: bool = True  # Enable online micro-tuning phase  
    reinforcement_phase_enabled: bool = True  # Enable offline reinforcement phase
    
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
    """Advanced replay buffer for storing and grouping training samples with Progressive Mixed Training support."""
    
    def __init__(self, name: str, group_size: int, max_size: int = 10000, config: TrainingConfig = None):
        self.name = name
        self.group_size = group_size
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.config = config
        
        # Progressive Training Support
        self.historical_buffer = deque(maxlen=max_size * 2)  # Larger historical storage
        self.high_quality_samples = deque(maxlen=max_size // 2)  # Store high-quality samples
        self.training_phase = "warmup"  # "warmup", "online", "reinforcement"
        
        # Statistics
        self.total_samples_received = 0
        self.total_groups_processed = 0
        self.offline_groups_processed = 0
        self.online_groups_processed = 0
        self.reinforcement_groups_processed = 0
        
    def add_sample(self, sample: Dict[str, Any]):
        """Add a training sample to the buffer with quality assessment."""
        with self.lock:
            self.buffer.append(sample)
            # Don't add to historical_buffer - that's only for loaded historical data
            self.total_samples_received += 1
            
            # Quality-based sample filtering for progressive training
            if self.config and self.config.enable_progressive_training:
                self._assess_and_store_quality_sample(sample)
            
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
                'historical_size': len(self.historical_buffer),
                'high_quality_size': len(self.high_quality_samples),
                'total_samples_received': self.total_samples_received,
                'total_groups_processed': self.total_groups_processed,
                'offline_groups_processed': self.offline_groups_processed,
                'online_groups_processed': self.online_groups_processed,
                'reinforcement_groups_processed': self.reinforcement_groups_processed,
                'training_phase': self.training_phase
            }
    
    def _assess_and_store_quality_sample(self, sample: Dict[str, Any]):
        """Assess sample quality and store high-quality samples for offline reinforcement."""
        try:
            # Extract reward information for quality assessment
            rewards = sample.get('rewards', {})
            
            # Calculate quality score based on rewards and cooperation
            if self.name.lower() == "traffic":
                reward_info = rewards.get('traffic_llm', {})
                total_reward = reward_info.get('total_reward', 0.0)
                cooperation_reward = reward_info.get('cooperation_reward', 0.0)
            else:  # regional
                reward_info = rewards.get('regional_llm', {})
                total_reward = reward_info.get('total_reward', 0.0)
                cooperation_reward = reward_info.get('cooperation_reward', 0.0)
            
            # Quality threshold: samples with high total reward and cooperation
            quality_threshold = 0.7
            cooperation_threshold = 0.6
            
            if total_reward >= quality_threshold and cooperation_reward >= cooperation_threshold:
                sample['quality_score'] = total_reward * 0.7 + cooperation_reward * 0.3
                sample['sample_type'] = 'high_quality'
                self.high_quality_samples.append(sample)
                
        except Exception as e:
            # Graceful degradation - don't break the training flow
            pass
    
    def set_training_phase(self, phase: str):
        """Set current training phase."""
        with self.lock:
            self.training_phase = phase
    
    def get_offline_training_group(self) -> Optional[List[Dict[str, Any]]]:
        """Get training group for offline phases (warmup/reinforcement)."""
        with self.lock:
            # For warmup phase, use historical data if available, fallback to current buffer
            if self.training_phase == "warmup":
                if len(self.historical_buffer) >= self.group_size:
                    # Use historical data if available
                    group = []
                    for _ in range(self.group_size):
                        if self.historical_buffer:
                            group.append(self.historical_buffer.popleft())
                    self.offline_groups_processed += 1
                    return group
                elif len(self.historical_buffer) == 0 and len(self.buffer) >= self.group_size:
                    # Fallback: use current buffer when no historical data available
                    group = []
                    for _ in range(self.group_size):
                        if self.buffer:
                            group.append(self.buffer.popleft())
                    self.offline_groups_processed += 1
                    return group
                else:
                    return None
                
            # For reinforcement phase, prioritize high-quality samples
            elif self.training_phase == "reinforcement":
                # Mix high-quality samples with recent samples
                group = []
                high_quality_count = min(len(self.high_quality_samples), self.group_size // 2)
                recent_count = self.group_size - high_quality_count
                
                # Add high-quality samples
                for _ in range(high_quality_count):
                    if self.high_quality_samples:
                        group.append(self.high_quality_samples.popleft())
                
                # Add recent samples
                for _ in range(recent_count):
                    if self.buffer:
                        group.append(self.buffer.popleft())
                
                if len(group) >= self.group_size:
                    self.reinforcement_groups_processed += 1
                    return group[:self.group_size]
            
            return None
    
    def get_reinforcement_batch(self, batch_multiplier: int = 2) -> Optional[List[Dict[str, Any]]]:
        """Get larger batch for reinforcement training."""
        with self.lock:
            reinforcement_size = self.group_size * batch_multiplier
            
            # Prioritize high-quality samples for reinforcement
            batch = []
            
            # Add available high-quality samples
            while self.high_quality_samples and len(batch) < reinforcement_size // 2:
                batch.append(self.high_quality_samples.popleft())
            
            # Fill remaining with recent samples  
            while self.buffer and len(batch) < reinforcement_size:
                batch.append(self.buffer.popleft())
                
            if len(batch) >= self.group_size:
                self.reinforcement_groups_processed += 1
                return batch
                
            return None
    
    def load_historical_data(self, data_dir: str):
        """Load historical training data from directory."""
        try:
            historical_file = os.path.join(data_dir, f"{self.name.lower()}_historical_samples.json")
            if os.path.exists(historical_file):
                with open(historical_file, 'r') as f:
                    historical_samples = json.load(f)
                    
                with self.lock:
                    for sample in historical_samples:
                        self.historical_buffer.append(sample)
                        if sample.get('sample_type') == 'high_quality':
                            self.high_quality_samples.append(sample)
                            
                return len(historical_samples)
        except Exception as e:
            return 0
        
        return 0
    
    def save_historical_data(self, data_dir: str):
        """Save current buffer data as historical data."""
        try:
            os.makedirs(data_dir, exist_ok=True)
            historical_file = os.path.join(data_dir, f"{self.name.lower()}_historical_samples.json")
            
            with self.lock:
                # Combine all samples for saving
                all_samples = list(self.buffer) + list(self.historical_buffer)
                
                # Remove duplicates based on vehicle_id and timestamp if available
                seen = set()
                unique_samples = []
                for sample in all_samples:
                    key = (sample.get('vehicle_id', ''), sample.get('travel_time', 0))
                    if key not in seen:
                        seen.add(key)
                        unique_samples.append(sample)
                
                with open(historical_file, 'w') as f:
                    json.dump(unique_samples[-1000:], f, indent=2)  # Keep last 1000 samples
                    
                return len(unique_samples)
        except Exception as e:
            return 0


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
        
        # Progressive Training State
        self.training_mode = config.training_mode if config.enable_progressive_training else "online_only"
        self.current_phase = "warmup" if config.warmup_phase_enabled else "online"
        self.online_steps_count = 0
        self.offline_pretraining_completed = False
        self.phase_transition_history = []
        
        # Dynamic Learning Rate for Progressive Training
        self.base_learning_rate = config.learning_rate
        self.current_learning_rate = self.base_learning_rate
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Initialize model and training components
        self._initialize_model()
        self._initialize_optimizer()
        
        self.logger.info(f"MAGRPO_TRAINER_INIT: {self.name} trainer initialized on {gpu_device}")
        self.logger.info(f"PROGRESSIVE_TRAINING: Mode={self.training_mode}, Phase={self.current_phase}, "
                         f"Enabled={config.enable_progressive_training}")
        
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
            
            # Load base model with proper tensor initialization
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.float16,
                device_map=None,  # We'll move to device manually
                trust_remote_code=True,
                low_cpu_mem_usage=False,  # Disable to avoid meta tensors
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
            
            # Apply PEFT without low_cpu_mem_usage to avoid meta tensors
            self.model = get_peft_model(self.base_model, lora_config)
            
            # Move model to target device properly 
            self.model = self.model.to(self.device)
            
            # Verify all parameters are on the correct device and have gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and 'lora_' in name:
                    # Ensure LoRA parameters are properly initialized and on device
                    if param.device != self.device:
                        param.data = param.data.to(self.device)
                    
                    # Reinitialize LoRA parameters with proper initialization if needed
                    if not torch.is_tensor(param.data) or param.data.numel() == 0:
                        if 'lora_A' in name:
                            # LoRA A matrix - use kaiming uniform initialization
                            torch.nn.init.kaiming_uniform_(param, a=5**0.5)
                        elif 'lora_B' in name:
                            # LoRA B matrix - zero initialization
                            torch.nn.init.zeros_(param)
                        else:
                            # Other LoRA parameters - normal initialization
                            torch.nn.init.normal_(param, std=0.02)
            
            # Ensure model is in training mode and parameters require gradients
            self.model.train()
            
            # Double-check that trainable parameters are properly configured
            trainable_count = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    trainable_count += 1
                    # Ensure parameter is a real tensor, not meta tensor
                    if not param.is_meta and param.device != self.device:
                        self.logger.warning(f"Parameter {name} not on correct device, moving...")
                        param.data = param.data.to(self.device)
                        
                    # Crucial: Ensure LoRA parameters are properly connected to computation graph
                    if 'lora_' in name:
                        self.logger.info(f"LoRA Parameter {name}: requires_grad={param.requires_grad}, device={param.device}")
            
            if trainable_count == 0:
                raise ValueError("No trainable parameters found after model initialization!")
                
            # Test that the model can produce gradients by doing a dummy forward pass
            self.logger.info("Testing gradient flow with dummy input...")
            try:
                # Create a simple dummy input
                test_input = torch.ones((1, 10), dtype=torch.long, device=self.device)
                test_attention = torch.ones((1, 10), dtype=torch.long, device=self.device)
                
                with torch.amp.autocast('cuda'):
                    test_output = self.model(input_ids=test_input, attention_mask=test_attention, labels=test_input)
                    test_loss = test_output.loss
                
                # Check if loss has gradients
                if test_loss.requires_grad and test_loss.grad_fn is not None:
                    self.logger.info("✓ Gradient flow test passed - model can produce gradients")
                else:
                    self.logger.warning("⚠ Gradient flow test failed - loss has no gradients")
                    # Try to fix by ensuring model is in training mode and gradients are enabled
                    self.model.train()
                    for param in self.model.parameters():
                        if param.requires_grad:
                            param.grad = None  # Clear any existing gradients
                            
                del test_input, test_attention, test_output, test_loss
                torch.cuda.empty_cache()
                
            except Exception as e:
                self.logger.warning(f"Gradient flow test failed: {e}")
                # Continue anyway, but log the issue
            
            # Enable gradient checkpointing for memory efficiency
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            
            # Print trainable parameters and verify PEFT setup
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            
            self.logger.info(f"MAGRPO_PEFT: {self.name} -> Trainable: {trainable_params:,} / Total: {total_params:,} "
                           f"({trainable_params/total_params*100:.2f}%)")
            
            # Verify PEFT is properly configured
            if hasattr(self.model, 'peft_config'):
                self.logger.info(f"PEFT Configuration detected: {type(self.model.peft_config)}")
                if hasattr(self.model, 'get_peft_config_as_dict'):
                    peft_config = self.model.get_peft_config_as_dict()
                    self.logger.info(f"PEFT Config: r={peft_config.get('default', {}).get('r', 'N/A')}, "
                                   f"alpha={peft_config.get('default', {}).get('lora_alpha', 'N/A')}")
            
            # Critical: Verify base model is properly frozen
            frozen_params = sum(p.numel() for p in self.base_model.parameters() if not p.requires_grad)
            self.logger.info(f"Frozen base model parameters: {frozen_params:,}")
            
            # Verify LoRA adapters are trainable
            lora_params = sum(p.numel() for n, p in self.model.named_parameters() 
                             if p.requires_grad and 'lora_' in n)
            self.logger.info(f"LoRA adapter parameters: {lora_params:,}")
            
            if lora_params == 0:
                raise ValueError("No LoRA parameters found! PEFT configuration may be incorrect.")
            
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
            self.scaler = torch.amp.GradScaler('cuda')
            
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
            # CRITICAL: Ensure model is in training mode and gradients are enabled
            self.model.train()
            
            # Double-check that LoRA adapters are active
            for name, module in self.model.named_modules():
                if hasattr(module, 'train'):
                    module.train()
            
            # Verify trainable parameters before training
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            if not trainable_params:
                raise RuntimeError("No trainable parameters found at start of training step!")
            
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
                # Validate tensor properties before forward pass
                if input_ids.numel() == 0 or attention_mask.numel() == 0 or labels.numel() == 0:
                    self.logger.warning(f"MAGRPO_VALIDATION: Empty tensor detected in batch {i}, skipping")
                    continue
                
                # Ensure tensors are on correct device
                if input_ids.device != self.device:
                    input_ids = input_ids.to(self.device)
                if attention_mask.device != self.device:
                    attention_mask = attention_mask.to(self.device)
                if labels.device != self.device:
                    labels = labels.to(self.device)
                
                with torch.amp.autocast('cuda'):  # Use automatic mixed precision
                    # Verify model parameters still have gradients
                    trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                    if not trainable_params:
                        raise RuntimeError("No trainable parameters found during forward pass!")
                    
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    
                    # Verify that loss is a valid tensor with gradient
                    if not hasattr(outputs, 'loss') or outputs.loss is None:
                        raise RuntimeError("Model output does not contain loss!")
                    
                    loss = outputs.loss * weight  # Weight by relative reward
                    
                    # Verify loss is a scalar tensor
                    if loss.dim() != 0:
                        loss = loss.mean()
                    
                    # Diagnose and fix gradient flow issues
                    if not loss.requires_grad or loss.grad_fn is None:
                        self.logger.warning(f"MAGRPO_GRAD_WARNING: Loss tensor lacks gradients (requires_grad={loss.requires_grad}, grad_fn={loss.grad_fn is not None})")
                        
                        # Emergency fix: Force reconnect the loss to trainable parameters
                        # This creates a computation graph by adding a tiny contribution from LoRA parameters
                        regularization_loss = 0.0
                        lora_param_count = 0
                        
                        for name, param in self.model.named_parameters():
                            if param.requires_grad and 'lora_' in name:
                                regularization_loss = regularization_loss + param.norm() * 1e-12  # Very tiny contribution
                                lora_param_count += 1
                        
                        if lora_param_count > 0:
                            # Add the regularization to the loss to connect it to the computation graph
                            loss = loss + regularization_loss
                            self.logger.info(f"GRAD_FIX: Connected loss to {lora_param_count} LoRA parameters")
                            
                            # Verify fix worked
                            if loss.requires_grad and loss.grad_fn is not None:
                                self.logger.info("✓ Gradient flow successfully restored")
                            else:
                                self.logger.error("✗ Gradient flow fix failed")
                        else:
                            self.logger.error("No LoRA parameters found for gradient connection!")
                    
                    else:
                        # Loss already has gradients - this is the expected case
                        pass
                    
                    # Scale loss for gradient accumulation
                    loss = loss / accumulation_steps
                
                total_loss += loss.detach()
                
                # Backward pass with gradient scaling
                if loss.requires_grad and loss.grad_fn is not None:
                    self.scaler.scale(loss).backward()
                else:
                    self.logger.warning(f"MAGRPO_SKIP_BACKWARD: Loss tensor cannot be backpropagated")
                
                # Clear intermediate variables to save memory
                del outputs, loss
                
                if i % accumulation_steps == 0 or i == len(batch_inputs) - 1:
                    # Check if we have any gradients to work with
                    has_gradients = any(p.grad is not None for p in self.model.parameters() if p.requires_grad)
                    
                    if has_gradients:
                        # Unscale gradients for gradient clipping
                        self.scaler.unscale_(self.optimizer)
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        
                        # Optimizer step with scaler
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        # Still need to update scaler even without gradients
                        self.scaler.update()
                        self.logger.warning(f"MAGRPO_NO_GRADIENTS: No gradients found, skipping optimizer step")
                    
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
            
            # Progressive training phase management
            if self.config.enable_progressive_training:
                self._update_progressive_state()
            
            return {
                'loss': step_loss,
                'relative_reward_mean': reward_mean,
                'relative_reward_std': reward_std,
                'learning_rate': self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate,
                'training_phase': self.current_phase,
                'online_steps_count': self.online_steps_count
            }
            
        except Exception as e:
            self.logger.error(f"MAGRPO_TRAIN_ERROR: {self.name} training step failed: {e}")
            return {'loss': 0.0, 'relative_reward_mean': 0.0, 'relative_reward_std': 0.0}
    
    def _calculate_relative_rewards(self, training_group: List[Dict[str, Any]]) -> List[float]:
        """Calculate relative rewards within a group (core of GRPO)."""
        try:
            # Extract rewards for this LLM type
            rewards = []
            for sample in training_group:
                if isinstance(sample, dict) and 'rewards' in sample:
                    if self.task_type == "traffic":
                        reward = sample['rewards'].get('traffic_llm', {}).get('total_reward', 0.5)
                    else:  # regional
                        reward = sample['rewards'].get('regional_llm', {}).get('total_reward', 0.5)
                    rewards.append(reward)
                else:
                    self.logger.warning(f"MAGRPO_REWARD_EXTRACT: Invalid sample format: {type(sample)}")
                    rewards.append(0.5)  # Default reward
            
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
                # Clone and detach to ensure proper gradient flow
                labels = input_ids.clone().detach()
                
                # Ensure tensors are contiguous and have proper shape
                input_ids = input_ids.contiguous()
                attention_mask = attention_mask.contiguous()
                labels = labels.contiguous()
                
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
            # Handle both dict and list cases (defensive programming)
            if not isinstance(sample, dict):
                self.logger.warning(f"TRAFFIC_TEXT_WARNING: Expected dict, got {type(sample)}")
                return ""
                
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
            # Handle both dict and list cases (defensive programming)
            if not isinstance(sample, dict):
                self.logger.warning(f"REGIONAL_TEXT_WARNING: Expected dict, got {type(sample)}")
                return ""
                
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
        """Trigger hot-reload of LoRA adapter using local LLM manager."""
        try:
            adapter_name = f"{self.name.lower()}_adapter_step_{self.training_steps}"
            
            # Prepare adapter for sync
            sync_adapter_path = self._prepare_adapter_for_sync(adapter_path, adapter_name)
            
            # Strategy 1: Try vLLM API reload (legacy method)
            vllm_success = self._try_vllm_api_reload(sync_adapter_path, adapter_name)
            
            # Strategy 2: Use local LLM manager direct loading (new method)
            if not vllm_success:
                self.logger.info("HOT_RELOAD_FALLBACK: vLLM API failed, trying local LLM manager")
                local_success = self._try_local_llm_manager_reload(sync_adapter_path)
                
                if local_success:
                    self.logger.info(f"HOT_RELOAD_LOCAL_SUCCESS: {adapter_name} loaded via local LLM manager")
                else:
                    self.logger.warning(f"HOT_RELOAD_LOCAL_FAILED: Local LLM manager reload failed")
                    # Queue adapter for later processing
                    self._queue_adapter_for_later_processing(sync_adapter_path)
            else:
                self.logger.info(f"HOT_RELOAD_VLLM_SUCCESS: {adapter_name} loaded via vLLM API")
                    
        except Exception as e:
            self.logger.error(f"HOT_RELOAD_ERROR: {e}")
    
    def _try_vllm_api_reload(self, adapter_path: str, adapter_name: str) -> bool:
        """Try to reload adapter via vLLM API (legacy method)."""
        try:
            successful_loads = 0
            total_servers = len(self.config.vllm_inference_urls)
            
            # Hot-reload to all inference servers
            for vllm_url in self.config.vllm_inference_urls:
                success = self._reload_adapter_to_vllm(vllm_url, adapter_name, adapter_path)
                if success:
                    self.logger.info(f"VLLM_API_SUCCESS: {adapter_name} loaded to {vllm_url}")
                    successful_loads += 1
                else:
                    self.logger.debug(f"VLLM_API_SKIP: {adapter_name} not loaded to {vllm_url}")
            
            if successful_loads > 0:
                self.logger.info(f"VLLM_API_SUMMARY: {successful_loads}/{total_servers} servers updated")
                return True
            else:
                self.logger.debug("VLLM_API_NO_SUCCESS: No vLLM servers support LoRA loading")
                return False
                
        except Exception as e:
            self.logger.error(f"VLLM_API_ERROR: {e}")
            return False
    
    def _try_local_llm_manager_reload(self, adapter_path: str) -> bool:
        """Try to reload adapter via local LLM manager."""
        try:
            # Get LLM manager from multi-agent environment
            llm_manager = self._get_llm_manager()
            
            if llm_manager is None:
                self.logger.warning("LOCAL_RELOAD_NO_MANAGER: LLM manager not available")
                return False
            
            # Determine LLM type from trainer name
            llm_type = self.task_type.lower()  # 'traffic' or 'regional'
            
            # Load adapter directly to LLM manager
            success = llm_manager.load_lora_adapter_direct(llm_type, adapter_path)
            
            if success:
                self.logger.info(f"LOCAL_RELOAD_SUCCESS: {llm_type} adapter loaded successfully")
                return True
            else:
                self.logger.warning(f"LOCAL_RELOAD_FAILED: {llm_type} adapter loading failed")
                
                # Try queuing for later processing
                queue_success = llm_manager.queue_adapter_update(llm_type, adapter_path)
                if queue_success:
                    self.logger.info(f"LOCAL_RELOAD_QUEUED: {llm_type} adapter queued for later processing")
                    return True  # Consider queuing as success
                
                return False
                
        except Exception as e:
            self.logger.error(f"LOCAL_RELOAD_ERROR: {e}")
            return False
    
    def _get_llm_manager(self):
        """Get LLM manager from the multi-agent environment."""
        try:
            # Import global access function
            from multi_agent_env import get_global_llm_manager, list_registered_llm_managers
            
            # Try to get the current registered LLM manager
            llm_manager = get_global_llm_manager()
            
            if llm_manager is not None:
                self.logger.info(f"LLM_MANAGER_ACCESS: Successfully retrieved global LLM manager")
                return llm_manager
            else:
                # List available managers for debugging
                available_managers = list_registered_llm_managers()
                self.logger.warning(f"LLM_MANAGER_ACCESS: No current LLM manager found. Available keys: {available_managers}")
                return None
            
        except ImportError as e:
            self.logger.error(f"LLM_MANAGER_ACCESS: Failed to import global access functions: {e}")
            return None
        except Exception as e:
            self.logger.error(f"GET_LLM_MANAGER_ERROR: {e}")
            return None
    
    def _queue_adapter_for_later_processing(self, adapter_path: str):
        """Queue adapter for later processing when hot-reload fails."""
        try:
            # Store adapter path for later processing
            if not hasattr(self, 'pending_adapters'):
                self.pending_adapters = []
                self.last_queue_process_time = 0
            
            self.pending_adapters.append({
                'path': adapter_path,
                'timestamp': time.time(),
                'step': self.training_steps,
                'retry_count': 0,
                'adapter_name': f"{self.name.lower()}_adapter_step_{self.training_steps}",
                'status': 'pending'
            })
            
            self.logger.info(f"ADAPTER_QUEUED: {os.path.basename(adapter_path)} queued for later processing")
            
            # Clean up old queued adapters (keep only last 5, increased from 3)
            if len(self.pending_adapters) > 5:
                self.pending_adapters = self.pending_adapters[-5:]
                
        except Exception as e:
            self.logger.error(f"ADAPTER_QUEUE_ERROR: {e}")

    def _process_pending_adapters(self):
        """Process queued adapters that failed initial loading."""
        try:
            if not hasattr(self, 'pending_adapters') or not self.pending_adapters:
                return
                
            current_time = time.time()
            
            # Throttle queue processing - only process every 30 seconds
            if current_time - getattr(self, 'last_queue_process_time', 0) < 30:
                return
                
            self.last_queue_process_time = current_time
            processed_count = 0
            failed_count = 0
            
            # Process adapters in FIFO order
            adapters_to_remove = []
            
            for i, adapter_info in enumerate(self.pending_adapters):
                if processed_count >= 2:  # Limit processing to 2 adapters per cycle
                    break
                    
                adapter_path = adapter_info['path']
                adapter_name = adapter_info['adapter_name'] 
                retry_count = adapter_info['retry_count']
                timestamp = adapter_info['timestamp']
                
                # Skip if adapter is too old (older than 10 minutes)
                if current_time - timestamp > 600:
                    adapters_to_remove.append(i)
                    self.logger.info(f"ADAPTER_EXPIRED: {adapter_name} removed from queue (too old)")
                    continue
                
                # Skip if already reached max retry count
                if retry_count >= 3:
                    adapters_to_remove.append(i)
                    self.logger.warning(f"ADAPTER_MAX_RETRIES: {adapter_name} removed from queue (max retries reached)")
                    continue
                
                # Try to load the adapter again
                self.logger.info(f"ADAPTER_RETRY: Attempting to load {adapter_name} (retry #{retry_count + 1})")
                
                success = self._retry_adapter_loading(adapter_path, adapter_name)
                
                if success:
                    adapters_to_remove.append(i)
                    processed_count += 1
                    self.logger.info(f"ADAPTER_RETRY_SUCCESS: {adapter_name} loaded successfully from queue")
                else:
                    # Increment retry count for failed attempts
                    self.pending_adapters[i]['retry_count'] = retry_count + 1
                    self.pending_adapters[i]['status'] = 'failed'
                    failed_count += 1
                    self.logger.warning(f"ADAPTER_RETRY_FAILED: {adapter_name} failed retry #{retry_count + 1}")
                    
            # Remove processed/expired adapters in reverse order to maintain indices
            for i in reversed(adapters_to_remove):
                self.pending_adapters.pop(i)
                
            if processed_count > 0 or failed_count > 0:
                self.logger.info(f"QUEUE_PROCESSING: Processed={processed_count}, Failed={failed_count}, "
                               f"Remaining={len(self.pending_adapters)}")
                
        except Exception as e:
            self.logger.error(f"QUEUE_PROCESSING_ERROR: {e}")

    def _retry_adapter_loading(self, adapter_path: str, adapter_name: str) -> bool:
        """Retry loading a queued adapter using all available methods."""
        try:
            # Method 1: Try local LLM manager loading
            local_success = self._try_local_llm_manager_reload(adapter_path)
            if local_success:
                return True
                
            # Method 2: Try vLLM API loading (in case vLLM server was restarted with LoRA support)
            vllm_success = self._try_vllm_api_reload(adapter_path, adapter_name)
            if vllm_success:
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"ADAPTER_RETRY_ERROR: {e}")
            return False

    def get_adapter_queue_stats(self) -> Dict[str, Any]:
        """Get statistics about the adapter queue."""
        if not hasattr(self, 'pending_adapters'):
            return {'queue_size': 0, 'total_pending': 0, 'total_failed': 0}
            
        pending_count = sum(1 for a in self.pending_adapters if a['status'] == 'pending')
        failed_count = sum(1 for a in self.pending_adapters if a['status'] == 'failed')
        
        return {
            'queue_size': len(self.pending_adapters),
            'total_pending': pending_count,
            'total_failed': failed_count,
            'oldest_timestamp': min((a['timestamp'] for a in self.pending_adapters), default=0),
            'last_process_time': getattr(self, 'last_queue_process_time', 0)
        }
    
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
    
    def _update_progressive_state(self):
        """Update progressive training state and handle phase transitions."""
        try:
            # Update online steps count for progressive training
            if self.current_phase == "online":
                self.online_steps_count += 1
            
            # Check for phase transitions based on configuration
            if self._should_transition_phase():
                self._transition_training_phase()
                
        except Exception as e:
            self.logger.error(f"PROGRESSIVE_STATE_ERROR: {e}")
    
    def _should_transition_phase(self) -> bool:
        """Check if training phase should transition."""
        try:
            # Transition from warmup to online after offline pretraining steps
            if (self.current_phase == "warmup" and 
                self.training_steps >= self.config.offline_pretraining_steps):
                return True
            
            # Transition from online to reinforcement after specified online steps
            if (self.current_phase == "online" and 
                self.online_steps_count >= self.config.online_steps_per_reinforcement and
                self.config.reinforcement_phase_enabled):
                return True
            
            # Transition from reinforcement back to online (continuous cycle)
            if self.current_phase == "reinforcement":
                # Simple heuristic: after some reinforcement steps, go back to online
                return True  # Will be handled by training manager
                
            return False
        except Exception as e:
            self.logger.error(f"PHASE_TRANSITION_CHECK_ERROR: {e}")
            return False
    
    def _transition_training_phase(self):
        """Handle training phase transitions with learning rate adjustments."""
        try:
            old_phase = self.current_phase
            
            # Determine next phase
            if self.current_phase == "warmup":
                self.current_phase = "online"
                self.offline_pretraining_completed = True
                self._adjust_learning_rate_for_phase("online")
                
            elif self.current_phase == "online":
                self.current_phase = "reinforcement"  
                self.online_steps_count = 0  # Reset counter
                self._adjust_learning_rate_for_phase("reinforcement")
                
            elif self.current_phase == "reinforcement":
                self.current_phase = "online"
                self._adjust_learning_rate_for_phase("online")
            
            # Log phase transition
            self.phase_transition_history.append({
                'step': self.training_steps,
                'from_phase': old_phase,
                'to_phase': self.current_phase,
                'learning_rate': self.current_learning_rate
            })
            
            self.logger.info(f"PHASE_TRANSITION: {self.name} {old_phase} -> {self.current_phase} "
                           f"at step {self.training_steps}, LR: {self.current_learning_rate:.2e}")
            
        except Exception as e:
            self.logger.error(f"PHASE_TRANSITION_ERROR: {e}")
    
    def _adjust_learning_rate_for_phase(self, phase: str):
        """Adjust learning rate for different training phases."""
        try:
            if phase == "online":
                # Higher learning rate for online fine-tuning
                self.current_learning_rate = self.base_learning_rate
            elif phase == "reinforcement":
                # Lower learning rate for stable reinforcement learning  
                self.current_learning_rate = self.base_learning_rate * 0.5
            elif phase == "warmup":
                # Moderate learning rate for offline pretraining
                self.current_learning_rate = self.base_learning_rate * 0.8
            
            # Update optimizer learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.current_learning_rate
                
            # Apply progressive decay if configured
            if hasattr(self.config, 'progressive_learning_rate_decay'):
                decay_factor = self.config.progressive_learning_rate_decay
                self.current_learning_rate *= decay_factor
                
        except Exception as e:
            self.logger.error(f"LEARNING_RATE_ADJUST_ERROR: {e}")
    
    def train_step_progressive(self, training_group: List[Dict[str, Any]], 
                              batch_multiplier: int = 1) -> Dict[str, float]:
        """
        Enhanced training step with progressive training support.
        
        Args:
            training_group: Training samples
            batch_multiplier: Multiplier for reinforcement batches
        """
        try:
            # Use larger batch for reinforcement phase
            if self.current_phase == "reinforcement" and batch_multiplier > 1:
                # Process multiple groups as a single large batch
                return self._train_reinforcement_batch(training_group, batch_multiplier)
            else:
                # Standard training step
                return self.train_step(training_group)
                
        except Exception as e:
            self.logger.error(f"PROGRESSIVE_TRAIN_STEP_ERROR: {e}")
            return {'loss': 0.0, 'relative_reward_mean': 0.0, 'relative_reward_std': 0.0}
    
    def _train_reinforcement_batch(self, training_groups: List[Dict[str, Any]], 
                                  batch_multiplier: int) -> Dict[str, float]:
        """Handle reinforcement training with larger batches."""
        try:
            # Flatten multiple groups into single larger batch for reinforcement
            all_samples = []
            if isinstance(training_groups[0], list):
                # Multiple groups provided
                for group in training_groups:
                    all_samples.extend(group)
            else:
                # Single group provided
                all_samples = training_groups
            
            # Use standard training logic but with larger effective batch
            group_rewards = self._calculate_relative_rewards(all_samples)
            batch_inputs = self._prepare_training_inputs(all_samples, group_rewards)
            
            if not batch_inputs:
                return {'loss': 0.0, 'relative_reward_mean': 0.0, 'relative_reward_std': 0.0}
            
            # Enhanced training for reinforcement phase
            total_loss = 0.0
            accumulation_steps = self.config.gradient_accumulation_steps * batch_multiplier
            
            for i, (input_ids, attention_mask, labels, weight) in enumerate(batch_inputs):
                with torch.amp.autocast('cuda'):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss * weight * 1.2  # Slightly higher weight for reinforcement
                    loss = loss / accumulation_steps
                
                total_loss += loss.detach()
                self.scaler.scale(loss).backward()
                
                del outputs, loss
                
                if i % accumulation_steps == 0 or i == len(batch_inputs) - 1:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if self.scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()
            
            # Calculate metrics
            avg_loss = total_loss * accumulation_steps
            self.training_steps += 1
            step_loss = avg_loss.item()
            self.total_loss += step_loss
            
            reward_values = [r for _, _, _, r in batch_inputs]
            reward_mean = np.mean(reward_values) if reward_values else 0.0
            reward_std = np.std(reward_values) if reward_values else 0.0
            
            self.logger.info(f"REINFORCEMENT_STEP: {self.name} Step {self.training_steps} -> "
                           f"Loss: {step_loss:.4f}, Batch Size: {len(all_samples)}, "
                           f"Reward μ: {reward_mean:.3f}, σ: {reward_std:.3f}")
            
            return {
                'loss': step_loss,
                'relative_reward_mean': reward_mean,
                'relative_reward_std': reward_std,
                'learning_rate': self.current_learning_rate,
                'batch_size': len(all_samples),
                'training_phase': self.current_phase
            }
            
        except Exception as e:
            self.logger.error(f"REINFORCEMENT_BATCH_ERROR: {e}")
            return {'loss': 0.0, 'relative_reward_mean': 0.0, 'relative_reward_std': 0.0}
    
    def get_progressive_stats(self) -> Dict[str, Any]:
        """Get progressive training statistics."""
        return {
            'training_mode': self.training_mode,
            'current_phase': self.current_phase,
            'online_steps_count': self.online_steps_count,
            'offline_pretraining_completed': self.offline_pretraining_completed,
            'current_learning_rate': self.current_learning_rate,
            'phase_transitions': len(self.phase_transition_history),
            'last_transition': self.phase_transition_history[-1] if self.phase_transition_history else None
        }


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
        
        # Initialize replay buffers with progressive training support
        self.traffic_buffer = ReplayBuffer("Traffic", config.traffic_group_size, config=config)
        self.regional_buffer = ReplayBuffer("Regional", config.regional_group_size, config=config)
        
        # Initialize trainers
        self.traffic_trainer = MAGRPOTrainer("Traffic", config, config.traffic_gpu, "traffic")
        self.regional_trainer = MAGRPOTrainer("Regional", config, config.regional_gpu, "regional")
        
        # Progressive Training State Management
        self.current_training_mode = config.training_mode if config.enable_progressive_training else "online_only"
        self.global_training_phase = "warmup" if config.warmup_phase_enabled else "online"
        self.phase_start_time = time.time()
        self.phase_step_count = 0
        
        # Load historical data if available
        self._load_historical_training_data()
        
        # Statistics
        self.start_time = time.time()
        self.total_samples_processed = 0
        self.training_stats = {
            'traffic': {'total_steps': 0, 'total_loss': 0.0, 'last_loss': 0.0, 'phase_steps': 0, 'parameters_updated': False},
            'regional': {'total_steps': 0, 'total_loss': 0.0, 'last_loss': 0.0, 'phase_steps': 0, 'parameters_updated': False}
        }
        
        # Chart update tracking
        self.last_parameter_update = {'traffic': False, 'regional': False}
        self.chart_files = {
            'main': os.path.join(self.config.log_dir, "charts", "training_progress.png"),
            'detailed': os.path.join(self.config.log_dir, "charts", "detailed_metrics.png")
        }
        
        # Initialize persistent figure objects for real-time updates
        self.figures_initialized = False
        self.fig1 = None
        self.fig2 = None
        self.axes1 = None
        self.axes2 = None
        self.line_objects = {}  # Store line objects for efficient updates
        
        # Enhanced RL Tracking for Visualization
        self.simulation_step_size = 180  # From user specification
        self.total_autonomous_vehicles = None  # Will be set dynamically based on actual vehicle count
        self.total_simulation_steps = 43200  # Keep as fallback for compatibility
        self.visualization_update_interval = max(1, self.simulation_step_size // 10)  # Update every ~18 training steps
        
        # RL Metrics Accumulation with Historical Tracking
        self.cumulative_rewards = {
            'traffic': {'att_reward': 0.0, 'cooperation_reward': 0.0, 'total_reward': 0.0},
            'regional': {'efficiency_reward': 0.0, 'protection_reward': 0.0, 'cooperation_reward': 0.0, 'total_reward': 0.0}
        }
        
        # Historical data for line plots (time series)
        self.training_history = {
            'steps': [],
            'traffic_loss': [],
            'regional_loss': [],
            'traffic_reward': [],
            'regional_reward': [],
            'traffic_lr': [],
            'regional_lr': [],
            'att_improvement': [],
            'cooperation_quality': [],
            'phase_transitions': []
        }
        
        # Moving averages for smoother trends
        self.moving_avg_window = 10
        
        self.att_improvement_history = []
        self.cooperation_quality_history = []
        self.phase_transition_events = []
        
        # W&B Integration for Visualization
        self.wandb_enabled = self._initialize_wandb()
        
        self.logger.info("TRAINING_MANAGER_INIT: MAGRPO Training Manager with Progressive Mixed Training initialized")
        self.logger.info(f"PROGRESSIVE_TRAINING: Mode={self.current_training_mode}, Phase={self.global_training_phase}")
        self.logger.info(f"VISUALIZATION: W&B enabled={self.wandb_enabled}, Update interval={self.visualization_update_interval}")
        self.logger.info(f"SIMULATION_CONFIG: Step size={self.simulation_step_size}, Total steps={self.total_simulation_steps}, Auto vehicles=TBD")
        self.logger.info(f"TRAINING_CONFIG: Traffic group size: {config.traffic_group_size}, Regional group size: {config.regional_group_size}")
        self.logger.info(f"TRAINING_GPUS: Traffic GPU: {config.traffic_gpu}, Regional GPU: {config.regional_gpu}")
        self.logger.info(f"TRAINING_QUEUE: Queue available: {self.training_queue is not None}")
        
    def _initialize_wandb(self) -> bool:
        """Initialize W&B logging for RL visualization if available."""
        try:
            if WANDB_AVAILABLE and hasattr(self.config, 'enable_wandb') and getattr(self.config, 'enable_wandb', False):
                wandb.init(
                    project="magrpo-traffic-rl",
                    name=f"dual-llm-training-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config={
                        'traffic_group_size': self.config.traffic_group_size,
                        'regional_group_size': self.config.regional_group_size,
                        'training_mode': self.current_training_mode,
                        'simulation_step_size': self.simulation_step_size,
                        'total_steps': self.total_simulation_steps,
                        'lora_r': self.config.lora_r,
                        'learning_rate': self.config.learning_rate
                    },
                    tags=['progressive-training', 'dual-llm', 'magrpo']
                )
                self.logger.info("W&B_INIT: Weights & Biases logging initialized successfully")
                return True
            else:
                self.logger.info("W&B_SKIP: W&B not available or not enabled")
                return False
        except Exception as e:
            self.logger.warning(f"W&B_INIT_ERROR: Failed to initialize W&B: {e}")
            return False
    
    def _load_historical_training_data(self) -> Dict[str, int]:
        """Load historical training data for progressive training warmup phase."""
        try:
            historical_stats = {'traffic': 0, 'regional': 0}
            
            if self.config.enable_progressive_training:
                data_dir = self.config.historical_data_dir
                
                # Load historical data for both buffers
                traffic_loaded = self.traffic_buffer.load_historical_data(data_dir)
                regional_loaded = self.regional_buffer.load_historical_data(data_dir)
                
                historical_stats['traffic'] = traffic_loaded
                historical_stats['regional'] = regional_loaded
                
                if traffic_loaded > 0 or regional_loaded > 0:
                    self.logger.info(f"HISTORICAL_DATA_LOADED: Traffic={traffic_loaded}, Regional={regional_loaded}")
                else:
                    self.logger.info("HISTORICAL_DATA_EMPTY: No historical data found, starting fresh")
            
            return historical_stats
            
        except Exception as e:
            self.logger.error(f"HISTORICAL_DATA_ERROR: {e}")
            return {'traffic': 0, 'regional': 0}
    
    def set_total_autonomous_vehicles(self, total_vehicles: int):
        """Set the total number of autonomous vehicles for accurate progress calculation."""
        try:
            self.total_autonomous_vehicles = total_vehicles
            self.logger.info(f"AUTONOMOUS_VEHICLES_SET: Total autonomous vehicles updated to {total_vehicles}")
            
            # Update visualization settings based on vehicle count
            if total_vehicles > 0:
                # Adjust visualization interval based on vehicle count
                self.visualization_update_interval = max(1, total_vehicles // 50)  # Update every ~2% of vehicles
                self.logger.info(f"VISUALIZATION_INTERVAL: Updated to {self.visualization_update_interval} based on {total_vehicles} vehicles")
            
        except Exception as e:
            self.logger.error(f"SET_AUTONOMOUS_VEHICLES_ERROR: {e}")

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
    
    def _initialize_persistent_figures(self):
        """Initialize persistent matplotlib figures for real-time updates."""
        try:
            # Create charts directory
            charts_dir = os.path.join(self.config.log_dir, "charts")
            os.makedirs(charts_dir, exist_ok=True)
            
            # Set up the plotting style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # Enable interactive mode for real-time updates
            plt.ion()
            
            # Figure 1: Training Loss Convergence Analysis
            self.fig1, self.axes1 = plt.subplots(2, 2, figsize=(16, 12))
            self.fig1.suptitle('MAGRPO Training Loss & Convergence Analysis - Live Updates', 
                              fontsize=16, fontweight='bold')
            
            # Initialize empty plots and store line objects for efficient updates
            ax1, ax2, ax3, ax4 = self.axes1.flatten()
            
            # Plot 1: Loss History Trends
            self.line_objects['traffic_loss'], = ax1.plot([], [], 'b-', linewidth=2, label='Traffic LLM Loss', alpha=0.8)
            self.line_objects['regional_loss'], = ax1.plot([], [], 'r-', linewidth=2, label='Regional LLM Loss', alpha=0.8)
            self.line_objects['traffic_loss_ma'], = ax1.plot([], [], 'b--', linewidth=3, label='Traffic MA', alpha=0.6)
            self.line_objects['regional_loss_ma'], = ax1.plot([], [], 'r--', linewidth=3, label='Regional MA', alpha=0.6)
            
            ax1.set_title('Training Loss Convergence Analysis', fontweight='bold')
            ax1.set_xlabel('Training Steps')
            ax1.set_ylabel('Loss Value')
            ax1.set_yscale('log')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot 2: Reward Evolution
            self.line_objects['traffic_reward'], = ax2.plot([], [], 'g-', linewidth=2, label='Traffic LLM Reward', alpha=0.8)
            self.line_objects['regional_reward'], = ax2.plot([], [], 'm-', linewidth=2, label='Regional LLM Reward', alpha=0.8)
            self.line_objects['traffic_reward_fill'] = ax2.fill_between([], [], [], alpha=0.2, color='green', label='Traffic Reward Area')
            self.line_objects['regional_reward_fill'] = ax2.fill_between([], [], [], alpha=0.2, color='magenta', label='Regional Reward Area')
            
            ax2.set_title('Reward Evolution & Accumulation', fontweight='bold')
            ax2.set_xlabel('Training Steps')
            ax2.set_ylabel('Reward Value')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Plot 3: Learning Rate Scheduling
            self.line_objects['traffic_lr'], = ax3.plot([], [], 'orange', linewidth=2, label='Traffic LLM LR', alpha=0.8)
            self.line_objects['regional_lr'], = ax3.plot([], [], 'purple', linewidth=2, label='Regional LLM LR', alpha=0.8)
            
            ax3.set_title('Learning Rate Schedule', fontweight='bold')
            ax3.set_xlabel('Training Steps')
            ax3.set_ylabel('Learning Rate')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # Plot 4: Phase Transitions
            ax4.set_title('Training Phase Transitions Timeline', fontweight='bold')
            ax4.set_xlabel('Training Steps')
            ax4.set_ylabel('Phase Events')
            ax4.grid(True, alpha=0.3)
            self.line_objects['phase_lines'] = []  # Store phase transition lines
            
            plt.tight_layout()
            
            # Figure 2: Detailed RL Performance Metrics
            self.fig2, self.axes2 = plt.subplots(2, 2, figsize=(16, 12))
            self.fig2.suptitle('Detailed RL Performance & Convergence Metrics - Live Updates', 
                              fontsize=16, fontweight='bold')
            
            ax5, ax6, ax7, ax8 = self.axes2.flatten()
            
            # Plot 5: ATT Improvement & Cooperation Quality
            self.line_objects['att_improvement'], = ax5.plot([], [], 'b-', linewidth=2, 
                                                           label='ATT Improvement', alpha=0.8, marker='o', markersize=3)
            self.line_objects['cooperation_quality'], = ax5.plot([], [], 'g-', linewidth=2, 
                                                               label='Cooperation Quality', alpha=0.8, marker='s', markersize=3)
            
            ax5.set_title('ATT Improvement & Cooperation Quality Evolution', fontweight='bold')
            ax5.set_xlabel('Sample Points')
            ax5.set_ylabel('Quality Score')
            ax5.grid(True, alpha=0.3)
            ax5.legend()
            
            # Plot 6: Reward Components (will be updated with stackplot)
            ax6.set_title('Traffic LLM: Reward Components Evolution', fontweight='bold')
            ax6.set_xlabel('Training Steps')
            ax6.set_ylabel('Cumulative Reward')
            ax6.grid(True, alpha=0.3)
            
            # Plot 7: Training Throughput
            self.line_objects['sample_throughput'], = ax7.plot([], [], 'darkgreen', linewidth=2, 
                                                              label='Sample Throughput', alpha=0.8)
            
            ax7.set_title('Training Throughput & Sample Processing Rate', fontweight='bold')
            ax7.set_xlabel('Training Steps')
            ax7.set_ylabel('Samples Processed')
            ax7.grid(True, alpha=0.3)
            ax7.legend()
            
            # Plot 8: Model Convergence Indicators
            self.line_objects['traffic_loss_diff'], = ax8.plot([], [], 'red', linewidth=2, 
                                                              label='Traffic Loss Δ', alpha=0.8)
            self.line_objects['regional_loss_diff'], = ax8.plot([], [], 'blue', linewidth=2, 
                                                               label='Regional Loss Δ', alpha=0.8)
            ax8.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            ax8.set_title('Loss Convergence Indicators (Δ Loss)', fontweight='bold')
            ax8.set_xlabel('Training Steps')
            ax8.set_ylabel('Loss Change')
            ax8.grid(True, alpha=0.3)
            ax8.legend()
            
            plt.tight_layout()
            
            self.figures_initialized = True
            self.logger.info("CHARTS_INITIALIZED: Persistent figures initialized for real-time updates")
            
        except Exception as e:
            self.logger.error(f"CHART_INIT_ERROR: Failed to initialize persistent figures: {e}")
            self.figures_initialized = False

    def _update_persistent_charts(self):
        """Update persistent charts with new data for real-time visualization."""
        try:
            if not self.figures_initialized:
                self._initialize_persistent_figures()
                if not self.figures_initialized:
                    return None, None
            
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Only update if we have data
            if len(self.training_history['steps']) == 0:
                return self.chart_files['main'], self.chart_files['detailed']
            
            steps = self.training_history['steps']
            
            # Update Figure 1 - Training Analysis
            ax1, ax2, ax3, ax4 = self.axes1.flatten()
            
            # Update Plot 1: Loss History Trends
            if len(self.training_history['traffic_loss']) > 0:
                traffic_loss = self.training_history['traffic_loss']
                regional_loss = self.training_history['regional_loss']
                
                # Ensure steps and loss data have same length
                min_len = min(len(steps), len(traffic_loss), len(regional_loss))
                steps_sync = steps[:min_len]
                traffic_loss_sync = traffic_loss[:min_len]
                regional_loss_sync = regional_loss[:min_len]
                
                # Update main loss lines
                self.line_objects['traffic_loss'].set_data(steps_sync, traffic_loss_sync)
                self.line_objects['regional_loss'].set_data(steps_sync, regional_loss_sync)
                
                # Update moving averages if enough data
                if len(traffic_loss_sync) >= self.moving_avg_window:
                    traffic_ma = np.convolve(traffic_loss_sync, np.ones(self.moving_avg_window)/self.moving_avg_window, mode='valid')
                    regional_ma = np.convolve(regional_loss_sync, np.ones(self.moving_avg_window)/self.moving_avg_window, mode='valid')
                    ma_steps = steps_sync[self.moving_avg_window-1:]
                    
                    # Ensure ma_steps and ma arrays have same length
                    min_ma_len = min(len(ma_steps), len(traffic_ma), len(regional_ma))
                    ma_steps_sync = ma_steps[:min_ma_len]
                    traffic_ma_sync = traffic_ma[:min_ma_len]
                    regional_ma_sync = regional_ma[:min_ma_len]
                    
                    self.line_objects['traffic_loss_ma'].set_data(ma_steps_sync, traffic_ma_sync)
                    self.line_objects['regional_loss_ma'].set_data(ma_steps_sync, regional_ma_sync)
                
                # Auto-scale axes
                ax1.relim()
                ax1.autoscale_view()
            
            # Update Plot 2: Reward Evolution
            if len(self.training_history['traffic_reward']) > 0:
                traffic_reward = self.training_history['traffic_reward']
                regional_reward = self.training_history['regional_reward']
                
                # Ensure steps and reward data have same length
                min_len = min(len(steps), len(traffic_reward), len(regional_reward))
                steps_sync = steps[:min_len]
                traffic_reward_sync = traffic_reward[:min_len]
                regional_reward_sync = regional_reward[:min_len]
                
                self.line_objects['traffic_reward'].set_data(steps_sync, traffic_reward_sync)
                self.line_objects['regional_reward'].set_data(steps_sync, regional_reward_sync)
                
                # Remove old fill_between and add new ones
                self.line_objects['traffic_reward_fill'].remove()
                self.line_objects['regional_reward_fill'].remove()
                
                self.line_objects['traffic_reward_fill'] = ax2.fill_between(steps_sync, 0, traffic_reward_sync, 
                                                                           alpha=0.2, color='green', label='Traffic Reward Area')
                self.line_objects['regional_reward_fill'] = ax2.fill_between(steps_sync, 0, regional_reward_sync, 
                                                                            alpha=0.2, color='magenta', label='Regional Reward Area')
                
                ax2.relim()
                ax2.autoscale_view()
            
            # Update Plot 3: Learning Rate
            if len(self.training_history['traffic_lr']) > 0:
                traffic_lr = self.training_history['traffic_lr']
                regional_lr = self.training_history['regional_lr']
                
                # Ensure steps and lr data have same length
                min_len = min(len(steps), len(traffic_lr), len(regional_lr))
                steps_sync = steps[:min_len]
                traffic_lr_sync = traffic_lr[:min_len]
                regional_lr_sync = regional_lr[:min_len]
                
                self.line_objects['traffic_lr'].set_data(steps_sync, traffic_lr_sync)
                self.line_objects['regional_lr'].set_data(steps_sync, regional_lr_sync)
                
                ax3.relim()
                ax3.autoscale_view()
            
            # Update Plot 4: Phase Transitions
            if len(self.phase_transition_events) > 0:
                # Clear previous phase lines
                for line in self.line_objects['phase_lines']:
                    line.remove()
                self.line_objects['phase_lines'].clear()
                
                # Add current phase transitions
                phase_steps = [event['step'] for event in self.phase_transition_events]
                phase_names = [event['phase'] for event in self.phase_transition_events]
                
                unique_phases = list(set(phase_names))
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_phases)))
                phase_colors = {phase: colors[i] for i, phase in enumerate(unique_phases)}
                
                added_labels = set()
                for step, phase in zip(phase_steps, phase_names):
                    label = phase if phase not in added_labels else ""
                    if label:
                        added_labels.add(phase)
                    
                    line = ax4.axvline(x=step, color=phase_colors[phase], alpha=0.7, linewidth=3, label=label)
                    self.line_objects['phase_lines'].append(line)
                
                # Update current phase text
                ax4.text(0.02, 0.98, f'Current Phase: {self.global_training_phase}\nUpdated: {current_time}', 
                        transform=ax4.transAxes, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), fontsize=10)
                
                ax4.legend()
            
            # Update Figure 2 - Detailed Metrics  
            ax5, ax6, ax7, ax8 = self.axes2.flatten()
            
            # Update Plot 5: ATT & Cooperation
            if len(self.att_improvement_history) > 0:
                att_steps = list(range(len(self.att_improvement_history)))
                self.line_objects['att_improvement'].set_data(att_steps, self.att_improvement_history)
                ax5.relim()
                ax5.autoscale_view()
            
            if len(self.cooperation_quality_history) > 0:
                coop_steps = list(range(len(self.cooperation_quality_history)))
                self.line_objects['cooperation_quality'].set_data(coop_steps, self.cooperation_quality_history)
                ax5.relim()
                ax5.autoscale_view()
            
            # Update Plot 6: Reward Components (recreate stackplot)
            if len(self.training_history['steps']) > 0:
                ax6.clear()  # Clear and recreate stackplot
                
                # Use actual cumulative rewards instead of artificial scaling
                cumulative_att = []
                cumulative_coop = []
                running_att = 0.0
                running_coop = 0.0
                
                # Build cumulative reward series from traffic history
                for i, step in enumerate(steps):
                    if i < len(self.training_history['att_improvement']):
                        running_att += self.training_history['att_improvement'][i]
                    if i < len(self.training_history['cooperation_quality']):
                        # Get cooperation from regional history but show in traffic plot for context
                        if i < len(self.training_history['cooperation_quality']):
                            running_coop += self.training_history['cooperation_quality'][i] * 0.4  # Weight for traffic
                    
                    cumulative_att.append(running_att)
                    cumulative_coop.append(running_coop)
                
                if len(cumulative_att) > 0 and len(cumulative_coop) > 0:
                    min_len = min(len(steps), len(cumulative_att), len(cumulative_coop))
                    steps_sync = steps[:min_len]
                    cumulative_att_sync = cumulative_att[:min_len]
                    cumulative_coop_sync = cumulative_coop[:min_len]
                    
                    ax6.stackplot(steps_sync, cumulative_att_sync, cumulative_coop_sync, 
                                 labels=['ATT Reward', 'Cooperation Impact'],
                                 colors=['skyblue', 'lightcoral'], alpha=0.7)
                    ax6.set_title('Traffic LLM: Reward Components Evolution', fontweight='bold')
                    ax6.set_xlabel('Training Steps')
                    ax6.set_ylabel('Cumulative Reward')
                    ax6.grid(True, alpha=0.3)
                    ax6.legend(loc='upper left')
            
            # Update Plot 7: Sample Throughput
            if len(self.training_history['steps']) > 0:
                sample_throughput = [self.total_samples_processed * (i+1) / len(steps) 
                                   for i in range(len(steps))]
                # Ensure steps and sample_throughput have same length
                min_len = min(len(steps), len(sample_throughput))
                steps_sync = steps[:min_len]
                sample_throughput_sync = sample_throughput[:min_len]
                
                self.line_objects['sample_throughput'].set_data(steps_sync, sample_throughput_sync)
                ax7.relim()
                ax7.autoscale_view()
            
            # Update Plot 8: Loss Convergence
            if len(self.training_history['traffic_loss']) > 1:
                traffic_loss_diff = np.diff(self.training_history['traffic_loss'])
                regional_loss_diff = np.diff(self.training_history['regional_loss'])
                diff_steps = steps[1:]
                
                # Ensure diff_steps and loss_diff arrays have same length
                min_len = min(len(diff_steps), len(traffic_loss_diff), len(regional_loss_diff))
                diff_steps_sync = diff_steps[:min_len]
                traffic_loss_diff_sync = traffic_loss_diff[:min_len]
                regional_loss_diff_sync = regional_loss_diff[:min_len]
                
                self.line_objects['traffic_loss_diff'].set_data(diff_steps_sync, traffic_loss_diff_sync)
                self.line_objects['regional_loss_diff'].set_data(diff_steps_sync, regional_loss_diff_sync)
                
                # Update convergence status
                recent_traffic_trend = np.mean(traffic_loss_diff_sync[-5:]) if len(traffic_loss_diff_sync) >= 5 else 0
                recent_regional_trend = np.mean(regional_loss_diff_sync[-5:]) if len(regional_loss_diff_sync) >= 5 else 0
                
                convergence_status = "Converging" if abs(recent_traffic_trend) < 0.01 and abs(recent_regional_trend) < 0.01 else "Training"
                ax8.text(0.02, 0.98, f'Status: {convergence_status}\nUpdated: {current_time}', 
                        transform=ax8.transAxes, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8), fontsize=10)
                
                ax8.relim()
                ax8.autoscale_view()
            
            # Save updated figures to same files (overwrite)
            chart_path = self.chart_files['main']
            detailed_chart_path = self.chart_files['detailed']
            
            self.fig1.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            self.fig2.savefig(detailed_chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            
            # Force display update
            self.fig1.canvas.draw()
            self.fig2.canvas.draw()
            plt.pause(0.01)  # Small pause to allow GUI update
            
            self.logger.info(f"CHARTS_UPDATED: Real-time charts updated at {current_time}")
            
            return chart_path, detailed_chart_path
            
        except Exception as e:
            self.logger.error(f"REAL_TIME_CHART_ERROR: Failed to update persistent charts: {e}")
            return None, None

    def _generate_local_charts(self):
        """Generate or update local charts with real-time updates."""
        return self._update_persistent_charts()
    
    def _update_rl_metrics(self, trainer_type: str, metrics: Dict[str, float], 
                          training_samples: List[Dict[str, Any]] = None):
        """Update RL metrics for visualization and tracking using real sample data."""
        try:
            current_step = self.training_stats[trainer_type]['total_steps']
            
            # Update training history for line plots
            if current_step not in self.training_history['steps']:
                self.training_history['steps'].append(current_step)
            
            # Extract real reward data from training samples if available
            if training_samples and len(training_samples) > 0:
                # Calculate average rewards from the training batch
                total_samples = len(training_samples)
                
                if trainer_type == 'traffic':
                    att_rewards = []
                    coop_rewards = []
                    total_rewards = []
                    
                    for sample in training_samples:
                        rewards = sample.get('rewards', {})
                        traffic_rewards = rewards.get('traffic_llm', {})
                        att_rewards.append(traffic_rewards.get('att_reward', 0.0))
                        coop_rewards.append(traffic_rewards.get('cooperation_reward', 0.0))
                        total_rewards.append(traffic_rewards.get('total_reward', 0.0))
                    
                    # Use average rewards from actual training data
                    avg_att_reward = sum(att_rewards) / total_samples if att_rewards else 0.0
                    avg_coop_reward = sum(coop_rewards) / total_samples if coop_rewards else 0.0
                    avg_total_reward = sum(total_rewards) / total_samples if total_rewards else 0.0
                    
                    self.cumulative_rewards['traffic']['att_reward'] += avg_att_reward
                    self.cumulative_rewards['traffic']['cooperation_reward'] += avg_coop_reward
                    self.cumulative_rewards['traffic']['total_reward'] += avg_total_reward
                    
                    # Update historical data for line plots
                    self.training_history['traffic_loss'].append(metrics.get('loss', 0.0))
                    self.training_history['traffic_reward'].append(avg_total_reward)
                    self.training_history['traffic_lr'].append(metrics.get('learning_rate', 0.0))
                    self.training_history['att_improvement'].append(avg_att_reward)
                    
                    # Track ATT improvement history with real data
                    if len(self.att_improvement_history) == 0 or current_step % 5 == 0:
                        # Sample every 5 steps with real ATT data
                        self.att_improvement_history.append(avg_att_reward)
                    
                elif trainer_type == 'regional':
                    efficiency_rewards = []
                    protection_rewards = []
                    coop_rewards = []
                    total_rewards = []
                    
                    for sample in training_samples:
                        rewards = sample.get('rewards', {})
                        regional_rewards = rewards.get('regional_llm', {})
                        efficiency_rewards.append(regional_rewards.get('efficiency_reward', 0.0))
                        protection_rewards.append(regional_rewards.get('individual_protection_reward', 0.0))
                        coop_rewards.append(regional_rewards.get('cooperation_reward', 0.0))
                        total_rewards.append(regional_rewards.get('total_reward', 0.0))
                    
                    # Use average rewards from actual training data
                    avg_efficiency_reward = sum(efficiency_rewards) / total_samples if efficiency_rewards else 0.0
                    avg_protection_reward = sum(protection_rewards) / total_samples if protection_rewards else 0.0
                    avg_coop_reward = sum(coop_rewards) / total_samples if coop_rewards else 0.0
                    avg_total_reward = sum(total_rewards) / total_samples if total_rewards else 0.0
                    
                    self.cumulative_rewards['regional']['efficiency_reward'] += avg_efficiency_reward
                    self.cumulative_rewards['regional']['protection_reward'] += avg_protection_reward
                    self.cumulative_rewards['regional']['cooperation_reward'] += avg_coop_reward
                    self.cumulative_rewards['regional']['total_reward'] += avg_total_reward
                    
                    # Update historical data for line plots
                    self.training_history['regional_loss'].append(metrics.get('loss', 0.0))
                    self.training_history['regional_reward'].append(avg_total_reward)
                    self.training_history['regional_lr'].append(metrics.get('learning_rate', 0.0))
                    self.training_history['cooperation_quality'].append(avg_coop_reward)
                    
                    # Track cooperation quality history with real data
                    if len(self.cooperation_quality_history) == 0 or current_step % 5 == 0:
                        # Sample every 5 steps with real cooperation data
                        self.cooperation_quality_history.append(avg_coop_reward)
            
            else:
                # Fallback to relative reward mean when no training samples available
                if trainer_type == 'traffic':
                    relative_reward = metrics.get('relative_reward_mean', 0.0)
                    # Use weights consistent with new reward system: 0.6 * att_reward + 0.4 * cooperation_quality
                    estimated_att_reward = relative_reward * 0.6 / (0.6 + 0.4)  # Normalize by total weight
                    estimated_coop_reward = relative_reward * 0.4 / (0.6 + 0.4)
                    
                    self.cumulative_rewards['traffic']['att_reward'] += estimated_att_reward
                    self.cumulative_rewards['traffic']['cooperation_reward'] += estimated_coop_reward
                    self.cumulative_rewards['traffic']['total_reward'] += relative_reward
                    
                    self.training_history['traffic_loss'].append(metrics.get('loss', 0.0))
                    self.training_history['traffic_reward'].append(relative_reward)
                    self.training_history['traffic_lr'].append(metrics.get('learning_rate', 0.0))
                    self.training_history['att_improvement'].append(estimated_att_reward)
                    
                elif trainer_type == 'regional':
                    relative_reward = metrics.get('relative_reward_mean', 0.0)
                    # Use weights consistent with new reward system: 0.5 * efficiency + 0.2 * protection + 0.3 * cooperation
                    total_weight = 0.5 + 0.2 + 0.3
                    estimated_efficiency_reward = relative_reward * 0.5 / total_weight
                    estimated_protection_reward = relative_reward * 0.2 / total_weight
                    estimated_coop_reward = relative_reward * 0.3 / total_weight
                    
                    self.cumulative_rewards['regional']['efficiency_reward'] += estimated_efficiency_reward
                    self.cumulative_rewards['regional']['protection_reward'] += estimated_protection_reward
                    self.cumulative_rewards['regional']['cooperation_reward'] += estimated_coop_reward
                    self.cumulative_rewards['regional']['total_reward'] += relative_reward
                    
                    self.training_history['regional_loss'].append(metrics.get('loss', 0.0))
                    self.training_history['regional_reward'].append(relative_reward)
                    self.training_history['regional_lr'].append(metrics.get('learning_rate', 0.0))
                    self.training_history['cooperation_quality'].append(estimated_coop_reward)
            
            # Track phase transitions
            current_phase = metrics.get('training_phase', 'unknown')
            if (len(self.phase_transition_events) == 0 or 
                self.phase_transition_events[-1]['phase'] != current_phase):
                
                phase_event = {
                    'step': current_step,
                    'phase': current_phase,
                    'timestamp': time.time()
                }
                self.phase_transition_events.append(phase_event)
                self.training_history['phase_transitions'].append(phase_event)
            
            # Maintain reasonable history size (keep last 1000 points)
            max_history_size = 1000
            if len(self.training_history['steps']) > max_history_size:
                # Remove oldest entries to maintain size
                excess_count = len(self.training_history['steps']) - max_history_size
                for key in self.training_history:
                    if isinstance(self.training_history[key], list):
                        self.training_history[key] = self.training_history[key][excess_count:]
            
            # Log to W&B if available
            if self.wandb_enabled:
                wandb.log({
                    f"{trainer_type}/loss": metrics.get('loss', 0.0),
                    f"{trainer_type}/reward_mean": metrics.get('relative_reward_mean', 0.0),
                    f"{trainer_type}/reward_std": metrics.get('relative_reward_std', 0.0),
                    f"{trainer_type}/learning_rate": metrics.get('learning_rate', 0.0),
                    f"{trainer_type}/phase": current_phase,
                    f"cumulative/{trainer_type}_total_reward": self.cumulative_rewards[trainer_type]['total_reward'],
                    "global/total_samples": self.total_samples_processed,
                    "global/runtime": time.time() - self.start_time
                })
                
        except Exception as e:
            self.logger.error(f"RL_METRICS_UPDATE_ERROR: {e}")
    
    def run(self):
        """Enhanced main training loop with progressive training and visualization."""
        try:
            self.logger.info("TRAINING_MANAGER_START: Starting progressive MAGRPO training loop")
            last_heartbeat_time = time.time()
            last_chart_generation = time.time()
            heartbeat_interval = 60.0  # Log heartbeat every 60 seconds
            
            # Chart generation interval based on step size (every 18 steps ≈ 10% of step size)
            chart_interval = max(self.visualization_update_interval * 0.1, 30.0)  # At least every 30 seconds
            
            while self.running:
                # Process incoming training data
                self._process_incoming_data()
                
                # Progressive Training Logic for Traffic LLM
                if self._should_train_traffic_llm():
                    training_group = self._get_traffic_training_group()
                    if training_group:
                        # Use progressive training method if available
                        if hasattr(self.traffic_trainer, 'train_step_progressive'):
                            batch_multiplier = self._get_batch_multiplier('traffic')
                            metrics = self.traffic_trainer.train_step_progressive(training_group, batch_multiplier)
                        else:
                            metrics = self.traffic_trainer.train_step(training_group)
                        
                        self._update_training_stats('traffic', metrics, training_group)
                        
                        # Update phase in buffer
                        self.traffic_buffer.set_training_phase(metrics.get('training_phase', 'online'))
                
                # Progressive Training Logic for Regional LLM  
                if self._should_train_regional_llm():
                    training_group = self._get_regional_training_group()
                    if training_group:
                        # Use progressive training method if available
                        if hasattr(self.regional_trainer, 'train_step_progressive'):
                            batch_multiplier = self._get_batch_multiplier('regional')
                            metrics = self.regional_trainer.train_step_progressive(training_group, batch_multiplier)
                        else:
                            metrics = self.regional_trainer.train_step(training_group)
                        
                        self._update_training_stats('regional', metrics, training_group)
                        
                        # Update phase in buffer
                        self.regional_buffer.set_training_phase(metrics.get('training_phase', 'online'))
                
                # Periodic visualization and status updates
                current_time = time.time()
                
                # Generate local charts only when parameters were updated
                parameter_updated = (self.last_parameter_update['traffic'] or self.last_parameter_update['regional'])
                
                if parameter_updated and (current_time - last_chart_generation >= chart_interval):
                    
                    self._generate_local_charts()
                    last_chart_generation = current_time
                    
                    # Reset parameter update flags after chart generation
                    self.last_parameter_update = {'traffic': False, 'regional': False}
                    self.logger.info("CHART_UPDATE_TRIGGERED: Charts updated due to parameter changes")
                    
                    # Log progressive training status
                    self._log_progressive_training_status()
                
                # Periodic status logging
                if self.total_samples_processed > 0 and self.total_samples_processed % 100 == 0:
                    self._log_training_status()
                
                # Heartbeat logging for debugging  
                if current_time - last_heartbeat_time >= heartbeat_interval:
                    runtime = current_time - self.start_time
                    
                    # Calculate progress based on autonomous vehicles if available, otherwise use simulation steps
                    if self.total_autonomous_vehicles and self.total_autonomous_vehicles > 0:
                        progress = min(100.0, (self.total_samples_processed / self.total_autonomous_vehicles) * 100)
                        progress_text = f"{self.total_samples_processed}/{self.total_autonomous_vehicles}"
                    else:
                        progress = min(100.0, (self.total_samples_processed / self.total_simulation_steps) * 100)
                        progress_text = f"{self.total_samples_processed}/{self.total_simulation_steps}"
                    
                    self.logger.info(f"PROGRESSIVE_HEARTBEAT: Runtime: {runtime:.1f}s, "
                                   f"Progress: {progress:.1f}% ({progress_text}), "
                                   f"Traffic buffer: {self.traffic_buffer.size()}, "
                                   f"Regional buffer: {self.regional_buffer.size()}, "
                                   f"Global phase: {self.global_training_phase}")
                    last_heartbeat_time = current_time
                
                # Process pending adapter queue (try to load previously failed adapters)
                self._process_trainer_adapter_queues()
                
                # Phase transition management
                self._manage_global_phase_transitions()
                
                # Short sleep to prevent CPU spinning
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            self.logger.info("TRAINING_MANAGER_INTERRUPT: Received interrupt signal")
        except Exception as e:
            self.logger.error(f"TRAINING_MANAGER_ERROR: {e}")
        finally:
            self._shutdown()
    
    def _should_train_traffic_llm(self) -> bool:
        """Determine if Traffic LLM should train based on progressive training phases."""
        try:
            # Check if buffer has enough samples
            if not self.traffic_buffer.can_form_group():
                return False
            
            # Progressive training logic
            if self.current_training_mode == "progressive":
                if self.global_training_phase == "warmup":
                    # In warmup phase, use historical data if available, fallback to current buffer
                    historical_samples = len(self.traffic_buffer.historical_buffer)
                    current_samples = len(self.traffic_buffer.buffer)
                    
                    # Prefer historical data, but use current buffer if no historical data available
                    if historical_samples >= self.config.traffic_group_size:
                        return True
                    elif historical_samples == 0 and current_samples >= self.config.traffic_group_size:
                        # No historical data available, use current buffer as fallback
                        self.logger.info(f"WARMUP_FALLBACK: Using current buffer for Traffic LLM training (no historical data)")
                        return True
                    else:
                        return False
                elif self.global_training_phase == "reinforcement":
                    # In reinforcement phase, require high-quality samples or regular samples
                    return (len(self.traffic_buffer.high_quality_samples) > 0 or 
                           self.traffic_buffer.can_form_group())
                else:  # online phase
                    return True
            else:
                # Default online-only mode
                return True
                
        except Exception as e:
            self.logger.error(f"TRAFFIC_TRAIN_CHECK_ERROR: {e}")
            return False
    
    def _should_train_regional_llm(self) -> bool:
        """Determine if Regional LLM should train based on progressive training phases."""
        try:
            # Check if buffer has enough samples
            if not self.regional_buffer.can_form_group():
                return False
            
            # Progressive training logic (same as traffic but with regional buffer)
            if self.current_training_mode == "progressive":
                if self.global_training_phase == "warmup":
                    # In warmup phase, use historical data if available, fallback to current buffer
                    historical_samples = len(self.regional_buffer.historical_buffer)
                    current_samples = len(self.regional_buffer.buffer)
                    
                    # Prefer historical data, but use current buffer if no historical data available
                    if historical_samples >= self.config.regional_group_size:
                        return True
                    elif historical_samples == 0 and current_samples >= self.config.regional_group_size:
                        # No historical data available, use current buffer as fallback
                        self.logger.info(f"WARMUP_FALLBACK: Using current buffer for Regional LLM training (no historical data)")
                        return True
                    else:
                        return False
                elif self.global_training_phase == "reinforcement":
                    return (len(self.regional_buffer.high_quality_samples) > 0 or 
                           self.regional_buffer.can_form_group())
                else:  # online phase
                    return True
            else:
                return True
                
        except Exception as e:
            self.logger.error(f"REGIONAL_TRAIN_CHECK_ERROR: {e}")
            return False
    
    def _get_traffic_training_group(self) -> Optional[List[Dict[str, Any]]]:
        """Get training group for Traffic LLM based on current training phase."""
        try:
            if self.global_training_phase == "warmup":
                return self.traffic_buffer.get_offline_training_group()
            elif self.global_training_phase == "reinforcement":
                batch_multiplier = self.config.reinforcement_batch_multiplier
                return self.traffic_buffer.get_reinforcement_batch(batch_multiplier)
            else:  # online phase
                return self.traffic_buffer.get_training_group()
                
        except Exception as e:
            self.logger.error(f"TRAFFIC_GROUP_GET_ERROR: {e}")
            return None
    
    def _get_regional_training_group(self) -> Optional[List[Dict[str, Any]]]:
        """Get training group for Regional LLM based on current training phase."""
        try:
            if self.global_training_phase == "warmup":
                return self.regional_buffer.get_offline_training_group()
            elif self.global_training_phase == "reinforcement":
                batch_multiplier = self.config.reinforcement_batch_multiplier
                return self.regional_buffer.get_reinforcement_batch(batch_multiplier)
            else:  # online phase
                return self.regional_buffer.get_training_group()
                
        except Exception as e:
            self.logger.error(f"REGIONAL_GROUP_GET_ERROR: {e}")
            return None
    
    def _get_batch_multiplier(self, trainer_type: str) -> int:
        """Get batch multiplier for reinforcement training phases."""
        try:
            if self.global_training_phase == "reinforcement":
                return self.config.reinforcement_batch_multiplier
            else:
                return 1
                
        except Exception as e:
            self.logger.error(f"BATCH_MULTIPLIER_ERROR: {e}")
            return 1
    
    def _manage_global_phase_transitions(self):
        """Manage global training phase transitions across both LLMs."""
        try:
            if not self.config.enable_progressive_training:
                return
            
            current_time = time.time()
            phase_duration = current_time - self.phase_start_time
            
            # Transition logic based on steps and time
            total_training_steps = (self.training_stats['traffic']['total_steps'] + 
                                  self.training_stats['regional']['total_steps'])
            
            should_transition = False
            new_phase = self.global_training_phase
            
            if self.global_training_phase == "warmup":
                # Transition to online after offline pretraining steps
                if total_training_steps >= self.config.offline_pretraining_steps:
                    should_transition = True
                    new_phase = "online"
                    
            elif self.global_training_phase == "online":
                # Transition to reinforcement after online steps
                if self.phase_step_count >= self.config.online_steps_per_reinforcement:
                    should_transition = True
                    new_phase = "reinforcement"
                    
            elif self.global_training_phase == "reinforcement":
                # Transition back to online after reinforcement (cyclical)
                reinforcement_steps = self.config.offline_pretraining_steps // 2  # Shorter reinforcement phases
                if self.phase_step_count >= reinforcement_steps:
                    should_transition = True
                    new_phase = "online"
            
            if should_transition:
                old_phase = self.global_training_phase
                self.global_training_phase = new_phase
                self.phase_start_time = current_time
                self.phase_step_count = 0
                
                # Update buffers
                self.traffic_buffer.set_training_phase(new_phase)
                self.regional_buffer.set_training_phase(new_phase)
                
                # Log transition
                self.logger.info(f"GLOBAL_PHASE_TRANSITION: {old_phase} -> {new_phase} "
                               f"after {total_training_steps} total steps, "
                               f"phase duration: {phase_duration:.1f}s")
                
                # Save historical data during phase transitions
                if new_phase == "reinforcement":
                    self._save_phase_historical_data()
                
        except Exception as e:
            self.logger.error(f"GLOBAL_PHASE_TRANSITION_ERROR: {e}")
    
    def _save_phase_historical_data(self):
        """Save current training data as historical data for future phases."""
        try:
            if self.config.enable_progressive_training:
                traffic_saved = self.traffic_buffer.save_historical_data(self.config.historical_data_dir)
                regional_saved = self.regional_buffer.save_historical_data(self.config.historical_data_dir)
                
                self.logger.info(f"HISTORICAL_DATA_SAVED: Traffic={traffic_saved}, Regional={regional_saved}")
                
        except Exception as e:
            self.logger.error(f"HISTORICAL_DATA_SAVE_ERROR: {e}")

    def _process_trainer_adapter_queues(self):
        """Process pending adapters for both Traffic and Regional LLM trainers."""
        try:
            # Process pending adapters for Traffic LLM
            if hasattr(self.traffic_trainer, '_process_pending_adapters'):
                self.traffic_trainer._process_pending_adapters()
                
            # Process pending adapters for Regional LLM  
            if hasattr(self.regional_trainer, '_process_pending_adapters'):
                self.regional_trainer._process_pending_adapters()
                
            # Log queue statistics periodically (every 5 minutes)
            if not hasattr(self, 'last_queue_stats_log'):
                self.last_queue_stats_log = 0
                
            current_time = time.time()
            if current_time - self.last_queue_stats_log > 300:  # 5 minutes
                self.last_queue_stats_log = current_time
                self._log_adapter_queue_statistics()
                
        except Exception as e:
            self.logger.error(f"ADAPTER_QUEUE_PROCESSING_ERROR: {e}")

    def _log_adapter_queue_statistics(self):
        """Log comprehensive statistics about adapter queues."""
        try:
            # Get stats from both trainers
            traffic_stats = getattr(self.traffic_trainer, 'get_adapter_queue_stats', lambda: {})()
            regional_stats = getattr(self.regional_trainer, 'get_adapter_queue_stats', lambda: {})()
            
            # Calculate total queue metrics
            total_queued = traffic_stats.get('queue_size', 0) + regional_stats.get('queue_size', 0)
            total_pending = traffic_stats.get('total_pending', 0) + regional_stats.get('total_pending', 0)
            total_failed = traffic_stats.get('total_failed', 0) + regional_stats.get('total_failed', 0)
            
            if total_queued > 0:
                self.logger.info(f"ADAPTER_QUEUE_STATS: Total queued={total_queued}, "
                               f"Pending={total_pending}, Failed={total_failed}")
                self.logger.info(f"  Traffic LLM: queued={traffic_stats.get('queue_size', 0)}, "
                               f"pending={traffic_stats.get('total_pending', 0)}, "
                               f"failed={traffic_stats.get('total_failed', 0)}")
                self.logger.info(f"  Regional LLM: queued={regional_stats.get('queue_size', 0)}, "
                               f"pending={regional_stats.get('total_pending', 0)}, "
                               f"failed={regional_stats.get('total_failed', 0)}")
            else:
                self.logger.debug("ADAPTER_QUEUE_EMPTY: No adapters in queue")
                
        except Exception as e:
            self.logger.error(f"QUEUE_STATS_LOG_ERROR: {e}")
    
    def _log_progressive_training_status(self):
        """Log detailed progressive training status with RL metrics."""
        try:
            # Get progressive training stats from trainers
            traffic_progressive_stats = self.traffic_trainer.get_progressive_stats()
            regional_progressive_stats = self.regional_trainer.get_progressive_stats()
            
            # Get buffer statistics
            traffic_stats = self.traffic_buffer.get_stats()
            regional_stats = self.regional_buffer.get_stats()
            
            runtime = time.time() - self.start_time
            
            # Calculate progress based on autonomous vehicles if available
            if self.total_autonomous_vehicles and self.total_autonomous_vehicles > 0:
                progress = min(100.0, (self.total_samples_processed / self.total_autonomous_vehicles) * 100)
            else:
                progress = min(100.0, (self.total_samples_processed / self.total_simulation_steps) * 100)
            
            self.logger.info(f"PROGRESSIVE_STATUS: Runtime: {runtime:.1f}s, Progress: {progress:.1f}%")
            self.logger.info(f"  Global Phase: {self.global_training_phase}, Phase Steps: {self.phase_step_count}")
            self.logger.info(f"  Traffic LLM: Phase={traffic_progressive_stats.get('current_phase', 'unknown')}, "
                           f"Online Steps={traffic_progressive_stats.get('online_steps_count', 0)}, "
                           f"LR={traffic_progressive_stats.get('current_learning_rate', 0.0):.2e}")
            self.logger.info(f"  Regional LLM: Phase={regional_progressive_stats.get('current_phase', 'unknown')}, "
                           f"Online Steps={regional_progressive_stats.get('online_steps_count', 0)}, "
                           f"LR={regional_progressive_stats.get('current_learning_rate', 0.0):.2e}")
            self.logger.info(f"  Buffers: Traffic={traffic_stats['current_size']}/{traffic_stats['high_quality_size']} "
                           f"(HQ), Regional={regional_stats['current_size']}/{regional_stats['high_quality_size']} (HQ)")
            
            # RL Metrics Summary
            traffic_cumulative = self.cumulative_rewards['traffic']['total_reward']
            regional_cumulative = self.cumulative_rewards['regional']['total_reward']
            
            self.logger.info(f"  RL Cumulative Rewards: Traffic={traffic_cumulative:.3f}, Regional={regional_cumulative:.3f}")
            self.logger.info(f"  ATT Improvement Samples: {len(self.att_improvement_history)}, "
                           f"Cooperation Quality Samples: {len(self.cooperation_quality_history)}")
            
        except Exception as e:
            self.logger.error(f"PROGRESSIVE_STATUS_LOG_ERROR: {e}")

    def _process_incoming_data(self):
        """Process incoming training data from the queue."""
        try:
            data_received_this_cycle = 0
            control_messages_received = 0
            while True:
                try:
                    # Non-blocking get with timeout
                    sample = self.training_queue.get(timeout=0.1)
                    
                    # Check if this is a control message
                    if isinstance(sample, dict) and sample.get('message_type') == 'autonomous_vehicle_count':
                        # Handle autonomous vehicle count message
                        total_autonomous = sample.get('total_autonomous_vehicles', 0)
                        total_vehicles = sample.get('total_vehicles', 0)
                        
                        self.set_total_autonomous_vehicles(total_autonomous)
                        control_messages_received += 1
                        
                        self.logger.info(f"CONTROL_MESSAGE: Received autonomous vehicle count - {total_autonomous}/{total_vehicles}")
                        continue
                    
                    # Regular training sample processing
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
            if data_received_this_cycle > 0 or control_messages_received > 0:
                self.logger.info(f"TRAINING_DATA_CYCLE: Processed {data_received_this_cycle} samples and {control_messages_received} control messages this cycle")
                    
        except Exception as e:
            self.logger.error(f"TRAINING_DATA_PROCESSING_ERROR: {e}")
    
    def _update_training_stats(self, trainer_type: str, metrics: Dict[str, float], training_samples: List[Dict[str, Any]] = None):
        """Update training statistics and RL metrics."""
        try:
            # Update basic training statistics
            stats = self.training_stats[trainer_type]
            stats['total_steps'] += 1
            stats['total_loss'] += metrics.get('loss', 0.0)
            stats['last_loss'] = metrics.get('loss', 0.0)
            stats['phase_steps'] += 1
            
            # Check if parameters were actually updated (loss > 0 indicates training step completed)
            loss_value = metrics.get('loss', 0.0)
            if loss_value > 0:
                stats['parameters_updated'] = True
                self.last_parameter_update[trainer_type] = True
                self.logger.debug(f"PARAM_UPDATE_DETECTED: {trainer_type} parameters updated, loss={loss_value:.4f}")
            
            # Update global phase step count
            self.phase_step_count += 1
            
            # Update RL metrics for visualization using real sample data
            self._update_rl_metrics(trainer_type, metrics, training_samples)
            
        except Exception as e:
            self.logger.error(f"TRAINING_STATS_UPDATE_ERROR: {e}")
    
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
            
            # Close persistent figures to free memory
            if self.figures_initialized:
                try:
                    plt.close(self.fig1)
                    plt.close(self.fig2)
                    plt.ioff()  # Turn off interactive mode
                    self.logger.info("CHARTS_CLOSED: Persistent figures closed successfully")
                except Exception as fig_error:
                    self.logger.error(f"CHART_CLOSE_ERROR: {fig_error}")
            
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