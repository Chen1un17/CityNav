#!/usr/bin/env python3
"""
Multi-Agent Traffic Control System Runner

This script provides easy ways to run the multi-agent traffic control system
with different configurations and parameters.

Default LLM: Local Qwen Model (/home/apulis-dev/userdata/Qwen)
- Uses local vLLM for high-performance inference
- Shared LLM architecture: Traffic LLM (GPU 2) + Regional LLM (GPU 3)
- Optimized for local deployment with multiple GPUs
- Fallback support for API mode when specified
"""

import argparse
import os
import sys
from datetime import datetime
os.environ["WANDB_MODE"] = "offline"
# os.environ["VLLM_USE_V1"] = "0"
from main import main


def check_local_model():
    """Check if local model is available."""
    model_path = "/home/apulis-dev/userdata/Qwen"
    if not os.path.exists(model_path):
        print(f"[错误] 本地模型路径不存在: {model_path}")
        print("请确保模型文件已正确放置")
        return False
    
    print(f"[成功] 本地模型路径已确认: {model_path}")
    
    # 检查GPU
    try:
        import torch
        if not torch.cuda.is_available():
            print("[警告] CUDA不可用，本地LLM性能可能受限")
            return False
        
        gpu_count = torch.cuda.device_count()
        print(f"[成功] 检测到 {gpu_count} 个GPU")
        
        if gpu_count >= 4:
            print("[推荐] GPU 2(Traffic LLM) + GPU 3(Regional LLM)")
        elif gpu_count >= 2:
            print("[警告] GPU数量不足4个，可能需要调整GPU分配")
        elif gpu_count == 1:
            print("[警告] 单GPU模式: Traffic和Regional LLM将轮流使用")
        
        return True
    except ImportError:
        print("[错误] PyTorch未安装，无法检查GPU状态")
        return False

def check_qwen_api_key():
    """Check if qwen API key is properly configured (for API mode)."""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("[错误] 未设置DASHSCOPE_API_KEY环境变量")
        print("请设置通义千问API密钥:")
        print("  export DASHSCOPE_API_KEY='sk-your-api-key'")
        return False
    
    if not api_key.startswith("sk-"):
        print("[错误] API密钥格式不正确，应该以'sk-'开头")
        return False
    
    print(f"[成功] 通义千问API密钥已配置: {api_key[:10]}...")
    return True


def run_quick_test():
    """Run a quick test with minimal settings."""
    print("Running Quick Test - Multi-Agent Traffic Control (Local LLM)")
    print("="*60)
    
    if not check_local_model():
        print("本地模型不可用，请检查模型路径和GPU状态")
        sys.exit(1)
    
    # Quick test parameters - 使用本地模型
    llm_path = "/home/apulis-dev/userdata/Qwen"
    batch_size = 8
    location = "NewYork"
    step_size = 60.0  # Faster steps for testing
    max_steps = 1800  # 30 minutes simulation
    use_local_llm = True  # 启用本地LLM模式
    enable_training = False  # 快速测试默认不启用训练
    
    main(llm_path, batch_size, location, use_reflection=True, 
         step_size=step_size, max_steps=max_steps, multi_agent=True, use_local_llm=use_local_llm, enable_training=enable_training)


def run_full_simulation():
    """Run a complete simulation with full settings."""
    print("Running Full Simulation - Multi-Agent Traffic Control (Local LLM)")
    print("="*60)
    
    if not check_local_model():
        print("本地模型不可用，请检查模型路径和GPU状态")
        sys.exit(1)
    
    # Full simulation parameters - 使用本地模型
    llm_path = "/home/apulis-dev/userdata/Qwen"
    batch_size = 8
    location = "NewYork"
    step_size = 180.0  # 3-minute decision intervals
    max_steps = 43200  # 12 hours simulation
    use_local_llm = True  # 启用本地LLM模式
    enable_training = False  # 完整仿真默认不启用训练
    
    main(llm_path, batch_size, location, use_reflection=True, 
         step_size=step_size, max_steps=max_steps, multi_agent=True, use_local_llm=use_local_llm, enable_training=enable_training)


def run_comparison():
    """Run both single-agent and multi-agent for comparison."""
    print("Running Comparison: Single-Agent vs Multi-Agent (Local LLM)")
    print("="*60)
    
    if not check_local_model():
        print("本地模型不可用，请检查模型路径和GPU状态")
        sys.exit(1)
    
    # Common parameters - 使用本地模型
    llm_path = "/home/apulis-dev/userdata/Qwen"
    batch_size = 8
    location = "NewYork"
    step_size = 180.0
    max_steps = 21600  # 6 hours simulation
    use_local_llm = True  # 启用本地LLM模式
    
    print("\n" + "="*40)
    print("RUNNING SINGLE-AGENT BASELINE (LOCAL LLM)")
    print("="*40)
    
    main(llm_path, batch_size, location, use_reflection=True, 
         step_size=step_size, max_steps=max_steps, multi_agent=False, use_local_llm=use_local_llm, enable_training=False)
    
    print("\n" + "="*40)
    print("RUNNING MULTI-AGENT SYSTEM (LOCAL LLM)")
    print("="*40)
    
    main(llm_path, batch_size, location, use_reflection=True, 
         step_size=step_size, max_steps=max_steps, multi_agent=True, use_local_llm=use_local_llm, enable_training=False)


def run_benchmark():
    """Run benchmark with different configurations."""
    print("Running Benchmark Suite (Local LLM)")
    print("="*60)
    
    if not check_local_model():
        print("本地模型不可用，请检查模型路径和GPU状态")
        sys.exit(1)
    
    configurations = [
        # (reflection, step_size, max_steps, description)
        (True, 180.0, 10800, "Full Features - 3 hours"),
        (False, 180.0, 10800, "No Reflection - 3 hours"),
        (True, 120.0, 7200, "Fast Decisions - 2 hours"),
        (True, 300.0, 14400, "Slow Decisions - 4 hours"),
    ]
    
    llm_path = "/home/apulis-dev/userdata/Qwen"  # 使用本地模型
    batch_size = 8
    location = "NewYork"
    use_local_llm = True  # 启用本地LLM模式
    
    for i, (reflection, step_size, max_steps, description) in enumerate(configurations):
        print(f"\n{'='*50}")
        print(f"BENCHMARK {i+1}/4: {description} (Local LLM)")
        print(f"{'='*50}")
        
        main(llm_path, batch_size, location, use_reflection=reflection, 
             step_size=step_size, max_steps=max_steps, multi_agent=True, use_local_llm=use_local_llm, enable_training=False)


def run_with_training():
    """Run multi-agent system with MAGRPO online training enabled."""
    print("Running Multi-Agent Traffic Control with MAGRPO Online Training")
    print("="*60)
    
    if not check_local_model():
        print("本地模型不可用，请检查模型路径和GPU状态")
        sys.exit(1)
    
    # Training parameters - optimized for memory and stability
    llm_path = "/home/apulis-dev/userdata/Qwen"
    batch_size = 4  # Reduced batch for training mode to save memory
    location = "Manhattan"
    step_size = 180.0  # 3-minute decision intervals
    max_steps = 86400.00  # 3 hours simulation (sufficient for training data collection)
    use_local_llm = True  # 必须使用本地LLM
    enable_training = True  # 启用训练
    start_time = 20880  # 从timestamp 20880开始仿真
    av_ratio = 0.015  # 从第一个路由文件中抽取1.5%的车辆作为自动驾驶车辆
    
    print(f"MAGRPO训练配置:")
    print(f"  - 模型路径: {llm_path}")
    print(f"  - 批处理大小: {batch_size} (为训练优化)")
    print(f"  - 仿真步长: {step_size}s")
    print(f"  - 起始时间: {start_time}s (timestamp)")
    print(f"  - 最大步数: {max_steps} (6小时)")
    print(f"  - 自动驾驶车辆比例: {av_ratio * 100}% (仅从第一个路由文件)")
    print(f"  - Traffic LLM训练GPU: cuda:2")
    print(f"  - Regional LLM训练GPU: cuda:3")
    print(f"  - 训练组大小: Traffic=8, Regional=12")
    print()
    
    print("注意:")
    print("- 训练进程将在独立进程中运行，使用GPU 2和3")
    print("- 推理仍使用GPU 0和1")
    print("- 训练数据通过队列实时传输")
    print("- 训练日志将保存在 logs/training_Manhattan/ 目录")
    print("- 模型检查点将每100步保存一次")
    print(f"- 仿真从 {start_time}s 开始，只使用第一个路由文件的 {av_ratio*100}% 车辆作为AV")
    print()
    
    main(llm_path, batch_size, location, use_reflection=True, 
         step_size=step_size, max_steps=max_steps, multi_agent=True, use_local_llm=use_local_llm, 
         enable_training=enable_training, start_time=start_time, av_ratio=av_ratio)


def run_chicago_with_lora():
    """Run Chicago experiment with pre-trained LoRA adapters."""
    print("Running Chicago Experiment with Pre-trained LoRA Adapters")
    print("="*60)
    
    if not check_local_model():
        print("本地模型不可用，请检查模型路径和GPU状态")
        sys.exit(1)
    
    # Chicago experiment parameters
    llm_path = "/home/apulis-dev/userdata/Qwen"
    batch_size = 8
    location = "Chicago"
    step_size = 180.0  # 3-minute decision intervals
    max_steps = 43200  # 12 hours simulation
    use_local_llm = True  # 使用本地LLM
    enable_training = False  # 不进行训练，只进行推理
    start_time = 0
    av_ratio = 0.02
    
    # LoRA adapter paths
    traffic_lora_path = "/home/apulis-dev/code/LLMNavigation/logs/training_Manhattan/lora_adapters/traffic_adapter_step_120"
    regional_lora_path = "/home/apulis-dev/code/LLMNavigation/logs/training_Manhattan/lora_adapters/regional_adapter_step_100"
    
    print(f"Chicago实验配置:")
    print(f"  - 模型路径: {llm_path}")
    print(f"  - 批处理大小: {batch_size}")
    print(f"  - 仿真步长: {step_size}s")
    print(f"  - 最大步数: {max_steps} (12小时)")
    print(f"  - 自动驾驶车辆比例: {av_ratio * 100}%")
    print(f"  - Traffic LoRA: {traffic_lora_path}")
    print(f"  - Regional LoRA: {regional_lora_path}")
    print()
    
    # Check if LoRA adapters exist
    if not os.path.exists(traffic_lora_path):
        print(f"[警告] Traffic LoRA adapter不存在: {traffic_lora_path}")
    else:
        print(f"[成功] Traffic LoRA adapter已确认")
    
    if not os.path.exists(regional_lora_path):
        print(f"[警告] Regional LoRA adapter不存在: {regional_lora_path}")
    else:
        print(f"[成功] Regional LoRA adapter已确认")
    
    print()
    print("注意:")
    print("- 使用Chicago路网和分区数据")
    print("- 仅进行推理，不进行训练")
    print("- 加载预训练的Manhattan LoRA adapters")
    print()
    
    main(llm_path, batch_size, location, use_reflection=True, 
         step_size=step_size, max_steps=max_steps, multi_agent=True, use_local_llm=use_local_llm, 
         enable_training=enable_training, start_time=start_time, av_ratio=av_ratio,
         traffic_lora_path=traffic_lora_path, regional_lora_path=regional_lora_path)


def run_manhattan_region1_with_lora():
    """Run Manhattan Region_1 experiment with pre-trained LoRA adapters."""
    print("Running Manhattan Region_1 Experiment with Pre-trained LoRA Adapters")
    print("="*60)
    
    if not check_local_model():
        print("本地模型不可用，请检查模型路径和GPU状态")
        sys.exit(1)
    
    # Manhattan Region_1 experiment parameters
    llm_path = "/home/apulis-dev/userdata/Qwen"  # qwen3-8b 模型路径
    batch_size = 8
    location = "Manhattan_Region1"  # 使用新的 location 配置
    step_size = 180.0  # 3-minute decision intervals
    max_steps = 43200  # 12 hours simulation
    use_local_llm = True  # 使用本地LLM
    enable_training = False  # 不进行训练，只进行推理
    start_time = 0
    av_ratio = 0.02
    
    # LoRA adapter paths
    traffic_lora_path = "/home/apulis-dev/code/LLMNavigation/logs/training_Manhattan/lora_adapters/traffic_adapter_step_120"
    regional_lora_path = "/home/apulis-dev/code/LLMNavigation/logs/training_Manhattan/lora_adapters/regional_adapter_step_100"
    
    print(f"Manhattan Region_1 实验配置:")
    print(f"  - 模型路径: {llm_path}")
    print(f"  - 批处理大小: {batch_size}")
    print(f"  - 仿真步长: {step_size}s")
    print(f"  - 最大步数: {max_steps} (12小时)")
    print(f"  - 自动驾驶车辆比例: {av_ratio * 100}%")
    print(f"  - SUMO配置: /home/apulis-dev/userdata/Region_1/Manhattan_sumo_config.sumocfg")
    print(f"  - 分区文件夹: /home/apulis-dev/userdata/Region_1/regions")
    print(f"  - Traffic LoRA: {traffic_lora_path}")
    print(f"  - Regional LoRA: {regional_lora_path}")
    print()
    
    # Check if LoRA adapters exist
    if not os.path.exists(traffic_lora_path):
        print(f"[警告] Traffic LoRA adapter不存在: {traffic_lora_path}")
    else:
        print(f"[成功] Traffic LoRA adapter已确认")
    
    if not os.path.exists(regional_lora_path):
        print(f"[警告] Regional LoRA adapter不存在: {regional_lora_path}")
    else:
        print(f"[成功] Regional LoRA adapter已确认")
    
    # Check if SUMO config exists
    sumo_config = "/home/apulis-dev/userdata/Region_1/Manhattan_sumo_config.sumocfg"
    if not os.path.exists(sumo_config):
        print(f"[警告] SUMO配置文件不存在: {sumo_config}")
    else:
        print(f"[成功] SUMO配置文件已确认")
    
    # Check if region data directory exists
    region_dir = "/home/apulis-dev/userdata/Region_1/regions"
    if not os.path.exists(region_dir):
        print(f"[警告] 分区文件夹不存在: {region_dir}")
    else:
        print(f"[成功] 分区文件夹已确认")
    
    print()
    print("注意:")
    print("- 使用 Manhattan Region_1 路网和分区数据")
    print("- 仅进行推理，不进行训练")
    print("- 加载预训练的 Manhattan LoRA adapters")
    print()
    
    main(llm_path, batch_size, location, use_reflection=True, 
         step_size=step_size, max_steps=max_steps, multi_agent=True, use_local_llm=use_local_llm, 
         enable_training=enable_training, start_time=start_time, av_ratio=av_ratio,
         traffic_lora_path=traffic_lora_path, regional_lora_path=regional_lora_path)


def run_nyc_with_lora():
    """Run NYC experiment with pre-trained LoRA adapters."""
    print("Running NYC Experiment with Pre-trained LoRA Adapters")
    print("="*60)

    if not check_local_model():
        print("本地模型不可用，请检查模型路径和GPU状态")
        sys.exit(1)

    # NYC experiment parameters
    llm_path = "/home/apulis-dev/userdata/Qwen"  # qwen3-8b 模型路径
    batch_size = 8
    location = "NewYork"  # 使用 NewYork location 配置
    step_size = 180.0  # 3-minute decision intervals
    max_steps = 43200  # 12 hours simulation
    use_local_llm = True  # 使用本地LLM
    enable_training = False  # 不进行训练，只进行推理
    start_time = 0
    av_ratio = 0.02  # 从第一个路由文件中抽取2%的车辆作为自动驾驶车辆

    # LoRA adapter paths
    traffic_lora_path = "/home/apulis-dev/code/LLMNavigation/logs/training_Manhattan/lora_adapters/traffic_adapter_step_120"
    regional_lora_path = "/home/apulis-dev/code/LLMNavigation/logs/training_Manhattan/lora_adapters/regional_adapter_step_100"

    print(f"NYC 实验配置:")
    print(f"  - 模型路径: {llm_path}")
    print(f"  - 批处理大小: {batch_size}")
    print(f"  - 仿真步长: {step_size}s")
    print(f"  - 最大步数: {max_steps} (12小时)")
    print(f"  - 自动驾驶车辆比例: {av_ratio * 100}% (仅从第一个路由文件)")
    print(f"  - SUMO配置: /home/apulis-dev/userdata/NYC/NewYork_sumo_config.sumocfg")
    print(f"  - 分区文件夹: /home/apulis-dev/userdata/NYC/New2")
    print(f"  - Traffic LoRA: {traffic_lora_path}")
    print(f"  - Regional LoRA: {regional_lora_path}")
    print()

    # Check if LoRA adapters exist
    if not os.path.exists(traffic_lora_path):
        print(f"[警告] Traffic LoRA adapter不存在: {traffic_lora_path}")
    else:
        print(f"[成功] Traffic LoRA adapter已确认")

    if not os.path.exists(regional_lora_path):
        print(f"[警告] Regional LoRA adapter不存在: {regional_lora_path}")
    else:
        print(f"[成功] Regional LoRA adapter已确认")

    # Check if SUMO config exists
    sumo_config = "/home/apulis-dev/userdata/NYC/NewYork_sumo_config.sumocfg"
    if not os.path.exists(sumo_config):
        print(f"[警告] SUMO配置文件不存在: {sumo_config}")
    else:
        print(f"[成功] SUMO配置文件已确认")

    # Check if region data directory exists
    region_dir = "/home/apulis-dev/userdata/NYC/New2"
    if not os.path.exists(region_dir):
        print(f"[警告] 分区文件夹不存在: {region_dir}")
    else:
        print(f"[成功] 分区文件夹已确认")

    print()
    print("注意:")
    print("- 使用 NYC 路网和分区数据")
    print("- 仅进行推理，不进行训练")
    print("- 加载预训练的 Manhattan LoRA adapters")
    print("- 从第一个路由文件中选取2%的车辆作为自动驾驶车辆")
    print()

    main(llm_path, batch_size, location, use_reflection=True,
         step_size=step_size, max_steps=max_steps, multi_agent=True, use_local_llm=use_local_llm,
         enable_training=enable_training, start_time=start_time, av_ratio=av_ratio,
         traffic_lora_path=traffic_lora_path, regional_lora_path=regional_lora_path)


def run_use_with_lora():
    """Run USE experiment with pre-trained LoRA adapters."""
    print("Running USE Experiment with Pre-trained LoRA Adapters")
    print("="*60)

    # Reset static data to allow loading USE-specific data
    from env_utils import reset_static_data
    reset_static_data()

    if not check_local_model():
        print("本地模型不可用，请检查模型路径和GPU状态")
        sys.exit(1)

    # USE experiment parameters
    llm_path = "/home/apulis-dev/userdata/Qwen"  # qwen3-8b 模型路径
    batch_size = 8
    location = "USE"  # 使用 USE location 配置
    step_size = 30.0  # 30-second decision intervals
    max_steps = 43200  # 12 hours simulation
    use_local_llm = True  # 使用本地LLM
    enable_training = False  # 不进行训练，只进行推理
    start_time = 0
    av_ratio = 0.02  # 从第一个路由文件中抽取2%的车辆作为自动驾驶车辆

    # LoRA adapter paths (from training checkpoints)
    traffic_lora_path = "/home/apulis-dev/code/LLMNavigation/logs/training_Manhattan/Traffic_checkpoints/step_140"
    regional_lora_path = "/home/apulis-dev/code/LLMNavigation/logs/training_Manhattan/Regional_checkpoints/step_100"

    print(f"USE 实验配置:")
    print(f"  - 模型路径: {llm_path}")
    print(f"  - 批处理大小: {batch_size}")
    print(f"  - 仿真步长: {step_size}s")
    print(f"  - 最大步数: {max_steps} (12小时)")
    print(f"  - 自动驾驶车辆比例: {av_ratio * 100}% (仅从第一个路由文件)")
    print(f"  - SUMO配置: /home/apulis-dev/userdata/USE/USE.sumocfg")
    print(f"  - 分区文件夹: /home/apulis-dev/userdata/USE/NewData")
    print(f"  - 分区Alpha值: 0.5")
    print(f"  - Traffic LoRA: {traffic_lora_path}")
    print(f"  - Regional LoRA: {regional_lora_path}")
    print(f"  - 推理GPU: GPU 0 (Traffic LLM) + GPU 1 (Regional LLM)")
    print()

    # Check if LoRA adapters exist
    if not os.path.exists(traffic_lora_path):
        print(f"[警告] Traffic LoRA adapter不存在: {traffic_lora_path}")
    else:
        print(f"[成功] Traffic LoRA adapter已确认")

    if not os.path.exists(regional_lora_path):
        print(f"[警告] Regional LoRA adapter不存在: {regional_lora_path}")
    else:
        print(f"[成功] Regional LoRA adapter已确认")

    # Check if SUMO config exists
    sumo_config = "/home/apulis-dev/userdata/USE/USE.sumocfg"
    if not os.path.exists(sumo_config):
        print(f"[警告] SUMO配置文件不存在: {sumo_config}")
    else:
        print(f"[成功] SUMO配置文件已确认")

    # Check if region data directory exists
    region_dir = "/home/apulis-dev/userdata/USE/NewData"
    if not os.path.exists(region_dir):
        print(f"[警告] 分区文件夹不存在: {region_dir}")
    else:
        print(f"[成功] 分区文件夹已确认")

    # Check if alpha 0.5 adjacency file exists
    adjacency_file = "/home/apulis-dev/userdata/USE/NewData/edge_adjacency_alpha_0.5.json"
    if not os.path.exists(adjacency_file):
        print(f"[警告] Alpha 0.5分区文件不存在: {adjacency_file}")
    else:
        print(f"[成功] Alpha 0.5分区文件已确认")

    print()
    print("注意:")
    print("- 使用 USE 路网和分区数据 (alpha=0.5)")
    print("- 仅进行推理，不进行训练")
    print("- 加载预训练的 Manhattan LoRA adapters (训练检查点)")
    print("- 从第一个路由文件中选取2%的车辆作为自动驾驶车辆")
    print("- 使用30秒的决策间隔 (step_size=30)")
    print()

    main(llm_path, batch_size, location, use_reflection=True,
         step_size=step_size, max_steps=max_steps, multi_agent=True, use_local_llm=use_local_llm,
         enable_training=enable_training, start_time=start_time, av_ratio=av_ratio,
         traffic_lora_path=traffic_lora_path, regional_lora_path=regional_lora_path)


def check_requirements():
    """Check if all required files and directories exist."""
    print("Checking System Requirements...")
    
    required_files = [
        "/home/apulis-dev/userdata/NYC/task_info.json",
        "/home/apulis-dev/userdata/NYC/NewYork_sumo_config.sumocfg",
        "/home/apulis-dev/userdata/NYC/NewYork_od_0.1.rou.alt.xml",
        "/home/apulis-dev/userdata/NYC/NewYork_road_info.json",
        "/home/apulis-dev/userdata/NYC/New2/edge_adjacency_alpha_2.json",
        "/home/apulis-dev/userdata/NYC/New2/boundary_edges_alpha_2.json",
        "/home/apulis-dev/userdata/NYC/New2/edge_to_region_alpha_2.json",
    ]
    
    required_dirs = [
        "./Data",
        "/home/apulis-dev/userdata/NYC/New2",
        "./agents",
        "./utils",
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(" Missing directories:")
        for dir_path in missing_dirs:
            print(f"   - {dir_path}")
    
    if missing_files:
        print(" Missing files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
    
    if not missing_files and not missing_dirs:
        print(" All required files and directories found!")
        return True
    else:
        print("\n  Some requirements are missing. Please ensure all data files are present.")
        return False


def main_runner():
    """Main runner function with argument parsing."""
    parser = argparse.ArgumentParser(description="Multi-Agent Traffic Control System Runner")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Quick test command
    quick_parser = subparsers.add_parser('quick', help='Run quick test')
    
    # Full simulation command
    full_parser = subparsers.add_parser('full', help='Run full simulation')
    
    # Comparison command
    comp_parser = subparsers.add_parser('compare', help='Compare single vs multi-agent')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run benchmark suite')
    
    # Training command
    training_parser = subparsers.add_parser('train', help='Run with MAGRPO online training')
    
    # Chicago experiment command
    chicago_parser = subparsers.add_parser('chicago', help='Run Chicago experiment with pre-trained LoRA adapters')
    
    # Manhattan Region_1 experiment command
    manhattan_region1_parser = subparsers.add_parser('manhattan_region1', help='Run Manhattan Region_1 experiment with pre-trained LoRA adapters')
    
    # NYC experiment command
    nyc_parser = subparsers.add_parser('nyc', help='Run NYC experiment with pre-trained LoRA adapters')

    # USE experiment command
    use_parser = subparsers.add_parser('use', help='Run USE experiment with pre-trained LoRA adapters')

    # Check requirements command
    check_parser = subparsers.add_parser('check', help='Check system requirements')
    
    # Custom command
    custom_parser = subparsers.add_parser('custom', help='Run with custom parameters')
    custom_parser.add_argument('--llm', type=str, default="/data/zhouyuping/Qwen/",
                              help='LLM model path or name (default: /data/zhouyuping/Qwen/)')
    custom_parser.add_argument('--batch-size', type=int, default=16,
                              help='Batch size for LLM (default: 16)')
    custom_parser.add_argument('--location', type=str, default="Manhattan",
                              help='Simulation location (default: NewYork)')
    custom_parser.add_argument('--step-size', type=float, default=180.0,
                              help='Simulation step size in seconds (default: 180.0)')
    custom_parser.add_argument('--max-steps', type=int, default=43200,
                              help='Maximum simulation steps (default: 43200)')
    custom_parser.add_argument('--no-reflection', action='store_true',
                              help='Disable LLM reflection')
    custom_parser.add_argument('--single-agent', action='store_true',
                              help='Use single-agent mode')
    custom_parser.add_argument('--use-api-llm', action='store_true',
                              help='Use API LLM instead of local LLM (requires API key)')
    custom_parser.add_argument('--enable-training', action='store_true',
                              help='Enable MAGRPO online training (requires local LLM)')
    custom_parser.add_argument('--disable-global-guidance', action='store_true',
                              help='Disable per-timestamp global macro guidance')
    custom_parser.add_argument('--traffic-lora-path', type=str, default=None,
                              help='Path to Traffic LLM LoRA adapter')
    custom_parser.add_argument('--regional-lora-path', type=str, default=None,
                              help='Path to Regional LLM LoRA adapter')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Print header
    print("Multi-Agent Traffic Control System")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    if args.command == 'quick':
        run_quick_test()
    elif args.command == 'full':
        run_full_simulation()
    elif args.command == 'compare':
        run_comparison()
    elif args.command == 'benchmark':
        run_benchmark()
    elif args.command == 'train':
        run_with_training()
    elif args.command == 'chicago':
        run_chicago_with_lora()
    elif args.command == 'manhattan_region1':
        run_manhattan_region1_with_lora()
    elif args.command == 'nyc':
        run_nyc_with_lora()
    elif args.command == 'use':
        run_use_with_lora()
    elif args.command == 'check':
        check_requirements()
    elif args.command == 'custom':
        use_local_llm = not args.use_api_llm
        enable_training = args.enable_training
        
        # Validate training configuration
        if enable_training and not use_local_llm:
            print("[ERROR] 强化学习训练需要本地LLM模式，请不要同时使用 --enable-training 和 --use-api-llm")
            sys.exit(1)
        
        if enable_training and args.single_agent:
            print("[ERROR] 强化学习训练需要多智能体模式，请不要同时使用 --enable-training 和 --single-agent")
            sys.exit(1)
        
        # Check requirements based on LLM mode
        if use_local_llm:
            if not check_local_model():
                print("本地模型不可用，请检查模型路径和GPU状态")
                sys.exit(1)
        else:
            # Check API key for API mode
            if "qwen" in args.llm.lower() and not check_qwen_api_key():
                sys.exit(1)
            
        if not check_requirements():
            sys.exit(1)
        
        use_reflection = not args.no_reflection
        use_multi_agent = not args.single_agent
        
        print(f"Custom Configuration:")
        print(f"  - LLM Mode: {'Local Shared LLMs' if use_local_llm else 'API/Traditional'}")
        print(f"  - LLM: {args.llm}")
        print(f"  - Batch Size: {args.batch_size}")
        print(f"  - Location: {args.location}")
        print(f"  - Step Size: {args.step_size}s")
        print(f"  - Max Steps: {args.max_steps}")
        print(f"  - Reflection: {'Enabled' if use_reflection else 'Disabled'}")
        print(f"  - Architecture: {'Multi-Agent' if use_multi_agent else 'Single-Agent'}")
        print(f"  - MAGRPO Training: {'ENABLED' if enable_training else 'Disabled'}")
        if enable_training:
            print(f"    * Traffic LLM Training GPU: cuda:2")
            print(f"    * Regional LLM Training GPU: cuda:3")
            print(f"    * Training Group Sizes: Traffic=8, Regional=12")
        print()
        
        # Propagate global guidance switch via environment variable
        try:
            if args.disable_global_guidance:
                os.environ['DISABLE_GLOBAL_GUIDANCE'] = '1'
            else:
                if 'DISABLE_GLOBAL_GUIDANCE' in os.environ:
                    del os.environ['DISABLE_GLOBAL_GUIDANCE']
        except Exception:
            pass

        main(args.llm, args.batch_size, args.location, use_reflection,
             args.step_size, args.max_steps, use_multi_agent, use_local_llm, enable_training,
             traffic_lora_path=args.traffic_lora_path, regional_lora_path=args.regional_lora_path)


if __name__ == "__main__":
    main_runner()