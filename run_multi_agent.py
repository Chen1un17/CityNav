#!/usr/bin/env python3
"""
Multi-Agent Traffic Control System Runner

This script provides easy ways to run the multi-agent traffic control system
with different configurations and parameters.

Default LLM: Local Qwen Model (/data/zhouyuping/Qwen/)
- Uses local vLLM for high-performance inference
- Shared LLM architecture: Traffic LLM (GPU 0) + Regional LLM (GPU 1)
- Optimized for local deployment with multiple GPUs
- Fallback support for API mode when specified
"""

import argparse
import os
import sys
from datetime import datetime
os.environ["WANDB_MODE"] = "offline"
from main import main


def check_local_model():
    """Check if local model is available."""
    model_path = "/data/zhouyuping/Qwen/"
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
        
        if gpu_count >= 2:
            print("[推荐] GPU 0(Traffic LLM) + GPU 1(Regional LLM)")
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
    llm_path = "/data/zhouyuping/Qwen/"
    batch_size = 4
    location = "NewYork"
    step_size = 60.0  # Faster steps for testing
    max_steps = 1800  # 30 minutes simulation
    use_local_llm = True  # 启用本地LLM模式
    
    main(llm_path, batch_size, location, use_reflection=True, 
         step_size=step_size, max_steps=max_steps, multi_agent=True, use_local_llm=use_local_llm)


def run_full_simulation():
    """Run a complete simulation with full settings."""
    print("Running Full Simulation - Multi-Agent Traffic Control (Local LLM)")
    print("="*60)
    
    if not check_local_model():
        print("本地模型不可用，请检查模型路径和GPU状态")
        sys.exit(1)
    
    # Full simulation parameters - 使用本地模型
    llm_path = "/data/zhouyuping/Qwen/"
    batch_size = 8
    location = "NewYork"
    step_size = 180.0  # 3-minute decision intervals
    max_steps = 43200  # 12 hours simulation
    use_local_llm = True  # 启用本地LLM模式
    
    main(llm_path, batch_size, location, use_reflection=True, 
         step_size=step_size, max_steps=max_steps, multi_agent=True, use_local_llm=use_local_llm)


def run_comparison():
    """Run both single-agent and multi-agent for comparison."""
    print("Running Comparison: Single-Agent vs Multi-Agent (Local LLM)")
    print("="*60)
    
    if not check_local_model():
        print("本地模型不可用，请检查模型路径和GPU状态")
        sys.exit(1)
    
    # Common parameters - 使用本地模型
    llm_path = "/data/zhouyuping/Qwen/"
    batch_size = 6
    location = "NewYork"
    step_size = 180.0
    max_steps = 21600  # 6 hours simulation
    use_local_llm = True  # 启用本地LLM模式
    
    print("\n" + "="*40)
    print("RUNNING SINGLE-AGENT BASELINE (LOCAL LLM)")
    print("="*40)
    
    main(llm_path, batch_size, location, use_reflection=True, 
         step_size=step_size, max_steps=max_steps, multi_agent=False, use_local_llm=use_local_llm)
    
    print("\n" + "="*40)
    print("RUNNING MULTI-AGENT SYSTEM (LOCAL LLM)")
    print("="*40)
    
    main(llm_path, batch_size, location, use_reflection=True, 
         step_size=step_size, max_steps=max_steps, multi_agent=True, use_local_llm=use_local_llm)


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
    
    llm_path = "/data/zhouyuping/Qwen/"  # 使用本地模型
    batch_size = 8
    location = "NewYork"
    use_local_llm = True  # 启用本地LLM模式
    
    for i, (reflection, step_size, max_steps, description) in enumerate(configurations):
        print(f"\n{'='*50}")
        print(f"BENCHMARK {i+1}/4: {description} (Local LLM)")
        print(f"{'='*50}")
        
        main(llm_path, batch_size, location, use_reflection=reflection, 
             step_size=step_size, max_steps=max_steps, multi_agent=True, use_local_llm=use_local_llm)


def check_requirements():
    """Check if all required files and directories exist."""
    print("Checking System Requirements...")
    
    required_files = [
        "/data/zhouyuping/LLMNavigation/Data/task_info.json",
        "/data/zhouyuping/LLMNavigation/Data/NYC//NewYork_sumo_config.sumocfg",
        "/data/zhouyuping/LLMNavigation/Data/NYC/NewYork_od_0.1.rou.alt.xml",
        "/data/zhouyuping/LLMNavigation/Data/NYC/NewYork_road_info.json",
        "/data/zhouyuping/LLMNavigation/Data/NYC/Region_1/edge_adjacency_alpha_1.json",
        "/data/zhouyuping/LLMNavigation/Data/NYC/Region_1/boundary_edges_alpha_1.json",
        "/data/zhouyuping/LLMNavigation/Data/NYC/Region_1/edge_to_region_alpha_1.json",
    ]
    
    required_dirs = [
        "./Data",
        "/data/zhouyuping/LLMNavigation/Data/NYC/Region_1",
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
    
    # Check requirements command
    check_parser = subparsers.add_parser('check', help='Check system requirements')
    
    # Custom command
    custom_parser = subparsers.add_parser('custom', help='Run with custom parameters')
    custom_parser.add_argument('--llm', type=str, default="/data/zhouyuping/Qwen/",
                              help='LLM model path or name (default: /data/zhouyuping/Qwen/)')
    custom_parser.add_argument('--batch-size', type=int, default=16,
                              help='Batch size for LLM (default: 16)')
    custom_parser.add_argument('--location', type=str, default="NewYork",
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
    elif args.command == 'check':
        check_requirements()
    elif args.command == 'custom':
        use_local_llm = not args.use_api_llm
        
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
        print()
        
        main(args.llm, args.batch_size, args.location, use_reflection,
             args.step_size, args.max_steps, use_multi_agent, use_local_llm)


if __name__ == "__main__":
    main_runner()