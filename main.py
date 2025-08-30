import argparse
import os
import sys
import multiprocessing as mp
from env import Simulation
from multi_agent_env import MultiAgentTrafficEnvironment
from utils.read_utils import load_json
from utils.language_model import LLM as LLM
from training_manager import TrainingConfig, run_training_manager
import wandb

class MultiAgent_Route_Planning(object):
    def __init__(self, batch_size, location, sumo_config, route_file, road_info_file, 
                 adjacency_file, task_info_file, llm_path_or_name, step_size, max_steps, 
                 use_reflection, region_data_dir, use_local_llm=True, enable_training=False):
        llm_name = llm_path_or_name.split("/")[-1]
        self.llm_name = llm_name
        self.location = location
        self.use_local_llm = use_local_llm
        self.enable_training = enable_training
        
        # Initialize training components
        self.training_queue = None
        self.training_process = None
        if enable_training and use_local_llm:
            print(f"\n=== 启用MAGRPO在线训练 ===")
            self._initialize_training_manager(llm_path_or_name)
        
        # Load task info
        task_info = load_json(task_info_file)
        
        # Initialize based on mode
        if use_local_llm:
            print(f"\n=== 初始化本地共享LLM架构 ===")
            print(f"模型路径: {llm_path_or_name}")
            
            # Initialize multi-agent environment with local LLM
            self.sim = MultiAgentTrafficEnvironment(
                location=location,
                sumo_config_file=sumo_config,
                route_file=route_file,
                road_info_file=road_info_file,
                adjacency_file=adjacency_file,
                region_data_dir=region_data_dir,
                model_path=llm_path_or_name,  # 使用本地模型路径
                step_size=step_size,
                max_steps=max_steps,
                log_dir=f"logs/{location}_{llm_name}_local{'_rl' if enable_training else ''}",
                task_info=task_info,
                use_local_llm=True,
                training_queue=self.training_queue  # Pass training queue for RL data collection
            )
        else:
            print(f"\n=== 初始化传统单一LLM模式 ===")
            # Initialize traditional single LLM
            self.llm_agent = LLM(llm_path_or_name, batch_size=batch_size, task_info=task_info, 
                               use_reflection=use_reflection)
            
            # Initialize multi-agent environment with single LLM
            self.sim = MultiAgentTrafficEnvironment(
                location=location,
                sumo_config_file=sumo_config,
                route_file=route_file,
                road_info_file=road_info_file,
                adjacency_file=adjacency_file,
                region_data_dir=region_data_dir,
                llm_agent=self.llm_agent,
                step_size=step_size,
                max_steps=max_steps,
                log_dir=f"logs/{location}_{llm_name}",
                use_local_llm=False,
                training_queue=self.training_queue  # None for non-local LLM mode
            )

        wandb.init(
            project="USTBench-MultiAgent-Route-Planning",
            group=f"{self.location}-{llm_name}{'-w/o reflection' if not use_reflection else ''}{'-RL' if enable_training else ''}",
            name="MultiAgent-Examination"
        )
    
    def _initialize_training_manager(self, llm_path):
        """Initialize the MAGRPO training manager in a separate process."""
        try:
            # Create multiprocessing queue for training data
            self.training_queue = mp.Queue(maxsize=1000)
            print(f"创建训练数据队列 (max_size=1000)")
            
            # Detect vLLM inference server URLs
            vllm_urls = self._detect_vllm_inference_servers()
            
            # Create training configuration
            training_config = TrainingConfig(
                model_path=llm_path,
                traffic_gpu="cuda:2",  # Traffic LLM training GPU
                regional_gpu="cuda:3",  # Regional LLM training GPU
                traffic_group_size=8,
                regional_group_size=12,
                learning_rate=1e-4,
                warmup_steps=100,
                max_grad_norm=1.0,
                weight_decay=0.01,
                gradient_accumulation_steps=4,
                save_steps=100,
                max_checkpoints=5,
                log_steps=10,
                log_dir=f"logs/training_{self.location}",
                vllm_inference_urls=vllm_urls,  # Pass vLLM server URLs
                adapter_sync_dir="lora_adapters",
                enable_hot_reload=True
            )
            
            print(f"训练配置:")
            print(f"  - 模型路径: {training_config.model_path}")
            print(f"  - Traffic LLM GPU: {training_config.traffic_gpu}")
            print(f"  - Regional LLM GPU: {training_config.regional_gpu}")
            print(f"  - 训练组大小: Traffic={training_config.traffic_group_size}, Regional={training_config.regional_group_size}")
            print(f"  - 学习率: {training_config.learning_rate}")
            print(f"  - 日志目录: {training_config.log_dir}")
            print(f"  - vLLM推理服务器: {training_config.vllm_inference_urls}")
            print(f"  - 热重载: {'启用' if training_config.enable_hot_reload else '禁用'}")
            
            # Start training manager in separate process
            print("启动MAGRPO训练管理器进程...")
            self.training_process = mp.Process(
                target=run_training_manager,
                args=(training_config.__dict__, self.training_queue),
                name="MAGRPO-TrainingManager"
            )
            self.training_process.daemon = False  # Ensure proper cleanup
            self.training_process.start()
            
            print(f"[SUCCESS] 训练管理器进程已启动 (PID: {self.training_process.pid})")
            print("训练数据将通过队列实时传输到训练进程")
            
        except Exception as e:
            print(f"[ERROR] 训练管理器初始化失败: {e}")
            self.training_queue = None
            self.training_process = None
            raise
    
    def _cleanup_training_manager(self):
        """Clean up training manager resources."""
        try:
            if self.training_process and self.training_process.is_alive():
                print("\n=== 清理训练管理器资源 ===")
                print(f"正在终止训练进程 (PID: {self.training_process.pid})")
                
                # Send termination signal and wait for graceful shutdown
                self.training_process.terminate()
                self.training_process.join(timeout=10)
                
                # Force kill if not terminated gracefully
                if self.training_process.is_alive():
                    print("强制终止训练进程...")
                    self.training_process.kill()
                    self.training_process.join()
                
                print("[SUCCESS] 训练进程已终止")
            
            # Clean up queue
            if self.training_queue:
                # Drain remaining items in queue
                try:
                    while not self.training_queue.empty():
                        self.training_queue.get_nowait()
                except:
                    pass
                print("[SUCCESS] 训练队列已清理")
        
        except Exception as e:
            print(f"[WARNING] 训练管理器清理过程中发生错误: {e}")
    
    def _detect_vllm_inference_servers(self):
        """Detect running vLLM inference servers for hot-reload."""
        import requests
        
        # Common vLLM server ports to check
        potential_urls = [
            "http://localhost:8000",  # Default vLLM port
            "http://127.0.0.1:8000",
            "http://localhost:8001",  # Alternative ports
            "http://127.0.0.1:8001",
        ]
        
        active_urls = []
        
        for url in potential_urls:
            try:
                # Check if vLLM server is running by querying /v1/models
                response = requests.get(f"{url}/v1/models", timeout=2)
                if response.status_code == 200:
                    active_urls.append(url)
                    print(f"[SUCCESS] 检测到vLLM推理服务器: {url}")
                    
                    # Log available models for debugging
                    try:
                        models_info = response.json()
                        if 'data' in models_info and len(models_info['data']) > 0:
                            model_names = [model.get('id', 'unknown') for model in models_info['data']]
                            print(f"  可用模型: {', '.join(model_names)}")
                    except:
                        pass
                        
            except requests.exceptions.RequestException:
                # Server not available at this URL
                continue
        
        if not active_urls:
            print("[WARNING] 未检测到运行中的vLLM推理服务器，热重载功能将被禁用")
            print("请确保vLLM服务器已启动并支持动态LoRA加载 (VLLM_ALLOW_RUNTIME_LORA_UPDATING=True)")
            active_urls = ["http://localhost:8000"]  # Fallback default
        
        return active_urls

    def run(self):
        try:
            average_travel_time, throughput = self.sim.run_simulation()
        finally:
            # Clean up training resources
            if self.enable_training:
                self._cleanup_training_manager()

        wandb.log({
            "average_travel_time": average_travel_time,
            "throughput": throughput
        })
        wandb.finish()


class Route_Planning(object):
    def __init__(self, batch_size, location, sumo_config, route_file, road_info_file, adjacency_file, task_info_file, llm_path_or_name, step_size, max_steps, use_reflection):
        llm_name = llm_path_or_name.split("/")[-1]
        self.llm_name = llm_name
        self.location = location

        self.sim = Simulation(
            location=location,
            sumo_config_file=sumo_config,
            route_file=route_file,
            road_info_file=road_info_file,
            adjacency_file=adjacency_file,
            step_size=step_size,
            max_steps=max_steps
        )
        self.sim.initialize()

        # initialize language model
        task_info = load_json(task_info_file)
        self.llm_agent = LLM(llm_path_or_name, batch_size=batch_size, task_info=task_info, use_reflection=use_reflection)

        wandb.init(
            project="USTBench-Route-Planning",
            group=f"{self.location}-{llm_name}{'-w/o reflection' if not use_reflection else ''}",
            name="Examination"
        )

    def run(self):
        average_travel_time, throughput = self.sim.run(self.llm_agent)

        wandb.log({
            "average_travel_time": average_travel_time,
            "throughput": throughput
        })
        wandb.finish()

def main(llm_path_or_name, batch_size, location, use_reflection=True, step_size=180.0, 
         max_steps=43200, multi_agent=True, use_local_llm=True, enable_training=False):
    """
    Main function to run traffic simulation.
    
    Args:
        llm_path_or_name: Path to or name of the language model
        batch_size: Batch size for LLM processing
        location: Location for simulation (e.g., /data/zhouyuping/LLMNavigation/Data/NYC/ttan)
        use_reflection: Whether to use reflection in LLM
        step_size: Simulation step size in seconds
        max_steps: Maximum simulation steps
        multi_agent: Whether to use multi-agent architecture (default: True)
        use_local_llm: Whether to use local shared LLM architecture (default: True)
        enable_training: Whether to enable MAGRPO online training (default: False)
    """
    # File paths - updated to use new data structure
    sumo_config = f"/data/zhouyuping/LLMNavigation/Data/NYC/NewYork_sumo_config.sumocfg"
    route_file = f"/data/zhouyuping/LLMNavigation/Data/NYC/NewYork_od_0.1.rou.alt.xml"
    road_info_file = f"/data/zhouyuping/LLMNavigation/Data/NYC/NewYork_road_info.json"
    adjacency_file = f"/data/zhouyuping/LLMNavigation/Data/NYC/Region_1/edge_adjacency_alpha_1.json"
    task_info_file = "/data/zhouyuping/LLMNavigation/Data/task_info.json"
    
    if multi_agent:
        # Use multi-agent architecture
        region_data_dir = f"/data/zhouyuping/LLMNavigation/Data/NYC/Region_1"  # Region partition data directory
        
        # Check if region data exists
        if not os.path.exists(region_data_dir):
            print(f"Warning: Region data directory {region_data_dir} not found. "
                  f"Falling back to single-agent mode.")
            multi_agent = False
    
    if multi_agent:
        print("Running Multi-Agent Traffic Control System")
        algo = MultiAgent_Route_Planning(
            batch_size, location, sumo_config, route_file, road_info_file, 
            adjacency_file, task_info_file, llm_path_or_name, step_size, 
            max_steps, use_reflection, region_data_dir, use_local_llm, enable_training
        )
    else:
        print("Running Traditional Single-Agent Route Planning")
        algo = Route_Planning(
            batch_size, location, sumo_config, route_file, road_info_file, 
            adjacency_file, task_info_file, llm_path_or_name, step_size, 
            max_steps, use_reflection
        )
    
    algo.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a SUMO simulation with autonomous vehicles.")
    parser.add_argument("--llm-path-or-name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", 
                       help="Path to the language model or its name.")
    parser.add_argument("--batch-size", type=int, default=16, 
                       help="Batch size for the language model.")
    parser.add_argument("--location", type=str, default="NewYork", 
                       help="Location of the simulation.")
    parser.add_argument("--step-size", type=float, default=180.0, 
                       help="Simulation step size in seconds.")
    parser.add_argument("--max-steps", type=int, default=43200, 
                       help="Maximum number of simulation steps.")
    parser.add_argument("--multi-agent", action="store_true", default=True,
                       help="Use multi-agent architecture (default: True)")
    parser.add_argument("--single-agent", action="store_true", default=False,
                       help="Use traditional single-agent architecture")
    parser.add_argument("--no-reflection", action="store_true", default=False,
                       help="Disable LLM reflection")
    parser.add_argument("--use-api-llm", action="store_true", default=False,
                       help="Use API LLM instead of local shared LLM")
    parser.add_argument("--enable-training", action="store_true", default=False,
                       help="Enable MAGRPO online training (requires local LLM)")
    
    args = parser.parse_args()
    
    # Determine which architecture to use
    use_multi_agent = args.multi_agent and not args.single_agent
    use_reflection = not args.no_reflection
    use_local_llm = not args.use_api_llm
    enable_training = args.enable_training
    
    # Validate training configuration
    if enable_training and not use_local_llm:
        print("[ERROR] 强化学习训练需要本地LLM模式，请不要同时使用 --enable-training 和 --use-api-llm")
        sys.exit(1)
    
    if enable_training and not use_multi_agent:
        print("[ERROR] 强化学习训练需要多智能体模式，请不要同时使用 --enable-training 和 --single-agent")
        sys.exit(1)
    
    print(f"Configuration:")
    print(f"  - Architecture: {'Multi-Agent' if use_multi_agent else 'Single-Agent'}")
    print(f"  - LLM Mode: {'Local Shared LLMs' if use_local_llm else 'API/Traditional'}")
    print(f"  - LLM: {args.llm_path_or_name}")
    print(f"  - Reflection: {'Enabled' if use_reflection else 'Disabled'}")
    print(f"  - MAGRPO Training: {'ENABLED' if enable_training else 'Disabled'}")
    if enable_training:
        print(f"    * Traffic LLM Training GPU: cuda:2")
        print(f"    * Regional LLM Training GPU: cuda:3")
        print(f"    * Training Group Sizes: Traffic=8, Regional=12")
    print(f"  - Location: {args.location}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Step Size: {args.step_size}s")
    print(f"  - Max Steps: {args.max_steps}")
    print()

    main(args.llm_path_or_name, args.batch_size, args.location, use_reflection, 
         args.step_size, args.max_steps, use_multi_agent, use_local_llm, enable_training)
