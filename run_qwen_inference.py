#!/usr/bin/env python3
"""
Qwen3-Max Inference Script for SUMO Traffic Simulation
仅进行推理实验，不触发训练，记录Token Usage

使用方法:
1. 设置环境变量: export DASHSCOPE_API_KEY="your-api-key-here"
2. 运行脚本: python run_qwen_inference.py
"""

import os
import sys
import argparse
import time
from datetime import datetime
from main import MultiAgent_Route_Planning
os.environ["WANDB_MODE"] = "offline"

def check_qwen_api_key() -> bool:
    """检查通义千问API密钥是否已设置"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("\n[错误] 未找到DASHSCOPE_API_KEY环境变量")
        print("请先设置API密钥: export DASHSCOPE_API_KEY='sk-your-api-key-here'")
        print("获取API密钥: https://dashscope.console.aliyun.com/apiKey\n")
        return False
    
    if not api_key.startswith("sk-"):
        print("\n[警告] DASHSCOPE_API_KEY格式可能不正确（通常以'sk-'开头）")
    
    print(f"[成功] 检测到DASHSCOPE_API_KEY: {api_key[:10]}...")
    return True


def run_qwen_inference(
    model_name: str = "qwen3-max",
    batch_size: int = 8,
    location: str = "Chicago",
    start_time: int = 28800,
    end_time: int = 43200,
    step_size: float = 180.0,
    use_reflection: bool = True,
    multi_agent: bool = True,
    sumo_config: str = None,
    output_dir: str = None
):
    """
    运行Qwen3-Max推理实验

    Args:
        model_name: Qwen模型名称 (默认: qwen3-max)
        batch_size: 批处理大小
        location: 仿真位置
        start_time: 仿真开始时间（秒）
        end_time: 仿真结束时间（秒）
        step_size: 仿真步长（秒）
        use_reflection: 是否使用反思机制
        multi_agent: 是否使用多智能体架构
        sumo_config: SUMO配置文件路径（可选）
        output_dir: 输出目录（可选）
    """
    # 检查API密钥
    if not check_qwen_api_key():
        sys.exit(1)

    if sumo_config is None:
        sumo_config = "/data/zhouyuping/LLMNavigation/Data/Chicago/Chicago_sumo_config.sumocfg"

    # 从sumo_config推断其他文件路径
    base_dir = os.path.dirname(sumo_config)
    route_file = os.path.join(base_dir, "Chicago_taxi_2015-01-01.rou.alt.xml")
    road_info_file = os.path.join(base_dir, "Chicago_road_info.json")

    # 如果路由和道路信息文件不存在，尝试使用默认路径
    if not os.path.exists(route_file):
        # 使用Chicago的默认路由文件
        route_file = "/data/zhouyuping/LLMNavigation/Data/Chicago/Chicago_taxi_2015-01-01.rou.alt.xml"
    if not os.path.exists(road_info_file):
        # 使用NYC的road_info作为备份（如果Chicago没有）
        road_info_file = "/data/zhouyuping/LLMNavigation/Data/Chicago/Chicago_road_info.json"

    adjacency_file = "/data/zhouyuping/LLMNavigation/Data/Chicago/Region/edge_adjacency_alpha_1.json"
    task_info_file = "/data/zhouyuping/LLMNavigation/Data/NYC/task_info.json"
    region_data_dir = "/data/zhouyuping/LLMNavigation/Data/Chicago/Region"
    
    # 检查必要文件是否存在
    required_files = [sumo_config, route_file, road_info_file, adjacency_file, task_info_file]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"[错误] 文件不存在: {file_path}")
            sys.exit(1)
    
    if multi_agent and not os.path.exists(region_data_dir):
        print(f"[警告] 区域数据目录不存在: {region_data_dir}")
        print("将切换到单智能体模式")
        multi_agent = False
    
    # 计算仿真步数
    max_steps = end_time  # SUMO使用绝对时间
    
    # 创建输出目录
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"logs/qwen_inference_{location}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Qwen3-Max 推理实验配置")
    print(f"{'='*60}")
    print(f"模型: {model_name}")
    print(f"API模式: DashScope API (OpenAI兼容)")
    print(f"仿真位置: {location}")
    print(f"仿真时间: {start_time}s - {end_time}s ({(end_time-start_time)/3600:.1f}小时)")
    print(f"步长: {step_size}s")
    print(f"批处理大小: {batch_size}")
    print(f"使用反思: {'是' if use_reflection else '否'}")
    print(f"多智能体: {'是' if multi_agent else '否'}")
    print(f"训练模式: 关闭（仅推理）")
    print(f"SUMO配置: {sumo_config}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}\n")
    
    # 创建算法实例
    try:
        if multi_agent:
            print("[信息] 初始化多智能体交通控制系统...")
            algo = MultiAgent_Route_Planning(
                batch_size=batch_size,
                location=location,
                sumo_config=sumo_config,
                route_file=route_file,
                road_info_file=road_info_file,
                adjacency_file=adjacency_file,
                task_info_file=task_info_file,
                llm_path_or_name="dashscope",  # 触发DashScope API模式
                step_size=step_size,
                max_steps=max_steps,
                use_reflection=use_reflection,
                region_data_dir=region_data_dir,
                use_local_llm=False,  # 使用API模式
                enable_training=False,  # 关闭训练
                start_time=start_time,
                av_ratio=0.02  # 自动驾驶车辆比例
            )
            
            # 设置真实的模型名称为qwen3-max
            if hasattr(algo, 'llm_agent') and algo.llm_agent is not None:
                try:
                    algo.llm_agent.llm_name = model_name
                    algo.llm_agent.provider_name = 'dashscope'  # 标记提供商
                    print(f"[信息] LLM Agent 模型设置为: {model_name}")
                except Exception as e:
                    print(f"[警告] 设置LLM Agent模型名失败: {e}")
            
            # 为多智能体环境中的LLM也设置模型名
            if hasattr(algo, 'sim') and algo.sim is not None:
                try:
                    # 关闭RL数据采集
                    algo.sim.rl_data_collection_enabled = False
                    algo.sim.enable_time_sliced_training = False
                    algo.sim.training_queue = None
                    
                    # 设置Traffic LLM模型名
                    if hasattr(algo.sim, 'traffic_llm') and algo.sim.traffic_llm:
                        algo.sim.traffic_llm.llm_name = model_name
                        algo.sim.traffic_llm.provider_name = 'dashscope'
                        print(f"[信息] Traffic LLM 模型设置为: {model_name}")
                    
                    # 设置Regional LLM模型名
                    if hasattr(algo.sim, 'regional_llm') and algo.sim.regional_llm:
                        algo.sim.regional_llm.llm_name = model_name
                        algo.sim.regional_llm.provider_name = 'dashscope'
                        print(f"[信息] Regional LLM 模型设置为: {model_name}")
                except Exception as e:
                    print(f"[警告] 配置多智能体LLM时出错: {e}")
        else:
            print("[信息] 初始化单智能体路径规划系统...")
            from main import Route_Planning
            algo = Route_Planning(
                batch_size=batch_size,
                location=location,
                sumo_config=sumo_config,
                route_file=route_file,
                road_info_file=road_info_file,
                adjacency_file=adjacency_file,
                task_info_file=task_info_file,
                llm_path_or_name="dashscope",
                step_size=step_size,
                max_steps=max_steps,
                use_reflection=use_reflection
            )
            # 设置真实的模型名称
            if hasattr(algo, 'llm_agent') and algo.llm_agent is not None:
                try:
                    algo.llm_agent.llm_name = model_name
                    algo.llm_agent.provider_name = 'dashscope'
                except Exception:
                    pass
        
        # 运行仿真
        print("\n[信息] 开始仿真...")
        start_run_time = time.time()
        
        average_travel_time, throughput = algo.run()
        
        end_run_time = time.time()
        total_time = end_run_time - start_run_time
        
        print(f"\n{'='*60}")
        print(f"仿真完成")
        print(f"{'='*60}")
        print(f"平均旅行时间: {average_travel_time:.2f}秒")
        print(f"吞吐量: {throughput:.2f}")
        print(f"总运行时间: {total_time/60:.2f}分钟")
        print(f"{'='*60}\n")
        
        # 保存Token Usage统计
        print("\n[信息] 保存Token使用统计...")
        if hasattr(algo, 'llm_agent') and algo.llm_agent:
            token_file = os.path.join(output_dir, "token_usage.json")
            token_data = algo.llm_agent.save_token_usage(token_file)
            print(f"[成功] Token使用统计已保存到: {token_file}")
            
            # 打印Token使用摘要
            if token_data:
                print(f"\nToken使用摘要:")
                print(f"  总Prompt Tokens: {token_data.get('total_prompt_tokens', 0)}")
                print(f"  总Completion Tokens: {token_data.get('total_completion_tokens', 0)}")
                print(f"  总Tokens: {token_data.get('total_tokens', 0)}")
                print(f"  API调用次数: {len(token_data.get('detailed_log', []))}")
        
        # 多智能体模式下保存所有LLM的token usage
        if hasattr(algo, 'sim') and hasattr(algo.sim, 'traffic_llm'):
            token_file_traffic = os.path.join(output_dir, "token_usage_traffic.json")
            token_file_regional = os.path.join(output_dir, "token_usage_regional.json")
            
            if algo.sim.traffic_llm:
                token_data = algo.sim.traffic_llm.save_token_usage(token_file_traffic)
                print(f"[成功] Traffic LLM Token使用统计已保存到: {token_file_traffic}")
                if token_data:
                    print(f"\nTraffic LLM Token使用摘要:")
                    print(f"  总Prompt Tokens: {token_data.get('total_prompt_tokens', 0)}")
                    print(f"  总Completion Tokens: {token_data.get('total_completion_tokens', 0)}")
                    print(f"  总Tokens: {token_data.get('total_tokens', 0)}")
            
            if hasattr(algo.sim, 'regional_llm') and algo.sim.regional_llm:
                token_data = algo.sim.regional_llm.save_token_usage(token_file_regional)
                print(f"[成功] Regional LLM Token使用统计已保存到: {token_file_regional}")
                if token_data:
                    print(f"\nRegional LLM Token使用摘要:")
                    print(f"  总Prompt Tokens: {token_data.get('total_prompt_tokens', 0)}")
                    print(f"  总Completion Tokens: {token_data.get('total_completion_tokens', 0)}")
                    print(f"  总Tokens: {token_data.get('total_tokens', 0)}")
        
        print(f"\n[成功] 实验完成！所有结果已保存到: {output_dir}\n")
        
    except Exception as e:
        print(f"\n[错误] 实验失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用Qwen3-Max进行SUMO交通仿真推理实验（不训练）"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="qwen3-max",
        help="Qwen模型名称 (默认: qwen3-max)"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=8,
        help="批处理大小 (默认: 8)"
    )
    
    parser.add_argument(
        "--location",
        type=str,
        default="Chicago",
        help="仿真位置 (默认: Chicago)"
    )

    parser.add_argument(
        "--start-time",
        type=int,
        default=28800,
        help="仿真开始时间（秒） (默认: 28800 = 8:00 AM)"
    )

    parser.add_argument(
        "--end-time",
        type=int,
        default=43200,
        help="仿真结束时间（秒） (默认: 43200 = 12:00 PM)"
    )

    parser.add_argument(
        "--step-size",
        type=float,
        default=180.0,
        help="仿真步长（秒） (默认: 180.0)"
    )

    parser.add_argument(
        "--no-reflection",
        action="store_true",
        help="禁用反思机制"
    )

    parser.add_argument(
        "--single-agent",
        action="store_true",
        help="使用单智能体模式"
    )

    parser.add_argument(
        "--sumo-config",
        type=str,
        default="/data/zhouyuping/LLMNavigation/Data/Chicago/Chicago_sumo_config.sumocfg",
        help="SUMO配置文件路径"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        default=None,
        help="输出目录（可选）"
    )
    
    args = parser.parse_args()
    
    run_qwen_inference(
        model_name=args.model,
        batch_size=args.batch_size,
        location=args.location,
        start_time=args.start_time,
        end_time=args.end_time,
        step_size=args.step_size,
        use_reflection=not args.no_reflection,
        multi_agent=not args.single_agent,
        sumo_config=args.sumo_config,
        output_dir=args.output_dir
    )

