#!/usr/bin/env python3
"""
DeepSeek-Chat Inference Script for SUMO Traffic Simulation
仅进行推理实验，不触发训练，记录Token Usage

默认配置:
- 路网: NYC Manhattan (NewYork_sumo_config.sumocfg)
- 时间: 28800s-43200s (8:00 AM - 12:00 PM, 4小时)
- 车辆抽取: 仅从第一个路由文件抽取2%进行LLM决策
- 模式: 多智能体

快速运行:
1. 设置环境变量: export DEEPSEEK_API_KEY="your-api-key-here"
2. 使用默认配置运行: python run_deepseek_inference.py
3. 自定义参数运行: python run_deepseek_inference.py --av-ratio 0.05 --batch-size 16

常用参数:
--av-ratio 0.02         # 从第一个路由文件抽取的车辆比例 (默认2%)
--batch-size 8          # 批处理大小 (默认8)
--model deepseek-chat   # 模型名称 (默认deepseek-reasoner)
--start-time 28800      # 仿真开始时间 (默认28800s = 8:00 AM)
--end-time 43200        # 仿真结束时间 (默认43200s = 12:00 PM)
"""

import os
import sys
import argparse
import time
from datetime import datetime
from main import MultiAgent_Route_Planning
os.environ["WANDB_MODE"] = "offline"

def check_deepseek_api_key() -> bool:
    """检查DeepSeek API密钥是否已设置"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("\n[错误] 未找到DEEPSEEK_API_KEY环境变量")
        print("请先设置API密钥: export DEEPSEEK_API_KEY='your-api-key-here'")
        print("获取API密钥: https://platform.deepseek.com/api_keys\n")
        return False
    
    if not api_key.startswith("sk-"):
        print("\n[警告] DEEPSEEK_API_KEY格式可能不正确（通常以'sk-'开头）")
    
    print(f"[成功] 检测到DEEPSEEK_API_KEY: {api_key[:10]}...")
    return True


def run_deepseek_inference(
    model_name: str = "deepseek-reasoner",
    batch_size: int = 8,
    location: str = "Manhattan",
    start_time: int = 28800,
    end_time: int = 43200,
    step_size: float = 180.0,
    use_reflection: bool = True,
    multi_agent: bool = True,
    sumo_config: str = None,
    output_dir: str = None,
    av_ratio: float = 0.02
):
    """
    运行DeepSeek-Chat推理实验

    Args:
        model_name: DeepSeek模型名称 (默认: deepseek-reasoner)
        batch_size: 批处理大小
        location: 仿真位置 (默认: Manhattan)
        start_time: 仿真开始时间（秒）
        end_time: 仿真结束时间（秒）
        step_size: 仿真步长（秒）
        use_reflection: 是否使用反思机制
        multi_agent: 是否使用多智能体架构
        sumo_config: SUMO配置文件路径（可选）
        output_dir: 输出目录（可选）
        av_ratio: 自动驾驶车辆比例，仅从第一个路由文件中抽取此比例的车辆进行LLM决策 (默认: 0.02)
    """
    # 检查API密钥
    if not check_deepseek_api_key():
        sys.exit(1)

    # 设置文件路径 - 默认使用NYC路网
    if sumo_config is None:
        sumo_config = "/data/zhouyuping/LLMNavigation/Data/NYC/NewYork_sumo_config.sumocfg"

    # 从sumo_config推断其他文件路径
    base_dir = os.path.dirname(sumo_config)

    # NYC路网配置
    route_file = os.path.join(base_dir, "NewYork_od_0.1.rou.alt.xml")
    road_info_file = os.path.join(base_dir, "NewYork_road_info.json")
    adjacency_file = os.path.join(base_dir, "Region_1/edge_adjacency_alpha_1.json")
    task_info_file = os.path.join(base_dir, "task_info.json")
    region_data_dir = os.path.join(base_dir, "Region_1")

    # 如果文件不存在，尝试使用备用路径
    if not os.path.exists(route_file):
        route_file = "/data/zhouyuping/LLMNavigation/Data/NYC/NewYork_od_0.1.rou.alt.xml"
    if not os.path.exists(road_info_file):
        road_info_file = "/data/zhouyuping/LLMNavigation/Data/NYC/NewYork_road_info.json"
    if not os.path.exists(adjacency_file):
        adjacency_file = "/data/zhouyuping/LLMNavigation/Data/NYC/Region_1/edge_adjacency_alpha_1.json"
    if not os.path.exists(task_info_file):
        task_info_file = "/data/zhouyuping/LLMNavigation/Data/NYC/task_info.json"
    if not os.path.exists(region_data_dir):
        region_data_dir = "/data/zhouyuping/LLMNavigation/Data/NYC/Region_1"
    
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
        output_dir = f"logs/deepseek_inference_{location}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"DeepSeek-Chat 推理实验配置")
    print(f"{'='*60}")
    print(f"模型: {model_name}")
    print(f"API模式: DeepSeek API (OpenAI兼容)")
    print(f"仿真位置: {location}")
    print(f"SUMO配置: {sumo_config}")
    print(f"路由文件: {route_file}")
    print(f"仿真时间: {start_time}s - {end_time}s ({(end_time-start_time)/3600:.1f}小时)")
    print(f"步长: {step_size}s")
    print(f"批处理大小: {batch_size}")
    print(f"AV比例: {av_ratio*100:.1f}% (仅从第一个路由文件抽取)")
    print(f"使用反思: {'是' if use_reflection else '否'}")
    print(f"多智能体: {'是' if multi_agent else '否'}")
    print(f"训练模式: 关闭（仅推理）")
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
                llm_path_or_name=model_name,  # 使用deepseek-chat
                step_size=step_size,
                max_steps=max_steps,
                use_reflection=use_reflection,
                region_data_dir=region_data_dir,
                use_local_llm=False,  # 使用API模式
                enable_training=False,  # 关闭训练
                start_time=start_time,
                av_ratio=av_ratio  # 自动驾驶车辆比例（从第一个路由文件抽取）
            )
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
                llm_path_or_name=model_name,
                step_size=step_size,
                max_steps=max_steps,
                use_reflection=use_reflection
            )
        
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
        if hasattr(algo, 'llm_agent') and algo.llm_agent:
            token_file = os.path.join(output_dir, "token_usage.json")
            token_data = algo.llm_agent.save_token_usage(token_file)
            print(f"[成功] Token使用统计已保存到: {token_file}")
        elif hasattr(algo, 'sim') and hasattr(algo.sim, 'traffic_llm'):
            # 多智能体模式下保存所有LLM的token usage
            token_file_traffic = os.path.join(output_dir, "token_usage_traffic.json")
            token_file_regional = os.path.join(output_dir, "token_usage_regional.json")
            
            if algo.sim.traffic_llm:
                algo.sim.traffic_llm.save_token_usage(token_file_traffic)
                print(f"[成功] Traffic LLM Token使用统计已保存到: {token_file_traffic}")
            
            if hasattr(algo.sim, 'regional_llm') and algo.sim.regional_llm:
                algo.sim.regional_llm.save_token_usage(token_file_regional)
                print(f"[成功] Regional LLM Token使用统计已保存到: {token_file_regional}")
        
        print(f"\n[成功] 实验完成！所有结果已保存到: {output_dir}\n")
        
    except Exception as e:
        print(f"\n[错误] 实验失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用DeepSeek-Chat进行SUMO交通仿真推理实验（不训练）"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="deepseek-reasoner",
        help="DeepSeek模型名称 (默认: deepseek-chat)"
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
        default="Manhattan",
        help="仿真位置 (默认: Manhattan)"
    )

    parser.add_argument(
        "--av-ratio",
        type=float,
        default=0.02,
        help="从第一个路由文件中抽取的车辆比例进行LLM决策 (默认: 0.02 = 2%%)"
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
        default=None,
        help="SUMO配置文件路径（可选）"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        default=None,
        help="输出目录（可选）"
    )
    
    args = parser.parse_args()
    
    run_deepseek_inference(
        model_name=args.model,
        batch_size=args.batch_size,
        location=args.location,
        start_time=args.start_time,
        end_time=args.end_time,
        step_size=args.step_size,
        use_reflection=not args.no_reflection,
        multi_agent=not args.single_agent,
        sumo_config=args.sumo_config,
        output_dir=args.output_dir,
        av_ratio=args.av_ratio
    )

