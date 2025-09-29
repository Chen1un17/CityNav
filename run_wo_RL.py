#!/usr/bin/env python3
"""
Run Multi-Agent Traffic Control with Qwen3-8B via DashScope API (no RL)

本脚本在保持与 run_multi_agent.py 相同的多智能体工作流和指标记录的前提下，
强制使用通义千问 DashScope OpenAI 兼容接口进行直接推理：
- 不采集任何 RL 数据
- 不进入 RL 训练流程
- 以 API LLM 替代本地共享 LLM

注意：请先配置环境变量 DASHSCOPE_API_KEY。
"""

import argparse
import os
import sys
from datetime import datetime

# 统一与 run_multi_agent.py 的 wandb 离线模式
os.environ["WANDB_MODE"] = "offline"

from main import MultiAgent_Route_Planning, Route_Planning  # 复用现有工作流
import xml.etree.ElementTree as ET


def check_qwen_api_key() -> bool:
    """校验 DashScope API Key（与 run_multi_agent.py 风格一致）。"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("[错误] 未设置DASHSCOPE_API_KEY环境变量")
        print("请设置通义千问API密钥:\n  export DASHSCOPE_API_KEY='sk-your-api-key'")
        return False
    if not api_key.startswith("sk-"):
        print("[错误] API密钥格式不正确，应该以'sk-'开头")
        return False
    print(f"[成功] 通义千问API密钥已配置: {api_key[:10]}...")
    return True

def check_requirements_api_mode(sumo_config: str,
                                road_info_file: str,
                                adjacency_file: str,
                                region_data_dir: str,
                                primary_route_file: str = None) -> bool:
    """最小化必需项检查：仅针对 API 推理模式与用户指定路径。
    校验以下文件/目录存在：sumocfg、(首个)route 文件、路网信息、邻接、分区目录。
    """
    print("Checking Inputs (API mode)...")
    missing = []
    if not os.path.exists(sumo_config):
        missing.append(sumo_config)
    if primary_route_file and not os.path.exists(primary_route_file):
        missing.append(primary_route_file)
    if not os.path.exists(road_info_file):
        missing.append(road_info_file)
    if not os.path.exists(adjacency_file):
        missing.append(adjacency_file)
    if not os.path.exists(region_data_dir):
        missing.append(region_data_dir)

    # 友好提示：分区目录下常见文件名（MultiAgent 环境默认读取 *_alpha_1）
    if os.path.isdir(region_data_dir):
        be1 = os.path.join(region_data_dir, "boundary_edges_alpha_2.json")
        er1 = os.path.join(region_data_dir, "edge_to_region_alpha_2.json")
        if not (os.path.exists(be1) and os.path.exists(er1)):
            # 容错：有些数据使用 *_alpha_2 命名
            be2 = os.path.join(region_data_dir, "boundary_edges_alpha_2.json")
            er2 = os.path.join(region_data_dir, "edge_to_region_alpha_2.json")
            if not (os.path.exists(be2) and os.path.exists(er2)):
                print("[警告] 分区目录中未发现 *_alpha_1 或 *_alpha_2 的边界/分区映射文件。")
                print("       MultiAgent 环境默认读取 *_alpha_1.json，请确认文件存在或命名一致。")

    if missing:
        print(" Missing files/dirs:")
        for m in missing:
            print(f"   - {m}")
        print("\n  Some inputs are missing. Please ensure all paths exist.")
        return False
    print(" All required inputs found!")
    return True


def run_api_multi_agent(llm_model: str,
                        batch_size: int,
                        location: str,
                        use_reflection: bool,
                        step_size: float,
                        max_steps: int,
                        disable_global_guidance: bool):
    """
    使用 DashScope API (Qwen3-8B) 运行多智能体工作流的便捷入口。
    - 强制使用 API LLM（不使用本地共享 LLM）
    - 显式关闭 RL 数据采集与训练
    - 复用 run_multi_agent/main 的数据与指标记录路径
    """

    # 使用用户指定的数据与分区
    sumo_config = "/data/zhouyuping/LLMNavigation/Data/NYC/NewYork_sumo_config.sumocfg"
    # 从 sumocfg 中解析 route-files，第一项作为 autonomous 候选来源
    primary_route_file = None
    background_route_file = None
    try:
        tree = ET.parse(sumo_config)
        root = tree.getroot()
        input_node = root.find('input')
        if input_node is not None:
            route_files_attr = input_node.find('route-files')
            if route_files_attr is not None:
                value = route_files_attr.get('value', '')
                routes = [p.strip() for p in value.split(',') if p.strip()]
                if routes:
                    primary_route_file = routes[0]
                    background_route_file = routes[1] if len(routes) > 1 else None
    except Exception as e:
        print(f"[WARN] 无法解析 sumocfg 的 route-files: {e}")

    # 主路由文件（仅用于抽取 autonomous vehicles）
    route_file = primary_route_file if primary_route_file else \
        "/data/zhouyuping/LLMNavigation/Data/NYC/NewYork_od_0.1.rou.alt.xml"

    # NYC 路网信息与全局边邻接（保持与现有脚本一致的文件结构）
    road_info_file = "/data/zhouyuping/LLMNavigation/Data/NYC/NewYork_road_info.json"
    adjacency_file = "/data/zhouyuping/LLMNavigation/Data/New/edge_adjacency_alpha_2.json"
    task_info_file = "/data/zhouyuping/LLMNavigation/Data/task_info.json"

    # 多智能体分区数据目录（用户指定的新分区目录）
    region_data_dir = "/data/zhouyuping/LLMNavigation/Data/New"

    # 全局宏观指导开关（与 run_multi_agent 自定义命令保持一致）
    try:
        if disable_global_guidance:
            os.environ['DISABLE_GLOBAL_GUIDANCE'] = '1'
        else:
            if 'DISABLE_GLOBAL_GUIDANCE' in os.environ:
                del os.environ['DISABLE_GLOBAL_GUIDANCE']
    except Exception:
        pass

    # 数据/依赖检查（使用 API 模式专用检查，避免 ./Data 误报）
    if not check_requirements_api_mode(
        sumo_config=sumo_config,
        road_info_file=road_info_file,
        adjacency_file=adjacency_file,
        region_data_dir=region_data_dir,
        primary_route_file=primary_route_file
    ):
        sys.exit(1)

    # API Key 检查
    if not check_qwen_api_key():
        sys.exit(1)

    # 使用 DashScope API（OpenAI 兼容）进行推理：
    # 关键点：传入一个包含 "dashscope" 的 llm 标识，以触发 LLM 类的 API 初始化分支，
    # 随后将 agent.llm_name 重置为用户期望的具体模型名（如 qwen3-8b）。
    # 这样既能获得正确的 DashScope 客户端，又能将真正的模型名传给 chat.completions。
    use_local_llm = False
    enable_training = False

    multi_agent = True
    if not os.path.exists(region_data_dir):
        print(f"Warning: Region data directory {region_data_dir} not found. Falling back to single-agent mode.")
        multi_agent = False

    if multi_agent:
        algo = MultiAgent_Route_Planning(
            batch_size=batch_size,
            location=location,
            sumo_config=sumo_config,
            route_file=route_file,
            road_info_file=road_info_file,
            adjacency_file=adjacency_file,
            task_info_file=task_info_file,
            llm_path_or_name="dashscope",  # 触发 DashScope API 模式
            step_size=step_size,
            max_steps=max_steps,
            use_reflection=use_reflection,
            region_data_dir=region_data_dir,
            use_local_llm=use_local_llm,
            enable_training=enable_training,
        )

        # 强制：使用 API，指定真实模型名，并关闭 RL 数据采集
        if hasattr(algo, 'llm_agent') and algo.llm_agent is not None:
            try:
                algo.llm_agent.llm_name = llm_model  # 确保 chat.completions 使用 qwen3-8b
            except Exception:
                pass
        if hasattr(algo, 'sim') and algo.sim is not None:
            try:
                # 显式关闭 RL 数据采集与训练相关机制
                algo.sim.rl_data_collection_enabled = False
                algo.sim.enable_time_sliced_training = False
                algo.sim.training_queue = None
            except Exception:
                pass

    else:
        # 单智能体降级（仍使用 DashScope API + 指定模型名）
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
            use_reflection=use_reflection,
        )
        if hasattr(algo, 'llm_agent') and algo.llm_agent is not None:
            try:
                algo.llm_agent.llm_name = llm_model
            except Exception:
                pass

    print("Multi-Agent Traffic Control System (API LLM: DashScope/Qwen)")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print("Configuration (No-RL API Inference):")
    print(f"  - Architecture: {'Multi-Agent' if multi_agent else 'Single-Agent'}")
    print(f"  - LLM Mode: API (DashScope)")
    print(f"  - Model: {llm_model}")
    if primary_route_file:
        print(f"  - Primary route file: {primary_route_file}")
    if background_route_file:
        print(f"  - Background route file: {background_route_file}")
    print(f"  - Reflection: {'Enabled' if use_reflection else 'Disabled'}")
    print(f"  - Location: {location}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Step Size: {step_size}s")
    print(f"  - Max Steps: {max_steps}")
    print(f"  - RL: Disabled (no data collection, no training)")
    print()

    algo.run()


def main_runner():
    parser = argparse.ArgumentParser(description="Run Multi-Agent (API LLM: Qwen3-8B via DashScope) without RL")
    parser.add_argument('--model', type=str, default='qwen3-8b',
                        help='DashScope 模型名（默认: qwen3-8b）')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='LLM 批处理大小（默认: 16）')
    parser.add_argument('--location', type=str, default='Manhattan',
                        help='仿真地点（默认: Manhattan）')
    parser.add_argument('--step-size', type=float, default=180.0,
                        help='仿真步长（秒，默认: 180.0）')
    parser.add_argument('--max-steps', type=int, default=86400,
                        help='最大仿真步数（默认: 43200）')
    parser.add_argument('--no-reflection', action='store_true',
                        help='禁用 LLM 自反思（默认启用）')
    parser.add_argument('--disable-global-guidance', action='store_true',
                        help='关闭每时间步的全局宏观指导')

    args = parser.parse_args()

    use_reflection = not args.no_reflection

    run_api_multi_agent(
        llm_model=args.model,
        batch_size=args.batch_size,
        location=args.location,
        use_reflection=use_reflection,
        step_size=args.step_size,
        max_steps=args.max_steps,
        disable_global_guidance=args.disable_global_guidance,
    )


if __name__ == "__main__":
    main_runner()


