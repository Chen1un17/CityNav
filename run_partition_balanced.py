import argparse
import os
import sys
import time
import random
from typing import Dict, List, Tuple, Optional, Set

from tqdm import tqdm
import networkx as nx
import traci
import wandb
import xml.etree.ElementTree as ET

os.environ["WANDB_MODE"] = "offline"
sys.path.append("../")

from env_utils import (
    parse_rou_file,
    get_multiple_edges_info,
    load_static_road_data,
)
from utils.read_utils import load_json


class PartitionBalancedPlanner(object):
    """
    分区感知的路径规划算法（独立实现，不依赖 SBP）。

    目标：
    - 仅从 路由1 中顺序选取 2% 车辆作为 AV（其余为背景车辆，不改路线）。
    - 使用 New 分区文件与邻接构建分区图，进行分区负载感知的加权最短路。
    - 实验配置：step_size=180，max_steps=43200。
    - 记录指标：
      平均 travel time（仅 AV）、平均等待时间（仅 AV，getAccumulatedWaitingTime）、
      平均 delay/timeLoss（仅 AV，getTimeLoss）、平均道路利用率（每步活跃边占比平均）、
      实时的已完成 AV 数。
    """

    def __init__(
        self,
        location: str,
        sumo_config: str,
        route1: str,
        route2: Optional[str],
        road_info_file: str,
        adjacency_file: str,
        partition_dir: str,
        step_size: float,
        max_steps: int,
        use_wandb: bool = False,
    ) -> None:
        self.location = location
        self.sumo_config = sumo_config
        self.route1 = route1
        self.route2 = route2
        self.road_info_file = road_info_file
        self.adjacency_file = adjacency_file
        self.partition_dir = partition_dir
        self.step_size = step_size
        self.max_steps = max_steps
        self.use_wandb = use_wandb

        # 车辆与指标追踪
        self.autonomous_vehicles: Set[str] = set()
        self.target_autonomous_count: int = 0
        self.processed_vehicles: Set[str] = set()
        self.vehicle_start_time: Dict[str, float] = {}
        self.vehicle_end_time: Dict[str, float] = {}
        self.vehicle_waiting_time_last: Dict[str, float] = {}
        self.vehicle_delay_last: Dict[str, float] = {}
        self.vehicle_waiting_time_final: Dict[str, float] = {}
        self.vehicle_delay_final: Dict[str, float] = {}

        self.completed_autonomous: int = 0
        self.average_travel_time: float = 0.0
        self.average_waiting_time: float = 0.0
        self.average_delay_time: float = 0.0
        self.step_utilization_values: List[float] = []

        # 路网与分区
        self.road_info: Dict = {}
        self.adjacency: Dict[str, List[str]] = {}
        self.road_network: nx.DiGraph = nx.DiGraph()
        self.edge_to_region: Dict[str, int] = {}
        self.boundary_edges: Dict[str, int] = {}
        self.partition_results: Dict = {}

        self.total_vehicles: int = 0
        self.route_lookup: Dict[str, Tuple[str, str]] = {}
        self.vehicle_routes: Dict[str, Dict] = {}

        print(f"Initialized PartitionBalancedPlanner for {location}")

    # ---------------------- 初始化 ----------------------
    def initialize(self) -> None:
        print("Starting SUMO simulation...")
        print(f"SUMO config: {self.sumo_config}")

        sumo_cmd = [
            "sumo", "-c", self.sumo_config,
            "--no-warnings",
            "--ignore-route-errors",
            "--no-step-log",
            "--time-to-teleport", "300",
            "--max-depart-delay", "900",
        ]

        traci.start(sumo_cmd)
        print("✓ SUMO started")

        # 加载路网数据
        self.road_info = load_json(self.road_info_file)
        self.adjacency = load_json(self.adjacency_file)

        # 分区数据（New 目录）
        edge_to_region_path = os.path.join(self.partition_dir, "edge_to_region_alpha_2.json")
        boundary_edges_path = os.path.join(self.partition_dir, "boundary_edges_alpha_2.json")
        partition_results_path = os.path.join(self.partition_dir, "partition_results_alpha_2_od_3.json")
        self.edge_to_region = load_json(edge_to_region_path)
        self.boundary_edges = load_json(boundary_edges_path)
        self.partition_results = load_json(partition_results_path)

        # 构建权重图：基础为道路长度，再叠加分区负载权重
        print("Building partition-aware road network graph...")
        self.road_network = nx.DiGraph()
        for edge in self.adjacency:
            if edge not in self.road_info:
                continue
            base_len = float(self.road_info[edge].get("road_len", 1.0))
            src_region = self.edge_to_region.get(edge, -1)
            for nxt in self.adjacency[edge]:
                if nxt not in self.road_info:
                    continue
                # 区域过边惩罚：跨区边在基础长度上乘以一个惩罚系数
                dst_region = self.edge_to_region.get(nxt, -1)
                cross_penalty = 1.0 if dst_region == src_region else 1.10
                weight = base_len * cross_penalty
                self.road_network.add_edge(edge, nxt, weight=weight)

        print(f"Road network: {self.road_network.number_of_nodes()} nodes, {self.road_network.number_of_edges()} edges")

        # 解析路由1用于 2% AV 选择与 trip 查表
        print("Parsing route1 for trips and counting vehicles...")
        trips1 = parse_rou_file(self.route1)
        self.total_vehicles = len(trips1)
        self.target_autonomous_count = int(0.02 * self.total_vehicles)
        self.route_lookup = {vid: (s, t) for vid, s, t, _ in trips1}
        print(f"Route1 vehicles: {self.total_vehicles} | Target AV (2%): {self.target_autonomous_count}")

        # 预加载静态数据到 env_utils，以便动态指标批量获取
        load_static_road_data()

        print("Initialization done. Ready to run.")

    # ---------------------- 路径规划核心 ----------------------
    def _current_partition_load_factor(self, edge_id: str) -> float:
        """根据当前分区负载返回一个权重放大因子（>1 更拥堵）。"""
        try:
            reg = self.edge_to_region.get(edge_id, -1)
            if reg == -1:
                return 1.0
            # 简化：使用该分区边上车辆数的归一化作为拥堵因子
            neighbors = list(self.road_network.successors(edge_id))[:8]
            edges = [edge_id] + neighbors
            info = get_multiple_edges_info(edges)
            total_veh = sum(max(0, info[e][1]) for e in info)
            # 以经验参数缩放，避免权重过大
            factor = 1.0 + min(total_veh / 60.0, 0.5)  # 上限 1.5x
            return factor
        except Exception:
            return 1.0

    def _compute_partition_aware_route(self, start_edge: str, end_edge: str) -> Optional[List[str]]:
        if start_edge == end_edge:
            return [start_edge]
        if start_edge not in self.road_network or end_edge not in self.road_network:
            return None
        # 在线调整边权：把邻近当前边的出边根据分区负载放大
        try:
            def dyn_weight(u, v, d):
                base = float(d.get("weight", 1.0))
                if u == start_edge:
                    return base * self._current_partition_load_factor(u)
                return base
            path = nx.dijkstra_path(self.road_network, source=start_edge, target=end_edge, weight=lambda u, v, d: dyn_weight(u, v, d))
            return path
        except Exception:
            try:
                res = traci.simulation.findRoute(fromEdge=start_edge, toEdge=end_edge)
                return list(res.edges) if res and hasattr(res, "edges") else None
            except Exception:
                return None

    def _apply_route(self, veh_id: str, route: List[str]) -> bool:
        try:
            if route and len(route) >= 1:
                traci.vehicle.setRoute(veh_id, route)
                return True
            return False
        except Exception:
            return False

    # ---------------------- 主循环 ----------------------
    def run(self) -> Tuple[float, int, float, float, float]:
        print("Starting Partition-Balanced simulation ...")
        print(f"Target autonomous (2% of route1): {self.target_autonomous_count}")

        random.seed(42)
        step = 0.0
        pbar = tqdm(total=self.max_steps, desc="Partition Simulation", unit="sec")

        try:
            while step < self.max_steps:
                traci.simulationStep(step)
                current_time = traci.simulation.getTime()

                vehicle_ids = traci.vehicle.getIDList()

                # 新出现车辆：按顺序从 route1 选择 2%
                for vid in vehicle_ids:
                    if vid in self.processed_vehicles:
                        continue
                    self.processed_vehicles.add(vid)

                    # 记录出发时间（可能延迟）
                    try:
                        self.vehicle_start_time[vid] = traci.vehicle.getDeparture(vid)
                    except Exception:
                        self.vehicle_start_time[vid] = current_time

                    # 仅从 route1 选 AV
                    if vid in self.route_lookup and len(self.autonomous_vehicles) < self.target_autonomous_count:
                        self.autonomous_vehicles.add(vid)
                        start_edge, end_edge = self.route_lookup[vid]
                        # 尽量用当前实际所在边作为起点
                        try:
                            cur_edge = traci.vehicle.getRoadID(vid) or start_edge
                        except Exception:
                            cur_edge = start_edge

                        route = self._compute_partition_aware_route(cur_edge, end_edge)
                        if route is not None and len(route) >= 1:
                            if self._apply_route(vid, route):
                                self.vehicle_routes[vid] = {"route": route, "start": cur_edge, "end": end_edge}
                        else:
                            # 失败则从 AV 集合移除
                            self.autonomous_vehicles.discard(vid)

                # 指标：等待与延迟的最后观测值
                present_set = set(vehicle_ids)
                for vid in list(self.autonomous_vehicles):
                    if vid not in present_set:
                        continue
                    try:
                        self.vehicle_waiting_time_last[vid] = traci.vehicle.getAccumulatedWaitingTime(vid)
                    except Exception:
                        pass
                    try:
                        self.vehicle_delay_last[vid] = traci.vehicle.getTimeLoss(vid)
                    except Exception:
                        pass

                # 到达车辆处理
                arrived = traci.simulation.getArrivedIDList()
                for vid in arrived:
                    if vid in self.vehicle_start_time and vid not in self.vehicle_end_time:
                        self.vehicle_end_time[vid] = current_time
                        if vid in self.autonomous_vehicles:
                            self.completed_autonomous += 1
                            if vid in self.vehicle_waiting_time_last:
                                self.vehicle_waiting_time_final[vid] = self.vehicle_waiting_time_last[vid]
                            if vid in self.vehicle_delay_last:
                                self.vehicle_delay_final[vid] = self.vehicle_delay_last[vid]

                # 道路利用率：活跃边/总边
                try:
                    active_edges = traci.edge.getIDList()
                    active_cnt = 0
                    veh_cnt = 0
                    for e in active_edges:
                        try:
                            if traci.edge.getLastStepVehicleNumber(e) > 0:
                                active_cnt += 1
                            veh_cnt += traci.edge.getLastStepVehicleNumber(e)
                        except Exception:
                            continue
                    total_edges = max(1, len(active_edges))
                    utilization = active_cnt / total_edges
                    self.step_utilization_values.append(utilization)
                except Exception:
                    pass

                # 步进
                step += self.step_size
                pbar.update(self.step_size)

                # 周期性输出
                if int(step) % 1800 == 0:
                    print(
                        f"Time {current_time:.0f}s | Active AV {len([v for v in vehicle_ids if v in self.autonomous_vehicles])} | "
                        f"Total AV {len(self.autonomous_vehicles)} | Completed AV {self.completed_autonomous}"
                    )

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        finally:
            pbar.close()
            self._finalize_metrics()

        avg_util = sum(self.step_utilization_values) / len(self.step_utilization_values) if self.step_utilization_values else 0.0
        return (
            self.average_travel_time,
            self.completed_autonomous,
            self.average_waiting_time,
            self.average_delay_time,
            avg_util,
        )

    # ---------------------- 结果处理 ----------------------
    def _finalize_metrics(self) -> None:
        # 仅对 AV 计算 travel time
        travel_times: List[float] = []
        for vid in self.autonomous_vehicles:
            if vid in self.vehicle_end_time and vid in self.vehicle_start_time:
                travel_times.append(self.vehicle_end_time[vid] - self.vehicle_start_time[vid])
        self.average_travel_time = sum(travel_times) / len(travel_times) if travel_times else 0.0

        waits = list(self.vehicle_waiting_time_final.values())
        self.average_waiting_time = sum(waits) / len(waits) if waits else 0.0

        delays = list(self.vehicle_delay_final.values())
        self.average_delay_time = sum(delays) / len(delays) if delays else 0.0

        avg_utilization = (
            sum(self.step_utilization_values) / len(self.step_utilization_values)
            if self.step_utilization_values
            else 0.0
        )

        print("\n" + "=" * 60)
        print("PARTITION-BALANCED ROUTING RESULTS")
        print("=" * 60)
        print(f"Average Travel Time (Autonomous): {self.average_travel_time:.2f}s")
        print(f"Average Waiting Time (Autonomous): {self.average_waiting_time:.2f}s")
        print(f"Average Delay/TimeLoss (Autonomous): {self.average_delay_time:.2f}s")
        print(f"Average Road Utilization (active edges): {avg_utilization:.4f}")
        print(f"Completed Autonomous Vehicles: {self.completed_autonomous}/{len(self.autonomous_vehicles)}")
        print("=" * 60)
        try:
            traci.close()
        except Exception:
            pass


# ---------------------- CLI ----------------------

def verify_required_files(files: List[str]) -> None:
    missing = [f for f in files if f and not os.path.exists(f)]
    if missing:
        print("ERROR: Missing required files:")
        for f in missing:
            print(f"  - {f}")
        sys.exit(1)


def main(location: str, step_size: float, max_steps: int, use_wandb: bool = False) -> Tuple[float, int]:
    # 用户给定 NYC 数据路径
    sumo_config = "/data/zhouyuping/LLMNavigation/Data/NYC/NewYork_sumo_config.sumocfg"
    route1 = "/data/zhouyuping/LLMNavigation/Data/NYC/NewYork_od_0.1.rou.alt.xml"
    route2 = "/data/zhouyuping/LLMNavigation/Data/NYC/NYC_routes_0.1_20250830_111509.alt.xml"
    road_info = "/data/zhouyuping/LLMNavigation/Data/NYC/NewYork_road_info.json"
    adjacency = "/data/zhouyuping/LLMNavigation/Data/New/edge_adjacency_alpha_2.json"
    partition_dir = "/data/zhouyuping/LLMNavigation/Data/New/"

    verify_required_files([sumo_config, route1, road_info, adjacency, partition_dir])
    if route2 and not os.path.exists(route2):
        print(f"⚠️ Optional route2 not found: {route2}")

    if use_wandb:
        wandb.init(
            project="USTBench-Route-Planning",
            group=f"{location}-Partition",
            name="Partition-Balanced-Experiment",
            config={
                "algorithm": "partition_balanced",
                "location": location,
                "step_size": step_size,
                "max_steps": max_steps,
                "autonomous_vehicle_ratio": 0.02,
            },
        )

    planner = PartitionBalancedPlanner(
        location=location,
        sumo_config=sumo_config,
        route1=route1,
        route2=route2,
        road_info_file=road_info,
        adjacency_file=adjacency,
        partition_dir=partition_dir,
        step_size=step_size,
        max_steps=max_steps,
        use_wandb=use_wandb,
    )

    planner.initialize()
    avg_tt, completed, avg_wait, avg_delay, avg_util = planner.run()

    if use_wandb:
        wandb.log(
            {
                "average_travel_time": avg_tt,
                "completed_autonomous": completed,
                "average_waiting_time": avg_wait,
                "average_delay_time": avg_delay,
                "average_road_utilization": avg_util,
            }
        )
        wandb.finish()

    return avg_tt, completed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run partition-aware routing algorithm on NYC data.")
    parser.add_argument("--location", type=str, default="NewYork", help="Simulation location label")
    parser.add_argument("--step-size", type=float, default=180.0, help="Simulation step size in seconds (default: 180)")
    parser.add_argument("--max-steps", type=int, default=43200, help="Simulation horizon in seconds (default: 43200)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")

    args = parser.parse_args()

    print("Partition-Aware Algorithm Experiment")
    print("=" * 50)
    print(f"Location: {args.location}")
    print(f"Step Size: {args.step_size}s")
    print(f"Max Steps: {args.max_steps}s")
    print(f"W&B Logging: {'Disabled' if args.no_wandb else 'Enabled'}")
    print("=" * 50)

    main(
        location=args.location,
        step_size=args.step_size,
        max_steps=args.max_steps,
        use_wandb=not args.no_wandb,
    )
