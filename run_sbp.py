import argparse
import os
import sys
import time
import random
import json
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


class SBPRoutePlanner(object):
    """
    Streaming Batch Planning (SBP) 路由规划实现（基于论文算法 1/2/3 的工程化版本）。

    设计要点：
    - 只从“路由1”中顺序选取前 2% 的车辆作为自动驾驶车辆（Autonomous Vehicles, AVs）。
    - 对 AV 到来时执行 InitSearch 生成初始路径；
    - 每隔 T 秒执行一次 BatchRefining，在不打断仿真的前提下为仍在行驶的 AV 进行成批微调；
    - 维护“边标签” L（简化为每条边当前被多少条计划路径使用），以近似流量影响；
    - 统计：平均 travel time、平均等待时间、平均 delay(timeLoss)、平均道路利用率、实时已完成 AV 数量。
    """

    def __init__(
        self,
        location: str,
        sumo_config: str,
        route1: str,
        route2: Optional[str],
        road_info_file: str,
        adjacency_file: str,
        step_size: float,
        max_steps: int,
        refine_interval: int = 900,
        epsilon: float = 0.05,
        use_wandb: bool = False,
    ) -> None:
        self.location = location
        self.sumo_config = sumo_config
        self.route1 = route1
        self.route2 = route2
        self.road_info_file = road_info_file
        self.adjacency_file = adjacency_file
        self.step_size = step_size
        self.max_steps = max_steps
        self.refine_interval = refine_interval
        self.epsilon = epsilon
        self.use_wandb = use_wandb

        # 网络与数据
        self.road_info: Dict[str, dict] = {}
        self.adjacency: Dict[str, List[str]] = {}
        self.graph: Optional[nx.DiGraph] = None
        self.graph_rev: Optional[nx.DiGraph] = None
        self.edges: List[str] = []
        self._heuristic_cache: Dict[str, Dict[str, float]] = {}

        # 车辆/路径追踪
        self.primary_vehicle_ids: Set[str] = set()  # 仅路由1车辆
        self.target_autonomous_count: int = 0
        self.autonomous_vehicles: Set[str] = set()  # 已选为 AV 的车辆 ID
        self.processed_vehicles: Set[str] = set()   # 已处理过的车辆（避免重复）
        self.vehicle_start_time: Dict[str, float] = {}
        self.vehicle_end_time: Dict[str, float] = {}
        self.vehicle_waiting_time_final: Dict[str, float] = {}
        self.vehicle_delay_final: Dict[str, float] = {}
        self.vehicle_waiting_time_last: Dict[str, float] = {}
        self.vehicle_delay_last: Dict[str, float] = {}

        # 路由：veh_id -> {route: List[str], cost: float, labels: List[Tuple[str, float, float]]}
        self.vehicle_routes: Dict[str, Dict] = {}
        # L: edge_id -> [(enter_time, exit_time, veh_id)]
        self.edge_labels: Dict[str, List[Tuple[float, float, str]]] = {}
        # 当前批次（refine 周期内新增）车辆 ID
        self.current_batch: List[str] = []

        # 交通流参数（论文 Equation 1 对应项）
        self.capacity_per_lane: float = 20.0  # 参考论文实验设置（20~100）
        self.background_flow_ratio: float = 0.4  # 非查询流量占容量比例
        self.beta_param: float = 2.0  # 对应论文中的 β
        self.gamma_param: float = 2.0  # 对应论文中的 γ

        # 统计
        self.completed_autonomous: int = 0
        self.total_vehicles_primary: int = 0
        self.step_utilization_values: List[float] = []
        self.average_travel_time: float = 0.0
        self.average_waiting_time: float = 0.0
        self.average_delay_time: float = 0.0
        self.utilization_interval: int = max(900, refine_interval)  # 降频：与精炼间隔一致或更大
        self.step_results: List[Dict] = []  # 存储每步的结果

        print(
            f"Initialized SBP Route Planner for {location} | step={step_size}s, horizon={max_steps}s, T={refine_interval}s, eps={epsilon}"
        )

    # ---------------------- 初始化 ----------------------
    def initialize(self) -> None:
        print("Starting SUMO simulation ...")
        print(f"SUMO config: {self.sumo_config}")

        # 启动 SUMO（增强稳定性设置）
        sumo_cmd = [
            "sumo",
            "-c",
            self.sumo_config,
            "--no-warnings",
            "--ignore-route-errors",
            "--no-step-log",
            "--time-to-teleport",
            "300",
            "--max-depart-delay",
            "900",
        ]
        traci.start(sumo_cmd)
        print("✓ SUMO started successfully")

        # 载入静态数据
        print("Loading static road network data ...")
        self.road_info = load_json(self.road_info_file)
        self.adjacency = load_json(self.adjacency_file)
        self.graph = nx.DiGraph()
        for edge, neighbors in self.adjacency.items():
            if edge not in self.road_info:
                continue
            for n in neighbors:
                if n in self.road_info:
                    # 使用自由流旅行时间作为基础权重
                    t_ff = self._free_flow_time(n)
                    self.graph.add_edge(edge, n, weight=t_ff)

        # 构建反向图用于启发式预计算（u->dest 的自由流时间 == 反向图从 dest 到 u 的距离）
        self.graph_rev = self.graph.reverse(copy=False)

        print(
            f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges"
        )

        # 解析 sumocfg 中 route-files，仅用于核对；主集来源 route1
        self._check_sumocfg_routes()

        # 只解析 route1 以确定可选 AV 集合
        print("Parsing PRIMARY route file for candidate AVs ...")
        trips1 = parse_rou_file(self.route1)
        self.primary_vehicle_ids = {vid for vid, _, _, _ in trips1}
        self.total_vehicles_primary = len(self.primary_vehicle_ids)
        self.target_autonomous_count = max(1, int(0.02 * self.total_vehicles_primary))
        print(
            f"Primary vehicles: {self.total_vehicles_primary} | Target AVs (2% of primary): {self.target_autonomous_count}"
        )

        # 边列表
        self.edges = list(traci.edge.getIDList())

        # 预热本地静态缓存（env_utils 使用）
        try:
            load_static_road_data()
        except Exception:
            pass

    def _check_sumocfg_routes(self) -> None:
        try:
            tree = ET.parse(self.sumo_config)
            root = tree.getroot()
            input_node = root.find("input")
            route_files_attr = input_node.find("route-files") if input_node is not None else None
            if route_files_attr is None:
                print("⚠️ SUMO config has no <route-files>. Continue anyway.")
                return
            value = route_files_attr.get("value", "")
            routes_in_cfg = [p.strip() for p in value.split(",") if p.strip()]
            expected = [p for p in [self.route1, self.route2] if p]
            if expected and set(os.path.abspath(p) for p in expected) - set(
                os.path.abspath(p) for p in routes_in_cfg
            ):
                print(
                    "⚠️ Provided route1/route2 paths differ from sumocfg <route-files>. We'll still run using sumocfg contents."
                )
        except Exception as e:
            print(f"SUMO config parse warning: {e}")

    # ---------------------- 核心代价/搜索 ----------------------
    def _free_flow_time(self, edge_id: str) -> float:
        info = self.road_info.get(edge_id, {})
        road_len = float(info.get("road_len", 100.0))
        speed_limit = float(info.get("speed_limit", 13.89))  # m/s (~50km/h)
        return road_len / max(speed_limit, 1e-3)

    def _edge_penalty_factor(self, edge_id: str) -> float:
        """
        根据边标签 L 对旅行时间进行放大。简化建模：
        time = t_ff * (1 + alpha * load/lanes)
        """
        load = self.edge_labels.get(edge_id, 0)
        lanes = max(1, int(self.road_info.get(edge_id, {}).get("lane_num", 1)))
        alpha = 0.3  # 经验系数
        return 1.0 + alpha * (load / lanes)

    def _estimate_edge_time(self, edge_id: str) -> float:
        return self._free_flow_time(edge_id) * self._edge_penalty_factor(edge_id)

    def _route_cost(self, route: List[str]) -> float:
        return sum(self._estimate_edge_time(e) for e in route)

    def _get_heuristic_map(self, dest_edge: str) -> Dict[str, float]:
        """返回从任意边到 dest_edge 的自由流最短时间，带缓存。"""
        if dest_edge in self._heuristic_cache:
            return self._heuristic_cache[dest_edge]
        if self.graph_rev is None:
            self._heuristic_cache[dest_edge] = {}
            return self._heuristic_cache[dest_edge]
        try:
            h_map: Dict[str, float] = nx.single_source_dijkstra_path_length(
                self.graph_rev, source=dest_edge, weight="weight"
            )
        except Exception:
            h_map = {}
        self._heuristic_cache[dest_edge] = h_map
        return h_map

    def _init_search(self, start_edge: str, end_edge: str) -> Optional[List[str]]:
        """
        Algorithm 1: InitSearch(G, q, L) 的工程化实现（以边为节点）。
        优化：改为 A*，启发式为自由流到达时间的下界 H（单源反向 Dijkstra 预计算并缓存）。
        """
        if start_edge == end_edge:
            return [start_edge]
        if (
            self.graph is None
            or start_edge not in self.graph
            or end_edge not in self.graph
        ):
            return None

        import heapq

        h_map = self._get_heuristic_map(end_edge)

        gscore: Dict[str, float] = {start_edge: 0.0}
        came_from: Dict[str, Optional[str]] = {start_edge: None}
        open_heap: List[Tuple[float, float, str]] = []  # (f, g, node)
        f0 = gscore[start_edge] + h_map.get(start_edge, 0.0)
        heapq.heappush(open_heap, (f0, 0.0, start_edge))
        closed: Set[str] = set()

        while open_heap:
            f_u, g_u, u = heapq.heappop(open_heap)
            if u in closed:
                continue
            closed.add(u)
            if u == end_edge:
                # 回溯
                route: List[str] = []
                cur = u
                while cur is not None:
                    route.append(cur)
                    cur = came_from[cur]
                route.reverse()
                return route

            for v in self.graph.successors(u):
                w = self._estimate_edge_time(v)
                tentative_g = g_u + w
                if tentative_g < gscore.get(v, float("inf")):
                    gscore[v] = tentative_g
                    came_from[v] = u
                    f_v = tentative_g + h_map.get(v, 0.0)
                    heapq.heappush(open_heap, (f_v, tentative_g, v))

        return None

    def _apply_route(self, veh_id: str, route: List[str]) -> bool:
        try:
            traci.vehicle.setRoute(veh_id, route)
            return True
        except Exception:
            return False

    def _add_labels_for_route(self, route: List[str], sign: int) -> None:
        for e in route:
            self.edge_labels[e] = max(0, self.edge_labels.get(e, 0) + sign)

    def _batch_refining(self, active_vehicles: List[str]) -> int:
        """
        Algorithm 2: 对当前批次 AV 的路径进行微调。若新路径 cost 明显更优(> eps)则替换。
        返回成功替换的数量。
        """
        replaced = 0
        for vid in active_vehicles:
            if vid not in self.vehicle_routes:
                continue
            old_info = self.vehicle_routes[vid]
            old_route = old_info.get("route", [])
            if not old_route:
                continue

            # 获取当前所在边与目的地
            try:
                current_edge = traci.vehicle.getRoadID(vid)
                dest_edge = traci.vehicle.getRoute(vid)[-1]
            except Exception:
                continue

            # 临时去除旧路径的边标签，避免自我影响
            self._add_labels_for_route(old_route, sign=-1)
            new_route = self._init_search(current_edge, dest_edge)
            if new_route is None or len(new_route) < 2:
                # 失败则恢复标签
                self._add_labels_for_route(old_route, sign=+1)
                continue

            old_cost = self._route_cost(old_route)
            new_cost = self._route_cost(new_route)
            # 接受条件：新成本显著更低
            if new_cost < old_cost * (1.0 - self.epsilon):
                if self._apply_route(vid, new_route):
                    # 应用成功：更新记录与标签
                    self.vehicle_routes[vid] = {"route": new_route, "cost": new_cost}
                    self._add_labels_for_route(new_route, sign=+1)
                    replaced += 1
                else:
                    # 应用失败，恢复旧标签
                    self._add_labels_for_route(old_route, sign=+1)
            else:
                # 恢复旧标签
                self._add_labels_for_route(old_route, sign=+1)

        return replaced

    # ---------------------- 主循环 ----------------------
    def run(self) -> Tuple[float, int]:
        print("Starting SBP traffic simulation ...")
        print(
            f"Target autonomous (2% of route1): {self.target_autonomous_count} | refine T={self.refine_interval}s, eps={self.epsilon}"
        )

        random.seed(42)
        step = 0.0
        last_refine_time = 0.0
        pbar = tqdm(total=self.max_steps, desc="SBP Simulation", unit="sec")

        try:
            while step < self.max_steps:
                traci.simulationStep(step)
                current_time = traci.simulation.getTime()

                # 新出现车辆：按顺序从 route1 选择 2%
                vehicle_ids = traci.vehicle.getIDList()
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
                    if (
                        vid in self.primary_vehicle_ids
                        and len(self.autonomous_vehicles) < self.target_autonomous_count
                    ):
                        # 选择为 AV，并立即进行初始规划
                        try:
                            route_edges = traci.vehicle.getRoute(vid)
                            if len(route_edges) < 2:
                                continue
                            start_edge = route_edges[0]
                            end_edge = route_edges[-1]
                            # 车辆当前边（更准确的起点）
                            try:
                                current_edge = traci.vehicle.getRoadID(vid)
                                if current_edge:
                                    start_edge = current_edge
                            except Exception:
                                pass

                            new_route = self._init_search(start_edge, end_edge)
                            if new_route is not None and len(new_route) >= 2:
                                applied = self._apply_route(vid, new_route)
                                if applied:
                                    self.autonomous_vehicles.add(vid)
                                    cost = self._route_cost(new_route)
                                    self.vehicle_routes[vid] = {
                                        "route": new_route,
                                        "cost": cost,
                                    }
                                    self._add_labels_for_route(new_route, sign=+1)
                        except Exception:
                            # 若任一步骤失败，则忽略该车的 AV 资格（继续作为背景车）
                            pass

                # 更新在途 AV 的实时等待/延迟（保存“最后一次看到”的累计值）
                present_set = set(vehicle_ids)
                for vid in list(self.autonomous_vehicles):
                    if vid not in present_set:
                        continue  # 仅对当前在仿真的车辆调用 TraCI，避免 Unknown Vehicle 错误日志
                    try:
                        self.vehicle_waiting_time_last[vid] = traci.vehicle.getAccumulatedWaitingTime(vid)
                    except Exception:
                        pass
                    try:
                        self.vehicle_delay_last[vid] = traci.vehicle.getTimeLoss(vid)
                    except Exception:
                        pass

                # 检查已到达车辆
                arrived = traci.simulation.getArrivedIDList()
                for vid in arrived:
                    if vid in self.vehicle_start_time and vid not in self.vehicle_end_time:
                        self.vehicle_end_time[vid] = current_time
                        if vid in self.autonomous_vehicles:
                            self.completed_autonomous += 1
                            # 最后一次观察到的累计等待/延迟作为该车终值
                            if vid in self.vehicle_waiting_time_last:
                                self.vehicle_waiting_time_final[vid] = self.vehicle_waiting_time_last[vid]
                            if vid in self.vehicle_delay_last:
                                self.vehicle_delay_final[vid] = self.vehicle_delay_last[vid]
                            # 从标签中移除其路径影响
                            if vid in self.vehicle_routes:
                                self._add_labels_for_route(self.vehicle_routes[vid].get("route", []), sign=-1)

                # 降频计算道路利用率（每 utilization_interval 计算一次），并对活跃边采样
                try:
                    if (
                        self.utilization_interval > 0
                        and int(current_time) % self.utilization_interval == 0
                    ):
                        active_edges = set()
                        for vid in vehicle_ids:
                            try:
                                e = traci.vehicle.getRoadID(vid)
                                if e:
                                    active_edges.add(e)
                            except Exception:
                                continue
                        if active_edges:
                            # 采样以减少 TraCI 订阅规模
                            sample_size = min(500, len(active_edges))
                            if sample_size < len(active_edges):
                                sampled = random.sample(list(active_edges), sample_size)
                            else:
                                sampled = list(active_edges)

                            multi = get_multiple_edges_info(sampled)
                            ratios = []
                            for e, tpl in multi.items():
                                try:
                                    lane_num, vehicle_num, _, _, _ = tpl
                                    cap = max(1, lane_num * 10)
                                    ratios.append(min(vehicle_num / cap, 1.0))
                                except Exception:
                                    pass
                            if ratios:
                                self.step_utilization_values.append(sum(ratios) / len(ratios))
                except Exception:
                    pass

                # 周期精炼
                if self.refine_interval > 0 and int(current_time) - int(last_refine_time) >= self.refine_interval:
                    # 仅对仍在仿真的 AV 执行精炼，避免 Unknown Vehicle 日志
                    current_present = set(traci.vehicle.getIDList())
                    active_avs = [vid for vid in self.autonomous_vehicles if vid in current_present]
                    replaced = self._batch_refining(active_avs)
                    last_refine_time = current_time
                    if replaced > 0:
                        print(f"Refined {replaced} routes at t={int(current_time)}s")

                # 进度与可视化输出
                if int(step) % max(1, int(600 / max(self.step_size, 1))) == 0:  # 大约每 ~10 分钟一次
                    print(
                        f"t={int(current_time)}s | AV selected={len(self.autonomous_vehicles)}/{self.target_autonomous_count} | completed AV={self.completed_autonomous}"
                    )

                # 保存当前步骤结果
                self._save_step_result(step, current_time)
                
                # 前进
                step += self.step_size
                pbar.update(self.step_size)

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        finally:
            pbar.close()
            self._finalize_metrics()

        return self.average_travel_time, self.completed_autonomous

    # ---------------------- 结果处理 ----------------------
    def _finalize_metrics(self) -> None:
        # 仅对 AV 计算
        travel_times: List[float] = []
        for vid in self.autonomous_vehicles:
            if vid in self.vehicle_end_time and vid in self.vehicle_start_time:
                travel_times.append(
                    self.vehicle_end_time[vid] - self.vehicle_start_time[vid]
                )
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
        print("SBP ROUTING SIMULATION RESULTS")
        print("=" * 60)
        print(f"Average Travel Time (Autonomous): {self.average_travel_time:.2f}s")
        print(f"Average Waiting Time (Autonomous): {self.average_waiting_time:.2f}s")
        print(f"Average Delay/TimeLoss (Autonomous): {self.average_delay_time:.2f}s")
        print(f"Average Road Utilization (active edges): {avg_utilization:.4f}")
        print(f"Completed Autonomous Vehicles: {self.completed_autonomous}/{len(self.autonomous_vehicles)}")
        print("=" * 60)
        
        # 保存最终结果
        self._save_final_results()

        # 关闭 SUMO
        try:
            traci.close()
        except Exception:
            pass

    def _save_step_result(self, current_step: float, current_time: float) -> None:
        """保存当前步骤的结果到内存，定期写入单个JSON文件"""
        try:
            # 计算当前已完成车辆的平均指标
            completed_avs = []
            for vid in self.autonomous_vehicles:
                if vid in self.vehicle_end_time:
                    travel_time = self.vehicle_end_time[vid] - self.vehicle_start_time.get(vid, 0)
                    wait_time = self.vehicle_waiting_time_final.get(vid, 0)
                    delay_time = self.vehicle_delay_final.get(vid, 0)
                    completed_avs.append({
                        'travel_time': travel_time,
                        'wait_time': wait_time,
                        'delay_time': delay_time
                    })

            # 当前在途车辆数量
            current_vehicles = len(traci.vehicle.getIDList())
            
            # 当前平均指标
            avg_travel_time = sum(av['travel_time'] for av in completed_avs) / len(completed_avs) if completed_avs else 0
            avg_wait_time = sum(av['wait_time'] for av in completed_avs) / len(completed_avs) if completed_avs else 0
            avg_delay_time = sum(av['delay_time'] for av in completed_avs) / len(completed_avs) if completed_avs else 0
            
            # 当前道路利用率
            current_utilization = self.step_utilization_values[-1] if self.step_utilization_values else 0.0
            
            step_data = {
                'step': int(current_step),
                'time': current_time,
                'current_vehicles': current_vehicles,
                'completed_autonomous': self.completed_autonomous,
                'total_autonomous_selected': len(self.autonomous_vehicles),
                'target_autonomous': self.target_autonomous_count,
                'avg_travel_time': avg_travel_time,
                'avg_wait_time': avg_wait_time,
                'avg_delay_time': avg_delay_time,
                'road_utilization': current_utilization,
                'edge_labels_count': len([e for e, count in self.edge_labels.items() if count > 0])
            }
            
            self.step_results.append(step_data)
            
            # 定期保存到单个JSON文件 (每100步或每30分钟)
            if (int(current_step) % 180 == 0) or (int(current_time) % 1800 == 0):
                os.makedirs("outputs", exist_ok=True)
                output_file = f"outputs/sbp_{self.location}_progress.json"
                progress_data = {
                    'location': self.location,
                    'last_update_time': current_time,
                    'last_update_step': int(current_step),
                    'step_results': self.step_results
                }
                with open(output_file, 'w') as f:
                    json.dump(progress_data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Failed to save step result: {e}")

    def _save_final_results(self) -> None:
        """保存最终的完整结果"""
        try:
            final_result = {
                'location': self.location,
                'simulation_config': {
                    'step_size': self.step_size,
                    'max_steps': self.max_steps,
                    'refine_interval': self.refine_interval,
                    'epsilon': self.epsilon
                },
                'final_metrics': {
                    'average_travel_time': self.average_travel_time,
                    'average_waiting_time': self.average_waiting_time,
                    'average_delay_time': self.average_delay_time,
                    'completed_autonomous': self.completed_autonomous,
                    'total_autonomous_selected': len(self.autonomous_vehicles),
                    'target_autonomous': self.target_autonomous_count,
                    'total_primary_vehicles': self.total_vehicles_primary,
                    'avg_road_utilization': sum(self.step_utilization_values) / len(self.step_utilization_values) if self.step_utilization_values else 0.0
                },
                'step_by_step_results': self.step_results
            }
            
            output_file = f"outputs/sbp_{self.location}_final_results.json"
            os.makedirs("outputs", exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(final_result, f, indent=2)
                
            print(f"\nResults saved to {output_file}")
            
        except Exception as e:
            print(f"Warning: Failed to save final results: {e}")


def verify_required_files(files: List[str]) -> None:
    missing = [f for f in files if f and not os.path.exists(f)]
    if missing:
        print("ERROR: Missing required files:")
        for f in missing:
            print(f"  - {f}")
        sys.exit(1)


def main(
    location: str,
    step_size: float,
    max_steps: int,
    epsilon: float,
    refine_interval: int,
    use_wandb: bool = False,
):
    # Region_1 Manhattan 数据（用户要求）
    sumo_config = "/data/zhouyuping/LLMNavigation/Data/Region_1/Manhattan_sumo_config.sumocfg"
    route1 = "/data/zhouyuping/LLMNavigation/Data/Region_1/Manhattan_od_0.01.rou.alt.xml"
    route2 = None  # 只使用一个路由文件
    road_info = "/data/zhouyuping/LLMNavigation/Data/Region_1/Manhattan_road_info.json"
    adjacency = "/data/zhouyuping/LLMNavigation/Data/Region_1/edge_adjacency_alpha_1.json"

    verify_required_files([sumo_config, route1, road_info, adjacency])
    # route2 设置为 None，只使用单一路由文件
    if route2 and not os.path.exists(route2):
        print(f"⚠️ Optional route2 not found: {route2}")
    else:
        print("Using single route file (route1 only)")

    if use_wandb:
        wandb.init(
            project="USTBench-Route-Planning",
            group=f"{location}-SBP",
            name=f"SBP-StreamingBatchPlanning",
            config={
                "algorithm": "sbp",
                "location": location,
                "step_size": step_size,
                "max_steps": max_steps,
                "epsilon": epsilon,
                "refine_interval": refine_interval,
                "autonomous_vehicle_ratio": 0.02,
            },
        )

    planner = SBPRoutePlanner(
        location=location,
        sumo_config=sumo_config,
        route1=route1,
        route2=route2,
        road_info_file=road_info,
        adjacency_file=adjacency,
        step_size=step_size,
        max_steps=max_steps,
        refine_interval=refine_interval,
        epsilon=epsilon,
        use_wandb=use_wandb,
    )
    planner.initialize()
    avg_tt, completed = planner.run()

    if use_wandb:
        wandb.log(
            {
                "average_travel_time": avg_tt,
                "completed_autonomous": completed,
            }
        )
        wandb.finish()

    return avg_tt, completed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SUMO simulation with Streaming Batch Planning (SBP) routing.",
    )
    parser.add_argument(
        "--location",
        type=str,
        default="Manhattan",
        help="Simulation location label",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=180.0,
        help="Simulation step size in seconds (default: 180)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=43200,
        help="Simulation horizon in seconds (default: 43200, i.e., 12 hours)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Batch refining acceptance threshold (fractional improvement)",
    )
    parser.add_argument(
        "--refine-interval",
        type=int,
        default=1800,
        help="Batch refining interval in seconds (default: 900)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )

    args = parser.parse_args()

    print("SBP Route Planning System")
    print("=" * 50)
    print(f"Location: {args.location}")
    print(f"Step Size: {args.step_size}s")
    print(f"Max Steps: {args.max_steps}s")
    print(f"Refine Interval: {args.refine_interval}s | Epsilon: {args.epsilon}")
    print(f"W&B Logging: {'Disabled' if args.no_wandb else 'Enabled'}")
    print("=" * 50)

    main(
        location=args.location,
        step_size=args.step_size,
        max_steps=args.max_steps,
        epsilon=args.epsilon,
        refine_interval=args.refine_interval,
        use_wandb=not args.no_wandb,
    )


