#!/usr/bin/env python3
"""
Dijkstra路径规划实验
使用指定的SUMO配置文件和路由文件，选择2%车辆进行Dijkstra最短路径规划
实时记录指标并保存到CSV文件
"""

import os
import sys
import csv
import random
import argparse
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import xml.etree.ElementTree as ET

import networkx as nx
import traci
from tqdm import tqdm

sys.path.append("../")
from env_utils import parse_rou_file


class DijkstraExperiment:
    """
    基于Dijkstra算法的路径规划实验
    - 选择2%车辆作为自动驾驶车辆进行路径规划
    - 其他车辆按原路由文件行驶
    - 实时记录各项指标并保存到CSV
    """
    
    def __init__(self, sumo_config: str, step_size: float = 1.0, max_steps: int = 43200, av_ratio: float = 0.02, use_astar: bool = False, location: str = "Manhattan"):
        self.sumo_config = sumo_config
        self.step_size = step_size
        self.max_steps = max_steps
        self.av_ratio = av_ratio
        self.use_astar = use_astar  # 是否使用A*算法
        self.location = location  # 仿真位置

        # 从sumocfg解析路径
        self.net_file = None
        self.route_file = None
        self._parse_sumo_config()

        # 路网数据
        self.road_network: nx.DiGraph = nx.DiGraph()
        self.edge_info: Dict = {}
        self.edge_coordinates: Dict[str, Tuple[float, float]] = {}  # 边的坐标（用于A*启发式）

        # 性能优化：LRU路径缓存和批量处理
        self._route_cache = OrderedDict()  # LRU缓存
        self._cache_hits = 0
        self._cache_misses = 0
        self._max_cache_size = 100000  # 增大缓存（NYC路网更大）

        # 预计算优化
        self._precomputed_paths = {}
        self._use_precompute = True

        # 车辆管理
        self.autonomous_vehicles: set = set()
        self.all_vehicles_in_route: List[str] = []
        self.vehicle_start_times: Dict[str, float] = {}
        self.vehicle_end_times: Dict[str, float] = {}
        self.processed_vehicles: set = set()
        self.completed_autonomous_vehicles: set = set()  # 跟踪已完成的autonomous vehicles
        self.vehicle_waiting_times: Dict[str, float] = {}  # 记录每个车辆的累积等待时间

        # 统计指标
        self.metrics_history: List[Dict] = []

        # CSV输出
        self.csv_file = None
        self.csv_writer = None
        
    def _parse_sumo_config(self):
        """解析SUMO配置文件获取网络文件和路由文件路径"""
        try:
            tree = ET.parse(self.sumo_config)
            root = tree.getroot()
            input_elem = root.find('input')

            if input_elem is not None:
                # 获取网络文件
                net_elem = input_elem.find('net-file')
                if net_elem is not None:
                    self.net_file = net_elem.get('value')
                    if not os.path.isabs(self.net_file):
                        self.net_file = os.path.join(os.path.dirname(self.sumo_config), self.net_file)

                # 获取路由文件（只取第一个）
                route_elem = input_elem.find('route-files')
                if route_elem is not None:
                    route_value = route_elem.get('value')
                    # 支持多个路由文件（逗号或空格分隔），只取第一个
                    route_files = route_value.replace(',', ' ').split()
                    if route_files:
                        self.route_file = route_files[0]
                        if not os.path.isabs(self.route_file):
                            self.route_file = os.path.join(os.path.dirname(self.sumo_config), self.route_file)

                        if len(route_files) > 1:
                            print(f"注意：配置文件中有{len(route_files)}个路由文件，只使用第一个进行车辆采样")

        except Exception as e:
            print(f"错误：解析SUMO配置文件失败: {e}")
            sys.exit(1)

        if not self.net_file or not self.route_file:
            print("错误：未能从SUMO配置文件中解析到网络文件或路由文件")
            sys.exit(1)

        print(f"网络文件: {self.net_file}")
        print(f"路由文件（采样源）: {self.route_file}")
    
    def _build_road_network(self):
        """构建路网图用于Dijkstra/A*算法"""
        algo_name = "A*" if self.use_astar else "Dijkstra"
        print(f"构建路网图（{algo_name}算法）...")

        # 解析SUMO网络文件
        try:
            tree = ET.parse(self.net_file)
            root = tree.getroot()

            # 添加所有边及其坐标信息
            for edge in root.findall('edge'):
                edge_id = edge.get('id')
                if edge_id and not edge_id.startswith(':'):  # 跳过内部边
                    # 获取边的长度作为权重
                    length = 0.0
                    for lane in edge.findall('lane'):
                        lane_length = float(lane.get('length', 0))
                        if lane_length > length:
                            length = lane_length

                    # 获取速度限制
                    speed = 13.89  # 默认速度 m/s
                    for lane in edge.findall('lane'):
                        lane_speed = float(lane.get('speed', speed))
                        speed = max(speed, lane_speed)

                    self.edge_info[edge_id] = {
                        'length': length,
                        'speed': speed
                    }

                    # 如果使用A*，获取边的坐标信息（用于启发式函数）
                    if self.use_astar:
                        for lane in edge.findall('lane'):
                            shape_str = lane.get('shape', '')
                            if shape_str:
                                try:
                                    # 解析形状字符串 "x1,y1 x2,y2 ..."
                                    points = []
                                    for point_str in shape_str.split():
                                        x, y = map(float, point_str.split(','))
                                        points.append((x, y))

                                    if points:
                                        # 计算中心点作为边的代表坐标
                                        center_x = sum(p[0] for p in points) / len(points)
                                        center_y = sum(p[1] for p in points) / len(points)
                                        self.edge_coordinates[edge_id] = (center_x, center_y)
                                    break
                                except:
                                    continue
            
            # 优化：批量添加连接关系
            connections_data = []
            for connection in root.findall('connection'):
                from_edge = connection.get('from')
                to_edge = connection.get('to')
                
                if (from_edge and to_edge and 
                    from_edge in self.edge_info and 
                    to_edge in self.edge_info and
                    not from_edge.startswith(':') and 
                    not to_edge.startswith(':')):
                    
                    weight = self.edge_info[from_edge]['length']
                    connections_data.append((from_edge, to_edge, weight))
            
            # 批量添加连接（更高效）
            self.road_network.add_weighted_edges_from(connections_data)
                        
        except Exception as e:
            print(f"警告：无法解析网络文件，将在运行时构建路网: {e}")
            
        print(f"路网图: {self.road_network.number_of_nodes()} 节点, {self.road_network.number_of_edges()} 边")
    
    def _build_network_from_traci(self):
        """从TraCI构建路网图（备用方法）"""
        print("从TraCI构建路网图...")
        
        edges = traci.edge.getIDList()
        for edge_id in edges:
            if not edge_id.startswith(':'):  # 跳过内部边
                try:
                    length = traci.lane.getLength(edge_id + '_0')
                    speed = traci.lane.getMaxSpeed(edge_id + '_0')
                    self.edge_info[edge_id] = {
                        'length': length,
                        'speed': speed
                    }
                except:
                    continue
        
        # 优化：批量构建连接关系
        connections_data = []
        for edge_id in list(self.edge_info.keys()):
            try:
                outgoing = traci.edge.getOutgoing(edge_id)
                weight = self.edge_info[edge_id]['length']
                for next_edge in outgoing:
                    if next_edge in self.edge_info:
                        connections_data.append((edge_id, next_edge, weight))
            except:
                continue
        
        # 批量添加连接
        self.road_network.add_weighted_edges_from(connections_data)
                
        print(f"路网图: {self.road_network.number_of_nodes()} 节点, {self.road_network.number_of_edges()} 边")
    
    def _euclidean_distance(self, edge1: str, edge2: str) -> float:
        """计算两个边之间的欧几里得距离（用于A*启发式）"""
        if edge1 not in self.edge_coordinates or edge2 not in self.edge_coordinates:
            return 0.0

        x1, y1 = self.edge_coordinates[edge1]
        x2, y2 = self.edge_coordinates[edge2]
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def _compute_dijkstra_route(self, start_edge: str, end_edge: str) -> Optional[List[str]]:
        """计算最短路径（支持Dijkstra和A*算法，带LRU缓存优化）"""
        if not start_edge or not end_edge:
            return None
        if start_edge == end_edge:
            return [start_edge]
        if start_edge not in self.road_network or end_edge not in self.road_network:
            return None

        # 检查LRU缓存
        cache_key = (start_edge, end_edge)
        if cache_key in self._route_cache:
            self._cache_hits += 1
            # LRU: 移到末尾表示最近使用
            self._route_cache.move_to_end(cache_key)
            return self._route_cache[cache_key]

        self._cache_misses += 1

        try:
            if self.use_astar:
                # 使用A*算法（带启发式函数，对大路网更高效）
                def heuristic(u, v):
                    # 欧几里得距离作为启发式函数
                    return self._euclidean_distance(u, v)

                path = nx.astar_path(
                    self.road_network,
                    source=start_edge,
                    target=end_edge,
                    heuristic=heuristic,
                    weight='weight'
                )
            else:
                # 使用双向Dijkstra算法（比单向快2倍以上）
                _, path = nx.bidirectional_dijkstra(
                    self.road_network,
                    source=start_edge,
                    target=end_edge,
                    weight='weight'
                )

            # LRU缓存：满了删除最旧的
            if len(self._route_cache) >= self._max_cache_size:
                self._route_cache.popitem(last=False)  # 删除最早的项
            self._route_cache[cache_key] = path

            return path
        except:
            # 缓存失败结果避免重复尝试
            if len(self._route_cache) >= self._max_cache_size:
                self._route_cache.popitem(last=False)
            self._route_cache[cache_key] = None
            return None
    
    def _replan_vehicle_route(self, vehicle_id: str):
        """优化的车辆路径重新规划方法"""
        if vehicle_id not in self.vehicle_od:
            return
            
        start_edge, end_edge = self.vehicle_od[vehicle_id]
        try:
            current_edge = traci.vehicle.getRoadID(vehicle_id)
            if current_edge and not current_edge.startswith(':'):
                new_route = self._compute_dijkstra_route(current_edge, end_edge)
                if new_route and len(new_route) > 1:
                    traci.vehicle.setRoute(vehicle_id, new_route)
        except:
            # 如果获取当前位置失败，使用起始边
            new_route = self._compute_dijkstra_route(start_edge, end_edge)
            if new_route:
                try:
                    traci.vehicle.setRoute(vehicle_id, new_route)
                except:
                    pass
    
    def _select_autonomous_vehicles(self):
        """选择2%车辆作为自动驾驶车辆，并预计算常用路径"""
        try:
            # 解析路由文件获取所有车辆
            trips = parse_rou_file(self.route_file)
            self.all_vehicles_in_route = [vid for vid, _, _, _ in trips]

            # 随机选择2%车辆
            random.seed(42)  # 确保可重复性
            num_av = int(len(self.all_vehicles_in_route) * self.av_ratio)
            selected_vehicles = random.sample(self.all_vehicles_in_route, num_av)
            self.autonomous_vehicles = set(selected_vehicles)

            print(f"总车辆数: {len(self.all_vehicles_in_route)}")
            print(f"选择的自动驾驶车辆数 ({self.av_ratio*100:.1f}%): {len(self.autonomous_vehicles)}")

            # 创建车辆OD映射
            self.vehicle_od = {}
            av_od_pairs = []
            for vid, start, end, _ in trips:
                self.vehicle_od[vid] = (start, end)
                if vid in self.autonomous_vehicles:
                    av_od_pairs.append((start, end))

            # 批量预计算AV的路径（性能优化）
            if self._use_precompute and av_od_pairs:
                print(f"预计算 {len(av_od_pairs)} 个AV路径...")
                for (start, end) in tqdm(av_od_pairs, desc="预计算路径", disable=len(av_od_pairs) < 100):
                    if (start, end) not in self._route_cache:
                        _ = self._compute_dijkstra_route(start, end)
                        # 路径已经在_compute_dijkstra_route中缓存
                print(f"预计算完成，缓存大小: {len(self._route_cache)}")

        except Exception as e:
            print(f"错误：解析路由文件失败: {e}")
            sys.exit(1)
    
    def _setup_csv_output(self):
        """设置CSV输出"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f"dijkstra_{self.location}_metrics_{timestamp}.csv"
        csv_path = os.path.join("outputs", csv_filename)
        
        # 确保outputs目录存在
        os.makedirs("outputs", exist_ok=True)
        
        self.csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        fieldnames = [
            'sim_time', 'current_vehicles', 'completed_vehicles', 'completed_autonomous_vehicles',
            'average_travel_time', 'average_delay_time', 'average_waiting_time'
        ]
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()
        self.csv_file.flush()
        
        print(f"CSV文件: {csv_path}")
        return csv_path
    
    def _calculate_metrics(self, current_time: float) -> Dict:
        """计算当前时刻的各项指标"""
        # 当前车辆数
        current_vehicles = len(traci.vehicle.getIDList())

        # 已完成车辆数
        completed_vehicles = len(self.vehicle_end_times)

        # 已完成的autonomous vehicles数量
        completed_autonomous_vehicles = len(self.completed_autonomous_vehicles)

        # 计算已完成的autonomous vehicles的旅行时间（修复：只统计AV）
        travel_times = []
        for vid in self.completed_autonomous_vehicles:
            if vid in self.vehicle_start_times and vid in self.vehicle_end_times:
                travel_time = self.vehicle_end_times[vid] - self.vehicle_start_times[vid]
                travel_times.append(travel_time)

        avg_travel_time = sum(travel_times) / len(travel_times) if travel_times else 0

        # 计算正在行驶的autonomous vehicles的delay time
        running_av_delay_times = []
        current_vehicle_ids = traci.vehicle.getIDList()

        for vid in current_vehicle_ids:
            # 只统计正在行驶的autonomous vehicles
            if vid in self.autonomous_vehicles and vid in self.vehicle_start_times and vid in self.vehicle_od:
                # 实际已用时间
                actual_time = current_time - self.vehicle_start_times[vid]

                # 计算理论最短时间
                start_edge, end_edge = self.vehicle_od[vid]
                shortest_path = self._compute_dijkstra_route(start_edge, end_edge)

                if shortest_path:
                    theoretical_time = 0
                    for edge in shortest_path:
                        if edge in self.edge_info:
                            edge_length = self.edge_info[edge]['length']
                            edge_speed = self.edge_info[edge]['speed']
                            theoretical_time += edge_length / edge_speed

                    # delay = 实际时间 - 理论时间
                    delay_time = max(0, actual_time - theoretical_time)
                    running_av_delay_times.append(delay_time)

        # 正在行驶的autonomous vehicles的平均delay time
        avg_delay_time = sum(running_av_delay_times) / len(running_av_delay_times) if running_av_delay_times else 0

        # 计算正在行驶的autonomous vehicles的平均等待时间
        running_av_waiting_times = []
        for vid in current_vehicle_ids:
            if vid in self.autonomous_vehicles and vid in self.vehicle_waiting_times:
                running_av_waiting_times.append(self.vehicle_waiting_times[vid])

        avg_waiting_time = sum(running_av_waiting_times) / len(running_av_waiting_times) if running_av_waiting_times else 0

        return {
            'sim_time': current_time,
            'current_vehicles': current_vehicles,
            'completed_vehicles': completed_vehicles,
            'completed_autonomous_vehicles': completed_autonomous_vehicles,
            'average_travel_time': avg_travel_time,
            'average_delay_time': avg_delay_time,  # 正在行驶的AVs的平均delay
            'average_waiting_time': avg_waiting_time  # 正在行驶的AVs的平均waiting time
        }
    
    def initialize(self):
        """初始化实验"""
        print("初始化Dijkstra路径规划实验...")
        
        # 启动SUMO
        sumo_cmd = [
            "sumo", "-c", self.sumo_config,
            "--no-warnings",
            "--ignore-route-errors",
            "--no-step-log",
            "--time-to-teleport", "-1",
            "--max-depart-delay", "900"
        ]
        
        print(f"启动SUMO: {' '.join(sumo_cmd)}")
        traci.start(sumo_cmd)
        
        # 构建路网
        self._build_road_network()
        if self.road_network.number_of_nodes() == 0:
            self._build_network_from_traci()
        
        # 选择自动驾驶车辆
        self._select_autonomous_vehicles()
        
        # 设置CSV输出
        csv_path = self._setup_csv_output()
        
        print("初始化完成")
        return csv_path
    
    def run(self):
        """运行实验"""
        print(f"开始实验: step_size={self.step_size}s, max_steps={self.max_steps}")
        
        step = 0
        pbar = tqdm(total=self.max_steps, desc="Dijkstra实验", unit="step")
        
        try:
            while step < self.max_steps:
                # 执行仿真步
                traci.simulationStep()
                current_time = traci.simulation.getTime()
                
                # 处理新车辆和更新等待时间
                current_vehicles = traci.vehicle.getIDList()
                for vid in current_vehicles:
                    if vid not in self.processed_vehicles:
                        self.processed_vehicles.add(vid)
                        self.vehicle_start_times[vid] = current_time
                        self.vehicle_waiting_times[vid] = 0.0  # 初始化累积等待时间
                        
                        # 如果是自动驾驶车辆，重新规划路径
                        if vid in self.autonomous_vehicles and vid in self.vehicle_od:
                            start_edge, end_edge = self.vehicle_od[vid]
                            try:
                                current_edge = traci.vehicle.getRoadID(vid)
                                if current_edge and not current_edge.startswith(':'):
                                    new_route = self._compute_dijkstra_route(current_edge, end_edge)
                                    if new_route and len(new_route) > 1:
                                        traci.vehicle.setRoute(vid, new_route)
                            except:
                                # 如果获取当前位置失败，尝试使用起始边
                                new_route = self._compute_dijkstra_route(start_edge, end_edge)
                                if new_route:
                                    try:
                                        traci.vehicle.setRoute(vid, new_route)
                                    except:
                                        pass
                    
                    # 更新车辆的累积等待时间（使用标准TraCI接口）
                    if vid in self.vehicle_waiting_times:
                        try:
                            # 使用正确的TraCI接口获取累积等待时间
                            accumulated_waiting = traci.vehicle.getAccumulatedWaitingTime(vid)
                            self.vehicle_waiting_times[vid] = accumulated_waiting
                        except:
                            pass
                
                # 处理到达车辆
                arrived_vehicles = traci.simulation.getArrivedIDList()

                for vid in arrived_vehicles:
                    if vid not in self.vehicle_end_times:
                        self.vehicle_end_times[vid] = current_time

                        # 检查是否是autonomous vehicle完成
                        if vid in self.autonomous_vehicles:
                            self.completed_autonomous_vehicles.add(vid)
                            print(f"Autonomous vehicle {vid} 完成旅程 (第{len(self.completed_autonomous_vehicles)}个)")

                            # 记录最终累积等待时间
                            final_waiting = self.vehicle_waiting_times.get(vid, 0)
                            print(f"  - 累积等待时间: {final_waiting:.2f}s")

                # 定期记录指标（每60秒或每次有AV完成时）
                if step % 60 == 0 or len(arrived_vehicles) > 0:
                    metrics = self._calculate_metrics(current_time)
                    self.metrics_history.append(metrics)
                    self.csv_writer.writerow(metrics)
                    self.csv_file.flush()

                step += 1
                pbar.update(1)
                
                # 检查是否还有车辆
                if len(current_vehicles) == 0 and step > 100:
                    print("\n所有车辆已完成，提前结束实验")
                    break
                    
        except KeyboardInterrupt:
            print("\n实验被用户中断")
        except Exception as e:
            print(f"\n实验运行错误: {e}")
        finally:
            pbar.close()
            self._finalize()
    
    def _finalize(self):
        """清理和总结"""
        try:
            traci.close()
        except:
            pass
        
        if self.csv_file:
            self.csv_file.close()
        
        print("\n" + "="*60)
        print("实验结束 - 最终统计")
        print("="*60)
        
        if self.metrics_history:
            final_metrics = self.metrics_history[-1]
            print(f"仿真时间: {final_metrics['sim_time']:.1f}s")
            print(f"已完成车辆: {final_metrics['completed_vehicles']}")
            print(f"已完成自动驾驶车辆: {final_metrics['completed_autonomous_vehicles']}/{len(self.autonomous_vehicles)}")
            print(f"平均旅行时间: {final_metrics['average_travel_time']:.2f}s")
            print(f"平均延误时间: {final_metrics['average_delay_time']:.2f}s")
            print(f"平均等待时间: {final_metrics['average_waiting_time']:.2f}s")
        else:
            print(f"总自动驾驶车辆数: {len(self.autonomous_vehicles)}")
            print(f"已完成自动驾驶车辆: {len(self.completed_autonomous_vehicles)}")
        
        # 性能优化统计
        total_requests = self._cache_hits + self._cache_misses
        if total_requests > 0:
            hit_rate = self._cache_hits / total_requests * 100
            print(f"路径缓存统计: {self._cache_hits}命中 / {total_requests}请求 ({hit_rate:.1f}% 命中率)")
            print(f"缓存大小: {len(self._route_cache)}/{self._max_cache_size}")
            print(f"性能提升: 缓存减少了 {self._cache_hits} 次Dijkstra计算")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Dijkstra路径规划实验（支持A*优化）")
    parser.add_argument("--config", type=str,
                       default="/data/XXXXX/LLMNavigation/Data/Chicago/Chicago_sumo_config.sumocfg",
                       help="SUMO配置文件路径")
    parser.add_argument("--location", type=str, default="Chicago", help="仿真位置（用于输出文件命名）")
    parser.add_argument("--step-size", type=float, default=1.0, help="仿真步长(秒)")
    parser.add_argument("--max-steps", type=int, default=43200, help="最大仿真步数")
    parser.add_argument("--av-ratio", type=float, default=0.02, help="自动驾驶车辆比例")
    parser.add_argument("--use-astar", action="store_true", help="使用A*算法替代Dijkstra（推荐用于大路网）")

    args = parser.parse_args()

    # 检查文件存在性
    if not os.path.exists(args.config):
        print(f"错误：SUMO配置文件不存在: {args.config}")
        sys.exit(1)

    # 创建并运行实验
    experiment = DijkstraExperiment(
        sumo_config=args.config,
        step_size=args.step_size,
        max_steps=args.max_steps,
        av_ratio=args.av_ratio,
        use_astar=args.use_astar,
        location=args.location
    )

    csv_path = experiment.initialize()
    algo_name = "A*" if args.use_astar else "Dijkstra"
    print(f"开始实验（{algo_name}算法），结果将保存到: {csv_path}")
    experiment.run()


if __name__ == "__main__":
    main()