#!/usr/bin/env python3
"""
贪心算法路径规划实验
进行两次实验：
1. 最小化距离 (minimize distance)
2. 最小化延迟 (minimize latency)
每次实验记录指标到单独的CSV文件
"""

import os
import sys
import csv
import random
import argparse
import math
from datetime import datetime
from typing import Dict, List, Optional
from collections import OrderedDict
import xml.etree.ElementTree as ET

import networkx as nx
import traci
from tqdm import tqdm

sys.path.append("../")
from env_utils import parse_rou_file


class GreedyExperiment:
    """
    基于贪心算法的路径规划实验
    - 选择2%车辆作为自动驾驶车辆进行路径规划
    - 支持两种贪心策略：最小化距离、最小化延迟
    - 实时记录各项指标并保存到CSV
    """
    
    def __init__(self, sumo_config: str, strategy: str, step_size: float = 1.0, max_steps: int = 43200, av_ratio: float = 0.02, location: str = "Manhattan"):
        self.sumo_config = sumo_config
        self.strategy = strategy  # "distance" or "latency"
        self.step_size = step_size
        self.max_steps = max_steps
        self.av_ratio = av_ratio
        self.location = location  # 仿真位置
        
        # 从sumocfg解析路径
        self.net_file = None
        self.route_file = None
        self._parse_sumo_config()
        
        # 路网数据
        self.road_network: nx.DiGraph = nx.DiGraph()
        self.edge_info: Dict = {}
        self.edge_coordinates: Dict = {}  # 边的坐标信息

        # 性能优化：LRU路径缓存和批量处理
        self._route_cache = OrderedDict()  # LRU缓存
        self._cache_hits = 0
        self._cache_misses = 0
        self._max_cache_size = 100000  # 大缓存提升性能

        # 预计算优化
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
    
    def _get_edge_coordinates(self, edge_id: str) -> tuple:
        """获取边的中心坐标"""
        if edge_id in self.edge_coordinates:
            return self.edge_coordinates[edge_id]
        
        try:
            # 获取边的第一个车道的形状点
            shape = traci.lane.getShape(edge_id + '_0')
            if shape:
                # 计算形状点的中心作为边的坐标
                x_coords = [point[0] for point in shape]
                y_coords = [point[1] for point in shape]
                center_x = sum(x_coords) / len(x_coords)
                center_y = sum(y_coords) / len(y_coords)
                self.edge_coordinates[edge_id] = (center_x, center_y)
                return (center_x, center_y)
        except:
            pass
        
        return (0, 0)  # 默认坐标
    
    def _calculate_distance(self, edge1: str, edge2: str) -> float:
        """计算两个边之间的欧几里得距离"""
        x1, y1 = self._get_edge_coordinates(edge1)
        x2, y2 = self._get_edge_coordinates(edge2)
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def _build_road_network(self):
        """构建路网图用于贪心算法"""
        print(f"构建路网图（策略: {self.strategy}）...")
        
        # 解析SUMO网络文件
        try:
            tree = ET.parse(self.net_file)
            root = tree.getroot()
            
            # 添加所有边及其坐标信息
            for edge in root.findall('edge'):
                edge_id = edge.get('id')
                if edge_id and not edge_id.startswith(':'):
                    # 获取边的长度和速度
                    length = 0.0
                    speed = 13.89  # 默认速度 m/s
                    
                    for lane in edge.findall('lane'):
                        lane_length = float(lane.get('length', 0))
                        lane_speed = float(lane.get('speed', speed))
                        if lane_length > length:
                            length = lane_length
                        speed = max(speed, lane_speed)
                    
                    self.edge_info[edge_id] = {
                        'length': length,
                        'speed': speed
                    }
                    
                    # 获取坐标信息（从shape属性）
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
                                    # 计算中心点
                                    center_x = sum(p[0] for p in points) / len(points)
                                    center_y = sum(p[1] for p in points) / len(points)
                                    self.edge_coordinates[edge_id] = (center_x, center_y)
                                break
                            except:
                                continue
            
            # 添加连接关系
            for connection in root.findall('connection'):
                from_edge = connection.get('from')
                to_edge = connection.get('to')
                
                if (from_edge and to_edge and 
                    from_edge in self.edge_info and 
                    to_edge in self.edge_info and
                    not from_edge.startswith(':') and 
                    not to_edge.startswith(':')):
                    
                    if self.strategy == "distance":
                        # 使用物理长度作为权重
                        weight = self.edge_info[from_edge]['length']
                    else:  # latency
                        # 使用旅行时间作为权重
                        length = self.edge_info[from_edge]['length']
                        speed = self.edge_info[from_edge]['speed']
                        weight = length / speed if speed > 0 else length / 13.89
                    
                    self.road_network.add_edge(from_edge, to_edge, weight=weight)
                        
        except Exception as e:
            print(f"警告：无法解析网络文件，将在运行时构建路网: {e}")
            
        print(f"路网图: {self.road_network.number_of_nodes()} 节点, {self.road_network.number_of_edges()} 边")
    
    def _build_network_from_traci(self):
        """从TraCI构建路网图（备用方法）"""
        print("从TraCI构建路网图...")
        
        edges = traci.edge.getIDList()
        for edge_id in edges:
            if not edge_id.startswith(':'):
                try:
                    length = traci.lane.getLength(edge_id + '_0')
                    speed = traci.lane.getMaxSpeed(edge_id + '_0')
                    self.edge_info[edge_id] = {
                        'length': length,
                        'speed': speed
                    }
                except:
                    continue
        
        # 构建连接关系
        for edge_id in list(self.edge_info.keys()):
            try:
                outgoing = traci.edge.getOutgoing(edge_id)
                for next_edge in outgoing:
                    if next_edge in self.edge_info:
                        if self.strategy == "distance":
                            weight = self.edge_info[edge_id]['length']
                        else:  # latency
                            length = self.edge_info[edge_id]['length']
                            speed = self.edge_info[edge_id]['speed']
                            weight = length / speed if speed > 0 else length / 13.89
                        self.road_network.add_edge(edge_id, next_edge, weight=weight)
            except:
                continue
                
        print(f"路网图: {self.road_network.number_of_nodes()} 节点, {self.road_network.number_of_edges()} 边")
    
    def _compute_greedy_route(self, start_edge: str, end_edge: str) -> Optional[List[str]]:
        """计算贪心路径（带LRU缓存优化）"""
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
            path = [start_edge]
            current = start_edge
            visited = set([start_edge])
            max_steps = 200  # 增加最大步数防止过早终止（大路网需要更多步骤）
            steps = 0

            while current != end_edge and steps < max_steps:
                steps += 1
                neighbors = list(self.road_network.successors(current))
                if not neighbors:
                    break

                # 过滤已访问的节点
                unvisited_neighbors = [n for n in neighbors if n not in visited]
                if not unvisited_neighbors:
                    # 如果所有邻居都已访问，选择最近的（允许回溯）
                    unvisited_neighbors = neighbors

                if self.strategy == "distance":
                    # 贪心策略：选择距离目标最近的邻居
                    best_neighbor = min(unvisited_neighbors,
                                      key=lambda n: self._calculate_distance(n, end_edge))
                else:  # latency
                    # 贪心策略：选择预期延迟最小的邻居
                    def latency_cost(neighbor):
                        # 计算到邻居的成本 + 估计到目标的成本
                        edge_cost = self.road_network.get_edge_data(current, neighbor, {}).get('weight', 1.0)
                        distance_to_target = self._calculate_distance(neighbor, end_edge)
                        # 估计剩余时间（假设平均速度）
                        estimated_remaining = distance_to_target / 13.89
                        return edge_cost + estimated_remaining

                    best_neighbor = min(unvisited_neighbors, key=latency_cost)

                path.append(best_neighbor)
                visited.add(best_neighbor)
                current = best_neighbor

                if current == end_edge:
                    break

            result = path if current == end_edge else None

            # LRU缓存：满了删除最旧的
            if len(self._route_cache) >= self._max_cache_size:
                self._route_cache.popitem(last=False)  # 删除最早的项
            self._route_cache[cache_key] = result

            return result

        except:
            # 缓存失败结果避免重复尝试
            if len(self._route_cache) >= self._max_cache_size:
                self._route_cache.popitem(last=False)
            self._route_cache[cache_key] = None
            return None
    
    def _select_autonomous_vehicles(self):
        """选择2%车辆作为自动驾驶车辆，并预计算常用路径"""
        try:
            trips = parse_rou_file(self.route_file)
            self.all_vehicles_in_route = [vid for vid, _, _, _ in trips]

            # 随机选择2%车辆
            random.seed(42)
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
                print(f"预计算 {len(av_od_pairs)} 个AV路径（{self.strategy}策略）...")
                for (start, end) in tqdm(av_od_pairs, desc=f"预计算路径({self.strategy})", disable=len(av_od_pairs) < 100):
                    if (start, end) not in self._route_cache:
                        _ = self._compute_greedy_route(start, end)
                        # 路径已经在_compute_greedy_route中缓存
                print(f"预计算完成，缓存大小: {len(self._route_cache)}")

        except Exception as e:
            print(f"错误：解析路由文件失败: {e}")
            sys.exit(1)
    
    def _setup_csv_output(self):
        """设置CSV输出"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f"greedy_{self.strategy}_{self.location}_metrics_{timestamp}.csv"
        csv_path = os.path.join("outputs", csv_filename)
        
        os.makedirs("outputs", exist_ok=True)
        
        self.csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        fieldnames = [
            'sim_time', 'current_vehicles', 'completed_vehicles', 'completed_autonomous_vehicles',
            'average_travel_time', 'average_delay_time', 'average_waiting_time'
        ]
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()
        self.csv_file.flush()
        
        print(f"CSV文件 ({self.strategy}): {csv_path}")
        return csv_path
    
    def _calculate_metrics(self, current_time: float) -> Dict:
        """计算当前时刻的各项指标"""
        current_vehicles = len(traci.vehicle.getIDList())
        completed_vehicles = len(self.vehicle_end_times)
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

                if self.strategy == "distance":
                    # 使用直线距离估算理论时间
                    theoretical_distance = self._calculate_distance(start_edge, end_edge)
                    theoretical_time = theoretical_distance / 13.89  # 假设平均速度
                else:  # latency
                    # 使用路网中的最短路径时间
                    try:
                        theoretical_time = nx.dijkstra_path_length(
                            self.road_network, start_edge, end_edge, weight='weight'
                        )
                    except:
                        # 如果无法计算，使用直线距离估算
                        theoretical_distance = self._calculate_distance(start_edge, end_edge)
                        theoretical_time = theoretical_distance / 13.89

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
        print(f"初始化贪心算法路径规划实验（策略: {self.strategy}）...")
        
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
        print(f"开始实验（{self.strategy}）: step_size={self.step_size}s, max_steps={self.max_steps}")
        
        step = 0
        pbar = tqdm(total=self.max_steps, desc=f"贪心实验({self.strategy})", unit="step")
        
        try:
            while step < self.max_steps:
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
                                    new_route = self._compute_greedy_route(current_edge, end_edge)
                                    if new_route and len(new_route) > 1:
                                        traci.vehicle.setRoute(vid, new_route)
                            except:
                                new_route = self._compute_greedy_route(start_edge, end_edge)
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
                            print(f"Autonomous vehicle {vid} 完成旅程 (第{len(self.completed_autonomous_vehicles)}个, {self.strategy}策略)")

                # 定期记录指标（每60秒或每次有AV完成时）
                if step % 60 == 0 or len(arrived_vehicles) > 0:
                    metrics = self._calculate_metrics(current_time)
                    self.metrics_history.append(metrics)
                    self.csv_writer.writerow(metrics)
                    self.csv_file.flush()

                step += 1
                pbar.update(1)
                
                if len(current_vehicles) == 0 and step > 100:
                    print(f"\n所有车辆已完成，提前结束实验（{self.strategy}）")
                    break
                    
        except KeyboardInterrupt:
            print(f"\n实验被用户中断（{self.strategy}）")
        except Exception as e:
            print(f"\n实验运行错误（{self.strategy}）: {e}")
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

        print(f"\n" + "="*60)
        print(f"实验结束 - 最终统计（{self.strategy.upper()}）")
        print("="*60)

        if self.metrics_history:
            final_metrics = self.metrics_history[-1]
            print(f"策略: {self.strategy}")
            print(f"仿真时间: {final_metrics['sim_time']:.1f}s")
            print(f"已完成车辆: {final_metrics['completed_vehicles']}")
            print(f"已完成自动驾驶车辆: {final_metrics['completed_autonomous_vehicles']}/{len(self.autonomous_vehicles)}")
            print(f"平均旅行时间（仅AV）: {final_metrics['average_travel_time']:.2f}s")
            print(f"平均延误时间: {final_metrics['average_delay_time']:.2f}s")
            print(f"平均等待时间: {final_metrics['average_waiting_time']:.2f}s")
        else:
            print(f"策略: {self.strategy}")
            print(f"总自动驾驶车辆数: {len(self.autonomous_vehicles)}")
            print(f"已完成自动驾驶车辆: {len(self.completed_autonomous_vehicles)}")

        # 性能优化统计
        total_requests = self._cache_hits + self._cache_misses
        if total_requests > 0:
            hit_rate = self._cache_hits / total_requests * 100
            print(f"路径缓存统计: {self._cache_hits}命中 / {total_requests}请求 ({hit_rate:.1f}% 命中率)")
            print(f"缓存大小: {len(self._route_cache)}/{self._max_cache_size}")
            print(f"性能提升: 缓存减少了 {self._cache_hits} 次贪心路径计算")

        print("="*60)


def run_both_experiments(sumo_config: str, step_size: float, max_steps: int, av_ratio: float, location: str):
    """运行两种贪心策略的实验"""
    strategies = ["distance", "latency"]

    for strategy in strategies:
        print(f"\n{'='*80}")
        print(f"开始 {strategy.upper()} 策略实验")
        print(f"{'='*80}")

        experiment = GreedyExperiment(
            sumo_config=sumo_config,
            strategy=strategy,
            step_size=step_size,
            max_steps=max_steps,
            av_ratio=av_ratio,
            location=location
        )

        csv_path = experiment.initialize()
        experiment.run()

        # 打印对比结果
        if experiment.metrics_history:
            metrics = experiment.metrics_history[-1]
            print(f"\n{strategy.upper()} 策略最终结果:")
            print(f"  平均旅行时间（仅AV）: {metrics['average_travel_time']:.2f}s")
            print(f"  平均延误时间: {metrics['average_delay_time']:.2f}s")
            print(f"  已完成车辆: {metrics['completed_vehicles']}")
            print(f"  CSV文件: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="贪心算法路径规划实验")
    parser.add_argument("--config", type=str,
                       default="/data/zhouyuping/LLMNavigation/Data/Chicago/Chicago_sumo_config.sumocfg",
                       help="SUMO配置文件路径")
    parser.add_argument("--location", type=str, default="Chicago", help="仿真位置（用于输出文件命名）")
    parser.add_argument("--strategy", type=str, choices=["distance", "latency", "both"],
                       default="both", help="贪心策略：distance, latency, 或 both")
    parser.add_argument("--step-size", type=float, default=1.0, help="仿真步长(秒)")
    parser.add_argument("--max-steps", type=int, default=43200, help="最大仿真步数")
    parser.add_argument("--av-ratio", type=float, default=0.02, help="自动驾驶车辆比例")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"错误：SUMO配置文件不存在: {args.config}")
        sys.exit(1)

    if args.strategy == "both":
        # 运行两种策略的实验
        results = run_both_experiments(
            sumo_config=args.config,
            step_size=args.step_size,
            max_steps=args.max_steps,
            av_ratio=args.av_ratio,
            location=args.location
        )
    else:
        # 运行单一策略实验
        experiment = GreedyExperiment(
            sumo_config=args.config,
            strategy=args.strategy,
            step_size=args.step_size,
            max_steps=args.max_steps,
            av_ratio=args.av_ratio,
            location=args.location
        )

        csv_path = experiment.initialize()
        print(f"开始实验（{args.strategy}策略），结果将保存到: {csv_path}")
        experiment.run()


if __name__ == "__main__":
    main()