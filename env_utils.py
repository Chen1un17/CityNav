import sys
import time
import json
import os
from typing import Dict, List, Tuple, Optional, Union

import numpy as np

sys.path.append("../")

import networkx as nx
import traci
import traci.constants as tc
import xml.etree.ElementTree as ET

# 尝试导入 libsumo 作为高性能替代方案
try:
    import libsumo
    LIBSUMO_AVAILABLE = True
except ImportError:
    LIBSUMO_AVAILABLE = False

# 全局缓存和配置
_edge_info_cache = {}
_static_road_data = {}  # 静态道路数据缓存
_adjacency_data = {}    # 邻接关系数据缓存
_region_mapping = {}    # 区域映射缓存
_valid_edges_cache = set()  # 有效边缘ID缓存
_use_batch_mode = True
_use_libsumo = False
_cache_timeout = 10.0  # 缓存超时时间（秒）
_static_data_loaded = False


def configure_edge_info_optimization(
    use_batch_mode: bool = True,
    use_libsumo: bool = False,
    cache_timeout: float = 10.0
):
    """
    配置边缘信息获取的优化选项
    
    Args:
        use_batch_mode: 是否使用批量模式获取边缘信息
        use_libsumo: 是否使用 libsumo（需要先安装）
        cache_timeout: 缓存超时时间（秒）
    """
    global _use_batch_mode, _use_libsumo, _cache_timeout
    
    _use_batch_mode = use_batch_mode
    _cache_timeout = cache_timeout
    
    if use_libsumo and LIBSUMO_AVAILABLE:
        _use_libsumo = True
        print("启用 libsumo 高性能模式")
    elif use_libsumo and not LIBSUMO_AVAILABLE:
        print("警告: libsumo 不可用，继续使用标准 traci")
        _use_libsumo = False
    else:
        _use_libsumo = False


def get_traci_interface():
    """获取 TraCI 接口（traci 或 libsumo）"""
    return libsumo if _use_libsumo else traci


def clear_edge_info_cache():
    """清空边缘信息缓存"""
    global _edge_info_cache
    _edge_info_cache.clear()


def reset_static_data():
    """重置静态数据加载状态，允许重新加载"""
    global _static_data_loaded, _static_road_data, _adjacency_data, _region_mapping, _valid_edges_cache
    _static_data_loaded = False
    _static_road_data = {}
    _adjacency_data = {}
    _region_mapping = {}
    _valid_edges_cache = set()
    print("STATIC_DATA_LOADER: Static data reset, ready for reload")


def load_static_road_data(data_dir: str = "Data/Region_1", road_info_file: str = None, adjacency_file: str = None):
    """
    预加载静态道路数据，大幅提升性能

    Args:
        data_dir: 数据目录路径
        road_info_file: 道路信息文件路径（可选，如果提供则使用，否则使用默认路径）
        adjacency_file: 邻接信息文件路径（可选，如果提供则使用，否则使用默认路径）
    """
    global _static_road_data, _adjacency_data, _region_mapping, _valid_edges_cache, _static_data_loaded

    if _static_data_loaded:
        return

    try:
        # 加载道路信息
        if road_info_file and os.path.exists(road_info_file):
            road_info_path = road_info_file
        else:
            road_info_path = "/data/XXXXX/LLMNavigation/Data/NYC/NewYork_road_info.json"

        if os.path.exists(road_info_path):
            with open(road_info_path, 'r') as f:
                _static_road_data = json.load(f)
            # 缓存有效边缘ID集合，用于快速验证
            _valid_edges_cache = set(_static_road_data.keys())
            print(f"STATIC_DATA_LOADER: Loaded {len(_static_road_data)} roads from {road_info_path}")
            print(f"STATIC_DATA_LOADER: Valid edges cache initialized with {len(_valid_edges_cache)} entries")
        else:
            print(f"STATIC_DATA_LOADER: WARNING - Road info file not found: {road_info_path}")

        # 加载邻接信息
        if adjacency_file and os.path.exists(adjacency_file):
            adj_info_path = adjacency_file
        else:
            adj_info_path = "/data/XXXXX/LLMNavigation/Data/NYC/Region_1/edge_adjacency_alpha_1.json"

        if os.path.exists(adj_info_path):
            with open(adj_info_path, 'r') as f:
                _adjacency_data = json.load(f)
            print(f"STATIC_DATA_LOADER: Loaded {len(_adjacency_data)} adjacency relationships from {adj_info_path}")

        # 加载区域映射
        region_mapping_path = os.path.join(data_dir, "edge_to_region_alpha_1.json")
        if os.path.exists(region_mapping_path):
            with open(region_mapping_path, 'r') as f:
                _region_mapping = json.load(f)
            print(f"STATIC_DATA_LOADER: Loaded {len(_region_mapping)} region mappings")

        _static_data_loaded = True
        print("STATIC_DATA_LOADER: Pre-loading completed successfully")

    except Exception as e:
        print(f"STATIC_DATA_LOADER: ERROR - Failed to load static data: {e}")
        _static_data_loaded = False


def get_static_road_info(edge_id: str) -> Tuple[int, float, float]:
    """
    从预加载的静态数据获取道路信息
    
    Args:
        edge_id: 边缘ID
        
    Returns:
        tuple: (车道数, 道路长度, 速度限制)
    """
    if not _static_data_loaded:
        load_static_road_data()
    
    if edge_id in _static_road_data:
        road_data = _static_road_data[edge_id]
        return (
            road_data.get("lane_num", 1),
            road_data.get("road_len", 100.0),
            road_data.get("speed_limit", 13.89)
        )
    else:
        return (1, 100.0, 13.89)


def get_adjacency_info(edge_id: str) -> List[str]:
    """
    从预加载数据获取邻接道路信息
    
    Args:
        edge_id: 边缘ID
        
    Returns:
        list: 邻接道路ID列表
    """
    if not _static_data_loaded:
        load_static_road_data()
    
    return _adjacency_data.get(edge_id, [])


def _is_cache_valid(cache_entry: dict, current_time: float) -> bool:
    """检查缓存条目是否仍然有效"""
    return (current_time - cache_entry.get('timestamp', 0)) < _cache_timeout


def get_edges_info_batch(edge_ids: List[str]) -> Dict[str, Tuple[int, int, float, float, float]]:
    """
    批量获取多个边缘的信息，优化版本：静态数据从JSON加载，只查询SUMO动态数据
    
    Args:
        edge_ids: 边缘ID列表
        
    Returns:
        字典，key为边缘ID，value为元组 (车道数, 车辆数, 车辆速度, 车辆长度, 道路长度)
    """
    if not edge_ids:
        return {}
    
    # 确保静态数据已加载
    if not _static_data_loaded:
        load_static_road_data()
    
    current_time = time.time()
    result = {}
    edges_to_fetch = []
    
    # 检查缓存中的有效数据
    for edge_id in edge_ids:
        if edge_id in _edge_info_cache and _is_cache_valid(_edge_info_cache[edge_id], current_time):
            result[edge_id] = _edge_info_cache[edge_id]['data']
        else:
            edges_to_fetch.append(edge_id)
    
    if not edges_to_fetch:
        return result
    
    try:
        traci_interface = get_traci_interface()
        
        # 只订阅动态变量，静态信息从JSON获取
        dynamic_variables = [
            tc.LAST_STEP_VEHICLE_NUMBER,  # 车辆数
            tc.LAST_STEP_MEAN_SPEED,      # 平均速度
            tc.LAST_STEP_LENGTH           # 车辆长度
        ]
        
        # 为每个边缘创建订阅
        valid_edges = []
        for edge_id in edges_to_fetch:
            if _is_edge_valid(edge_id):
                valid_edges.append(edge_id)
                traci_interface.edge.subscribe(edge_id, dynamic_variables)
        
        # 获取订阅结果
        subscription_results = traci_interface.edge.getAllSubscriptionResults()
        
        # 处理订阅结果
        for edge_id in edges_to_fetch:
            # 从静态数据获取车道数和道路长度
            lane_num, road_len, speed_limit = get_static_road_info(edge_id)
            
            if edge_id in subscription_results:
                # 从订阅结果获取动态数据
                edge_data = subscription_results[edge_id]
                vehicle_num = edge_data.get(tc.LAST_STEP_VEHICLE_NUMBER, 0)
                vehicle_speed = edge_data.get(tc.LAST_STEP_MEAN_SPEED, 0.0)
                vehicle_length = edge_data.get(tc.LAST_STEP_LENGTH, 0.0)
            else:
                # 使用默认动态数据
                vehicle_num, vehicle_speed, vehicle_length = 0, 0.0, 0.0
            
            edge_info = (lane_num, vehicle_num, vehicle_speed, vehicle_length, road_len)
            result[edge_id] = edge_info
            
            # 更新缓存
            _edge_info_cache[edge_id] = {
                'data': edge_info,
                'timestamp': current_time
            }
    
    except Exception as e:
        print(f"批量获取边缘信息失败: {e}")
        # 回退到逐个获取
        for edge_id in edges_to_fetch:
            try:
                edge_info = get_edge_info_single(edge_id)
                result[edge_id] = edge_info
                _edge_info_cache[edge_id] = {
                    'data': edge_info,
                    'timestamp': current_time
                }
            except Exception as single_error:
                print(f"获取边缘 {edge_id} 信息失败: {single_error}")
                # 即使失败也使用静态数据
                lane_num, road_len, speed_limit = get_static_road_info(edge_id)
                result[edge_id] = (lane_num, 0, 0.0, 0.0, road_len)
    
    return result


def get_dynamic_data_batch(edge_ids: List[str]) -> Dict[str, Tuple[int, float, float]]:
    """
    批量获取边缘的动态数据（车辆数、速度、车辆长度），最高性能版本
    
    Args:
        edge_ids: 边缘ID列表
        
    Returns:
        字典，key为边缘ID，value为元组 (车辆数, 车辆平均速度, 车辆长度)
    """
    if not edge_ids:
        return {}
    
    current_time = time.time()
    result = {}
    edges_to_fetch = []
    
    # 检查缓存中的有效动态数据
    for edge_id in edge_ids:
        cache_key = f"dynamic_{edge_id}"
        if cache_key in _edge_info_cache and _is_cache_valid(_edge_info_cache[cache_key], current_time):
            result[edge_id] = _edge_info_cache[cache_key]['data']
        else:
            edges_to_fetch.append(edge_id)
    
    if not edges_to_fetch:
        return result
    
    try:
        traci_interface = get_traci_interface()
        
        # 使用libsumo的批量接口如果可用
        if _use_libsumo and LIBSUMO_AVAILABLE:
            # 一次性过滤有效边缘
            if not _static_data_loaded:
                load_static_road_data()
            valid_edges = [edge_id for edge_id in edges_to_fetch if edge_id in _valid_edges_cache]
            
            for edge_id in valid_edges:
                try:
                    vehicle_num = traci_interface.edge.getLastStepVehicleNumber(edge_id)
                    vehicle_speed = traci_interface.edge.getLastStepMeanSpeed(edge_id)
                    vehicle_length = traci_interface.edge.getLastStepLength(edge_id)
                    
                    dynamic_data = (vehicle_num, vehicle_speed, vehicle_length)
                    result[edge_id] = dynamic_data
                    
                    # 缓存动态数据
                    cache_key = f"dynamic_{edge_id}"
                    _edge_info_cache[cache_key] = {
                        'data': dynamic_data,
                        'timestamp': current_time
                    }
                except Exception:
                    result[edge_id] = (0, 0.0, 0.0)
            
            # 为无效边缘设置默认值
            for edge_id in edges_to_fetch:
                if edge_id not in valid_edges:
                    result[edge_id] = (0, 0.0, 0.0)
        else:
            # 使用订阅批量获取
            dynamic_variables = [
                tc.LAST_STEP_VEHICLE_NUMBER,
                tc.LAST_STEP_MEAN_SPEED,
                tc.LAST_STEP_LENGTH
            ]
            
            # 一次性过滤有效边缘，避免循环中重复验证
            if not _static_data_loaded:
                load_static_road_data()
            valid_edges = [edge_id for edge_id in edges_to_fetch if edge_id in _valid_edges_cache]
            invalid_edges = [edge_id for edge_id in edges_to_fetch if edge_id not in _valid_edges_cache]
            
            # print(f"DYNAMIC_BATCH_QUERY: Processing {len(edges_to_fetch)} edges - {len(valid_edges)} valid, {len(invalid_edges)} invalid")
            # if invalid_edges:
            #     print(f"DYNAMIC_BATCH_QUERY: Invalid edges detected: {invalid_edges[:5]}{'...' if len(invalid_edges) > 5 else ''}")
            
            # 创建订阅
            for edge_id in valid_edges:
                traci_interface.edge.subscribe(edge_id, dynamic_variables)
            
            # 获取结果
            subscription_results = traci_interface.edge.getAllSubscriptionResults()
            # print(f"DYNAMIC_BATCH_QUERY: Received subscription results for {len(subscription_results)} edges")
            
            for edge_id in edges_to_fetch:
                if edge_id in subscription_results:
                    edge_data = subscription_results[edge_id]
                    vehicle_num = edge_data.get(tc.LAST_STEP_VEHICLE_NUMBER, 0)
                    vehicle_speed = edge_data.get(tc.LAST_STEP_MEAN_SPEED, 0.0)
                    vehicle_length = edge_data.get(tc.LAST_STEP_LENGTH, 0.0)
                    
                    dynamic_data = (vehicle_num, vehicle_speed, vehicle_length)
                    result[edge_id] = dynamic_data
                    
                    # 缓存动态数据
                    cache_key = f"dynamic_{edge_id}"
                    _edge_info_cache[cache_key] = {
                        'data': dynamic_data,
                        'timestamp': current_time
                    }
                else:
                    result[edge_id] = (0, 0.0, 0.0)
                    
    except Exception as e:
        # print(f"DYNAMIC_BATCH_QUERY: ERROR - Batch query failed: {e}")
        # 回退到默认值
        for edge_id in edges_to_fetch:
            result[edge_id] = (0, 0.0, 0.0)
    
    return result


def get_edge_info_single(edge_id: str) -> Tuple[int, int, float, float, float]:
    """
    获取单个边缘信息（优化版本：静态数据从JSON获取）
    """
    # 确保静态数据已加载
    if not _static_data_loaded:
        load_static_road_data()
        
    traci_interface = get_traci_interface()
    
    try:
        # 从静态数据获取车道数和道路长度
        lane_num, road_len, speed_limit = get_static_road_info(edge_id)
        
        if not _is_edge_valid(edge_id):
            return (lane_num, 0, 0.0, 0.0, road_len)
        
        # 只查询动态数据
        vehicle_num = traci_interface.edge.getLastStepVehicleNumber(edge_id)
        vehicle_speed = traci_interface.edge.getLastStepMeanSpeed(edge_id)
        vehicle_length = traci_interface.edge.getLastStepLength(edge_id)
        
        return (lane_num, vehicle_num, vehicle_speed, vehicle_length, road_len)
    
    except Exception as e:
        print(f"EDGE_INFO_SINGLE: ERROR - Failed to get edge {edge_id} info: {e}")
        # 即使失败也使用静态数据
        lane_num, road_len, speed_limit = get_static_road_info(edge_id)
        return (lane_num, 0, 0.0, 0.0, road_len)

# 获取边的车道数
def get_edge_lane_info(edge_id, lane_id):
    traci_interface = get_traci_interface()
    lane_num = traci_interface.edge.getLaneNumber(edge_id)  # 获取边上的所有车道ID
    vehicle_num = traci_interface.edge.getLastStepVehicleNumber(edge_id)
    vehicle_speed = traci_interface.edge.getLastStepMeanSpeed(edge_id)
    vehicle_length = traci_interface.edge.getLastStepLength(edge_id)
    speed_limit = traci_interface.lane.getMaxSpeed(lane_id)

    # route = traci_interface.simulation.findRoute(fromEdge=edge_id, toEdge=edge_id)
    road_len = traci_interface.lane.getLength(lane_id)

    return lane_num, vehicle_num, vehicle_speed, vehicle_length, road_len, speed_limit


def get_edge_info(edge_id, retry_count=0, max_retries=3):
    """
    获取边缘信息，包含错误处理和重试机制
    现在支持静态数据预加载和批量模式优化
    
    Args:
        edge_id: 边缘ID
        retry_count: 当前重试次数
        max_retries: 最大重试次数
    
    Returns:
        tuple: (车道数, 车辆数, 车辆速度, 车辆长度, 道路长度)
    """
    # 确保静态数据已加载
    if not _static_data_loaded:
        load_static_road_data()
    
    if _use_batch_mode:
        # 使用高性能批量模式
        try:
            batch_result = get_multiple_edges_info([edge_id])
            result = batch_result.get(edge_id)
            if result:
                lane_num, vehicle_num, vehicle_speed, vehicle_length, road_len = result
                print(f"EDGE_INFO: {edge_id} -> lanes:{lane_num}, vehicles:{vehicle_num}, speed:{vehicle_speed:.2f}, length:{road_len:.2f}")
                return result
        except Exception as e:
            print(f"EDGE_INFO: ERROR - Batch mode failed, fallback to single query: {e}")
    
    # 单个获取模式（使用优化版本）
    try:
        result = get_edge_info_single(edge_id)
        lane_num, vehicle_num, vehicle_speed, vehicle_length, road_len = result
        print(f"EDGE_INFO: {edge_id} -> lanes:{lane_num}, vehicles:{vehicle_num}, speed:{vehicle_speed:.2f}, length:{road_len:.2f}")
        return result
        
    except Exception as e:
        error_msg = f"ERROR: Failed to get edge info for {edge_id} (attempt {retry_count + 1}/{max_retries + 1}): {e}"
        print(error_msg)
        
        if retry_count < max_retries:
            print(f"RETRY: Retrying edge info query for {edge_id}")
            import time
            time.sleep(0.1)  # 短暂延迟后重试
            return get_edge_info(edge_id, retry_count + 1, max_retries)
        else:
            print(f"FALLBACK: Using static data for edge {edge_id}")
            # 最终回退：使用静态数据
            lane_num, road_len, speed_limit = get_static_road_info(edge_id)
            return lane_num, 0, 0.0, 0.0, road_len


def _is_edge_valid(edge_id):
    """检查边缘是否有效和可访问（使用缓存数据，避免重复查询）"""
    try:
        # 确保静态数据已加载
        if not _static_data_loaded:
            load_static_road_data()
        
        # 使用预加载的静态数据进行快速验证
        if edge_id in _valid_edges_cache:
            return True
        
        # 如果静态数据中没有，则该边缘无效
        return False
        
    except Exception as e:
        print(f"EDGE_VALIDATION: ERROR - Edge validation failed for {edge_id}: {e}")
        return False


def get_multiple_edges_info(edge_ids: List[str]) -> Dict[str, Tuple[int, int, float, float, float]]:
    """
    高效获取多个边缘信息的便利函数（最高性能版本）
    静态数据从JSON预加载，动态数据批量查询
    
    Args:
        edge_ids: 边缘ID列表
        
    Returns:
        字典，key为边缘ID，value为元组 (车道数, 车辆数, 车辆速度, 车辆长度, 道路长度)
        
    Example:
        >>> edges = ['edge1', 'edge2', 'edge3']
        >>> results = get_multiple_edges_info(edges)
        >>> print(results['edge1'])  # (2, 5, 13.8, 4.5, 100.0)
    """
    if not edge_ids:
        return {}
    
    # 确保静态数据已加载
    if not _static_data_loaded:
        load_static_road_data()
    
    result = {}
    
    if _use_batch_mode:
        try:
            # 批量获取动态数据
            dynamic_data = get_dynamic_data_batch(edge_ids)
            
            # 组合静态和动态数据
            for edge_id in edge_ids:
                # 获取静态数据
                lane_num, road_len, speed_limit = get_static_road_info(edge_id)
                
                # 获取动态数据
                if edge_id in dynamic_data:
                    vehicle_num, vehicle_speed, vehicle_length = dynamic_data[edge_id]
                else:
                    vehicle_num, vehicle_speed, vehicle_length = 0, 0.0, 0.0
                
                result[edge_id] = (lane_num, vehicle_num, vehicle_speed, vehicle_length, road_len)
            
        except Exception as e:
            print(f"MULTI_EDGES_INFO: ERROR - Batch mode failed, fallback to original method: {e}")
            return get_edges_info_batch(edge_ids)
    else:
        # 如果不使用批量模式，仍然逐个获取
        for edge_id in edge_ids:
            try:
                result[edge_id] = get_edge_info(edge_id)
            except Exception as e:
                print(f"MULTI_EDGES_INFO: ERROR - Failed to get edge {edge_id} info: {e}")
                # 使用静态数据作为回退
                lane_num, road_len, speed_limit = get_static_road_info(edge_id)
                result[edge_id] = (lane_num, 0, 0.0, 0.0, road_len)
    
    return result


def update_edges_info_for_observation(road_network, current_edges: List[str]) -> Dict[str, dict]:
    """
    为观察（observation）更新边缘信息，最高性能版本
    
    Args:
        road_network: 道路网络图
        current_edges: 当前需要更新的边缘列表
        
    Returns:
        更新后的边缘信息字典
    """
    if not current_edges:
        return {}
    
    # 确保静态数据已加载
    if not _static_data_loaded:
        load_static_road_data()
    
    # 批量获取动态数据
    print(f"OBSERVATION_UPDATE: Updating {len(current_edges)} edges for observation")
    dynamic_data = get_dynamic_data_batch(current_edges)
    
    # 构建结果字典
    edge_dict = {}
    for edge_id in current_edges:
        # 获取静态数据
        lane_num, road_len, speed_limit = get_static_road_info(edge_id)
        
        # 获取动态数据
        if edge_id in dynamic_data:
            vehicle_num, vehicle_speed, vehicle_length = dynamic_data[edge_id]
        else:
            vehicle_num, vehicle_speed, vehicle_length = 0, 0.0, 0.0
        
        # 计算拥塞水平
        try:
            # 简单的拥塞率计算：基于车辆数和车道数
            if lane_num > 0:
                congestion_rate = min(vehicle_num / (lane_num * 10), 1.0)  # 假设每车道最大10辆车
            else:
                congestion_rate = 0.0
            
            congestion_level = get_congestion_level(congestion_rate)
        except Exception:
            congestion_level = 0
        
        edge_dict[edge_id] = {
            'lane_num': lane_num,
            'vehicle_num': vehicle_num,
            'vehicle_speed': vehicle_speed,
            'vehicle_length': vehicle_length,
            'road_len': road_len,
            'congestion_level': congestion_level
        }
    
    return edge_dict


def get_congestion_level(congestion_rate):
    if 0.0 <= congestion_rate <= 0.60:
        return 0
    elif 0.60 < congestion_rate <= 0.70:
        return 1
    elif 0.70 < congestion_rate <= 0.80:
        return 2
    elif 0.80 < congestion_rate <= 0.90:
        return 3
    elif 0.90 < congestion_rate <= 1.0:
        return 4
    else:
        return 5


def parse_rou_file(file_path):
    """
    解析路由文件(.rou.xml 或 .rou.alt.xml)，提取车辆的起终点信息
    兼容特性：
    - 忽略 XML 命名空间（默认 xmlns）
    - 支持 vehicle 内联 <route> / <routeDistribution>
    - 支持 vehicle 的 route="id" / routeDistribution="id" 属性引用顶层定义
    """
    trips = []
    tree = ET.parse(file_path)
    root = tree.getroot()

    # 去除命名空间，便于通过本地名匹配
    for elem in root.iter():
        if isinstance(elem.tag, str) and '}' in elem.tag:
            elem.tag = elem.tag.rsplit('}', 1)[1]

    # 调试信息：输出根元素和车辆数量
    print(f"DEBUG: 解析文件 {file_path}")
    print(f"DEBUG: 根元素标签: {root.tag}")

    # 构建顶层 route(id->edges) 与 routeDistribution(id->list[(edges,prob)]) 映射
    route_id_to_edges = {}

    def _parse_route_distribution(rd_elem):
        entries = []  # (edges_seq, prob)
        probs = []
        for ch in list(rd_elem):
            if ch.tag != 'route':
                continue
            prob = ch.get('probability') or ch.get('prob') or ch.get('weight') or ch.get('p')
            try:
                prob = float(prob) if prob is not None else None
            except Exception:
                prob = None
            edges_attr = ch.get('edges')
            if edges_attr:
                edges_seq = edges_attr.strip().split()
            else:
                ref_id = ch.get('refId') or ch.get('id') or ch.get('route')
                edges_seq = route_id_to_edges.get(ref_id, [])
            if edges_seq:
                entries.append((edges_seq, prob))
                probs.append(prob)
        # 概率补全/归一化
        if entries:
            if any(p is None for _, p in entries):
                n = len(entries)
                entries = [(seq, 1.0 / n) for seq, _ in entries]
            else:
                s = sum(p for _, p in entries)
                if s > 0:
                    entries = [(seq, p / s) for seq, p in entries]
                else:
                    n = len(entries)
                    entries = [(seq, 1.0 / n) for seq, _ in entries]
        return entries

    route_dist_map = {}
    for elem in root.iter():
        if elem.tag == 'route':
            rid = elem.get('id')
            edges_attr = elem.get('edges')
            if rid and edges_attr:
                route_id_to_edges[rid] = edges_attr.strip().split()
        elif elem.tag == 'routeDistribution':
            rdid = elem.get('id')
            if rdid:
                route_dist_map[rdid] = _parse_route_distribution(elem)

    vehicles = root.findall('.//vehicle')
    print(f"DEBUG: 找到 {len(vehicles)} 个vehicle标签")

    vehicle_count = 0
    debug_limit = 5  # 只对前5个车辆输出详细调试信息

    for vehicle in vehicles:
        vehicle_count += 1
        vehicle_id = vehicle.get('id')
        debug_output = vehicle_count <= debug_limit

        if debug_output:
            print(f"DEBUG: 处理车辆 {vehicle_count}: {vehicle_id}")

        def _append_trip_from_edges(edges_str: str) -> bool:
            if not edges_str:
                return False
            edge_list = edges_str.strip().split()
            if not edge_list:
                return False
            start_edge = edge_list[0]
            end_edge = edge_list[-1]
            try:
                depart_time = float(vehicle.get('depart', 0.0))
            except Exception:
                depart_time = 0.0
            trips.append((vehicle_id, start_edge, end_edge, depart_time))
            if debug_output:
                print(f"DEBUG: 成功解析车辆 {vehicle_id}: {start_edge} -> {end_edge}")
            return True

        parsed = False

        # 1) 处理属性引用：route / routeDistribution
        if not parsed:
            ref_route_id = vehicle.get('route')
            if ref_route_id and ref_route_id in route_id_to_edges:
                parsed = _append_trip_from_edges(' '.join(route_id_to_edges[ref_route_id]))
                if debug_output and parsed:
                    print(f"DEBUG: 通过属性 route=\"{ref_route_id}\" 解析")

        if not parsed:
            ref_rd_id = vehicle.get('routeDistribution') or vehicle.get('routeDist')
            if ref_rd_id and ref_rd_id in route_dist_map and route_dist_map[ref_rd_id]:
                # 选择概率最大的，若相等取第一个
                entries = route_dist_map[ref_rd_id]
                best = max(entries, key=lambda x: x[1] if x[1] is not None else 0.0)
                parsed = _append_trip_from_edges(' '.join(best[0]))
                if debug_output and parsed:
                    print(f"DEBUG: 通过属性 routeDistribution=\"{ref_rd_id}\" 解析")

        # 2) 处理子元素 routeDistribution
        if not parsed:
            route_distribution = vehicle.find('routeDistribution')
            if route_distribution is not None:
                if debug_output:
                    print(f"DEBUG: 找到routeDistribution")
                # 优先取首个有效 route 子元素
                chosen_edges = None
                for ch in list(route_distribution):
                    if ch.tag != 'route':
                        continue
                    if ch.get('edges'):
                        chosen_edges = ch.get('edges')
                        break
                    # route 引用 id 的情况
                    ref_id = ch.get('refId') or ch.get('id') or ch.get('route')
                    if ref_id and ref_id in route_id_to_edges:
                        chosen_edges = ' '.join(route_id_to_edges[ref_id])
                        break
                if chosen_edges:
                    parsed = _append_trip_from_edges(chosen_edges)
                else:
                    if debug_output:
                        print(f"DEBUG: routeDistribution 中没有有效的 route 条目")

        # 3) 处理子元素 route
        if not parsed:
            route = vehicle.find('route')
            if route is not None:
                if debug_output:
                    print(f"DEBUG: 找到直接route标签")
                edges = route.get('edges')
                if not edges:
                    # 可能是引用形式 <route id="r1"/>
                    ref = route.get('id') or route.get('refId') or route.get('route')
                    if ref and ref in route_id_to_edges:
                        edges = ' '.join(route_id_to_edges[ref])
                if edges:
                    parsed = _append_trip_from_edges(edges)
                else:
                    if debug_output:
                        print(f"DEBUG: 直接route没有edges属性也没有有效引用")
            else:
                if debug_output:
                    print(f"DEBUG: 没有找到直接route标签")

        # 在调试限制处输出提示
        if vehicle_count == debug_limit and len(vehicles) > debug_limit:
            print(f"DEBUG: 继续处理剩余 {len(vehicles) - debug_limit} 个车辆（不输出详细信息）...")

    print(f"从路由文件 {file_path} 解析到 {len(trips)} 个车辆路径")
    return trips


def get_1_2_hop_neighbors(graph, query_edge):
    one_hop_neighbors = set(graph.neighbors(query_edge))

    two_hop_neighbors = set()
    for neighbor in one_hop_neighbors:
        two_hop_neighbors.update(graph.neighbors(neighbor))

    return one_hop_neighbors, two_hop_neighbors

def get_subgraph(graph, query_edges):
    sub_graph = graph.subgraph(query_edges)

    return sub_graph

def get_autonomous_vehicle_observation(vehicle_ids, autonomous_vehicles, road_info, road_network):
    traci_interface = get_traci_interface()
    update_vehicle_info = [[], [], [], [], [], []]
    for veh_id in vehicle_ids:
        if veh_id in autonomous_vehicles:
            # get observation
            current_edge = traci_interface.vehicle.getRoadID(veh_id)
            end_edge = traci_interface.vehicle.getRoute(veh_id)[-1]
            veh_trip = (veh_id, current_edge, end_edge)
            road_candidates, data_text, answer_option_form = get_observation_text(veh_trip, road_network, road_info)

            if len(road_candidates) < 2:
                continue

            update_vehicle_info[0].append(veh_id)
            update_vehicle_info[1].append(data_text)
            update_vehicle_info[2].append(answer_option_form)
            update_vehicle_info[3].append(current_edge)
            update_vehicle_info[4].append(end_edge)
            update_vehicle_info[5].append(road_candidates)

    return update_vehicle_info


def get_observation(trip, road_network, edge_dict):
    _, start_edge, end_edge = trip

    edge_candidates, _ = get_1_2_hop_neighbors(road_network, start_edge)

    edge_candidates = list(edge_candidates)[:10]
    edge_candidate_info = {}
    for edge_can in edge_candidates:
        congestion_level = edge_dict[edge_can]["congestion_level"]

        try:
            shortest_route_len = nx.dijkstra_path_length(road_network, source=edge_can, target=end_edge)
        except Exception:
            continue

        one_hop_neighbors, two_hop_neighbors = get_1_2_hop_neighbors(road_network, edge_can)
        nei_dict = {}
        for nei in one_hop_neighbors:
            if len(nei_dict) >= 10:
                break
            nei_congestion_level = edge_dict[nei]['congestion_level']
            nei_dict[nei] = {
                "hop": 1,
                "congestion_level": nei_congestion_level
            }
        for nei in two_hop_neighbors:
            if len(nei_dict) >= 10:
                break
            nei_congestion_level = edge_dict[nei]['congestion_level']
            nei_dict[nei] = {
                "hop": 2,
                "congestion_level": nei_congestion_level
            }

        edge_candidate_info[edge_can] = {
            "congestion_level": congestion_level,
            "shortest_route_len": shortest_route_len,
            "neighbors": nei_dict
        }

    return edge_candidate_info

def get_observation_text(trip, road_network, edge_dict):
    # get observation
    road_candidates = get_observation(trip, road_network, edge_dict)

    candidate_roads_texts = []
    answer_option_form = "\"" + "/".join([edge_can for edge_can in road_candidates]) + "\""
    for edge_can in road_candidates:
        can_road_text = (f"road: {edge_can}\n"
                         f"- congestion_level: {str(road_candidates[edge_can]['congestion_level'])}\n"
                         f"- shortest_route_length: {str(round(road_candidates[edge_can]['shortest_route_len'], 2))}m\n"
                         f"- road_length: {str(round(edge_dict[edge_can]['road_len'], 2))}m")
        candidate_roads_texts.append(can_road_text)

    candidate_roads_text = ("Candidate roads:\n\n" +
                            "\n\n".join([can_road_text for can_road_text in candidate_roads_texts]))

    # Get subgraph adj & nearby edges
    nearby_roads = []
    for edge_can in road_candidates:
        nearby_roads += list([nei for nei in road_candidates[edge_can]["neighbors"]])
    nearby_roads = list(set(nearby_roads))
    subgraph_roads = set([edge_can for edge_can in road_candidates] + nearby_roads)
    subgraph = get_subgraph(road_network, subgraph_roads)

    nearby_road_texts = []
    for nei_road in nearby_roads:
        road_text = (f"road {nei_road}:\n"
                     f"- congestion_level: {edge_dict[nei_road]['congestion_level']}\n"
                     f"- road_length: {edge_dict[nei_road]['road_len']}m")
        nearby_road_texts.append(road_text)
    nearby_roads_text = ("Nearby roads:\n\n" +
                         "\n\n".join(nearby_road_texts))

    # adjacency info
    edges_str = [f"({u}, {v}, {round(d['weight'], 2)}m)" for u, v, d in subgraph.edges(data=True)]
    adj_info_text = (f"Connectivity:\n"
                     "[" + ", ".join(edges_str) + "]")

    obs_text = candidate_roads_text + "\n\n" + nearby_roads_text + "\n\n" + adj_info_text

    return road_candidates, obs_text, answer_option_form


def update_route(current_edge, next_edge, end_edge):
    traci_interface = get_traci_interface()
    current_route = traci_interface.simulation.findRoute(fromEdge=current_edge, toEdge=end_edge)
    try:
        candidate_route = traci_interface.simulation.findRoute(fromEdge=next_edge, toEdge=end_edge)
        new_route = [current_edge] + list(candidate_route.edges)
        return new_route

    except Exception as e:
        print(f"Route Switch Failed: {e}")
        return current_route.edges


def get_env_change_text(current_road, candidate_roads):
    average_congestion_level = np.mean([candidate_roads[road]["congestion_level"] for road in candidate_roads])
    average_route_length = np.mean([candidate_roads[road]["shortest_route_len"] for road in candidate_roads])

    nei_1_congestion = []
    nei_2_congestion = []
    for road in candidate_roads:
        for nei in candidate_roads[road]["neighbors"]:
            if candidate_roads[road]["neighbors"][nei]["hop"] == 1:
                nei_1_congestion.append(candidate_roads[road]["neighbors"][nei]["congestion_level"])
            else:
                nei_2_congestion.append(candidate_roads[road]["neighbors"][nei]["congestion_level"])

    average_nei_1_congestion = np.mean(nei_1_congestion)
    average_nei_2_congestion = np.mean(nei_2_congestion)

    env_change_text = (f"Candidate roads after passing road {current_road}:\n"
                       f"- average_congestion_level: {str(round(average_congestion_level))}\n"
                       f"- average_route_length: {str(round(average_route_length, 2))}m\n"
                       f"- average_1_hop_connected_road_congestion_level: {str(round(average_nei_1_congestion))}\n"
                       f"- average_2_hop_connected_road_congestion_level: {str(round(average_nei_2_congestion))}")

    return env_change_text


# ================================
# 性能优化使用示例和测试
# ================================

def performance_test_edge_info(edge_ids: List[str], iterations: int = 5):
    """
    性能测试函数：比较原始方法和批量方法的性能
    
    Args:
        edge_ids: 要测试的边缘ID列表
        iterations: 测试迭代次数
    """
    if not edge_ids:
        print("没有提供边缘ID进行测试")
        return
    
    print(f"开始性能测试: {len(edge_ids)} 个边缘, {iterations} 次迭代")
    
    # 测试原始方法
    print("\n=== 测试原始逐个获取方法 ===")
    start_time = time.time()
    
    for _ in range(iterations):
        configure_edge_info_optimization(use_batch_mode=False)
        clear_edge_info_cache()
        
        for edge_id in edge_ids:
            try:
                get_edge_info(edge_id)
            except Exception as e:
                print(f"原始方法获取 {edge_id} 失败: {e}")
    
    original_time = time.time() - start_time
    
    # 测试批量方法
    print("\n=== 测试批量获取方法 ===")
    start_time = time.time()
    
    for _ in range(iterations):
        configure_edge_info_optimization(use_batch_mode=True)
        clear_edge_info_cache()
        
        try:
            get_multiple_edges_info(edge_ids)
        except Exception as e:
            print(f"批量方法失败: {e}")
    
    batch_time = time.time() - start_time
    
    # 输出结果
    print(f"\n=== 性能测试结果 ===")
    print(f"原始方法总时间: {original_time:.3f} 秒")
    print(f"批量方法总时间: {batch_time:.3f} 秒")
    
    if batch_time > 0:
        speedup = original_time / batch_time
        print(f"性能提升: {speedup:.2f}x")
        print(f"时间节省: {((original_time - batch_time) / original_time * 100):.1f}%")
    
    # 恢复默认设置
    configure_edge_info_optimization(use_batch_mode=True)


def usage_example():
    """
    使用示例函数，展示如何使用优化后的边缘信息获取功能
    """
    print("=== 边缘信息获取优化使用示例 ===\n")
    
    # 1. 配置优化选项
    print("1. 配置优化选项:")
    print("   configure_edge_info_optimization(use_batch_mode=True, use_libsumo=False, cache_timeout=10.0)")
    configure_edge_info_optimization(
        use_batch_mode=True,
        use_libsumo=False,  # 如果安装了 libsumo 可以设为 True
        cache_timeout=10.0
    )
    
    # 2. 单个边缘信息获取（自动使用批量模式和缓存）
    print("\n2. 获取单个边缘信息（自动优化）:")
    print("   result = get_edge_info('edge_1')")
    
    # 3. 批量获取多个边缘信息
    print("\n3. 批量获取多个边缘信息:")
    print("   edges = ['edge_1', 'edge_2', 'edge_3']")
    print("   results = get_multiple_edges_info(edges)")
    
    # 4. 为观察更新边缘信息
    print("\n4. 为观察功能优化边缘信息:")
    print("   edge_dict = update_edges_info_for_observation(road_network, current_edges)")
    
    # 5. 缓存管理
    print("\n5. 缓存管理:")
    print("   clear_edge_info_cache()  # 清空缓存")
    
    print("\n=== 关键优化特性 ===")
    print("✓ 使用 TraCI subscriptions 批量获取，减少网络调用")
    print("✓ 智能缓存机制，避免重复获取相同数据")
    print("✓ 支持 libsumo 高性能模式")
    print("✓ 完全向后兼容，无需修改现有代码")
    print("✓ 自动错误处理和回退机制")
    print("✓ 可配置的缓存超时时间")


# 为了向后兼容，确保现有函数继续工作
def get_edge_lane_info_optimized(edge_id, lane_id):
    """
    优化版本的 get_edge_lane_info，使用批量获取
    """
    edge_info = get_edge_info(edge_id)
    lane_num, vehicle_num, vehicle_speed, vehicle_length, road_len = edge_info
    
    # 获取速度限制
    try:
        traci_interface = get_traci_interface()
        speed_limit = traci_interface.lane.getMaxSpeed(lane_id)
    except Exception:
        speed_limit = 13.89  # 默认速度限制
    
    return lane_num, vehicle_num, vehicle_speed, vehicle_length, road_len, speed_limit
