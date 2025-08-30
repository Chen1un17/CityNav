import sys
sys.path.append("../")
sys.path.append("/home/zhouyuping/program/LLMNavigation_basline/LLMNavigation")

import os.path
import random
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime
import multiprocessing as mp
from functools import partial

import geopandas as gpd
import numpy as np
import pyproj
import sumolib
from shapely.geometry import Point
from shapely.strtree import STRtree
from tqdm import tqdm

# 尝试导入GPU加速库
try:
    import cupy as cp
    import cupyx.scipy.spatial
    # 测试GPU是否真正可用
    test_array = cp.array([1, 2, 3])
    GPU_AVAILABLE = True
    print("GPU acceleration available (CuPy detected)")
except (ImportError, Exception) as e:
    GPU_AVAILABLE = False
    print(f"GPU acceleration not available: {e}")
    print("Using CPU-only processing")

# from utils.gravity_model import GravityGenerator  # No longer needed, using existing OD matrix
from utils.read_utils import load_json, dump_json

print("Starting efficient trip generation without SUMO simulation...")

# 定义行车道的类型和允许的车辆类型
driveable_types = {"highway", "motorway", "primary", "secondary", "tertiary", "residential"}
allowed_vehicles = {"passenger", "bus", "truck"}  # 常见的行车车辆类型

def calculate_midpoint(shape):
    num_points = len(shape)
    if num_points == 0:
        return None
    mid_index = num_points // 2
    if num_points % 2 == 0:
        # 如果有偶数个点，计算中间两点的平均
        midpoint = ((shape[mid_index-1][0] + shape[mid_index][0]) / 2,
                    (shape[mid_index-1][1] + shape[mid_index][1]) / 2)
    else:
        # 如果有奇数个点，取中间点
        midpoint = shape[mid_index]
    return midpoint

def parse_edges(net):
    edges = net.getEdges()
    edges_midpoints = []

    for edge in tqdm(edges):
        shape = edge.getShape()
        midpoint = calculate_midpoint(shape)
        edges_midpoints.append({'id': edge.getID(), 'point': midpoint, 'type': edge.getType(), 'lanes': edge._lanes})

    return edges_midpoints


def check_edge_driveable(edge):
    """检查edge是否可行车"""
    # 检查edge类型
    for ty in driveable_types:
        if ty in edge['type']:
            return True
    
    # 检查车道是否允许车辆通行
    for lane in edge['lanes']:
        if set(lane._allowed).intersection(allowed_vehicles):
            return True
    
    return False


def process_edge_batch(args):
    """处理一批edges的区域分配 - 修复序列化问题"""
    edge_batch, areas_geoms, areas_indices = args
    
    from shapely.strtree import STRtree
    from shapely.geometry import Point
    import geopandas as gpd
    
    # 重建几何对象
    geometries = []
    for geom_wkt in areas_geoms:
        from shapely import wkt
        geometries.append(wkt.loads(geom_wkt))
    
    # 构建空间索引
    spatial_index = STRtree(geometries)
    
    batch_results = {}
    
    for edge_data in edge_batch:
        edge_id, lon, lat, is_driveable = edge_data
        
        if not is_driveable:
            continue
            
        query_point = Point(lon, lat)
        
        # 使用空间索引快速查找
        possible_matches_idx = spatial_index.query(query_point)
        
        # 检查实际包含关系
        for idx in possible_matches_idx:
            if geometries[idx].contains(query_point):
                region_idx = areas_indices[idx]  # 获取原始索引
                if region_idx not in batch_results:
                    batch_results[region_idx] = []
                batch_results[region_idx].append(edge_id)
                break
    
    return batch_results


def assign_edges_to_areas_parallel(edges, areas, net, num_processes=None):
    """并行分配edges到areas - 修复序列化问题"""
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 8)
    
    print(f"Using {num_processes} processes for parallel processing...")
    
    # 预处理edges数据
    print("Pre-processing edges data...")
    processed_edges = []
    for edge in tqdm(edges, desc="Converting coordinates"):
        point = Point(*edge['point'])
        lon, lat = net.convertXY2LonLat(point.x, point.y)
        is_driveable = check_edge_driveable(edge)
        processed_edges.append((edge['id'], lon, lat, is_driveable))
    
    # 序列化areas几何数据为WKT格式（可序列化）
    print("Preparing areas data...")
    areas_geoms = [geom.wkt for geom in areas.geometry.values]
    areas_indices = list(areas.index.values)
    
    # 分批处理
    batch_size = max(len(processed_edges) // (num_processes * 4), 100)
    edge_batches = [processed_edges[i:i + batch_size] for i in range(0, len(processed_edges), batch_size)]
    
    print(f"Split {len(processed_edges)} edges into {len(edge_batches)} batches")
    
    # 准备参数
    batch_args = [(batch, areas_geoms, areas_indices) for batch in edge_batches]
    
    # 并行处理
    assigned_edges = {}
    
    with mp.Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_edge_batch, batch_args),
            total=len(batch_args),
            desc="Processing edge batches"
        ))
    
    # 合并结果
    for batch_result in results:
        for region_idx, edge_list in batch_result.items():
            if region_idx not in assigned_edges:
                assigned_edges[region_idx] = []
            assigned_edges[region_idx].extend(edge_list)
    
    final_result = {str(k): v for k, v in assigned_edges.items()}
    print(f"Assigned {sum(len(v) for v in final_result.values())} edges to {len(final_result)} regions")
    
    return final_result


def assign_edges_to_areas_gpu(edges, areas, net):
    """GPU加速版本（如果CuPy可用）"""
    if not GPU_AVAILABLE:
        return assign_edges_to_areas(edges, areas, net)
    
    print("Using GPU acceleration for edge assignment...")
    
    # 预处理坐标数据
    print("Converting coordinates on CPU...")
    edge_coords = []
    edge_ids = []
    
    for edge in tqdm(edges, desc="Processing edges"):
        if not check_edge_driveable(edge):
            continue
            
        point = Point(*edge['point'])
        lon, lat = net.convertXY2LonLat(point.x, point.y)
        edge_coords.append([lon, lat])
        edge_ids.append(edge['id'])
    
    # 转换为GPU数组
    edge_coords_gpu = cp.array(edge_coords, dtype=cp.float32)
    
    print(f"Processing {len(edge_coords)} edges on GPU...")
    
    assigned_edges = {}
    
    # 对每个区域使用GPU进行并行点-多边形测试
    for idx, area in tqdm(areas.iterrows(), total=len(areas), desc="Processing areas on GPU"):
        # 获取区域边界框进行快速筛选
        bounds = area['geometry'].bounds
        minx, miny, maxx, maxy = bounds
        
        # 在GPU上进行边界框筛选
        mask = ((edge_coords_gpu[:, 0] >= minx) & 
                (edge_coords_gpu[:, 0] <= maxx) &
                (edge_coords_gpu[:, 1] >= miny) & 
                (edge_coords_gpu[:, 1] <= maxy))
        
        # 将mask转回CPU进行精确的几何测试
        candidates = cp.asnumpy(cp.where(mask)[0])
        
        if len(candidates) > 0:
            # 对候选点进行精确的包含测试
            for candidate_idx in candidates:
                lon, lat = edge_coords[candidate_idx]
                point = Point(lon, lat)
                if area['geometry'].contains(point):
                    if idx not in assigned_edges:
                        assigned_edges[idx] = []
                    assigned_edges[idx].append(edge_ids[candidate_idx])
    
    final_result = {str(k): v for k, v in assigned_edges.items()}
    print(f"GPU processing assigned {sum(len(v) for v in final_result.values())} edges to {len(final_result)} regions")
    
    return final_result


def assign_edges_to_areas(edges, areas, net):
    """原始的单线程版本（作为备用）"""
    assigned_edges = {}

    for edge in tqdm(edges, desc="Processing edges (single-threaded)"):
        if not check_edge_driveable(edge):
            continue
            
        point = Point(*edge['point'])
        lon, lat = net.convertXY2LonLat(point.x, point.y)
        point = Point([lon, lat])
        
        for idx, area in areas.iterrows():
            if area['geometry'].contains(point):
                if idx not in assigned_edges:
                    assigned_edges[idx] = []
                assigned_edges[idx].append(edge['id'])
                break

    return {str(k): v for k, v in assigned_edges.items()}


def generate_trips_efficient(od_matrix, assigned_edges, departure_prob):
    """
    高效生成trip数据，内存优化版本
    适用于大规模数据（百万级轨迹）
    """
    total_trips = int(od_matrix.sum())
    print(f"Generating {total_trips:,} trips from OD matrix...")
    
    # 内存优化：使用生成器而非列表
    def trip_generator():
        trip_count = 0
        for i in range(od_matrix.shape[0]):
            for j in range(od_matrix.shape[1]):
                if i != j and od_matrix[i, j] > 0:
                    # 检查区域是否有可用边
                    if str(i) not in assigned_edges or str(j) not in assigned_edges:
                        continue
                        
                    from_edges = assigned_edges[str(i)]
                    to_edges = assigned_edges[str(j)]
                    
                    if not from_edges or not to_edges:
                        continue
                    
                    # 生成该OD对的所有trips
                    num_trips = int(od_matrix[i, j])
                    
                    # 批量生成出发时间和边选择以提高效率
                    start_hours = np.random.choice(range(24), size=num_trips, p=departure_prob)
                    start_seconds = np.random.randint(0, 3600, size=num_trips)
                    
                    for trip_idx in range(num_trips):
                        start_time = start_hours[trip_idx] * 3600 + start_seconds[trip_idx]
                        
                        # 随机选择起终点边
                        from_edge = random.choice(from_edges)
                        to_edge = random.choice(to_edges)
                        
                        trip_id = f"trip_{i}_{j}_{trip_idx}"
                        
                        yield {
                            'id': trip_id,
                            'start_time': start_time,
                            'from': from_edge, 
                            'to': to_edge,
                            'from_region': i,
                            'to_region': j
                        }
                        
                        trip_count += 1
                        if trip_count % 100000 == 0:
                            print(f"Generated {trip_count:,} trips...")
    
    # 转换为列表（如果内存不足，可以考虑直接写入文件）
    print("Converting generator to list...")
    trips = list(tqdm(trip_generator(), total=total_trips, desc="Generating trips"))
    
    print(f"Generated {len(trips):,} valid trips")
    return trips


def generate_trips_memory_efficient(od_matrix, assigned_edges, departure_prob, output_file):
    """
    内存高效版本：直接写入文件，不在内存中保存所有trips
    适用于超大规模数据
    """
    total_trips = int(od_matrix.sum())
    print(f"Generating {total_trips:,} trips directly to file...")
    
    # 创建XML根节点
    root = ET.Element('routes')
    
    # 添加车辆类型定义
    vtype = ET.SubElement(root, 'vType')
    vtype.set('id', 'passenger')
    vtype.set('vClass', 'passenger')
    vtype.set('maxSpeed', '50')
    vtype.set('speedFactor', '1.0')
    
    trip_count = 0
    
    with tqdm(total=total_trips, desc="Writing trips") as pbar:
        for i in range(od_matrix.shape[0]):
            for j in range(od_matrix.shape[1]):
                if i != j and od_matrix[i, j] > 0:
                    # 检查区域是否有可用边
                    if str(i) not in assigned_edges or str(j) not in assigned_edges:
                        continue
                        
                    from_edges = assigned_edges[str(i)]
                    to_edges = assigned_edges[str(j)]
                    
                    if not from_edges or not to_edges:
                        continue
                    
                    # 生成该OD对的所有trips
                    num_trips = int(od_matrix[i, j])
                    
                    # 批量生成时间
                    start_hours = np.random.choice(range(24), size=num_trips, p=departure_prob)
                    start_seconds = np.random.randint(0, 3600, size=num_trips)
                    
                    for trip_idx in range(num_trips):
                        start_time = start_hours[trip_idx] * 3600 + start_seconds[trip_idx]
                        
                        # 随机选择起终点边
                        from_edge = random.choice(from_edges)
                        to_edge = random.choice(to_edges)
                        
                        trip_id = f"trip_{i}_{j}_{trip_idx}"
                        
                        # 直接创建XML元素
                        trip_elem = ET.SubElement(root, 'trip')
                        trip_elem.set('id', trip_id)
                        trip_elem.set('depart', str(start_time))
                        trip_elem.set('from', from_edge)
                        trip_elem.set('to', to_edge)
                        trip_elem.set('type', 'passenger')
                        
                        trip_count += 1
                        pbar.update(1)
    
    # 写入文件
    print(f"Writing {trip_count:,} trips to {output_file}")
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    
    print(f"Successfully generated {trip_count:,} trips")
    return trip_count

def write_trips_to_xml(trips, output_file):
    """
    生成trip文件用于duarouter批量路径计算
    """
    root = ET.Element('routes')
    
    # 添加车辆类型定义
    vtype = ET.SubElement(root, 'vType')
    vtype.set('id', 'passenger')
    vtype.set('vClass', 'passenger')
    vtype.set('maxSpeed', '50')
    vtype.set('speedFactor', '1.0')
    
    print(f"Writing {len(trips)} trips to XML...")
    sorted_trips = sorted(trips, key=lambda x: x['start_time'])
    
    # 使用trip而非flow，让duarouter计算路径
    for trip in tqdm(sorted_trips, desc="Writing trips"):
        trip_elem = ET.SubElement(root, 'trip')
        trip_elem.set('id', trip['id'])
        trip_elem.set('depart', str(trip['start_time']))
        trip_elem.set('from', trip['from'])
        trip_elem.set('to', trip['to'])
        trip_elem.set('type', 'passenger')

    # 格式化XML输出
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    
    print(f"Saving trips to {output_file}")
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"Trip file generated successfully!")


def run_duarouter(trip_file, network_file, output_file):
    """
    使用duarouter批量计算路径
    """
    print(f"Running duarouter for {trip_file}...")
    
    duarouter_cmd = [
        'duarouter',
        '--trip-files', trip_file,
        '--net-file', network_file, 
        '--output-file', output_file,
        '--ignore-errors',
        '--no-warnings',
        '--repair',
        '--remove-loops'
    ]
    
    try:
        print("Executing:", ' '.join(duarouter_cmd))
        result = subprocess.run(duarouter_cmd, 
                              capture_output=True, 
                              text=True, 
                              check=True)
        
        print("duarouter completed successfully!")
        if result.stdout:
            print("STDOUT:", result.stdout)
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"duarouter failed with return code {e.returncode}")
        print("STDERR:", e.stderr)
        print("STDOUT:", e.stdout)
        return False
    except FileNotFoundError:
        print("Error: duarouter not found. Please make sure SUMO is installed and duarouter is in PATH")
        return False


def main():
    """
    主函数：高效生成大规模轨迹数据
    """
    # 配置文件路径
    roadnet_file = "/data/zhouyuping/LLMNavigation/Data/NYC/NewYork.net.xml"
    area_file = "/data/zhouyuping/LLMNavigation/Data/NYC_shp/NYC_areas.shp"
    od_matrix_file = "/data/zhouyuping/LLMNavigation/Data/NYC_shp/NYC_od_matrix.npy"
    
    # 根据您的需求设置scaling，生成2,399,701条轨迹
    # 原始OD矩阵总和约为23,997,010，所以scaling约为0.1
    scaling = 0.1
    
    print(f"=== 大规模轨迹生成系统 ===")
    print(f"目标轨迹数量: ~2,400,000")
    print(f"Scaling factor: {scaling}")

    # 加载网络和区域数据
    print("Loading network and region data...")
    roadnet = sumolib.net.readNet(roadnet_file)
    areas = gpd.read_file(area_file)

    # 解析道路网络并分配边到区域（使用并行优化）
    edges_file = "/data/zhouyuping/LLMNavigation/Data/NYC/NYC_area2edge.json"
    if not os.path.exists(edges_file):
        print("Parsing network edges and assigning to regions...")
        edges = parse_edges(roadnet)
        
        # 尝试不同的加速方法，按优先级顺序
        try:
            if GPU_AVAILABLE:
                print("Attempting GPU-accelerated processing...")
                assigned_edges = assign_edges_to_areas_gpu(edges, areas, roadnet)
            else:
                print("Attempting CPU parallel processing...")
                assigned_edges = assign_edges_to_areas_parallel(edges, areas, roadnet)
        except Exception as e:
            print(f"Accelerated processing failed: {e}")
            print("Falling back to CPU parallel processing...")
            try:
                assigned_edges = assign_edges_to_areas_parallel(edges, areas, roadnet)
            except Exception as e2:
                print(f"Parallel processing also failed: {e2}")
                print("Falling back to single-threaded processing...")
                assigned_edges = assign_edges_to_areas(edges, areas, roadnet)
            
        dump_json(assigned_edges, edges_file)
        print(f"Saved edge assignments to {edges_file}")
    else:
        print("Loading existing edge assignments...")
    
    assigned_edges = load_json(edges_file)
    print(f"Loaded edge assignments for {len(assigned_edges)} regions")

    # 加载OD矩阵并应用scaling
    print("Loading OD matrix...")
    od_matrix_raw = np.load(od_matrix_file)
    od_matrix = np.int64(od_matrix_raw * scaling)
    
    print(f"OD matrix shape: {od_matrix.shape}")
    print(f"Total trips after scaling: {od_matrix.sum():,}")
    print(f"Non-zero OD pairs: {np.count_nonzero(od_matrix):,}")

    # 设置出发时间分布（与您要求的一致）
    departure_time_curve = [1, 1, 1, 1, 2, 3, 4, 5, 6, 5, 4, 0.5, 1, 1, 0.5, 1, 1, 0.5, 1, 1, 0.5, 1, 1, 0.5]
    sum_times = sum(departure_time_curve)
    departure_prob = np.array([d / sum_times for d in departure_time_curve])
    
    print("Departure time distribution (24 hours):")
    for i, prob in enumerate(departure_prob):
        print(f"  Hour {i:2d}: {prob:.3f}")

    # 输出文件路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trip_file = f"/data/zhouyuping/LLMNavigation/Data/NYC/NYC_trips_{scaling}_{timestamp}.xml"
    route_file = f"/data/zhouyuping/LLMNavigation/Data/NYC/NYC_routes_{scaling}_{timestamp}.xml"

    # 检查预计内存使用量，选择最佳生成方法
    estimated_memory_gb = od_matrix.sum() * 200 / (1024**3)  # 每个trip约200字节
    
    print(f"\n=== 开始生成trips ===")
    print(f"预计内存使用: ~{estimated_memory_gb:.1f} GB")
    
    if estimated_memory_gb > 8.0:  # 如果预计使用超过8GB内存
        print("使用内存高效模式（直接写入文件）...")
        trip_count = generate_trips_memory_efficient(od_matrix, assigned_edges, departure_prob, trip_file)
        print(f"生成了 {trip_count:,} 条轨迹")
    else:
        print("使用标准模式...")
        trips = generate_trips_efficient(od_matrix, assigned_edges, departure_prob)
        
        # 写入trip文件
        print(f"\n=== 写入trip文件 ===")
        write_trips_to_xml(trips, trip_file)

    # 使用duarouter批量计算路径
    print(f"\n=== 使用duarouter计算路径 ===")
    success = run_duarouter(trip_file, roadnet_file, route_file)
    
    if success:
        print(f"\n=== 生成完成 ===")
        print(f"Trip file: {trip_file}")
        print(f"Route file: {route_file}")
        print(f"Total trajectories: {len(trips):,}")
    else:
        print(f"\n=== 路径计算失败 ===")
        print(f"Trip file generated: {trip_file}")
        print("Please run duarouter manually or check SUMO installation")


if __name__ == "__main__":
    # 多进程支持需要
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 已经设置过了
    main()