import sys
sys.path.append("../")

import os.path
import random
import threading
import time
import xml.etree.ElementTree as ET

import geopandas as gpd
import numpy as np
import pyproj
import sumolib
import traci
from shapely.geometry import Point
from tqdm import tqdm

from utils.gravity_model import GravityGenerator
from utils.read_utils import load_json, dump_json

# 启动 SUMO 仿真
sumoCmd = ["sumo", "-c", "../scripts/sumo_config.sumocfg", "--no-warnings", "--ignore-route-errors"]
traci.start(sumoCmd)
print("Simulation starts.")

# 定义行车道的类型和允许的车辆类型
driveable_types = {"highway", "motorway", "primary", "secondary", "tertiary", "residential"}
allowed_vehicles = {"passenger", "bus", "truck"}  # 常见的行车车辆类型

trips = []

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


def assign_edges_to_areas(edges, areas, net):
    assigned_edges = {}

    for edge in tqdm(edges):
        point = Point(*edge['point'])
        lon, lat = net.convertXY2LonLat(point.x, point.y)
        point = Point([lon, lat])
        for idx, area in areas.iterrows():
            if area['geometry'].contains(point):
                # 检查edge类型
                is_driveable = False
                for ty in driveable_types:
                    if ty in edge['type']:
                        is_driveable = True
                        break

                for lane in edge['lanes']:
                    if set(lane._allowed).intersection(allowed_vehicles):
                        is_driveable = True
                        break

                if is_driveable:
                    if idx not in assigned_edges:
                        assigned_edges[idx] = []
                    assigned_edges[idx].append(edge['id'])
                break

    return assigned_edges


def build_area_routes(i, j, from_edges, to_edges, num_trips, departure_prob):
    # print("==========================================")
    # print(f"Extracting area {i} and {j}.")
    for _ in range(num_trips):
        start_time = np.random.choice(range(24), p=departure_prob) * 3600 + random.randint(0, 3599)
        for k in range(10):
            from_edge = random.choice(from_edges)
            to_edge = random.choice(to_edges)
            is_valid = routing(from_edge, to_edge)

            if is_valid:
                # print("==================================")
                # print(f"Route find from from: {from_edge}, to: {to_edge}!")
                trips.append({'start_time': start_time, 'from': from_edge, 'to': to_edge})
                break


def generate_trips(od_matrix, assigned_edges, departure_prob, agent_num):
    threads = []

    for i in tqdm(range(od_matrix.shape[0])):
        for j in range(od_matrix.shape[1]):
            t1 = time.time()
            if i != j and od_matrix[i, j] > 0 and str(i) in assigned_edges and str(j) in assigned_edges:
                from_edges = assigned_edges[str(i)]
                to_edges = assigned_edges[str(j)]

                # Scale the number of trips by the scaling factor
                num_trips = od_matrix[i, j]

                if len(threads) < 10:
                    threads.append(threading.Thread(target=build_area_routes, args=(i, j, from_edges, to_edges, num_trips, departure_prob)))

                else:
                    for t in threads:
                        t.start()

                    for t in threads:
                        t.join()

                    threads = [threading.Thread(target=build_area_routes,
                                                args=(i, j, from_edges, to_edges, num_trips, departure_prob))]

                    t2 = time.time()
                    print("====================================")
                    print(f"Area {i}, from {j-10} - {j} done. It tasks {(t2 - t1) / 60}min.")
                    print("====================================")

    return trips


def routing(start_edge, end_edge):
    flag = False

    try:
        route = traci.simulation.findRoute(fromEdge=start_edge, toEdge=end_edge)
        if route.edges:
            flag = True
    except Exception as e:
        # print(f"无法计算从 {start_edge} 到 {end_edge} 的路径: {str(e)}")
        pass

    return flag

def write_trips_to_xml(trips, output_file):
    root = ET.Element('routes')
    sorted_trips = sorted(trips, key=lambda person: person['start_time'])
    for idx, trip in tqdm(enumerate(sorted_trips)):
        flow = ET.SubElement(root, 'flow')
        flow.set('id', f'{idx}')
        flow.set('begin', str(trip['start_time']))
        flow.set('end', str(trip['start_time'] + 3600))
        flow.set('number', '1')
        flow.set('from', trip['from'])
        flow.set('to', trip['to'])

    tree = ET.ElementTree(root)
    tree.write(output_file)


def main():
    area = "ttan"
    roadnet_file = f"/data/zhouyuping/LLMNavigation/Data/NYC/Maps/{area}.net.xml"
    area_file = f"/data/zhouyuping/LLMNavigation/Data/NYC/Maps/{area}.shp"
    pop_file = f"/data/zhouyuping/LLMNavigation/Data/NYC/Maps/{area}_population.npy"
    scaling = 0.1

    # Load and convert shapefile
    roadnet = sumolib.net.readNet(roadnet_file)
    areas = gpd.read_file(area_file)
    pops = np.load(pop_file)

    # Parse road network and assign edges to areas
    if not os.path.exists(f"/data/zhouyuping/LLMNavigation/Data/NYC/Maps/{area}_area2edge.json"):
        edges = parse_edges(roadnet)
        assigned_edges = assign_edges_to_areas(edges, areas, roadnet)
        dump_json(assigned_edges, f"/data/zhouyuping/LLMNavigation/Data/NYC/Maps/{area}_area2edge.json")
    assigned_edges = load_json(f"/data/zhouyuping/LLMNavigation/Data/NYC/Maps/{area}_area2edge.json")

    # Generate OD matrix (you can use your own OD matrix generation method here)
    gravity_generator = GravityGenerator(Lambda=0.2, Alpha=0.5, Beta=0.5, Gamma=0.5)
    gravity_generator.load_area(areas)
    od_matrix = np.int64(gravity_generator.generate(pops) * scaling)

    # Generate trips based on OD matrix and departure probability
    departure_time_curve = [1, 1, 1, 1, 2, 3, 4, 5, 6, 5, 4, 0.5, 1, 1, 0.5, 1, 1, 0.5, 1, 1, 0.5, 1, 1, 0.5]
    sum_times = sum(departure_time_curve)
    departure_prob = np.array([d / sum_times for d in departure_time_curve])
    trips = generate_trips(od_matrix, assigned_edges, departure_prob, agent_num=10000)

    # Write trips to .xml file
    output_file = f"/data/zhouyuping/LLMNavigation/Data/NYC/traffic/{area}_od_{scaling}.trips.xml"
    write_trips_to_xml(trips, output_file)


if __name__ == "__main__":
    main()
