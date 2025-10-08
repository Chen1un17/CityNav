#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simulation-based Dynamic Traffic Assignment (SBDTA) with Simulated Annealing for AV rerouting.

Requirements/Assumptions:
- SUMO/TraCI installed and available on PATH, or SUMO_HOME is set.
- sumo configuration references both route files; we only parse the route files to identify which vehicles originate from route1 to select 2% AVs.

Functionality:
- Load SUMO via TraCI using provided sumocfg.
- Parse route1 to identify vehicles/flows/trips belonging to route1; estimate total count for exact 2% target.
- At each vehicle departure, determine whether it comes from route1; select 2% (exact in expectation; forced selection near end) as Autonomous Vehicles (AVs).
- Every step is 1s; every step_size seconds (default 180) run Simulated Annealing rerouting for AVs currently in the network.
- Metrics recorded periodically to CSV: completed AV count (cumulative), mean travel time, mean waiting time, mean time loss (delay), and mean edge occupancy (road utilization proxy).

CLI defaults:
- net: /data/XXXXX/LLMNavigation/Data/NYC/NewYork.net.xml
- route1: /data/XXXXX/LLMNavigation/Data/NYC/NewYork_od_0.1.rou.alt.xml
- route2: /data/XXXXX/LLMNavigation/Data/NYC/NYC_routes_0.1_20250830_111509.alt.xml
- sumocfg: /data/XXXXX/LLMNavigation/Data/NYC/NewYork_sumo_config.sumocfg
- step_size: 180; max_steps: 43200; av_ratio: 0.02
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _ensure_sumo_on_path() -> None:
    """Ensure SUMO TraCI is importable, trying SUMO_HOME if needed."""
    try:
        import traci  # noqa: F401
        return
    except Exception:
        pass
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        tools = os.path.join(sumo_home, "tools")
        if tools not in sys.path:
            sys.path.append(tools)
    try:
        import traci  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "无法导入 TraCI。请确认已安装 SUMO 并设置环境变量 SUMO_HOME，或将 SUMO 的 tools 目录加入 PYTHONPATH。"
        ) from exc


def _check_binary(gui: bool) -> str:
    import traci
    try:
        from sumolib.checkBinary import checkBinary
    except Exception:
        # Fallback: assume 'sumo' / 'sumo-gui' available on PATH
        return "sumo-gui" if gui else "sumo"
    return checkBinary("sumo-gui" if gui else "sumo")


@dataclass
class Route1Signature:
    vehicle_ids: Set[str]
    flow_ids: Set[str]
    trip_ids: Set[str]
    route_ids: Set[str]
    estimated_total_count: int


def parse_route_file_signature(route_path: str) -> Route1Signature:
    """Parse a SUMO route file to collect IDs and estimate total vehicles.

    Estimation rules for <flow>:
    - If attribute 'number' present: use it directly
    - Else if 'vehsPerHour' with [begin,end]: count = vehsPerHour/3600 * (end-begin)
    - Else if 'period' with [begin,end]: count = (end-begin)/period
    - Else if 'probability' with [begin,end]: count = probability * (end-begin)
    - Otherwise unknown -> ignore for estimate (conservative)
    """
    tree = ET.parse(route_path)
    root = tree.getroot()

    vehicle_ids: Set[str] = set()
    flow_ids: Set[str] = set()
    trip_ids: Set[str] = set()
    route_ids: Set[str] = set()

    estimated_total = 0

    def _to_float(val: Optional[str]) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(val)
        except Exception:
            return None

    for elem in root.iter():
        tag = elem.tag
        if tag.endswith("vehicle"):
            vid = elem.attrib.get("id")
            if vid:
                vehicle_ids.add(vid)
                estimated_total += 1
        elif tag.endswith("flow"):
            fid = elem.attrib.get("id")
            if fid:
                flow_ids.add(fid)
            number = elem.attrib.get("number")
            if number is not None:
                try:
                    estimated_total += int(float(number))
                except Exception:
                    pass
            else:
                begin = _to_float(elem.attrib.get("begin"))
                end = _to_float(elem.attrib.get("end"))
                vehs_per_hour = _to_float(elem.attrib.get("vehsPerHour"))
                period = _to_float(elem.attrib.get("period"))
                probability = _to_float(elem.attrib.get("probability"))
                if begin is not None and end is not None and end > begin:
                    duration = max(0.0, end - begin)
                    if vehs_per_hour is not None and vehs_per_hour > 0:
                        estimated_total += int(round(vehs_per_hour / 3600.0 * duration))
                    elif period is not None and period > 0:
                        estimated_total += int(round(duration / period))
                    elif probability is not None and probability >= 0:
                        # probability per second
                        estimated_total += int(round(probability * duration))
        elif tag.endswith("trip"):
            tid = elem.attrib.get("id")
            if tid:
                trip_ids.add(tid)
                estimated_total += 1
        elif tag.endswith("route"):
            rid = elem.attrib.get("id")
            if rid:
                route_ids.add(rid)

    return Route1Signature(
        vehicle_ids=vehicle_ids,
        flow_ids=flow_ids,
        trip_ids=trip_ids,
        route_ids=route_ids,
        estimated_total_count=max(estimated_total, len(vehicle_ids)),
    )


def is_from_route1(veh_id: str, route_id: str, sig: Route1Signature) -> bool:
    """Decide whether a vehicle belongs to route1 based on IDs/signature."""
    if veh_id in sig.vehicle_ids:
        return True
    base = veh_id.split(".")[0]
    if base in sig.flow_ids or base in sig.trip_ids:
        return True
    # routeId match as fallback
    if route_id and route_id in sig.route_ids:
        return True
    return False


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class Metrics:
    completed_av_count: int = 0
    total_travel_time: float = 0.0
    total_wait_time: float = 0.0
    total_time_loss: float = 0.0

    def mean_travel_time(self) -> float:
        return self.total_travel_time / self.completed_av_count if self.completed_av_count > 0 else 0.0

    def mean_wait_time(self) -> float:
        return self.total_wait_time / self.completed_av_count if self.completed_av_count > 0 else 0.0

    def mean_time_loss(self) -> float:
        return self.total_time_loss / self.completed_av_count if self.completed_av_count > 0 else 0.0


def estimate_route_travel_time(edges: Sequence[str]) -> float:
    import traci
    total = 0.0
    for e in edges:
        try:
            total += float(traci.edge.getTraveltime(e))
        except Exception:
            # Unknown edge? skip
            continue
    return total


def simulated_annealing_accept(current_cost: float, candidate_cost: float, temperature: float, rng: random.Random) -> bool:
    if candidate_cost < current_cost:
        return True
    if temperature <= 1e-9:
        return False
    delta = candidate_cost - current_cost
    prob = math.exp(-delta / max(temperature, 1e-9))
    return rng.random() < prob


def run_experiment(
    sumocfg: str,
    route1: str,
    route2: str,
    step_size: int,
    max_steps: int,
    av_ratio: float,
    output_csv: str,
    seed: Optional[int],
    gui: bool,
    initial_temperature: float,
    cooling_rate: float,
    allow_teleport: bool,
) -> None:
    _ensure_sumo_on_path()
    import traci

    rng = random.Random(seed)

    # Parse route1 signature for identification and AV count estimation
    sig1 = parse_route_file_signature(route1)
    if sig1.estimated_total_count <= 0:
        print("[WARN] 路由1无法估计车辆总数，回退为依出发流在线采样 (期望2%)。")
    target_av_count = int(round(av_ratio * sig1.estimated_total_count)) if sig1.estimated_total_count > 0 else None

    # Prepare TraCI
    sumo_binary = _check_binary(gui)
    cmd = [sumo_binary, "-c", sumocfg, "--step-length", "1"]
    # Disable SUMO teleport mechanisms unless explicitly allowed
    if not allow_teleport:
        cmd += [
            "--time-to-teleport", "-1",           # disable jam teleport
            "--max-depart-delay", "-1",           # disable depart teleport
            "--collision.action", "remove",       # avoid teleport on collision
        ]
    # Make runs deterministic if seed provided
    if seed is not None:
        cmd += ["--seed", str(seed)]

    print(f"[INFO] 启动SUMO: {' '.join(cmd)}")
    traci.start(cmd)

    # CSV init
    ensure_dir(output_csv)
    csv_file = open(output_csv, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "sim_time",
        "completed_av_count",
        "avg_travel_time",
        "avg_wait_time",
        "avg_delay",
        "avg_edge_occupancy",
    ])
    csv_file.flush()

    # State
    av_selected: Set[str] = set()
    av_active: Set[str] = set()
    route1_departed_count = 0
    av_selected_count = 0
    metrics = Metrics()
    depart_time_map: Dict[str, float] = {}
    # Per-vehicle metric snapshots while active (since arrived vehicles cannot be queried)
    last_wait_time_map: Dict[str, float] = {}
    last_time_loss_map: Dict[str, float] = {}

    # SA schedule
    temperature = float(initial_temperature)
    sa_epoch = 0

    # Precompute non-internal edges for utilization
    all_edges = [e for e in traci.edge.getIDList() if not e.startswith(":")]
    last_log_time = -1

    try:
        for step in range(int(max_steps)):
            # Advance the simulation by 1s, then read the updated sim time
            traci.simulationStep()
            sim_time = traci.simulation.getTime()

            # Handle departures
            for vid in traci.simulation.getDepartedIDList():
                try:
                    rid = traci.vehicle.getRouteID(vid)
                except Exception:
                    rid = ""
                if is_from_route1(vid, rid, sig1):
                    route1_departed_count += 1
                    # Compute dynamic acceptance probability targeting exact quota if possible
                    is_av = False
                    if av_ratio <= 0:
                        is_av = False
                    elif target_av_count is None:
                        # Online sampling with fixed probability
                        is_av = rng.random() < av_ratio
                    else:
                        remaining_est = max(1, sig1.estimated_total_count - route1_departed_count + 1)
                        needed = max(0, target_av_count - av_selected_count)
                        if needed >= remaining_est:
                            is_av = True
                        else:
                            p = max(0.0, min(1.0, needed / remaining_est))
                            is_av = rng.random() < p
                    if is_av:
                        av_selected.add(vid)
                        av_active.add(vid)
                        av_selected_count += 1
                # Record depart time for AVs only (or for all; we will check later)
                try:
                    depart_time_map[vid] = float(traci.vehicle.getDeparture(vid))
                except Exception:
                    depart_time_map[vid] = sim_time

            # Update metric snapshots for active AVs (every step)
            active_ids_all = traci.vehicle.getIDList()
            for vid in active_ids_all:
                if vid in av_selected:
                    try:
                        last_wait_time_map[vid] = float(traci.vehicle.getAccumulatedWaitingTime(vid))
                    except Exception:
                        pass
                    try:
                        last_time_loss_map[vid] = float(traci.vehicle.getTimeLoss(vid))
                    except Exception:
                        pass

            # Handle arrivals
            for vid in traci.simulation.getArrivedIDList():
                if vid in av_active:
                    av_active.discard(vid)
                if vid in av_selected:
                    # Aggregate metrics at completion
                    travel_time = max(0.0, float(sim_time) - float(depart_time_map.get(vid, sim_time)))
                    # Read last known snapshots captured while the vehicle was active
                    wait_time = float(last_wait_time_map.get(vid, 0.0))
                    time_loss = float(last_time_loss_map.get(vid, max(0.0, travel_time - 1e-9)))

                    metrics.completed_av_count += 1
                    metrics.total_travel_time += travel_time
                    metrics.total_wait_time += wait_time
                    metrics.total_time_loss += time_loss

            # SA rerouting every step_size seconds
            if int(sim_time) % int(step_size) == 0 and int(sim_time) != last_log_time:
                # Update SA temperature
                if sa_epoch > 0:
                    temperature = max(1e-6, float(temperature) * float(cooling_rate))
                sa_epoch += 1

                # Reroute AVs currently in the network
                active_ids = [vid for vid in traci.vehicle.getIDList() if vid in av_selected]
                for vid in active_ids:
                    try:
                        route = traci.vehicle.getRoute(vid)
                        if not route:
                            continue
                        idx = traci.vehicle.getRouteIndex(vid)
                        idx = max(0, int(idx))
                        # Determine fromEdge/destEdge
                        from_edge = route[idx] if idx < len(route) else route[-1]
                        dest_edge = route[-1]

                        # Compute candidate route from current to destination
                        vt = traci.vehicle.getTypeID(vid)
                        candidate_edges: List[str] = []
                        candidate_tt: Optional[float] = None
                        try:
                            res = traci.simulation.findRoute(from_edge, dest_edge, vt)
                            # Handle multiple possible return types across SUMO versions
                            if isinstance(res, (list, tuple)):
                                # SUMO may return (edges, travelTime, cost)
                                if len(res) >= 2 and isinstance(res[0], (list, tuple)):
                                    candidate_edges = list(res[0])
                                    candidate_tt = float(res[1])
                                elif len(res) >= 1 and isinstance(res[0], (list, tuple)):
                                    candidate_edges = list(res[0])
                            elif hasattr(res, "edges"):
                                candidate_edges = list(getattr(res, "edges"))
                                if hasattr(res, "travelTime"):
                                    candidate_tt = float(getattr(res, "travelTime"))
                        except Exception:
                            candidate_edges = []
                            candidate_tt = None
                        if not candidate_edges:
                            continue
                        # Current remaining route segment
                        current_edges = route[idx:]
                        current_tt = estimate_route_travel_time(current_edges)
                        if candidate_tt is None:
                            candidate_tt = estimate_route_travel_time(candidate_edges)

                        # If candidate equals current route, skip
                        if candidate_edges == current_edges:
                            continue

                        # SA acceptance
                        if simulated_annealing_accept(current_tt, candidate_tt, temperature, rng):
                            try:
                                traci.vehicle.setRoute(vid, candidate_edges)
                            except Exception:
                                pass
                    except Exception:
                        continue

                # Log metrics at this boundary
                occ_values: List[float] = []
                for e in all_edges:
                    try:
                        occ_values.append(float(traci.edge.getLastStepOccupancy(e)))
                    except Exception:
                        continue
                avg_occupancy = sum(occ_values) / len(occ_values) if occ_values else 0.0
                csv_writer.writerow([
                    int(sim_time),
                    metrics.completed_av_count,
                    round(metrics.mean_travel_time(), 3),
                    round(metrics.mean_wait_time(), 3),
                    round(metrics.mean_time_loss(), 3),
                    round(avg_occupancy, 6),
                ])
                csv_file.flush()
                last_log_time = int(sim_time)

            # Optional early break if simulation has no vehicles and no pending
            if traci.simulation.getMinExpectedNumber() <= 0 and int(sim_time) >= int(max_steps):
                break

        # Final log at the end if not aligned
        sim_time = traci.simulation.getTime()
        if last_log_time != int(sim_time):
            occ_values: List[float] = []
            for e in all_edges:
                try:
                    occ_values.append(float(traci.edge.getLastStepOccupancy(e)))
                except Exception:
                    continue
            avg_occupancy = sum(occ_values) / len(occ_values) if occ_values else 0.0
            csv_writer.writerow([
                int(sim_time),
                metrics.completed_av_count,
                round(metrics.mean_travel_time(), 3),
                round(metrics.mean_wait_time(), 3),
                round(metrics.mean_time_loss(), 3),
                round(avg_occupancy, 6),
            ])
            csv_file.flush()
    finally:
        try:
            traci.close()
        except Exception:
            pass
        try:
            csv_file.close()
        except Exception:
            pass

    print(
        "[DONE] 结果写入: {} | 完成AV数量: {} | 平均TT: {:.2f}s | 平均等待: {:.2f}s | 平均延误: {:.2f}s".format(
            output_csv, metrics.completed_av_count, metrics.mean_travel_time(), metrics.mean_wait_time(), metrics.mean_time_loss()
        )
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "基于模拟退火(SA)的SBDTA单脚本实验：每180s对2%路由1车辆(AV)进行重路由，统计关键指标。"
        )
    )
    parser.add_argument(
        "--sumo-cfg",
        type=str,
        default="/data/XXXXX/LLMNavigation/Data/NYC/NewYork_sumo_config.sumocfg",
        help="SUMO 配置文件 (.sumocfg)",
    )
    parser.add_argument(
        "--route1",
        type=str,
        default="/data/XXXXX/LLMNavigation/Data/NYC/NewYork_od_0.1.rou.alt.xml",
        help="路由1：仅在此源中抽样2% AV",
    )
    parser.add_argument(
        "--route2",
        type=str,
        default="/data/XXXXX/LLMNavigation/Data/NYC/NYC_routes_0.1_20250830_111509.alt.xml",
        help="路由2：仅作为环境车辆源",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=720,
        help="重路由与指标记录的间隔(秒)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=43200,
        help="仿真步数(每步1秒); 默认12小时=43200步",
    )
    parser.add_argument(
        "--av-ratio",
        type=float,
        default=0.02,
        help="仅对route1来源车辆选取的AV比例",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="outputs/sbdta_sa_metrics.csv",
        help="指标输出CSV路径",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="使用 sumo-gui",
    )
    parser.add_argument(
        "--initial-temperature",
        type=float,
        default=300.0,
        help="模拟退火初始温度",
    )
    parser.add_argument(
        "--cooling-rate",
        type=float,
        default=0.98,
        help="每个SA epoch的降温系数(0<r<1)",
    )
    parser.add_argument(
        "--allow-teleport",
        action="store_true",
        help="允许 SUMO 瞬移 (默认禁用：time-to-teleport=-1, max-depart-delay=-1, collision.action=remove)",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    # Basic validations
    for p in [args.sumo_cfg, args.route1, args.route2]:
        if not os.path.exists(p):
            print(f"[WARN] 文件不存在: {p}")

    run_experiment(
        sumocfg=args.sumo_cfg,
        route1=args.route1,
        route2=args.route2,
        step_size=args.step_size,
        max_steps=args.max_steps,
        av_ratio=args.av_ratio,
        output_csv=args.output_csv,
        seed=args.seed,
        gui=bool(args.gui),
        initial_temperature=args.initial_temperature,
        cooling_rate=args.cooling_rate,
        allow_teleport=bool(args.allow_teleport),
    )


if __name__ == "__main__":
    main()


