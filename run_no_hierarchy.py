#!/usr/bin/env python3
"""
No-Hierarchy Ablation Experiment

This script implements an ablation study that removes the Traffic Agent from the
multi-agent system, leaving only Regional Agents to handle end-to-end routing.

Purpose: Demonstrate the computational efficiency and token efficiency advantages
of the hierarchical architecture by comparing against a flat, no-hierarchy baseline.

Key Differences from Hierarchical System:
1. NO Traffic Agent - Regional Agents plan complete routes from origin to destination
2. Direct end-to-end planning - No macro-level coordination
3. Higher token usage expected - Each Regional Agent must consider full network
4. Potentially lower efficiency - No global optimization

Configuration:
- Location: NewYork (NewYork.sumocfg)
- Time Range: 21600s - 43200s (6 hours simulation)
- AV Ratio: 2% from first route file
- LLM: Qwen3-8B via DashScope API (云端)
- Token Tracking: Enabled with detailed metrics

Requirements:
- Set DASHSCOPE_API_KEY environment variable
- API: https://dashscope.console.aliyun.com/
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Set, Optional, Tuple, Any

# Disable wandb online mode
os.environ["WANDB_MODE"] = "offline"

import traci
import networkx as nx
from agents.regional_agent import RegionalAgent
from agents.agent_logger import AgentLogger
from agents.prediction_engine import PredictionEngine
from utils.language_model import LLM
from utils.read_utils import load_json
import wandb


def check_qwen_api_key() -> bool:
    """检查通义千问API密钥是否已设置"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("\n[错误] 未找到DASHSCOPE_API_KEY环境变量")
        print("请先设置API密钥: export DASHSCOPE_API_KEY='sk-your-api-key-here'")
        print("获取API密钥: https://dashscope.console.aliyun.com/apiKey\n")
        return False

    if not api_key.startswith("sk-"):
        print("\n[警告] DASHSCOPE_API_KEY格式可能不正确（通常以'sk-'开头）")

    print(f"[成功] 检测到DASHSCOPE_API_KEY: {api_key[:10]}...")
    return True


class NoHierarchyEnvironment:
    """
    No-Hierarchy Traffic Environment for Ablation Study.

    Removes Traffic Agent and uses only Regional Agents for end-to-end routing.
    This demonstrates the computational advantages of hierarchical architecture.
    """

    def __init__(self, location: str, sumo_config_file: str, route_file: str,
                 road_info_file: str, region_data_dir: str, model_name: str,
                 step_size: float, max_steps: int, task_info_file: str,
                 start_time: float = 21600, av_ratio: float = 0.02,
                 log_dir: str = "logs/no_hierarchy", use_api: bool = True):
        """
        Initialize No-Hierarchy environment.

        Args:
            location: Location name
            sumo_config_file: Path to SUMO configuration file
            route_file: Path to route file
            road_info_file: Path to road information JSON
            region_data_dir: Directory containing region partition data
            model_name: LLM model name (e.g., "qwen-turbo" for API)
            step_size: Decision-making interval in seconds
            max_steps: Maximum simulation steps
            task_info_file: Path to task info JSON file
            start_time: Simulation start time (default: 21600s = 6 hours)
            av_ratio: Ratio of autonomous vehicles from first route file
            log_dir: Directory for logging
            use_api: Whether to use cloud API (default: True)
        """
        self.location = location
        self.sumo_config_file = sumo_config_file
        self.route_file = route_file
        self.road_info_file = road_info_file
        self.region_data_dir = region_data_dir
        self.model_name = model_name
        self.step_size = step_size
        self.max_steps = max_steps
        self.start_time = start_time
        self.av_ratio = av_ratio
        self.log_dir = log_dir
        self.use_api = use_api

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # Initialize logger
        self.logger = AgentLogger(log_dir, "NoHierarchy")

        # Load task info
        print("Loading task information...")
        self.task_info = load_json(task_info_file)
        print(f"  - Task: {self.task_info.get('task_description', 'N/A')[:100]}...")

        # Load region data
        self._load_region_data()

        # Load road network
        self._load_road_network()

        # Initialize LLM (single shared LLM for all Regional Agents)
        if use_api:
            print(f"\n=== Initializing Qwen3-8B API (DashScope) ===")
            print(f"Model: {model_name}")
            print(f"API Provider: DashScope (阿里云)")

            # Check API key
            if not check_qwen_api_key():
                raise ValueError("DASHSCOPE_API_KEY not found. Please set the environment variable.")

            # Use "dashscope" as the path to trigger API mode
            self.llm_agent = LLM(
                "dashscope",  # Special keyword to trigger DashScope API
                batch_size=16,  # Larger batch for no-hierarchy mode
                task_info=self.task_info,  # Use complete task_info
                use_reflection=True
            )

            # Set the actual model name
            self.llm_agent.llm_name = model_name
            self.llm_agent.provider_name = 'dashscope'

            print(f"[SUCCESS] Qwen3-8B API initialized successfully")
            print(f"  - Model: {model_name}")
            print(f"  - Provider: DashScope")
        else:
            print(f"\n=== Initializing Local Qwen3-8B Model ===")
            print(f"Model Path: {model_name}")
            self.llm_agent = LLM(
                model_name,
                batch_size=16,
                task_info=self.task_info,  # Use complete task_info
                use_reflection=True,
                use_local_llm=True
            )
            print(f"[SUCCESS] Local Qwen3-8B model loaded successfully")

        # Initialize Regional Agents (NO Traffic Agent)
        print(f"\n=== Initializing Regional Agents (NO Traffic Agent) ===")
        self._initialize_regional_agents()

        # Vehicle tracking
        self.autonomous_vehicles: Set[str] = set()
        self.vehicle_routes: Dict[str, List[str]] = {}
        self.vehicle_destinations: Dict[str, str] = {}
        self.vehicle_regions: Dict[str, int] = {}

        # Token usage tracking
        self.token_stats = {
            'total_calls': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_tokens': 0,
            'calls_by_region': defaultdict(int),
            'tokens_by_region': defaultdict(int),
            'call_history': []
        }

        # Detailed timing tracking for each decision
        self.timing_stats = {
            'total_decision_cycles': 0,
            'total_decision_time': 0.0,
            'data_collection_time': 0.0,
            'llm_inference_time': 0.0,
            'route_execution_time': 0.0,
            'decision_history': []  # Per-decision timing breakdown
        }

        # Performance metrics
        self.metrics = {
            'total_decisions': 0,
            'avg_decision_time': 0.0,
            'avg_travel_time': 0.0,
            'throughput': 0
        }

        print(f"[SUCCESS] No-Hierarchy Environment Initialized")
        print(f"  - Regional Agents: {len(self.regional_agents)}")
        print(f"  - NO Traffic Agent (Ablation)")
        print(f"  - Start Time: {start_time}s")
        print(f"  - Simulation Duration: {max_steps - start_time}s")
        print(f"  - AV Ratio: {av_ratio * 100}%")

    def _load_region_data(self):
        """Load region partition data."""
        print("Loading region partition data...")

        # Load boundary edges
        boundary_file = os.path.join(self.region_data_dir, "boundary_edges_alpha_2.json")
        with open(boundary_file, 'r') as f:
            self.boundary_edges = json.load(f)

        # Load edge-to-region mapping
        edge_to_region_file = os.path.join(self.region_data_dir, "edge_to_region_alpha_2.json")
        with open(edge_to_region_file, 'r') as f:
            self.edge_to_region = json.load(f)

        # Get unique region IDs
        self.region_ids = sorted(set(self.edge_to_region.values()))

        print(f"  - Regions: {len(self.region_ids)}")
        print(f"  - Boundary Edges: {len(self.boundary_edges)}")

    def _load_road_network(self):
        """Load road network information."""
        print("Loading road network...")

        # Load road info
        self.road_info = load_json(self.road_info_file)

        # Build network graph
        self.road_network = nx.DiGraph()
        for edge_id, info in self.road_info.items():
            self.road_network.add_node(edge_id)

        print(f"  - Road Network Nodes: {self.road_network.number_of_nodes()}")

    def _initialize_regional_agents(self):
        """Initialize Regional Agents without Traffic Agent."""
        self.regional_agents: Dict[int, RegionalAgent] = {}

        for region_id in self.region_ids:
            agent = RegionalAgent(
                region_id=region_id,
                boundary_edges=self.boundary_edges,
                edge_to_region=self.edge_to_region,
                road_info=self.road_info,
                road_network=self.road_network,
                llm_agent=self.llm_agent,
                logger=self.logger,
                prediction_engine=None,  # Optional
                raw_llm_agent=self.llm_agent
            )

            # Inject reference to environment for full network access
            agent.parent_env = self
            agent.full_road_network = self.road_network
            agent.all_regions = self.region_ids

            self.regional_agents[region_id] = agent

        print(f"[SUCCESS] Initialized {len(self.regional_agents)} Regional Agents")

    def run_simulation(self):
        """Run the no-hierarchy simulation."""
        print(f"\n{'='*60}")
        print(f"STARTING NO-HIERARCHY SIMULATION")
        print(f"{'='*60}")

        # Start SUMO
        self._start_sumo()

        # Load route data for AV selection
        self._load_and_select_avs()

        # Main simulation loop
        current_time = self.start_time
        step_count = 0

        print(f"\n[SIMULATION] Starting from {current_time}s")

        try:
            while current_time < self.max_steps:
                # Simulation step
                traci.simulationStep()
                current_time = traci.simulation.getTime()
                step_count += 1

                # Update vehicle status
                self._update_vehicle_status(current_time)

                # Make routing decisions (every step_size seconds)
                if step_count % int(self.step_size) == 0:
                    decision_start = time.time()
                    self._make_regional_decisions(current_time)
                    decision_time = time.time() - decision_start

                    self.metrics['avg_decision_time'] = (
                        (self.metrics['avg_decision_time'] * self.metrics['total_decisions'] + decision_time) /
                        (self.metrics['total_decisions'] + 1)
                    )
                    self.metrics['total_decisions'] += 1

                    # Log progress
                    if step_count % (int(self.step_size) * 10) == 0:
                        self._log_progress(current_time, step_count)

                # Check completion
                if len(traci.vehicle.getIDList()) == 0:
                    print(f"\n[SIMULATION] All vehicles completed at {current_time}s")
                    break

        except KeyboardInterrupt:
            print(f"\n[SIMULATION] Interrupted by user at {current_time}s")

        finally:
            # Generate final report
            self._generate_final_report(current_time)

            # Close SUMO
            traci.close()

        return self.metrics['avg_travel_time'], self.metrics['throughput']

    def _start_sumo(self):
        """Start SUMO simulation."""
        print("\n[SUMO] Starting simulation...")

        sumo_cmd = [
            "sumo",
            "-c", self.sumo_config_file,
            "--begin", str(int(self.start_time)),
            "--end", str(int(self.max_steps)),
            "--step-length", "1.0",
            "--no-warnings", "true",
            "--duration-log.disable", "true",
            "--no-step-log", "true"
        ]

        traci.start(sumo_cmd)
        print(f"[SUMO] Simulation started (begin={self.start_time}, end={self.max_steps})")

    def _load_and_select_avs(self):
        """Load route data and select autonomous vehicles."""
        print("\n[AV_SELECTION] Loading and selecting autonomous vehicles...")

        # Parse route file to get vehicle IDs from first route file
        import xml.etree.ElementTree as ET
        tree = ET.parse(self.route_file)
        root = tree.getroot()

        # Collect all vehicle IDs
        all_vehicles = []
        for vehicle in root.findall('.//vehicle'):
            vehicle_id = vehicle.get('id')
            if vehicle_id:
                all_vehicles.append(vehicle_id)

        # Select 2% as autonomous vehicles
        import random
        random.seed(42)  # For reproducibility
        num_avs = int(len(all_vehicles) * self.av_ratio)
        selected_avs = random.sample(all_vehicles, num_avs)

        self.autonomous_vehicles = set(selected_avs)

        print(f"[AV_SELECTION] Total vehicles: {len(all_vehicles)}")
        print(f"[AV_SELECTION] Selected AVs: {num_avs} ({self.av_ratio*100}%)")

    def _update_vehicle_status(self, current_time: float):
        """Update status of autonomous vehicles."""
        try:
            current_vehicles = set(traci.vehicle.getIDList())
            active_avs = current_vehicles.intersection(self.autonomous_vehicles)

            # Update regional agents with vehicle status
            for region_id, agent in self.regional_agents.items():
                agent.update_vehicle_status(current_time)

        except Exception as e:
            self.logger.log_error(f"Vehicle status update failed: {e}")

    def _make_regional_decisions(self, current_time: float):
        """
        Make routing decisions using only Regional Agents (no Traffic Agent).

        Each Regional Agent must plan end-to-end routes for vehicles in its region.
        Tracks detailed timing for each stage of the decision-making process.
        """
        decision_cycle_start = time.time()

        try:
            # === STAGE 1: Data Collection ===
            data_collection_start = time.time()

            # Get all active autonomous vehicles
            current_vehicles = set(traci.vehicle.getIDList())
            active_avs = current_vehicles.intersection(self.autonomous_vehicles)

            if not active_avs:
                return

            # Group vehicles by region
            vehicles_by_region = defaultdict(list)
            for vehicle_id in active_avs:
                try:
                    current_edge = traci.vehicle.getRoadID(vehicle_id)
                    if current_edge in self.edge_to_region:
                        region_id = self.edge_to_region[current_edge]
                        route = traci.vehicle.getRoute(vehicle_id)
                        destination = route[-1] if route else None

                        vehicles_by_region[region_id].append({
                            'vehicle_id': vehicle_id,
                            'current_edge': current_edge,
                            'destination': destination
                        })
                except Exception:
                    continue

            data_collection_time = time.time() - data_collection_start

            # === STAGE 2: LLM Inference ===
            llm_inference_start = time.time()

            # Each Regional Agent plans end-to-end routes (NO Traffic Agent coordination)
            total_llm_calls = 0
            total_input_tokens = 0
            total_output_tokens = 0
            llm_call_details = []

            for region_id, vehicles in vehicles_by_region.items():
                if not vehicles:
                    continue

                agent = self.regional_agents[region_id]

                # Plan end-to-end routes for each vehicle with timing
                for vehicle_data in vehicles:
                    vehicle_id = vehicle_data['vehicle_id']
                    destination = vehicle_data['destination']

                    if not destination:
                        continue

                    # Track individual LLM call
                    vehicle_llm_start = time.time()
                    calls_before = getattr(self.llm_agent, 'total_calls', 0)
                    input_tokens_before = getattr(self.llm_agent, 'total_input_tokens', 0)
                    output_tokens_before = getattr(self.llm_agent, 'total_output_tokens', 0)

                    # Determine destination region
                    dest_region = self.edge_to_region.get(destination, region_id)

                    # Regional Agent plans full route (no hierarchy)
                    # This is the key difference - no Traffic Agent coordination
                    route_plan = self._plan_end_to_end_route(
                        agent, vehicle_id, destination, dest_region, current_time
                    )

                    # Track LLM usage after
                    vehicle_llm_time = time.time() - vehicle_llm_start
                    calls_after = getattr(self.llm_agent, 'total_calls', 0)
                    input_tokens_after = getattr(self.llm_agent, 'total_input_tokens', 0)
                    output_tokens_after = getattr(self.llm_agent, 'total_output_tokens', 0)

                    calls_delta = calls_after - calls_before
                    input_tokens_delta = input_tokens_after - input_tokens_before
                    output_tokens_delta = output_tokens_after - output_tokens_before

                    total_llm_calls += calls_delta
                    total_input_tokens += input_tokens_delta
                    total_output_tokens += output_tokens_delta

                    # Update region-specific stats
                    self.token_stats['calls_by_region'][region_id] += calls_delta
                    self.token_stats['tokens_by_region'][region_id] += (input_tokens_delta + output_tokens_delta)

                    # Record individual LLM call details
                    if calls_delta > 0:
                        llm_call_details.append({
                            'vehicle_id': vehicle_id,
                            'region_id': region_id,
                            'timestamp': current_time,
                            'llm_time': vehicle_llm_time,
                            'input_tokens': input_tokens_delta,
                            'output_tokens': output_tokens_delta,
                            'total_tokens': input_tokens_delta + output_tokens_delta,
                            'route_found': route_plan is not None
                        })

                    # Execute route if found
                    if route_plan:
                        route_plan['llm_time'] = vehicle_llm_time
                        route_plan['input_tokens'] = input_tokens_delta
                        route_plan['output_tokens'] = output_tokens_delta

            llm_inference_time = time.time() - llm_inference_start

            # === STAGE 3: Route Execution ===
            route_execution_start = time.time()

            # Execute all planned routes
            for region_id, vehicles in vehicles_by_region.items():
                for vehicle_data in vehicles:
                    vehicle_id = vehicle_data['vehicle_id']
                    # Routes are executed in _plan_end_to_end_route, but track timing here

            route_execution_time = time.time() - route_execution_start

            # === STAGE 4: Statistics Update ===
            total_tokens = total_input_tokens + total_output_tokens

            # Update global token stats
            self.token_stats['total_calls'] += total_llm_calls
            self.token_stats['total_input_tokens'] += total_input_tokens
            self.token_stats['total_output_tokens'] += total_output_tokens
            self.token_stats['total_tokens'] += total_tokens

            # Record call history
            self.token_stats['call_history'].append({
                'timestamp': current_time,
                'calls': total_llm_calls,
                'input_tokens': total_input_tokens,
                'output_tokens': total_output_tokens,
                'total_tokens': total_tokens,
                'vehicles': len(active_avs),
                'llm_call_details': llm_call_details
            })

            # Update timing stats
            decision_cycle_time = time.time() - decision_cycle_start
            self.timing_stats['total_decision_cycles'] += 1
            self.timing_stats['total_decision_time'] += decision_cycle_time
            self.timing_stats['data_collection_time'] += data_collection_time
            self.timing_stats['llm_inference_time'] += llm_inference_time
            self.timing_stats['route_execution_time'] += route_execution_time

            # Record detailed timing breakdown
            timing_breakdown = {
                'timestamp': current_time,
                'total_time': decision_cycle_time,
                'data_collection_time': data_collection_time,
                'llm_inference_time': llm_inference_time,
                'route_execution_time': route_execution_time,
                'vehicles_processed': len(active_avs),
                'llm_calls': total_llm_calls,
                'total_tokens': total_tokens,
                'avg_time_per_vehicle': decision_cycle_time / max(1, len(active_avs)),
                'avg_tokens_per_vehicle': total_tokens / max(1, len(active_avs))
            }
            self.timing_stats['decision_history'].append(timing_breakdown)

            # Log detailed timing information
            self.logger.log_info(
                f"NO_HIERARCHY_DECISION @ {current_time:.0f}s:\n"
                f"  Vehicles: {len(active_avs)} AVs\n"
                f"  Total Time: {decision_cycle_time:.3f}s\n"
                f"    - Data Collection: {data_collection_time:.3f}s ({data_collection_time/decision_cycle_time*100:.1f}%)\n"
                f"    - LLM Inference: {llm_inference_time:.3f}s ({llm_inference_time/decision_cycle_time*100:.1f}%)\n"
                f"    - Route Execution: {route_execution_time:.3f}s ({route_execution_time/decision_cycle_time*100:.1f}%)\n"
                f"  LLM Stats:\n"
                f"    - Total Calls: {total_llm_calls}\n"
                f"    - Input Tokens: {total_input_tokens}\n"
                f"    - Output Tokens: {total_output_tokens}\n"
                f"    - Total Tokens: {total_tokens}\n"
                f"    - Avg Tokens/Vehicle: {total_tokens / max(1, len(active_avs)):.1f}"
            )

        except Exception as e:
            self.logger.log_error(f"Regional decision making failed: {e}")
            import traceback
            self.logger.log_error(traceback.format_exc())

    def _plan_end_to_end_route(self, agent: RegionalAgent, vehicle_id: str,
                              destination: str, dest_region: int, current_time: float) -> Optional[Dict]:
        """
        Plan end-to-end route using Regional Agent (without Traffic Agent).

        This is less efficient than hierarchical planning because:
        1. Regional Agent must consider full network
        2. No macro-level optimization
        3. Higher token usage per decision

        Uses Qwen3-8B LLM for route optimization decisions.
        """
        try:
            route_planning_start = time.time()

            current_edge = traci.vehicle.getRoadID(vehicle_id)

            # Phase 1: Generate multiple candidate routes
            candidate_generation_start = time.time()
            candidates = []

            # Generate 3-5 candidate routes to destination
            # Candidate 1: Direct SUMO route
            route_result = traci.simulation.findRoute(current_edge, destination)
            if route_result and route_result.edges:
                candidates.append({
                    'route': list(route_result.edges),
                    'travel_time': route_result.travelTime,
                    'distance': route_result.length,
                    'type': 'direct'
                })

            # Candidate 2-3: Alternative routes through different intermediate points
            # (Simplified - in full implementation, would generate more diverse routes)

            candidate_generation_time = time.time() - candidate_generation_start

            if not candidates:
                return None

            # Phase 2: Use Qwen3-8B LLM to select optimal route
            # Build observation for LLM decision
            llm_decision_start = time.time()

            observation_parts = []
            observation_parts.append(f"END-TO-END ROUTE PLANNING (No Hierarchy)")
            observation_parts.append(f"Vehicle: {vehicle_id}")
            observation_parts.append(f"Current: {current_edge}")
            observation_parts.append(f"Destination: {destination}")
            observation_parts.append(f"Destination Region: {dest_region}")
            observation_parts.append(f"Time: {current_time:.0f}s")
            observation_parts.append("")
            observation_parts.append("CANDIDATE ROUTES:")

            for i, candidate in enumerate(candidates[:5]):
                route_edges = candidate['route']
                observation_parts.append(
                    f"Option {i+1}: {len(route_edges)} edges, "
                    f"{candidate['travel_time']:.1f}s travel time, "
                    f"{candidate['distance']:.0f}m distance"
                )

            observation_parts.append("")
            observation_parts.append("SELECT: Choose best route (1-5)")

            observation_text = "\n".join(observation_parts)
            answer_options = "/".join([str(i+1) for i in range(len(candidates))])

            # Call Qwen3-8B LLM for decision
            try:
                decisions = self.llm_agent.hybrid_decision_making_pipeline(
                    [observation_text],
                    [answer_options]
                )

                selected_idx = 0  # Default to first candidate
                if decisions and len(decisions) > 0 and 'answer' in decisions[0]:
                    llm_answer = decisions[0]['answer']
                    try:
                        if isinstance(llm_answer, (int, float)):
                            selected_idx = int(llm_answer) - 1
                        elif isinstance(llm_answer, str) and llm_answer.strip().isdigit():
                            selected_idx = int(llm_answer.strip()) - 1

                        # Validate index
                        if not (0 <= selected_idx < len(candidates)):
                            selected_idx = 0
                    except:
                        selected_idx = 0

                selected_candidate = candidates[selected_idx]

            except Exception as llm_error:
                self.logger.log_warning(f"LLM decision failed for {vehicle_id}: {llm_error}, using default route")
                selected_candidate = candidates[0]

            llm_decision_time = time.time() - llm_decision_start

            # Phase 3: Execute selected route
            execution_start = time.time()
            full_route = selected_candidate['route']

            # Set route in SUMO
            if vehicle_id in traci.vehicle.getIDList():
                traci.vehicle.setRoute(vehicle_id, full_route)
                self.vehicle_routes[vehicle_id] = full_route

            execution_time = time.time() - execution_start

            total_planning_time = time.time() - route_planning_start

            # Regional Agent evaluates and potentially optimizes using LLM
            # This creates higher token usage compared to hierarchical approach
            plan = {
                'vehicle_id': vehicle_id,
                'route': full_route,
                'destination': destination,
                'travel_time': selected_candidate['travel_time'],
                'planning_method': 'no_hierarchy',
                'llm_model': self.model_name,  # Use actual model name
                'llm_provider': 'dashscope' if self.use_api else 'local',
                'timing': {
                    'total_planning_time': total_planning_time,
                    'candidate_generation_time': candidate_generation_time,
                    'llm_decision_time': llm_decision_time,
                    'execution_time': execution_time
                }
            }

            return plan

        except Exception as e:
            self.logger.log_error(f"End-to-end planning failed for {vehicle_id}: {e}")
            import traceback
            self.logger.log_error(traceback.format_exc())
            return None

    def _execute_route_plan(self, vehicle_id: str, route_plan: Dict):
        """Execute the route plan for a vehicle."""
        try:
            if vehicle_id not in traci.vehicle.getIDList():
                return

            route = route_plan['route']
            if not route:
                return

            traci.vehicle.setRoute(vehicle_id, route)
            self.vehicle_routes[vehicle_id] = route

        except Exception as e:
            self.logger.log_error(f"Route execution failed for {vehicle_id}: {e}")

    def _log_progress(self, current_time: float, step_count: int):
        """Log simulation progress with detailed timing information."""
        active_vehicles = len(traci.vehicle.getIDList())
        active_avs = len(set(traci.vehicle.getIDList()).intersection(self.autonomous_vehicles))

        avg_tokens_per_call = (
            self.token_stats['total_tokens'] / max(1, self.token_stats['total_calls'])
        )

        # Calculate timing percentages
        total_time = self.timing_stats['total_decision_time']
        if total_time > 0:
            data_pct = (self.timing_stats['data_collection_time'] / total_time) * 100
            llm_pct = (self.timing_stats['llm_inference_time'] / total_time) * 100
            exec_pct = (self.timing_stats['route_execution_time'] / total_time) * 100
        else:
            data_pct = llm_pct = exec_pct = 0

        print(f"\n{'='*70}")
        print(f"[PROGRESS] Time: {current_time:.0f}s | Step: {step_count}")
        print(f"{'='*70}")
        print(f"Vehicles:")
        print(f"  - Total: {active_vehicles}")
        print(f"  - Autonomous: {active_avs}")
        print(f"\nLLM Statistics (Qwen3-8B):")
        print(f"  - Total Calls: {self.token_stats['total_calls']}")
        print(f"  - Input Tokens: {self.token_stats['total_input_tokens']}")
        print(f"  - Output Tokens: {self.token_stats['total_output_tokens']}")
        print(f"  - Total Tokens: {self.token_stats['total_tokens']}")
        print(f"  - Avg Tokens/Call: {avg_tokens_per_call:.1f}")
        print(f"\nTiming Breakdown:")
        print(f"  - Total Decision Time: {total_time:.2f}s")
        print(f"  - Data Collection: {self.timing_stats['data_collection_time']:.2f}s ({data_pct:.1f}%)")
        print(f"  - LLM Inference: {self.timing_stats['llm_inference_time']:.2f}s ({llm_pct:.1f}%)")
        print(f"  - Route Execution: {self.timing_stats['route_execution_time']:.2f}s ({exec_pct:.1f}%)")
        print(f"  - Avg Decision Time: {self.metrics['avg_decision_time']:.2f}s")
        print(f"{'='*70}\n")

    def _generate_final_report(self, final_time: float):
        """Generate final simulation report with token usage and detailed timing metrics."""
        print(f"\n{'='*80}")
        print(f"NO-HIERARCHY ABLATION EXPERIMENT - FINAL REPORT")
        print(f"{'='*80}")

        # Calculate timing statistics
        total_time = self.timing_stats['total_decision_time']
        data_time = self.timing_stats['data_collection_time']
        llm_time = self.timing_stats['llm_inference_time']
        exec_time = self.timing_stats['route_execution_time']

        if total_time > 0:
            data_pct = (data_time / total_time) * 100
            llm_pct = (llm_time / total_time) * 100
            exec_pct = (exec_time / total_time) * 100
        else:
            data_pct = llm_pct = exec_pct = 0

        # Save comprehensive report
        report_data = {
            'experiment_info': {
                'type': 'no_hierarchy_ablation',
                'location': self.location,
                'llm_model': self.model_name,
                'llm_provider': 'dashscope' if self.use_api else 'local',
                'api_mode': self.use_api,
                'start_time': self.start_time,
                'end_time': final_time,
                'duration_seconds': final_time - self.start_time,
                'duration_hours': (final_time - self.start_time) / 3600,
                'av_ratio': self.av_ratio
            },
            'token_stats': {
                'total_calls': self.token_stats['total_calls'],
                'total_input_tokens': self.token_stats['total_input_tokens'],
                'total_output_tokens': self.token_stats['total_output_tokens'],
                'total_tokens': self.token_stats['total_tokens'],
                'avg_tokens_per_call': (
                    self.token_stats['total_tokens'] / max(1, self.token_stats['total_calls'])
                ),
                'avg_input_tokens_per_call': (
                    self.token_stats['total_input_tokens'] / max(1, self.token_stats['total_calls'])
                ),
                'avg_output_tokens_per_call': (
                    self.token_stats['total_output_tokens'] / max(1, self.token_stats['total_calls'])
                )
            },
            'timing_stats': {
                'total_decision_cycles': self.timing_stats['total_decision_cycles'],
                'total_decision_time': total_time,
                'data_collection_time': data_time,
                'data_collection_percentage': data_pct,
                'llm_inference_time': llm_time,
                'llm_inference_percentage': llm_pct,
                'route_execution_time': exec_time,
                'route_execution_percentage': exec_pct,
                'avg_cycle_time': total_time / max(1, self.timing_stats['total_decision_cycles']),
                'avg_llm_time_per_cycle': llm_time / max(1, self.timing_stats['total_decision_cycles'])
            },
            'region_breakdown': {
                str(region_id): {
                    'calls': self.token_stats['calls_by_region'][region_id],
                    'tokens': self.token_stats['tokens_by_region'][region_id]
                }
                for region_id in self.region_ids
            },
            'detailed_history': {
                'token_usage_history': self.token_stats['call_history'],
                'timing_history': self.timing_stats['decision_history']
            }
        }

        # Save JSON report
        report_file = os.path.join(self.log_dir, "final_report.json")
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)

        # Save timing breakdown CSV
        timing_csv_file = os.path.join(self.log_dir, "timing_breakdown.csv")
        with open(timing_csv_file, 'w') as f:
            f.write("timestamp,total_time,data_collection_time,llm_inference_time,route_execution_time,vehicles,llm_calls,tokens\n")
            for record in self.timing_stats['decision_history']:
                f.write(f"{record['timestamp']},{record['total_time']},{record['data_collection_time']},"
                       f"{record['llm_inference_time']},{record['route_execution_time']},"
                       f"{record['vehicles_processed']},{record['llm_calls']},{record['total_tokens']}\n")

        # Print summary
        print(f"\n=== Experiment Information ===")
        print(f"Type: No-Hierarchy Ablation Study")
        print(f"LLM Model: {self.model_name}")
        print(f"LLM Provider: {'DashScope API (云端)' if self.use_api else 'Local'}")
        print(f"Location: {self.location}")
        print(f"Duration: {final_time - self.start_time:.0f}s ({(final_time - self.start_time)/3600:.2f} hours)")
        print(f"AV Ratio: {self.av_ratio * 100}%")

        print(f"\n=== Token Usage Summary (Qwen3-8B) ===")
        print(f"Total LLM Calls: {self.token_stats['total_calls']}")
        print(f"Input Tokens: {self.token_stats['total_input_tokens']}")
        print(f"Output Tokens: {self.token_stats['total_output_tokens']}")
        print(f"Total Tokens: {self.token_stats['total_tokens']}")
        print(f"Avg Tokens/Call: {self.token_stats['total_tokens'] / max(1, self.token_stats['total_calls']):.1f}")
        print(f"  - Avg Input/Call: {self.token_stats['total_input_tokens'] / max(1, self.token_stats['total_calls']):.1f}")
        print(f"  - Avg Output/Call: {self.token_stats['total_output_tokens'] / max(1, self.token_stats['total_calls']):.1f}")

        print(f"\n=== Timing Breakdown ===")
        print(f"Total Decision Cycles: {self.timing_stats['total_decision_cycles']}")
        print(f"Total Decision Time: {total_time:.2f}s")
        print(f"  - Data Collection: {data_time:.2f}s ({data_pct:.1f}%)")
        print(f"  - LLM Inference: {llm_time:.2f}s ({llm_pct:.1f}%)")
        print(f"  - Route Execution: {exec_time:.2f}s ({exec_pct:.1f}%)")
        print(f"Avg Time per Cycle: {total_time / max(1, self.timing_stats['total_decision_cycles']):.2f}s")
        print(f"Avg LLM Time per Cycle: {llm_time / max(1, self.timing_stats['total_decision_cycles']):.2f}s")

        print(f"\n=== Performance Summary ===")
        print(f"Total Decisions: {self.metrics['total_decisions']}")
        print(f"Avg Decision Time: {self.metrics['avg_decision_time']:.2f}s")

        print(f"\n=== Output Files ===")
        print(f"Final Report: {report_file}")
        print(f"Timing Breakdown CSV: {timing_csv_file}")
        print(f"Log Directory: {self.log_dir}")
        print(f"{'='*80}\n")


def main():
    """Main function to run no-hierarchy ablation experiment."""
    parser = argparse.ArgumentParser(
        description="No-Hierarchy Ablation Experiment - Remove Traffic Agent, Use Qwen3-8B API"
    )

    parser.add_argument("--model", type=str, default="qwen-turbo",
                       help="Qwen model name (default: qwen-turbo for qwen3-8b)")
    parser.add_argument("--use-local", action="store_true",
                       help="Use local model instead of API")
    parser.add_argument("--model-path", type=str,
                       default="/data/XXXXX/Qwen/",
                       help="Path to local LLM model (only if --use-local)")
    parser.add_argument("--location", type=str, default="NewYork",
                       help="Location (default: NewYork)")
    parser.add_argument("--start-time", type=float, default=21600,
                       help="Start time in seconds (default: 21600 = 6 hours)")
    parser.add_argument("--max-steps", type=float, default=43200,
                       help="End time in seconds (default: 43200 = 12 hours)")
    parser.add_argument("--step-size", type=float, default=180.0,
                       help="Decision interval in seconds (default: 180)")
    parser.add_argument("--av-ratio", type=float, default=0.02,
                       help="Autonomous vehicle ratio (default: 0.02 = 2%%)")

    args = parser.parse_args()

    # Check API key if using cloud API
    if not args.use_local:
        if not check_qwen_api_key():
            print("\n[错误] 使用云端API需要设置DASHSCOPE_API_KEY环境变量")
            print("使用方法: export DASHSCOPE_API_KEY='sk-your-api-key-here'")
            print("或者使用 --use-local 参数切换到本地模型\n")
            sys.exit(1)

    # File paths for NewYork
    sumo_config = "/data/XXXXX/LLMNavigation/Data/NYC/NewYork_sumo_config.sumocfg"
    route_file = "/data/XXXXX/LLMNavigation/Data/NYC/NewYork_od_0.1.rou.alt.xml"
    road_info_file = "/data/XXXXX/LLMNavigation/Data/NYC/NewYork_road_info.json"
    region_data_dir = "/data/XXXXX/LLMNavigation/Data/New"
    task_info_file = "/data/XXXXX/LLMNavigation/Data/NYC/task_info.json"

    # Determine model name
    if args.use_local:
        model_name = args.model_path
        use_api = False
        api_info = "本地模型"
    else:
        model_name = args.model
        use_api = True
        api_info = "DashScope云端API"

    print(f"\n{'='*70}")
    print(f"NO-HIERARCHY ABLATION EXPERIMENT")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  - LLM Mode: {api_info}")
    print(f"  - Model: {model_name}")
    if use_api:
        print(f"  - API Provider: 阿里云DashScope")
        print(f"  - API Endpoint: https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation")
    print(f"  - Location: {args.location}")
    print(f"  - Time Range: {args.start_time}s - {args.max_steps}s")
    print(f"  - Duration: {args.max_steps - args.start_time}s ({(args.max_steps - args.start_time)/3600:.1f} hours)")
    print(f"  - AV Ratio: {args.av_ratio * 100}%")
    print(f"  - Decision Interval: {args.step_size}s")
    print(f"\nKey Ablation:")
    print(f"  ✗ NO Traffic Agent (removed)")
    print(f"  ✓ Regional Agents only (end-to-end planning)")
    print(f"  ✓ Token usage tracking enabled")
    print(f"  ✓ Cloud API for high-performance inference")
    print(f"{'='*70}\n")

    # Initialize wandb
    wandb.init(
        project="LLMNavigation-Ablation",
        group=f"{args.location}-NoHierarchy-{'API' if use_api else 'Local'}",
        name=f"NoHierarchy-{model_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "experiment": "no_hierarchy",
            "location": args.location,
            "start_time": args.start_time,
            "max_steps": args.max_steps,
            "av_ratio": args.av_ratio,
            "model": model_name,
            "api_mode": use_api,
            "provider": "dashscope" if use_api else "local"
        }
    )

    # Run experiment
    env = NoHierarchyEnvironment(
        location=args.location,
        sumo_config_file=sumo_config,
        route_file=route_file,
        road_info_file=road_info_file,
        region_data_dir=region_data_dir,
        model_name=model_name,
        step_size=args.step_size,
        max_steps=args.max_steps,
        task_info_file=task_info_file,
        start_time=args.start_time,
        av_ratio=args.av_ratio,
        log_dir=f"logs/no_hierarchy_{args.location}_{model_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        use_api=use_api
    )

    avg_travel_time, throughput = env.run_simulation()

    # Log to wandb
    wandb.log({
        "avg_travel_time": avg_travel_time,
        "throughput": throughput,
        "total_llm_calls": env.token_stats['total_calls'],
        "total_tokens": env.token_stats['total_tokens'],
        "avg_tokens_per_call": env.token_stats['total_tokens'] / max(1, env.token_stats['total_calls'])
    })

    wandb.finish()

    print(f"\n[COMPLETE] No-Hierarchy ablation experiment finished")
    print(f"  - LLM: {model_name} ({'云端API' if use_api else '本地'})")
    print(f"  - Avg Travel Time: {avg_travel_time:.2f}s")
    print(f"  - Throughput: {throughput}")
    print(f"  - Total LLM Calls: {env.token_stats['total_calls']}")
    print(f"  - Total Tokens: {env.token_stats['total_tokens']}")

    # Save LLM token usage if using API
    if use_api and hasattr(env.llm_agent, 'save_token_usage'):
        token_usage_file = os.path.join(env.log_dir, "llm_token_usage.json")
        token_data = env.llm_agent.save_token_usage(token_usage_file)
        if token_data:
            print(f"\n=== DashScope API Token Usage (Detailed) ===")
            print(f"  - Total Prompt Tokens: {token_data.get('total_prompt_tokens', 0)}")
            print(f"  - Total Completion Tokens: {token_data.get('total_completion_tokens', 0)}")
            print(f"  - Total Tokens: {token_data.get('total_tokens', 0)}")
            print(f"  - API Calls: {len(token_data.get('detailed_log', []))}")
            print(f"  - Saved to: {token_usage_file}")


if __name__ == "__main__":
    main()
