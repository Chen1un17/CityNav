"""
Multi-Agent Traffic Control Environment

This module implements the multi-agent architecture for traffic simulation,
integrating Regional Agents, Traffic Agent, and Prediction Engine.
"""

import os
import sys
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Set, Optional, Tuple, Any
import networkx as nx
import traci

from utils.read_utils import load_json
from agents import RegionalAgent, TrafficAgent, PredictionEngine, AgentLogger
from env_utils import parse_rou_file, get_edge_info, get_multiple_edges_info, get_dynamic_data_batch, get_congestion_level

# Global LLM Manager Registry for Progressive Training
_global_llm_manager_registry = {}


class MultiAgentTrafficEnvironment:
    """
    Main environment class for multi-agent traffic control system.
    
    Integrates Regional Agents, Traffic Agent, and Prediction Engine
    for coordinated traffic management.
    """
    
    def __init__(self, location: str, sumo_config_file: str, route_file: str,
                 road_info_file: str, adjacency_file: str, region_data_dir: str,
                 model_path: str = None, llm_agent=None, step_size: float = 1.0, max_steps: int = 1000,
                 log_dir: str = "logs", task_info=None, use_local_llm: bool = True, training_queue=None):
        """
        Initialize multi-agent traffic environment.
        
        Args:
            location: Location name (e.g., Manhattan)
            sumo_config_file: Path to SUMO configuration file
            route_file: Path to route file
            road_info_file: Path to road information file
            adjacency_file: Path to adjacency information file
            region_data_dir: Directory containing region partition data
            model_path: Path to local model (for local LLM mode)
            llm_agent: Language model agent for decision making (legacy mode)
            step_size: Simulation step size in seconds
            max_steps: Maximum simulation steps
            log_dir: Directory for log files
            task_info: Task information for LLM initialization
            use_local_llm: Whether to use local shared LLM architecture
            training_queue: Multiprocessing queue for RL training data (optional)
        """
        self.location = location
        self.sumo_config_file = sumo_config_file
        self.route_file = route_file
        self.road_info_file = road_info_file
        self.adjacency_file = adjacency_file
        self.region_data_dir = region_data_dir
        self.model_path = model_path
        self.llm_agent = llm_agent  # 兼容性保留
        self.step_size = step_size
        self.max_steps = max_steps
        self.task_info = task_info
        self.use_local_llm = use_local_llm
        
        # 本地LLM管理器
        self.llm_manager = None
        self.traffic_llm = None
        self.regional_llm = None
        
        # Initialize logger
        self.logger = AgentLogger(log_dir=log_dir, console_output=True)
        
        # RL Training Integration - Initialize training queue
        self.training_queue = training_queue  # Multiprocessing queue for training data
        self.completed_vehicle_times = {}  # Store completion times for ATT calculation
        self.rl_data_collection_enabled = True  # Flag to control RL data collection
        
        # Hot-reload mechanism for LoRA adapters
        self.current_lora_adapters = {
            'traffic': None,  # Current Traffic LLM LoRA adapter name
            'regional': None  # Current Regional LLM LoRA adapter name
        }
        self.lora_update_lock = threading.Lock()  # Thread-safe adapter updates
        self.llm_call_lock = threading.Lock()    # 防止多LLM实例同时调用的互斥锁
        
        # === 优化加载顺序：先加载模型后加载路网 ===
        print("\n=== Step 1: 初始化LLM模型 ===")
        self._initialize_llms_first()
        
        print("\n=== Step 2: 加载路网数据 ===")
        # Load region and road data
        self._load_region_data()
        self._load_road_data()
        
        # Load static road data for performance optimization
        print(f"MULTI_AGENT_ENV: Loading static road data from {self.region_data_dir}")
        from env_utils import load_static_road_data
        load_static_road_data(data_dir=self.region_data_dir)
        print("MULTI_AGENT_ENV: Static road data loading completed")
        
        # Event-driven coordination parameters - only LLM makes decisions
        # Real-time event-driven system, no fixed intervals
        self.vehicle_birth_events = {}  # Track when vehicles enter simulation
        self.region_change_events = {}  # Track when vehicles change regions
        self.macro_planning_requests = []  # Queue for macro planning requests
        self.regional_planning_requests = {}  # Queue for regional planning by region
        
        # Event tracking for LLM decision points
        self.last_vehicle_region_check = 0.0
        self.pending_macro_decisions = {}
        self.pending_regional_decisions = {}
        
        # Time tracking for components that still need periodic updates
        self.prediction_update_interval = 15.0  # Prediction engine updates every 15s
        self.last_prediction_update_time = 0.0
        self.last_regional_decision_time = {}  # Track regional agent decision times
        
        # Initialize agents (LLMs already loaded, now create agent instances)
        print("\n=== Step 3: 初始化智能体 ===")
        self._initialize_agents_with_existing_llms()
        
        # Simulation state
        self.autonomous_vehicles = set()
        self.vehicle_start_times = {}
        self.vehicle_end_times = {}
        self.vehicle_regions = {}  # Track which region each vehicle is in
        self.vehicle_macro_routes = {}  # Track macro routes for each vehicle
        self.vehicle_destinations = {}  # Track final destinations
        self.total_vehicles = 0
        self.completed_vehicles = 0
        
        # Communication and broadcasting system - enhanced for LLM coordination
        self.boundary_vehicle_plans = {}  # Track vehicles planning to reach each boundary
        self.region_vehicle_plans = {}    # Track vehicles planning to reach each region
        self.communication_log = []       # Log all communication events
        self.broadcast_messages = []       # Queue for broadcast messages
        self.vehicle_plan_updates = {}     # Track real-time vehicle plan updates
        
        # Real-time tracking for LLM decisions
        self.vehicle_current_plans = {}    # Current macro routes for each vehicle
        self.vehicle_regional_plans = {}   # Current regional routes for each vehicle
        self.vehicle_travel_metrics = {}   # Real-time travel metrics per vehicle
        
        # Threading for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Async LLM calling and caching system
        self._initialize_async_llm_system()
        
        # Real-time monitoring state
        self.current_step = 0
        self.current_sim_time = 0.0
        self.active_autonomous_vehicles = 0
        self.att_calculation = 0.0
        self.system_throughput = 0.0
        
    def _initialize_async_llm_system(self):
        """Initialize asynchronous LLM calling system."""
        from concurrent.futures import Future
        
        # Async LLM call management
        self.pending_llm_calls = {}  # Future objects for ongoing LLM calls
        self.llm_results_queue = []  # Queue for completed LLM results
        self.temp_decisions = {}  # Temporary heuristic decisions while waiting for LLM
        
        # LLM call statistics
        self.llm_call_stats = {
            'macro_calls': 0,
            'regional_calls': 0,
            'async_calls': 0,
            'total_time_saved': 0.0
        }
    
    def _async_llm_call_macro_route(self, vehicle_id: str, start_region: int, dest_region: int, 
                                  candidates: List[List[int]], current_time: float) -> Optional[List[int]]:
        """Asynchronous macro route selection - always calls LLM for dynamic decisions."""
        try:
            # Check if already processing this request
            call_key = f"macro_{vehicle_id}"
            if call_key in self.pending_llm_calls:
                # Return heuristic decision for now
                heuristic_route = candidates[0] if candidates else None
                self.temp_decisions[vehicle_id] = {
                    'type': 'macro',
                    'route': heuristic_route,
                    'timestamp': current_time
                }
                self.logger.log_info(f"ASYNC_PENDING: Using heuristic route for {vehicle_id} while LLM processes")
                return heuristic_route
            
            # Submit async LLM call
            future = self.executor.submit(
                self._llm_select_macro_route_sync,
                vehicle_id, start_region, dest_region, candidates, current_time
            )
            
            self.pending_llm_calls[call_key] = {
                'future': future,
                'vehicle_id': vehicle_id,
                'type': 'macro',
                'start_time': time.time(),
                'candidates': candidates
            }
            
            self.llm_call_stats['async_calls'] += 1
            self.llm_call_stats['macro_calls'] += 1
            
            # Return heuristic decision for immediate use
            heuristic_route = candidates[0] if candidates else None
            self.temp_decisions[vehicle_id] = {
                'type': 'macro',
                'route': heuristic_route,
                'timestamp': current_time
            }
            
            self.logger.log_info(f"ASYNC_SUBMIT: Macro route LLM call submitted for {vehicle_id}, using heuristic temporarily")
            return heuristic_route
            
        except Exception as e:
            self.logger.log_error(f"ASYNC_MACRO_ERROR: {e}")
            return candidates[0] if candidates else None
    
    def _async_llm_call_regional_route(self, vehicle_id: str, current_edge: str, 
                                     route_candidates: List[dict], target_region: int, 
                                     region_id: int, current_time: float) -> Optional[dict]:
        """Asynchronous regional route selection - always calls LLM for dynamic decisions."""
        try:
            # Check if already processing this request
            call_key = f"regional_{vehicle_id}"
            if call_key in self.pending_llm_calls:
                # Return heuristic decision for now
                heuristic_plan = route_candidates[0] if route_candidates else None
                self.temp_decisions[vehicle_id] = {
                    'type': 'regional',
                    'plan': heuristic_plan,
                    'timestamp': current_time
                }
                self.logger.log_info(f"ASYNC_PENDING: Using heuristic plan for {vehicle_id} while LLM processes")
                return heuristic_plan
            
            # Submit async LLM call through regional agent
            if region_id in self.regional_agents:
                regional_agent = self.regional_agents[region_id]
                future = self.executor.submit(
                    regional_agent._llm_select_regional_route,
                    vehicle_id, current_edge, route_candidates, target_region, current_time
                )
                
                self.pending_llm_calls[call_key] = {
                    'future': future,
                    'vehicle_id': vehicle_id,
                    'type': 'regional',
                    'start_time': time.time(),
                    'candidates': route_candidates
                }
                
                self.llm_call_stats['async_calls'] += 1
                self.llm_call_stats['regional_calls'] += 1
            
            # Return heuristic decision for immediate use
            heuristic_plan = route_candidates[0] if route_candidates else None
            self.temp_decisions[vehicle_id] = {
                'type': 'regional',
                'plan': heuristic_plan,
                'timestamp': current_time
            }
            
            self.logger.log_info(f"ASYNC_SUBMIT: Regional route LLM call submitted for {vehicle_id}, using heuristic temporarily")
            return heuristic_plan
            
        except Exception as e:
            self.logger.log_error(f"ASYNC_REGIONAL_ERROR: {e}")
            return route_candidates[0] if route_candidates else None
    
    def _process_completed_llm_calls(self, current_time: float):
        """Process completed async LLM calls and update vehicle routes."""
        try:
            completed_calls = []
            
            for call_key, call_info in self.pending_llm_calls.items():
                future = call_info['future']
                
                if future.done():
                    completed_calls.append(call_key)
                    
                    try:
                        result = future.result()
                        vehicle_id = call_info['vehicle_id']
                        call_type = call_info['type']
                        elapsed_time = time.time() - call_info['start_time']
                        
                        # Update vehicle route with LLM result
                        if result and vehicle_id in self.temp_decisions:
                            self._apply_llm_result(vehicle_id, call_type, result, current_time)
                            
                        # Clean up temporary decision
                        if vehicle_id in self.temp_decisions:
                            del self.temp_decisions[vehicle_id]
                        
                        self.llm_call_stats['total_time_saved'] += max(0, elapsed_time - 1.0)  # Assume 1s for heuristic
                        
                        self.logger.log_info(f"ASYNC_COMPLETE: {call_type} LLM call for {vehicle_id} completed in {elapsed_time:.1f}s")
                        
                    except Exception as e:
                        self.logger.log_error(f"ASYNC_RESULT_ERROR: Failed to process LLM result for {call_key}: {e}")
            
            # Remove completed calls
            for call_key in completed_calls:
                del self.pending_llm_calls[call_key]
                
        except Exception as e:
            self.logger.log_error(f"ASYNC_PROCESS_ERROR: {e}")
    
    def _apply_llm_result(self, vehicle_id: str, call_type: str, result, current_time: float):
        """Apply LLM result to vehicle routing."""
        try:
            if call_type == 'macro' and isinstance(result, list):
                # Update macro route
                if vehicle_id in self.vehicle_current_plans:
                    self.vehicle_current_plans[vehicle_id]['macro_route'] = result
                    self.logger.log_info(f"LLM_UPDATE: Updated macro route for {vehicle_id}: {result}")
                    
            elif call_type == 'regional' and isinstance(result, dict):
                # Update regional route
                if vehicle_id in self.vehicle_regional_plans:
                    self.vehicle_regional_plans[vehicle_id] = result
                    
                    # Apply route to SUMO if vehicle is still active
                    if vehicle_id in traci.vehicle.getIDList():
                        route = result.get('route', [])
                        if route and len(route) > 1:
                            try:
                                # Safe route setting - ensure current edge is included
                                current_edge = traci.vehicle.getRoadID(vehicle_id)
                                safe_route = self._create_safe_route(current_edge, route)
                                if safe_route:
                                    traci.vehicle.setRoute(vehicle_id, safe_route)
                                    self.logger.log_info(f"LLM_UPDATE: Applied regional route for {vehicle_id}")
                                else:
                                    self.logger.log_warning(f"LLM_UPDATE: Cannot create safe route for {vehicle_id} from {current_edge}")
                            except Exception as route_error:
                                self.logger.log_warning(f"LLM_UPDATE: Failed to apply route for {vehicle_id}: {route_error}")
                        
        except Exception as e:
            self.logger.log_error(f"LLM_APPLY_ERROR: {e}")
    
    def _load_region_data(self):
        """Load region partition data."""
        # Load boundary edges
        boundary_file = os.path.join(self.region_data_dir, "boundary_edges_alpha_1.json")
        self.boundary_edges = load_json(boundary_file)
        
        # Load edge to region mapping
        edge_region_file = os.path.join(self.region_data_dir, "edge_to_region_alpha_1.json")
        self.edge_to_region = load_json(edge_region_file)
        
        # Determine number of regions
        self.num_regions = len(set(self.edge_to_region.values()))
        
        self.logger.log_info(f"Loaded region data: {self.num_regions} regions, "
                            f"{len(self.boundary_edges)} boundary edges")
    
    def _check_and_apply_latest_adapters(self):
        """Check for and apply latest LoRA adapters from training process."""
        if not hasattr(self, 'llm_manager') or not self.llm_manager:
            return
            
        try:
            import glob
            import os
            
            # Check for latest adapters in the sync directory
            log_dir = getattr(self, 'log_dir', 'logs')
            adapter_sync_dir = os.path.join(log_dir, 'lora_adapters')
            
            if not os.path.exists(adapter_sync_dir):
                return
            
            with self.lora_update_lock:
                # Check Traffic LLM adapters
                traffic_adapters = glob.glob(os.path.join(adapter_sync_dir, 'traffic_adapter_step_*'))
                if traffic_adapters:
                    latest_traffic = max(traffic_adapters, key=lambda x: int(x.split('_')[-1]))
                    adapter_name = os.path.basename(latest_traffic)
                    
                    if self.current_lora_adapters['traffic'] != adapter_name:
                        self.current_lora_adapters['traffic'] = adapter_name
                        self.logger.log_info(f"LORA_UPDATE_TRAFFIC: 应用最新Traffic LLM适配器 {adapter_name}")
                
                # Check Regional LLM adapters  
                regional_adapters = glob.glob(os.path.join(adapter_sync_dir, 'regional_adapter_step_*'))
                if regional_adapters:
                    latest_regional = max(regional_adapters, key=lambda x: int(x.split('_')[-1]))
                    adapter_name = os.path.basename(latest_regional)
                    
                    if self.current_lora_adapters['regional'] != adapter_name:
                        self.current_lora_adapters['regional'] = adapter_name
                        self.logger.log_info(f"LORA_UPDATE_REGIONAL: 应用最新Regional LLM适配器 {adapter_name}")
                        
        except Exception as e:
            self.logger.log_error(f"LORA_UPDATE_ERROR: {e}")
    
    # ===== PROGRESSIVE TRAINING: Global LLM Manager Registry =====
    
    def _register_llm_manager_globally(self):
        """Register LLM manager globally for access by training managers."""
        try:
            global _global_llm_manager_registry
            
            # Create unique key for this environment
            env_key = f"{self.location}_{id(self)}"
            
            _global_llm_manager_registry[env_key] = self.llm_manager
            
            # Also register as 'current' for easy access
            _global_llm_manager_registry['current'] = self.llm_manager
            
            print(f"LLM_MANAGER_REGISTERED: {env_key}")
            
        except Exception as e:
            print(f"LLM_MANAGER_REGISTRY_ERROR: {e}")
    
    def _load_road_data(self):
        """Load road network and information."""
        try:
            # Load road information
            self.road_info = load_json(self.road_info_file)
            
            # Load adjacency matrix and build network graph
            adjacency_matrix = load_json(self.adjacency_file)
            self.road_network = nx.DiGraph()
            
            for edge in adjacency_matrix:
                if edge in self.road_info:
                    road_len = self.road_info[edge]['road_len']
                    for neighbor_edge in adjacency_matrix[edge]:
                        self.road_network.add_edge(edge, neighbor_edge, weight=road_len)
            
            self.logger.log_info(f"Loaded road data: {len(self.road_info)} edges, "
                               f"{self.road_network.number_of_edges()} connections")
            
        except Exception as e:
            self.logger.log_error(f"Failed to load road data: {e}")
            raise
    
    def _initialize_llms_first(self):
        """优先初始化LLM模型（最耗时的步骤）"""
        try:
            # 初始化本地LLM管理器或使用传统模式
            if self.use_local_llm and self.model_path:
                print("\n=== 使用本地共享LLM架构 ===")
                from utils.language_model import LocalLLMManager
                
                # 创建本地LLM管理器
                self.llm_manager = LocalLLMManager(
                    model_path=self.model_path,
                    task_info=self.task_info
                )
                
                # 初始化共享LLM实例（这是最耗时的步骤）
                print("正在初始化共享LLM实例...")
                self.traffic_llm, self.regional_llm = self.llm_manager.initialize_llms()
                
                # 打印GPU状态
                self.llm_manager.print_gpu_status()
                
                # Initialize LoRA management for progressive training
                if hasattr(self.llm_manager, 'initialize_lora_management'):
                    self.llm_manager.initialize_lora_management()
                
                # Register LLM manager globally for training manager access
                self._register_llm_manager_globally()
                
            elif self.llm_agent:
                print("\n=== 使用传统单一LLM模式 ===")
                self.traffic_llm = self.llm_agent
                self.regional_llm = self.llm_agent
            else:
                raise ValueError("必须提供 model_path 或 llm_agent 参数")
                
            print("[SUCCESS] LLM模型初始化完成")
            
        except Exception as e:
            self.logger.log_error(f"Failed to initialize LLMs: {e}")
            raise
    
    def _initialize_agents_with_existing_llms(self):
        """使用已初始化的LLM创建智能体实例"""
        try:
            # Initialize prediction engine
            edge_list = list(self.road_info.keys())
            self.prediction_engine = PredictionEngine(edge_list, self.logger)
            
            # Initialize Traffic Agent with appropriate LLM (already loaded)
            self.traffic_agent = TrafficAgent(
                boundary_edges=self.boundary_edges,
                edge_to_region=self.edge_to_region,
                road_info=self.road_info,
                num_regions=self.num_regions,
                llm_agent=self.traffic_llm,  # 使用已初始化的Traffic LLM
                logger=self.logger,
                prediction_engine=self.prediction_engine
            )
            
            # Initialize Regional Agents with shared Regional LLM (already loaded)
            self.regional_agents = {}
            for region_id in range(self.num_regions):
                regional_agent = RegionalAgent(
                    region_id=region_id,
                    boundary_edges=self.boundary_edges,
                    edge_to_region=self.edge_to_region,
                    road_info=self.road_info,
                    road_network=self.road_network,
                    llm_agent=self.regional_llm,  # 所有区域共享已初始化的LLM
                    logger=self.logger,
                    prediction_engine=self.prediction_engine
                )
                self.regional_agents[region_id] = regional_agent
                self.last_regional_decision_time[region_id] = 0.0
            
            if self.use_local_llm:
                self.logger.log_info(f"Initialized {len(self.regional_agents)} Regional Agents "
                                   f"(sharing 1 Local Regional LLM), "
                                   f"1 Traffic Agent (with 1 Local Traffic LLM), "
                                   f"and 1 Prediction Engine")
            else:
                self.logger.log_info(f"Initialized {len(self.regional_agents)} Regional Agents, "
                                   f"1 Traffic Agent, and 1 Prediction Engine")
            
        except Exception as e:
            self.logger.log_error(f"Failed to initialize agents: {e}")
            raise
    
    def initialize_simulation(self):
        """Initialize SUMO simulation."""
        try:
            # Start SUMO
            sumo_cmd = ["sumo", "-c", self.sumo_config_file, "--no-warnings", 
                       "--ignore-route-errors"]
            traci.start(sumo_cmd)
            
            # Parse only the first route file for autonomous vehicle selection
            # This ensures only taxi vehicles from NewYork_od_0.1.rou.alt.xml can be autonomous
            # The second file NYC_routes_0.1_20250830_111509.alt.xml is loaded as environment traffic only
            first_route_vehicles = parse_rou_file(self.route_file)
            first_route_vehicle_ids = [veh_id for veh_id, _, _ in first_route_vehicles]
            
            # Get total vehicles from SUMO simulation (includes both route files)
            # We need to wait a bit for SUMO to load all vehicles
            import time
            time.sleep(0.1)  # Small delay to ensure all vehicles are loaded
            
            # Count total vehicles that will be in simulation
            self.total_vehicles = len(first_route_vehicles)  # Base count from first file
            
            # Try to get additional vehicles from the simulation
            try:
                # Get all vehicle IDs that SUMO knows about (including from both files)
                all_sumo_vehicles = set()
                for _ in range(10):  # Try multiple times to get all vehicles
                    current_vehicles = set(traci.vehicle.getIDList())
                    # Also get vehicles that are loaded but not yet departed
                    loaded_vehicles = set(traci.simulation.getLoadedIDList())
                    all_sumo_vehicles.update(current_vehicles)
                    all_sumo_vehicles.update(loaded_vehicles)
                    traci.simulationStep()  # Step forward to load more vehicles
                
                # Reset simulation to start
                traci.load(sumo_cmd[2:])  # Reload with same config
                
                # Update total count with environment vehicles
                environment_vehicles = all_sumo_vehicles - set(first_route_vehicle_ids)
                self.total_vehicles = len(all_sumo_vehicles)
                
                self.logger.log_info(f"Vehicle loading: {len(first_route_vehicle_ids)} from primary route file, "
                                   f"{len(environment_vehicles)} from environment route file, "
                                   f"{self.total_vehicles} total vehicles")
                
            except Exception as count_error:
                self.logger.log_warning(f"Could not count environment vehicles: {count_error}")
                # Fall back to just the first route file count
                pass
            
            # Select 2% of vehicles as autonomous - ONLY from the first route file
            import random
            self.autonomous_vehicles = set(random.sample(first_route_vehicle_ids, 
                                                       int(0.02 * len(first_route_vehicle_ids))))
            
            # Initialize edge information
            print("MULTI_AGENT_ENV: Initializing edge list from SUMO")
            self.edges = traci.edge.getIDList()
            print(f"MULTI_AGENT_ENV: Retrieved {len(self.edges)} edges from SUMO")
            
            # Update logger with initial vehicle count
            self.logger.update_vehicle_count(self.total_vehicles, 0.0)
            
            self.logger.log_info(f"Simulation initialized: {self.total_vehicles} total vehicles, "
                               f"{len(self.autonomous_vehicles)} autonomous vehicles")
            
            # Send autonomous vehicle count to training manager if queue is available
            if self.training_queue is not None:
                try:
                    autonomous_count_message = {
                        'message_type': 'autonomous_vehicle_count',
                        'total_autonomous_vehicles': len(self.autonomous_vehicles),
                        'total_vehicles': self.total_vehicles,
                        'timestamp': time.time()
                    }
                    self.training_queue.put(autonomous_count_message, block=False)
                    self.logger.log_info(f"TRAINING_QUEUE: Sent autonomous vehicle count ({len(self.autonomous_vehicles)}) to training manager")
                except Exception as queue_error:
                    self.logger.log_warning(f"TRAINING_QUEUE: Failed to send vehicle count: {queue_error}")
            
        except Exception as e:
            self.logger.log_error(f"Failed to initialize simulation: {e}")
            raise
    
    def handle_vehicle_birth_macro_planning(self, vehicle_id: str, current_time: float):
        """
        Handle macro route planning when an autonomous vehicle is born.
        Enhanced with CORY cooperative decision framework:
        1. State space construction
        2. Action space generation 
        3. Pioneer-Observer cooperative decision making
        4. J1-Judge quality evaluation
        """
        try:
            self.logger.log_info(f"CORY_VEHICLE_BIRTH: Processing cooperative macro planning for {vehicle_id}")
            
            # [Phase 1: State Space Construction] - Following CLAUDE.md specifications
            state_context = self._construct_state_space(vehicle_id, current_time)
            if not state_context:
                self.logger.log_warning(f"CORY_STATE_CONSTRUCTION: Failed to construct state space for {vehicle_id}")
                return
            
            start_region = state_context['start_region']
            dest_region = state_context['dest_region']
            
            self.logger.log_info(f"CORY_STATE_SPACE: Vehicle {vehicle_id} state constructed - "
                               f"Start: Region {start_region}, Dest: Region {dest_region}, "
                               f"Current traffic: {state_context['regional_congestion']}, "
                               f"Predictions available: {len(state_context['traffic_predictions'])} time windows")
            
            # Handle single-region case with CORY logging
            if start_region == dest_region:
                self.logger.log_info(f"CORY_SINGLE_REGION: {vehicle_id} intra-region route in Region {dest_region}")
                single_region_route = [start_region]
                self.vehicle_current_plans[vehicle_id] = {
                    'macro_route': single_region_route,
                    'current_region_index': 0,
                    'creation_time': current_time,
                    'last_update': current_time,
                    'single_region': True,
                    'cory_decision_type': 'single_region',
                    'cooperation_quality': 1.0  # Perfect cooperation for single region
                }
                self.logger.log_info(f"CORY_SINGLE_REGION: {vehicle_id} assigned single-region macro route [{start_region}]")
                return
            
            # [Phase 2: Action Space Generation] - Generate candidate macro routes
            macro_candidates = self._generate_macro_route_candidates(
                start_region, dest_region, current_time, state_context
            )
            
            if not macro_candidates:
                self.logger.log_warning(f"CORY_ACTION_SPACE: No macro candidates found for {vehicle_id}")
                emergency_route = [start_region]
                self.vehicle_current_plans[vehicle_id] = {
                    'macro_route': emergency_route,
                    'current_region_index': 0,
                    'creation_time': current_time,
                    'last_update': current_time,
                    'emergency_route': True,
                    'original_dest_region': dest_region,
                    'cory_decision_type': 'emergency',
                    'cooperation_quality': 0.1  # Low quality due to emergency
                }
                return
            
            self.logger.log_info(f"CORY_ACTION_SPACE: Generated {len(macro_candidates)} route candidates for {vehicle_id}: {macro_candidates}")
            
            # [Phase 3: CORY Cooperative Decision Making]
            cory_result = self._cory_cooperative_decision(
                vehicle_id, state_context, macro_candidates, current_time
            )
            
            selected_macro_route = cory_result.get('final_route')
            cooperation_quality = cory_result.get('cooperation_quality', 0.5)
            pioneer_decision = cory_result.get('pioneer_decision', {})
            observer_feedback = cory_result.get('observer_feedback', {})
            j1_judge_evaluation = cory_result.get('j1_judge_evaluation', {})
            
            self.logger.log_info(f"CORY_DECISION: Vehicle {vehicle_id} cooperative decision completed - "
                               f"Route: {selected_macro_route}, Quality: {cooperation_quality:.3f}, "
                               f"Pioneer confidence: {pioneer_decision.get('confidence', 'N/A')}, "
                               f"Observer acceptance: {observer_feedback.get('acceptance', 'N/A')}")
            
            if selected_macro_route:
                # Store enhanced macro route with CORY metadata
                self.vehicle_current_plans[vehicle_id] = {
                    'macro_route': selected_macro_route,
                    'current_region_index': 0,
                    'creation_time': current_time,
                    'last_update': current_time,
                    'cory_decision_type': 'cooperative',
                    'cooperation_quality': cooperation_quality,
                    'pioneer_decision': pioneer_decision,
                    'observer_feedback': observer_feedback,
                    'j1_judge_evaluation': j1_judge_evaluation,
                    'state_context': state_context  # Store for RL training
                }
                
                # Update communication system
                self._broadcast_vehicle_macro_plan(vehicle_id, selected_macro_route, current_time)
                self._log_vehicle_decision(vehicle_id, "CORY_MACRO_PLANNING", selected_macro_route, current_time)
                
                self.logger.log_info(f"CORY_SUCCESS: {vehicle_id} assigned cooperative macro route {selected_macro_route} "
                                   f"(Quality: {cooperation_quality:.3f})")
            else:
                self.logger.log_error(f"CORY_FAILURE: Failed cooperative decision for {vehicle_id}")
                
        except Exception as e:
            self.logger.log_error(f"CORY_ERROR: Cooperative macro planning failed for {vehicle_id}: {e}")
            import traceback
            self.logger.log_error(f"CORY_TRACEBACK: {traceback.format_exc()}")
            # Fallback to original system if CORY fails
            self._fallback_original_planning(vehicle_id, current_time)
    
    def _create_safe_route(self, current_edge: str, target_route: List[str]) -> Optional[List[str]]:
        """
        Create a safe route that includes the vehicle's current edge as the first element.
        This is required by SUMO's setRoute() function.
        
        Args:
            current_edge: Vehicle's current edge ID
            target_route: Desired route
            
        Returns:
            Safe route with current edge as first element, or None if impossible
        """
        try:
            # Skip junction edges (internal edges)
            if current_edge.startswith(':'):
                self.logger.log_warning(f"SAFE_ROUTE: Vehicle on junction edge {current_edge}, cannot set route safely")
                return None
            
            # If target route is empty, return None
            if not target_route:
                return None
                
            # If current edge is already the first element, return as-is
            if target_route[0] == current_edge:
                return target_route
            
            # If current edge is somewhere in the middle of the route, extract remaining part
            if current_edge in target_route:
                current_index = target_route.index(current_edge)
                safe_route = target_route[current_index:]
                self.logger.log_info(f"SAFE_ROUTE: Extracted remaining route from current position")
                return safe_route
            
            # Current edge not in target route - need to connect
            # Try to find a route from current edge to the first edge of target route
            try:
                # Use SUMO's route finding to connect current edge to target route
                connection_route = traci.simulation.findRoute(fromEdge=current_edge, toEdge=target_route[0])
                if connection_route and hasattr(connection_route, 'edges'):
                    # Combine connection with target route (avoid duplication of first target edge)
                    safe_route = list(connection_route.edges) + target_route[1:]
                    self.logger.log_info(f"SAFE_ROUTE: Connected current edge to target route")
                    return safe_route
                else:
                    self.logger.log_warning(f"SAFE_ROUTE: No connection found from {current_edge} to {target_route[0]}")
                    return None
            except Exception as route_error:
                self.logger.log_warning(f"SAFE_ROUTE: Route finding failed from {current_edge} to {target_route[0]}: {route_error}")
                return None
                
        except Exception as e:
            self.logger.log_error(f"SAFE_ROUTE: Error creating safe route for {current_edge}: {e}")
            return None

    def _validate_route_setting(self, vehicle_id: str, route: List[str]) -> bool:
        """
        Validate if a route can be safely set for a vehicle.
        
        Args:
            vehicle_id: Vehicle ID
            route: Proposed route
            
        Returns:
            True if route can be set safely, False otherwise
        """
        try:
            # Check if vehicle exists
            if vehicle_id not in traci.vehicle.getIDList():
                self.logger.log_warning(f"ROUTE_VALIDATION: Vehicle {vehicle_id} not found")
                return False
            
            # Check if route is valid
            if not route or len(route) == 0:
                self.logger.log_warning(f"ROUTE_VALIDATION: Empty route for {vehicle_id}")
                return False
            
            # Get current edge
            current_edge = traci.vehicle.getRoadID(vehicle_id)
            
            # Skip validation if on junction
            if current_edge.startswith(':'):
                self.logger.log_warning(f"ROUTE_VALIDATION: Vehicle {vehicle_id} on junction {current_edge}")
                return False
            
            # Check if current edge is in route or can be connected
            if current_edge not in route:
                # Try to find connection
                try:
                    connection_route = traci.simulation.findRoute(fromEdge=current_edge, toEdge=route[0])
                    if not connection_route or not hasattr(connection_route, 'edges'):
                        self.logger.log_warning(f"ROUTE_VALIDATION: No connection from {current_edge} to route start {route[0]}")
                        return False
                except Exception:
                    self.logger.log_warning(f"ROUTE_VALIDATION: Route finding failed from {current_edge} to {route[0]}")
                    return False
            
            # Check if all edges in route are valid
            for edge in route:
                try:
                    # Try to get basic info about the edge
                    traci.edge.getLaneNumber(edge)
                except Exception:
                    self.logger.log_warning(f"ROUTE_VALIDATION: Invalid edge {edge} in route")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.log_error(f"ROUTE_VALIDATION: Error validating route for {vehicle_id}: {e}")
            return False

    def _construct_state_space(self, vehicle_id: str, current_time: float) -> Dict[str, Any]:
        """
        Construct comprehensive state space for CORY cooperative decision making.
        Following CLAUDE.md Phase 1: Decision Need Generation & Environment State Construction.
        
        Returns:
            Complete state context including:
            - Vehicle individual state (position, route, region mapping)
            - Traffic state perception (regional congestion, boundary flows)
            - System global state (vehicle counts, ATT, predictions)
            - Candidate action space preparation
        """
        try:
            # Get vehicle's route and position information
            route = traci.vehicle.getRoute(vehicle_id)
            if not route:
                self.logger.log_error(f"CORY_STATE: No route found for {vehicle_id}")
                return None
            
            start_edge = route[0]
            dest_edge = route[-1]
            
            # Map edges to regions
            start_region = self.edge_to_region.get(start_edge)
            dest_region = self.edge_to_region.get(dest_edge)
            
            if start_region is None or dest_region is None:
                self.logger.log_error(f"CORY_STATE: Region mapping failed for {vehicle_id} - "
                                     f"start_edge: {start_edge} -> {start_region}, "
                                     f"dest_edge: {dest_edge} -> {dest_region}")
                return None
            
            # [Vehicle Individual State] - Extract and abstract to region level
            vehicle_state = {
                'vehicle_id': vehicle_id,
                'start_edge': start_edge,
                'dest_edge': dest_edge,
                'start_region': start_region,
                'dest_region': dest_region,
                'route_length': len(route),
                'creation_time': current_time
            }
            
            # [Traffic State Perception] - Multi-level data collection
            regional_congestion = {}
            boundary_flows = {}
            
            # Calculate regional congestion levels
            for region_id in range(self.num_regions):
                region_edges = [edge for edge, reg in self.edge_to_region.items() if reg == region_id]
                if region_edges:
                    occupancy_sum = 0
                    valid_edges = 0
                    for edge in region_edges:
                        try:
                            occupancy = traci.edge.getLastStepOccupancy(edge)
                            occupancy_sum += occupancy
                            valid_edges += 1
                        except Exception:
                            continue
                    regional_congestion[region_id] = occupancy_sum / max(valid_edges, 1)
                else:
                    regional_congestion[region_id] = 0.0
            
            # Calculate boundary flows
            for boundary_info in self.boundary_edges:
                edge_id = boundary_info['edge_id']
                try:
                    vehicle_count = traci.edge.getLastStepVehicleNumber(edge_id)
                    boundary_flows[edge_id] = {
                        'vehicle_count': vehicle_count,
                        'from_region': boundary_info['from_region'],
                        'to_region': boundary_info['to_region']
                    }
                except Exception:
                    boundary_flows[edge_id] = {'vehicle_count': 0, 
                                              'from_region': boundary_info['from_region'],
                                              'to_region': boundary_info['to_region']}
            
            # [System Global State] - Maintain global variables
            global_state = {
                'current_time': current_time,
                'total_vehicles': len(traci.vehicle.getIDList()),
                'completed_vehicles': len(self.completed_vehicle_times),
                'autonomous_vehicles_active': len([v for v in self.autonomous_vehicles if v in traci.vehicle.getIDList()]),
                'current_avg_travel_time': self._calculate_current_att()
            }
            
            # [Traffic Predictions] - Multi-horizon predictions
            traffic_predictions = {}
            try:
                # Get predictions for 15min, 30min, 45min, 60min windows
                prediction_horizons = [900, 1800, 2700, 3600]  # seconds
                boundary_edge_ids = [info['edge_id'] for info in self.boundary_edges]
                
                for horizon in prediction_horizons:
                    predictions = self.prediction_engine.get_congestion_forecast(
                        boundary_edge_ids, horizon
                    )
                    traffic_predictions[f'{horizon}s'] = predictions
                
                self.logger.log_info(f"CORY_STATE_PREDICTIONS: Generated predictions for {len(prediction_horizons)} time windows")
            except Exception as e:
                self.logger.log_warning(f"CORY_STATE_PREDICTIONS: Failed to generate predictions: {e}")
                traffic_predictions = {}
            
            # Compile comprehensive state context
            state_context = {
                'vehicle_state': vehicle_state,
                'start_region': start_region,
                'dest_region': dest_region,
                'regional_congestion': regional_congestion,
                'boundary_flows': boundary_flows,
                'global_state': global_state,
                'traffic_predictions': traffic_predictions,
                'state_construction_time': current_time
            }
            
            self.logger.log_info(f"CORY_STATE_SUCCESS: Constructed state space for {vehicle_id} - "
                               f"Regions: {start_region}->{dest_region}, "
                               f"Global vehicles: {global_state['total_vehicles']}, "
                               f"Regional congestion range: [{min(regional_congestion.values()):.3f}, {max(regional_congestion.values()):.3f}]")
            
            return state_context
            
        except Exception as e:
            self.logger.log_error(f"CORY_STATE_ERROR: State space construction failed for {vehicle_id}: {e}")
            return None
    
    def _calculate_current_att(self) -> float:
        """Calculate current average travel time from completed vehicles."""
        if not self.completed_vehicle_times:
            # Use adaptive baseline based on system state
            return self._estimate_adaptive_baseline_time()
        
        total_time = sum(self.completed_vehicle_times.values())
        return total_time / len(self.completed_vehicle_times)
    
    def _estimate_adaptive_baseline_time(self) -> float:
        """Estimate adaptive baseline time based on current system state."""
        # Get current global congestion level
        global_congestion = self._get_global_congestion_level()
        
        # Estimate network diameter (average path length)
        avg_path_complexity = self._estimate_network_complexity()
        
        # Base time per region with congestion adjustment
        base_time_per_region = 120 + (global_congestion * 100)  # 120-220s per region
        
        # Adaptive baseline based on network characteristics
        estimated_baseline = avg_path_complexity * base_time_per_region
        
        # Ensure reasonable bounds (200s - 1200s)
        return max(200.0, min(1200.0, estimated_baseline))
    
    def _get_global_congestion_level(self) -> float:
        """Calculate global congestion level across all regions."""
        if not hasattr(self, 'latest_regional_report') or not self.latest_regional_report:
            return 0.3  # Default moderate congestion
        
        congestion_values = []
        for region_data in self.latest_regional_report.values():
            if isinstance(region_data, dict) and 'congestion_level' in region_data:
                congestion_values.append(region_data['congestion_level'])
        
        return sum(congestion_values) / len(congestion_values) if congestion_values else 0.3
    
    def _estimate_network_complexity(self) -> float:
        """Estimate average path complexity (number of regions)."""
        # Use region graph connectivity to estimate typical path lengths
        if hasattr(self.traffic_agent, 'region_connections') and self.traffic_agent.region_connections:
            # Estimate based on network diameter
            total_regions = len(set(self.edge_to_region.values()))
            connectivity_ratio = len(self.traffic_agent.region_connections) / max(1, total_regions * (total_regions - 1) / 2)
            
            # Higher connectivity = shorter paths, lower connectivity = longer paths
            complexity_factor = 2.0 + (1.0 - connectivity_ratio) * 2.0  # Range: 2-4 regions average
            return complexity_factor
        
        return 3.0  # Default assumption: 3 regions average path
    
    def _get_adaptive_expected_time(self, cooperation_quality: float) -> float:
        """Get adaptive expected time based on system state and cooperation quality."""
        base_time = self._estimate_adaptive_baseline_time()
        
        # High cooperation quality reduces expected time (better coordination)
        cooperation_factor = 1.0 - (cooperation_quality - 0.5) * 0.2  # ±10% based on cooperation
        cooperation_factor = max(0.8, min(1.2, cooperation_factor))
        
        return base_time * cooperation_factor
    
    def _get_route_baseline_time(self, route_length: int, cooperation_quality: float) -> float:
        """Get baseline time for a specific route length."""
        # Get system-wide baseline
        system_baseline = self._estimate_adaptive_baseline_time()
        
        # Adjust for route complexity
        if route_length <= 1:
            route_factor = 0.5  # Single region routes are much faster
        elif route_length <= 2:
            route_factor = 0.7  # Direct routes
        else:
            route_factor = 1.0 + (route_length - 3) * 0.2  # Each additional region adds complexity
        
        # Cooperation quality impact
        cooperation_factor = 1.0 - (cooperation_quality - 0.5) * 0.15
        cooperation_factor = max(0.85, min(1.15, cooperation_factor))
        
        return system_baseline * route_factor * cooperation_factor
    
    def _get_adaptive_fairness_threshold(self, cooperation_quality: float) -> float:
        """Get adaptive fairness threshold based on system state."""
        base_threshold = self._estimate_adaptive_baseline_time() * 1.8  # 80% above baseline
        
        # Better cooperation allows for slightly higher expectations (lower threshold)
        cooperation_adjustment = (cooperation_quality - 0.5) * 0.2
        adjusted_threshold = base_threshold * (1.0 - cooperation_adjustment)
        
        # Ensure reasonable bounds (300s - 2000s)
        return max(300.0, min(2000.0, adjusted_threshold))
    
    def _fallback_original_planning(self, vehicle_id: str, current_time: float):
        """Fallback to original planning when CORY fails."""
        try:
            self.logger.log_warning(f"CORY_FALLBACK: Using original planning for {vehicle_id}")
            
            # Get vehicle route
            route = traci.vehicle.getRoute(vehicle_id)
            if not route:
                return
            
            start_edge = route[0]
            dest_edge = route[-1]
            start_region = self.edge_to_region.get(start_edge)
            dest_region = self.edge_to_region.get(dest_edge)
            
            if start_region is None or dest_region is None:
                return
            
            # Generate candidates using original method
            candidates = self._generate_macro_route_candidates_original(
                start_region, dest_region, current_time
            )
            
            if candidates:
                selected_route = candidates[0]  # Use first candidate
                self.vehicle_current_plans[vehicle_id] = {
                    'macro_route': selected_route,
                    'current_region_index': 0,
                    'creation_time': current_time,
                    'last_update': current_time,
                    'fallback_planning': True
                }
                self.logger.log_info(f"CORY_FALLBACK: Assigned route {selected_route} for {vehicle_id}")
                
        except Exception as e:
            self.logger.log_error(f"CORY_FALLBACK_ERROR: {e}")
    
    def _cory_cooperative_decision(self, vehicle_id: str, state_context: Dict[str, Any], 
                                 macro_candidates: List[List[int]], current_time: float) -> Dict[str, Any]:
        """
        CORY Cooperative Decision Making Framework Implementation.
        Following CLAUDE.md Phase 2: CORY协作决策机制的深入分析.
        
        Pioneer-Observer协作哲学:
        - Pioneer (Traffic LLM): 宏观战略决策，全局优化
        - Observer (Regional LLM): 区域协调，个体保护，局部优化
        - J1-Judge: 协作质量评估
        
        Returns:
            Dict containing:
            - final_route: Selected macro route
            - cooperation_quality: Quality score of cooperation
            - pioneer_decision: Pioneer's original decision
            - observer_feedback: Observer's evaluation and suggestions
            - j1_judge_evaluation: Quality assessment
        """
        try:
            self.logger.log_info(f"CORY_COOPERATIVE: Starting cooperative decision for {vehicle_id}")
            
            # [Phase 2.1: Pioneer Decision Process] - Traffic LLM as Pioneer
            pioneer_result = self._pioneer_decision(
                vehicle_id, state_context, macro_candidates, current_time
            )
            
            if not pioneer_result or 'selected_route' not in pioneer_result:
                self.logger.log_error(f"CORY_PIONEER_FAILED: Pioneer decision failed for {vehicle_id}")
                return {'final_route': macro_candidates[0] if macro_candidates else None,
                       'cooperation_quality': 0.1}
            
            # [Phase 2.2: Observer Feedback Process] - Regional LLM as Observer
            observer_result = self._observer_feedback(
                vehicle_id, state_context, pioneer_result, current_time
            )
            
            if not observer_result:
                self.logger.log_warning(f"CORY_OBSERVER_FAILED: Observer feedback failed for {vehicle_id}, using Pioneer decision")
                observer_result = {'acceptance': True, 'improvements': [], 'conflicts': []}
            
            # [Phase 2.3: J1-Judge Quality Evaluation]
            j1_judge_result = self._j1_judge_evaluation(
                vehicle_id, pioneer_result, observer_result, current_time
            )
            
            # [Phase 2.4: Final Decision Synthesis]
            final_result = self._synthesize_final_decision(
                vehicle_id, pioneer_result, observer_result, j1_judge_result, current_time
            )
            
            self.logger.log_info(f"CORY_COOPERATIVE_SUCCESS: Completed cooperative decision for {vehicle_id} - "
                               f"Final route: {final_result.get('final_route')}, "
                               f"Quality: {final_result.get('cooperation_quality', 0):.3f}")
            
            return final_result
            
        except Exception as e:
            self.logger.log_error(f"CORY_COOPERATIVE_ERROR: Cooperative decision failed for {vehicle_id}: {e}")
            return {'final_route': macro_candidates[0] if macro_candidates else None,
                   'cooperation_quality': 0.1,
                   'error': str(e)}
    
    def _pioneer_decision(self, vehicle_id: str, state_context: Dict[str, Any], 
                        macro_candidates: List[List[int]], current_time: float) -> Dict[str, Any]:
        """
        Pioneer Decision Process - Traffic LLM as Pioneer.
        
        Following CLAUDE.md: Traffic LLM担任Pioneer的角色，负责基于宏观交通状态提出初始的路径选择方案。
        这个阶段的重点是全局优化，考虑的是如何最大化整个系统的效率。
        
        Returns:
            Pioneer decision with route selection, reasoning, and confidence
        """
        try:
            self.logger.log_info(f"CORY_PIONEER: Starting Pioneer decision for {vehicle_id}")
            
            # [Macro Context Construction] - 专门为Traffic LLM设计的输入格式
            macro_context = self._prepare_macro_context(state_context, macro_candidates)
            
            # [Decision Strategy] - Traffic LLM学习全局优化偏好
            decision_input = {
                'vehicle_id': vehicle_id,
                'current_time': current_time,
                'start_region': state_context['start_region'],
                'dest_region': state_context['dest_region'],
                'route_candidates': macro_candidates,
                'regional_congestion': state_context['regional_congestion'],
                'boundary_flows': state_context['boundary_flows'],
                'traffic_predictions': state_context['traffic_predictions'],
                'global_state': state_context['global_state']
            }
            
            # Check LLM health before calling
            if self.traffic_llm is None or not hasattr(self.traffic_llm, 'model') or self.traffic_llm.model is None:
                self.logger.log_error(f"CORY_PIONEER_LLM_UNAVAILABLE: Traffic LLM is None or damaged for {vehicle_id}")
                return {
                    'selected_route': macro_candidates[0] if macro_candidates else [],
                    'reasoning': 'LLM unavailable - using first route candidate',
                    'confidence': 0.2,
                    'decision_type': 'llm_unavailable',
                    'timestamp': current_time
                }
            
            # Use Traffic LLM for Pioneer decision (带锁保护)
            with self.llm_call_lock:
                pioneer_response = self._call_traffic_llm_pioneer(decision_input, macro_context)
            
            if not pioneer_response:
                self.logger.log_error(f"CORY_PIONEER_LLM_FAILED: Traffic LLM call failed for {vehicle_id}")
                # Fallback to heuristic decision
                return {
                    'selected_route': macro_candidates[0] if macro_candidates else [],
                    'reasoning': 'Fallback heuristic selection (LLM failed)',
                    'confidence': 0.3,
                    'decision_type': 'fallback',
                    'timestamp': current_time
                }
            
            # Extract Pioneer decision
            selected_route = pioneer_response.get('selected_route', macro_candidates[0] if macro_candidates else [])
            reasoning = pioneer_response.get('reasoning', 'Pioneer strategic decision')
            confidence = pioneer_response.get('confidence', 0.7)
            
            pioneer_result = {
                'selected_route': selected_route,
                'reasoning': reasoning,
                'confidence': confidence,
                'decision_type': 'pioneer_strategic',
                'macro_context': macro_context,
                'timestamp': current_time,
                'alternative_routes': [r for r in macro_candidates if r != selected_route]
            }
            
            self.logger.log_info(f"CORY_PIONEER_SUCCESS: Pioneer selected route {selected_route} for {vehicle_id} "
                               f"(confidence: {confidence:.3f}, reasoning: {reasoning[:100]}...)")
            
            return pioneer_result
            
        except Exception as e:
            self.logger.log_error(f"CORY_PIONEER_ERROR: Pioneer decision failed for {vehicle_id}: {e}")
            return {
                'selected_route': macro_candidates[0] if macro_candidates else [],
                'reasoning': f'Error in Pioneer decision: {e}',
                'confidence': 0.1,
                'decision_type': 'error_fallback',
                'timestamp': current_time
            }
    
    def _prepare_macro_context(self, state_context: Dict[str, Any], macro_candidates: List[List[int]]) -> Dict[str, Any]:
        """
        Prepare macro context specifically for Traffic LLM Pioneer decision.
        
        Following CLAUDE.md: 这个格式包含区域拥堵映射表、边界流量信息、预测数据等关键信息
        """
        regional_congestion = state_context['regional_congestion']
        boundary_flows = state_context['boundary_flows']
        traffic_predictions = state_context['traffic_predictions']
        
        # Smooth congestion values using exponential moving average
        smoothed_congestion = {}
        alpha = 0.7  # Smoothing factor
        for region_id, congestion in regional_congestion.items():
            # For now, just use current value (could enhance with historical data)
            smoothed_congestion[region_id] = congestion
        
        # Calculate boundary flow intensity
        boundary_flow_map = {}
        for edge_id, flow_info in boundary_flows.items():
            boundary_flow_map[edge_id] = {
                'intensity': flow_info['vehicle_count'],
                'from_region': flow_info['from_region'],
                'to_region': flow_info['to_region']
            }
        
        # Format prediction data for different time horizons
        formatted_predictions = {}
        for horizon, predictions in traffic_predictions.items():
            formatted_predictions[horizon] = predictions
        
        macro_context = {
            'regional_congestion_smoothed': smoothed_congestion,
            'boundary_flow_intensity': boundary_flow_map,
            'multi_horizon_predictions': formatted_predictions,
            'route_candidates': macro_candidates,
            'num_regions': len(smoothed_congestion),
            'context_type': 'macro_strategic'
        }
        
        return macro_context
    
    def _handle_stuck_vehicle_replanning(self, vehicle_id: str, current_time: float):
        """Handle replanning for stuck vehicles with improved safety checks and route validation."""
        try:
            # Check if vehicle still exists in simulation
            if vehicle_id not in traci.vehicle.getIDList():
                self.logger.log_warning(f"STUCK_REPLAN: Vehicle {vehicle_id} no longer exists, skipping replanning")
                return
            
            # Check if vehicle already being replanned recently to avoid spam
            if hasattr(self, '_vehicle_replan_times'):
                last_replan = self._vehicle_replan_times.get(vehicle_id, 0)
                if current_time - last_replan < 120:  # Avoid replanning same vehicle within 2 minutes
                    self.logger.log_info(f"STUCK_REPLAN: Vehicle {vehicle_id} was replanned recently, skipping")
                    return
            else:
                self._vehicle_replan_times = {}
            
            # Get vehicle's current position and route info
            try:
                current_edge = traci.vehicle.getRoadID(vehicle_id)
                original_route = traci.vehicle.getRoute(vehicle_id)
                speed = traci.vehicle.getSpeed(vehicle_id)
                
                self.logger.log_info(f"STUCK_REPLAN: Initiating replanning for stuck vehicle {vehicle_id} - "
                                   f"current_edge: {current_edge}, speed: {speed:.2f}, original_route_length: {len(original_route)}")
                
                # Skip if vehicle is on junction (internal edge)
                if current_edge.startswith(':'):
                    self.logger.log_warning(f"STUCK_REPLAN: Vehicle {vehicle_id} on junction edge {current_edge}, waiting for exit")
                    return
                    
            except Exception as pos_error:
                self.logger.log_error(f"STUCK_REPLAN: Cannot get position info for {vehicle_id}: {pos_error}")
                return
            
            # Check if this is an emergency route vehicle
            is_emergency_route = False
            original_dest_region = None
            if vehicle_id in self.vehicle_current_plans:
                plan = self.vehicle_current_plans[vehicle_id]
                is_emergency_route = plan.get('emergency_route', False)
                original_dest_region = plan.get('original_dest_region')
                
                if is_emergency_route:
                    self.logger.log_info(f"STUCK_REPLAN: {vehicle_id} is on emergency route, attempting to find path to original destination {original_dest_region}")
            
            # Clear existing plans and routes
            if vehicle_id in self.vehicle_current_plans:
                old_route = self.vehicle_current_plans[vehicle_id]['macro_route']
                self.logger.log_info(f"STUCK_REPLAN: Clearing old macro route {old_route} for {vehicle_id}")
                
                # Clean up old plan counts
                for region_id in old_route[1:] if len(old_route) > 1 else []:
                    if region_id in self.region_vehicle_plans:
                        self.region_vehicle_plans[region_id] = max(0, self.region_vehicle_plans[region_id] - 1)
                
                del self.vehicle_current_plans[vehicle_id]
            
            if vehicle_id in self.vehicle_regional_plans:
                del self.vehicle_regional_plans[vehicle_id]
            
            # Record replan time
            self._vehicle_replan_times[vehicle_id] = current_time
            
            # Try to create emergency route first
            try:
                dest_edge = original_route[-1] if original_route else None
                if dest_edge and current_edge != dest_edge:
                    # Find alternative route using SUMO's route finder
                    emergency_route = traci.simulation.findRoute(fromEdge=current_edge, toEdge=dest_edge)
                    if emergency_route and hasattr(emergency_route, 'edges') and len(emergency_route.edges) > 1:
                        emergency_route_list = list(emergency_route.edges)
                        
                        # Set emergency route directly
                        traci.vehicle.setRoute(vehicle_id, emergency_route_list)
                        
                        self.logger.log_info(f"STUCK_REPLAN: Applied emergency route for {vehicle_id} - "
                                           f"length: {len(emergency_route_list)}, travel_time: {emergency_route.travelTime:.1f}s")
                        
                        # Update vehicle plans with emergency route info
                        current_region = self.edge_to_region.get(current_edge, -1)
                        dest_region = self.edge_to_region.get(dest_edge, -1)
                        
                        self.vehicle_current_plans[vehicle_id] = {
                            'macro_route': [current_region, dest_region] if current_region != -1 and dest_region != -1 else [0, 0],
                            'detailed_route': emergency_route_list,
                            'emergency_route': True,
                            'original_dest_region': dest_region,
                            'replan_time': current_time,
                            'expected_travel_time': emergency_route.travelTime
                        }
                        
                        # Successful emergency reroute - return early
                        return
                        
                    else:
                        self.logger.log_warning(f"STUCK_REPLAN: No alternative route found from {current_edge} to {dest_edge}")
                        
            except Exception as emergency_error:
                self.logger.log_warning(f"STUCK_REPLAN: Emergency routing failed for {vehicle_id}: {emergency_error}")
            
            # If emergency routing failed, trigger macro planning as if vehicle just born
            self.logger.log_info(f"STUCK_REPLAN: Falling back to full replanning for {vehicle_id}")
            self.handle_vehicle_birth_macro_planning(vehicle_id, current_time)
            
            # Also trigger regional planning if vehicle has macro route now
            if vehicle_id in self.vehicle_regions and vehicle_id in self.vehicle_current_plans:
                current_region = self.vehicle_regions[vehicle_id]
                self.handle_vehicle_regional_planning(vehicle_id, current_region, current_time)
            
            self.logger.log_info(f"STUCK_REPLAN: Completed replanning for vehicle {vehicle_id}")
            
        except Exception as e:
            self.logger.log_error(f"STUCK_REPLAN: Failed for {vehicle_id}: {e}")
            import traceback
            self.logger.log_error(f"STUCK_REPLAN_TRACEBACK: {traceback.format_exc()}")
    
    def _call_traffic_llm_pioneer(self, decision_input: Dict[str, Any], macro_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call Traffic LLM for Pioneer decision making.
        
        Uses the enhanced macro route planning capability or falls back to hybrid decision making.
        """
        try:
            if hasattr(self.traffic_llm, 'macro_route_planning'):
                # Use enhanced LLM method
                global_state = {
                    'current_time': decision_input['current_time'],
                    'total_vehicles': decision_input['global_state']['total_vehicles'],
                    'regional_congestion': decision_input['regional_congestion'],
                    'boundary_congestion': macro_context['boundary_flow_intensity'],
                    'avg_travel_time': decision_input['global_state']['current_avg_travel_time']
                }
                
                route_requests = [{
                    'vehicle_id': decision_input['vehicle_id'],
                    'start_region': decision_input['start_region'],
                    'end_region': decision_input['dest_region'],
                    'possible_routes': decision_input['route_candidates'],
                    'route_urgency': 'normal',
                    'special_requirements': None
                }]
                
                regional_conditions = {}
                for region_id, congestion in decision_input['regional_congestion'].items():
                    regional_conditions[region_id] = {
                        'congestion_level': congestion,
                        'capacity_utilization': min(1.0, congestion / 5.0),
                        'vehicle_count': 0,  # Could be enhanced
                        'status': 'congested' if congestion > 3.0 else 'normal'
                    }
                
                llm_result = self.traffic_llm.macro_route_planning(
                    global_state=global_state,
                    route_requests=route_requests,
                    regional_conditions=regional_conditions,
                    boundary_analysis={},  # Simplified for now
                    flow_predictions={'time_horizon': 1800, 'regional_trend': 'stable'},
                    coordination_needs={'load_balancing_required': True},
                    region_routes={}
                )
                
                # Extract result
                macro_routes = llm_result.get('macro_routes', [])
                if macro_routes:
                    # Convert selected option number to actual route
                    selected_option = macro_routes[0].get('selected_route_option', macro_routes[0].get('planned_route', '1'))
                    selected_route = self._parse_route_option(selected_option, decision_input['route_candidates'])
                    
                    return {
                        'selected_route': selected_route,
                        'reasoning': macro_routes[0].get('reasoning', 'Strategic macro planning'),
                        'confidence': 0.8
                    }
            
            # Fallback to hybrid decision making
            observation_text = self._create_pioneer_observation(decision_input, macro_context)
            
            # Create simple option indices for LLM selection (1, 2, 3, etc.)
            num_candidates = len(decision_input['route_candidates'])
            answer_options = "/".join([str(i+1) for i in range(num_candidates)])
            
            decisions = self.traffic_llm.hybrid_decision_making_pipeline(
                [observation_text], [f'"{answer_options}"']
            )
            
            if decisions and len(decisions) > 0:
                decision = decisions[0]
                selected_option_str = decision.get('answer', '1')
                
                # Parse selected option number to actual route
                selected_route = self._parse_route_option(selected_option_str, decision_input['route_candidates'])
                
                return {
                    'selected_route': selected_route,
                    'reasoning': decision.get('summary', 'Hybrid decision making'),
                    'confidence': 0.7
                }
            
            return None
            
        except Exception as e:
            self.logger.log_error(f"CORY_TRAFFIC_LLM_ERROR: {e}")
            return None
    
    def _parse_route_option(self, selected_option, route_candidates):
        """Parse LLM selected option number to actual route."""
        try:
            # Handle different formats of option selection
            option_num = None
            
            if isinstance(selected_option, (list, tuple)):
                # If LLM returned the actual route, try to find it in candidates
                for i, candidate in enumerate(route_candidates):
                    if candidate == selected_option:
                        return candidate
                # If not found, use first option
                return route_candidates[0] if route_candidates else []
            
            elif isinstance(selected_option, str):
                # Parse option number from string
                if selected_option.isdigit():
                    option_num = int(selected_option)
                elif 'option' in selected_option.lower():
                    # Extract number from "Option1", "option2", etc.
                    import re
                    match = re.search(r'(\d+)', selected_option)
                    if match:
                        option_num = int(match.group(1))
                else:
                    # Try to parse as route directly
                    try:
                        import ast
                        parsed_route = ast.literal_eval(selected_option)
                        if isinstance(parsed_route, list):
                            # Check if this route exists in candidates
                            for candidate in route_candidates:
                                if candidate == parsed_route:
                                    return candidate
                    except:
                        pass
            
            elif isinstance(selected_option, int):
                option_num = selected_option
            
            # Convert option number to route (1-indexed)
            if option_num is not None:
                if 1 <= option_num <= len(route_candidates):
                    return route_candidates[option_num - 1]
            
            # Fallback to first candidate
            return route_candidates[0] if route_candidates else []
            
        except Exception as e:
            self.logger.log_error(f"ROUTE_OPTION_PARSE_ERROR: {e}, selected_option: {selected_option}")
            return route_candidates[0] if route_candidates else []
    
    def _create_pioneer_observation(self, decision_input: Dict[str, Any], macro_context: Dict[str, Any]) -> str:
        """Create compressed observation text for Pioneer LLM decision - PROMPT OPTIMIZATION."""
        # Compress vehicle and route info
        v_id = str(decision_input['vehicle_id'])[-4:]  # Use last 4 chars of vehicle ID
        s_reg = decision_input['start_region']
        d_reg = decision_input['dest_region']
        time_s = decision_input['current_time']
        
        observation = f"V{v_id} R{s_reg}->R{d_reg} T{time_s:.0f}s\n\nCONG:"
        
        # Compress congestion info - only show relevant regions
        relevant_regions = set([s_reg, d_reg])
        for route in decision_input['route_candidates']:
            relevant_regions.update(route)
        
        for region_id, congestion in decision_input['regional_congestion'].items():
            if region_id in relevant_regions:
                status = "H" if congestion > 3.0 else "M" if congestion > 1.5 else "L"
                observation += f" R{region_id}:{congestion:.1f}({status})"
        
        observation += f"\n\nOPTS:"
        for i, route in enumerate(decision_input['route_candidates']):
            observation += f" {i+1}:{route}"
        
        # Compress global state to essentials
        observation += f"\n\nGLB: V{decision_input['global_state']['total_vehicles']} ATT{decision_input['global_state']['current_avg_travel_time']:.0f}s"
        
        return observation
    
    def _generate_macro_route_candidates_original(self, start_region: int, dest_region: int, current_time: float) -> List[List[int]]:
        """Original macro route candidate generation for fallback."""
        return self._generate_macro_route_candidates(start_region, dest_region, current_time)
    
    def _generate_macro_route_candidates(self, start_region: int, dest_region: int, current_time: float, state_context: Dict[str, Any] = None) -> List[List[int]]:
        """
        Enhanced macro route candidate generation for CORY framework.
        Considers reachability, connectivity, congestion, distance, and state context.
        
        Args:
            state_context: Enhanced state information from CORY state construction
        """
        try:
            candidates = []
            
            # Direct route if possible
            if self.traffic_agent.region_graph.has_edge(start_region, dest_region):
                candidates.append([start_region, dest_region])
            
            # Find alternative routes using NetworkX
            try:
                # Get all simple paths up to length 5
                all_paths = list(nx.all_simple_paths(
                    self.traffic_agent.region_graph, 
                    start_region, 
                    dest_region, 
                    cutoff=5
                ))
                
                # Limit to best candidates based on different criteria
                for path in all_paths[:10]:  # Limit to 10 paths
                    if path not in candidates:
                        candidates.append(path)
                        
            except nx.NetworkXNoPath:
                # If no path found, try to supplement region connections and retry
                self.logger.log_warning(f"MACRO_CANDIDATES: No path found from region {start_region} to {dest_region}, attempting to supplement connections")
                
                # Try to supplement region connections
                self._supplement_region_connections_emergency(start_region, dest_region)
                
                # Retry pathfinding
                try:
                    all_paths = list(nx.all_simple_paths(
                        self.traffic_agent.region_graph, 
                        start_region, 
                        dest_region, 
                        cutoff=5
                    ))
                    
                    for path in all_paths[:10]:
                        if path not in candidates:
                            candidates.append(path)
                            
                except nx.NetworkXNoPath:
                    self.logger.log_error(f"MACRO_CANDIDATES: Still no path found from region {start_region} to {dest_region} after supplementing")
                    return []
            
            # Evaluate candidates based on multiple factors
            evaluated_candidates = []
            
            for candidate in candidates:
                score = self._evaluate_macro_route_candidate(candidate, current_time)
                evaluated_candidates.append((candidate, score))
            
            # Sort by score (higher is better) and return top candidates
            evaluated_candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = [route for route, score in evaluated_candidates[:5]]  # Top 5 candidates
            
            # Enhanced logging with state context if available
            if state_context:
                regional_congestion = state_context.get('regional_congestion', {})
                avg_congestion = sum(regional_congestion.get(r, 0) for route in top_candidates for r in route) / max(sum(len(route) for route in top_candidates), 1)
                self.logger.log_info(f"CORY_MACRO_CANDIDATES: Generated {len(top_candidates)} candidates for regions {start_region}->{dest_region}: {top_candidates} "
                                   f"(Avg route congestion: {avg_congestion:.3f})")
            else:
                self.logger.log_info(f"MACRO_CANDIDATES: Generated {len(top_candidates)} candidates for {start_region}->{dest_region}")
            
            return top_candidates
            
        except Exception as e:
            self.logger.log_error(f"CORY_MACRO_CANDIDATES_ERROR: Failed to generate candidates: {e}")
            return []
    
    def _observer_feedback(self, vehicle_id: str, state_context: Dict[str, Any], 
                         pioneer_result: Dict[str, Any], current_time: float) -> Dict[str, Any]:
        """
        Observer Feedback Process - Regional LLM as Observer.
        
        Following CLAUDE.md: Regional LLM担任Observer的角色，负责从区域协调和个体保护的角度
        对Pioneer的方案进行评估和改进。这个阶段的重点是局部优化和公平性考虑。
        
        多维度评估框架:
        - 可行性评估：检查Pioneer提出的路径在当前交通状况下是否真正可行
        - 效率评估：从区域交通管理的角度评估路径的效率
        - 公平性评估：评估Pioneer的决策是否可能导致个别车辆的过度牺牲
        
        Returns:
            Observer feedback with acceptance, improvements, conflicts, and suggestions
        """
        try:
            self.logger.log_info(f"CORY_OBSERVER: Starting Observer feedback for {vehicle_id}")
            
            # [Regional Context Construction] - 为Regional LLM创建输入
            regional_context = self._prepare_regional_context(state_context, pioneer_result)
            
            # [Multi-dimensional Evaluation Framework]
            # 1. Feasibility Assessment
            feasibility_score = self._assess_route_feasibility(
                pioneer_result['selected_route'], state_context
            )
            
            # 2. Efficiency Assessment 
            efficiency_score = self._assess_route_efficiency(
                pioneer_result['selected_route'], state_context
            )
            
            # 3. Fairness Assessment
            fairness_score = self._assess_route_fairness(
                pioneer_result['selected_route'], state_context
            )
            
            # [Improvement Suggestions Generation]
            improvements = self._generate_improvement_suggestions(
                pioneer_result, feasibility_score, efficiency_score, fairness_score
            )
            
            # [Conflict Identification & Resolution]
            conflicts = self._identify_conflicts(
                pioneer_result['selected_route'], state_context
            )
            
            # [Regional LLM Feedback Call] (带锁保护)
            with self.llm_call_lock:
                observer_llm_result = self._call_regional_llm_observer(
                    vehicle_id, regional_context, pioneer_result, 
                    feasibility_score, efficiency_score, fairness_score,
                    improvements, conflicts, current_time
                )
            
            # [Compile Observer Result]
            observer_result = {
                'acceptance': observer_llm_result.get('acceptance', True),
                'feasibility_score': feasibility_score,
                'efficiency_score': efficiency_score,
                'fairness_score': fairness_score,
                'improvements': improvements,
                'conflicts': conflicts,
                'llm_feedback': observer_llm_result,
                'refined_routes': observer_llm_result.get('refined_routes', []),
                'observer_reasoning': observer_llm_result.get('reasoning', 'Observer evaluation completed'),
                'timestamp': current_time
            }
            
            self.logger.log_info(f"CORY_OBSERVER_SUCCESS: Observer feedback completed for {vehicle_id} - "
                               f"Acceptance: {observer_result['acceptance']}, "
                               f"Scores: F={feasibility_score:.2f}, E={efficiency_score:.2f}, Fa={fairness_score:.2f}, "
                               f"Improvements: {len(improvements)}, Conflicts: {len(conflicts)}")
            
            return observer_result
            
        except Exception as e:
            self.logger.log_error(f"CORY_OBSERVER_ERROR: Observer feedback failed for {vehicle_id}: {e}")
            return {
                'acceptance': True,  # Default to acceptance on error
                'feasibility_score': 0.5,
                'efficiency_score': 0.5,
                'fairness_score': 0.5,
                'improvements': [],
                'conflicts': [],
                'error': str(e),
                'timestamp': current_time
            }
    
    def _prepare_regional_context(self, state_context: Dict[str, Any], pioneer_result: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare regional context for Observer evaluation."""
        selected_route = pioneer_result['selected_route']
        
        # Get detailed regional information for the selected route
        route_regional_details = {}
        for region_id in selected_route:
            route_regional_details[region_id] = {
                'congestion_level': state_context['regional_congestion'].get(region_id, 0),
                'region_edges': [edge for edge, reg in self.edge_to_region.items() if reg == region_id],
                'expected_vehicles': 0  # Could be enhanced with actual counts
            }
        
        # Calculate route-specific metrics
        route_length = len(selected_route)
        route_congestion_sum = sum(state_context['regional_congestion'].get(r, 0) for r in selected_route)
        route_avg_congestion = route_congestion_sum / max(route_length, 1)
        
        regional_context = {
            'pioneer_selected_route': selected_route,
            'pioneer_reasoning': pioneer_result.get('reasoning', ''),
            'pioneer_confidence': pioneer_result.get('confidence', 0.5),
            'route_regional_details': route_regional_details,
            'route_metrics': {
                'length': route_length,
                'avg_congestion': route_avg_congestion,
                'total_congestion': route_congestion_sum
            },
            'alternative_routes': pioneer_result.get('alternative_routes', []),
            'context_type': 'regional_observer'
        }
        
        return regional_context
    
    def _assess_route_feasibility(self, selected_route: List[int], state_context: Dict[str, Any]) -> float:
        """Assess the feasibility of the selected route."""
        if not selected_route:
            return 0.0
        
        feasibility_score = 1.0
        
        # Check if route is physically connected
        for i in range(len(selected_route) - 1):
            current_region = selected_route[i]
            next_region = selected_route[i + 1]
            
            # Check if regions are connected
            if not self.traffic_agent.region_graph.has_edge(current_region, next_region):
                feasibility_score *= 0.5  # Penalize disconnected regions
        
        # Check congestion levels (heavily congested regions reduce feasibility)
        for region_id in selected_route:
            congestion = state_context['regional_congestion'].get(region_id, 0)
            if congestion > 4.0:  # Very high congestion
                feasibility_score *= 0.8
            elif congestion > 2.0:  # Moderate congestion
                feasibility_score *= 0.9
        
        return max(feasibility_score, 0.1)  # Minimum feasibility
    
    def _assess_route_efficiency(self, selected_route: List[int], state_context: Dict[str, Any]) -> float:
        """Assess the efficiency of the selected route from regional management perspective."""
        if not selected_route:
            return 0.0
        
        # Calculate route efficiency based on length and congestion
        route_length = len(selected_route)
        total_congestion = sum(state_context['regional_congestion'].get(r, 0) for r in selected_route)
        avg_congestion = total_congestion / max(route_length, 1)
        
        # Efficiency is higher for shorter routes with less congestion
        length_efficiency = max(0, 1.0 - (route_length - 2) * 0.1)  # Penalize long routes
        congestion_efficiency = max(0, 1.0 - avg_congestion * 0.2)  # Penalize congested routes
        
        efficiency_score = (length_efficiency + congestion_efficiency) / 2
        return max(efficiency_score, 0.1)
    
    def _assess_route_fairness(self, selected_route: List[int], state_context: Dict[str, Any]) -> float:
        """Assess fairness - whether the route might cause excessive individual sacrifice."""
        if not selected_route:
            return 1.0  # Neutral fairness for empty route
        
        # Estimate expected travel time based on route and congestion
        route_length = len(selected_route)
        total_congestion = sum(state_context['regional_congestion'].get(r, 0) for r in selected_route)
        
        # Intelligent fairness assessment without hard-coded time estimates
        # Instead of estimating travel time, assess route characteristics directly
        
        # Route complexity assessment
        complexity_score = min(1.0, 3.0 / max(1, route_length))  # Shorter routes score higher
        
        # Congestion resilience (routes through less congested areas score higher)
        if selected_route:
            avg_congestion = total_congestion / len(selected_route)
            congestion_score = max(0.0, 1.0 - avg_congestion)
        else:
            congestion_score = 0.5
        
        # Weighted fairness score combining complexity and congestion factors
        fairness_score = 0.6 * complexity_score + 0.4 * congestion_score
        
        return fairness_score
    
    def _generate_improvement_suggestions(self, pioneer_result: Dict[str, Any], 
                                        feasibility_score: float, efficiency_score: float, 
                                        fairness_score: float) -> List[Dict[str, Any]]:
        """Generate improvement suggestions based on assessment scores."""
        improvements = []
        
        if feasibility_score < 0.7:
            improvements.append({
                'type': 'feasibility',
                'suggestion': 'Consider alternative routes due to connectivity or high congestion issues',
                'priority': 'high'
            })
        
        if efficiency_score < 0.6:
            improvements.append({
                'type': 'efficiency',
                'suggestion': 'Route may be suboptimal - consider shorter or less congested alternatives',
                'priority': 'medium'
            })
        
        if fairness_score < 0.8:
            improvements.append({
                'type': 'fairness',
                'suggestion': 'Route may cause excessive individual travel time - consider individual protection',
                'priority': 'high'
            })
        
        return improvements
    
    def _identify_conflicts(self, selected_route: List[int], state_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential conflicts in the selected route."""
        conflicts = []
        
        # Intelligent conflict detection based on route characteristics
        route_length = len(selected_route)
        congestion_levels = [state_context['regional_congestion'].get(r, 0) for r in selected_route]
        avg_congestion = sum(congestion_levels) / len(congestion_levels) if congestion_levels else 0
        max_congestion = max(congestion_levels) if congestion_levels else 0
        
        # Route complexity analysis
        is_complex_route = route_length > 5
        has_high_congestion = max_congestion > 0.8
        has_sustained_congestion = sum(1 for c in congestion_levels if c > 0.6) > 2
        
        # Only flag conflicts for genuinely problematic routes
        if is_complex_route and has_high_congestion and has_sustained_congestion:
            conflicts.append({
                'type': 'route_complexity_conflict',
                'description': f'Complex route ({route_length} regions) with sustained high congestion (max: {max_congestion:.2f}, avg: {avg_congestion:.2f})',
                'severity': 'medium',
                'resolution_needed': False  # Let ATT reward handle optimization naturally
            })
        elif has_high_congestion and route_length > 3:
            conflicts.append({
                'type': 'congestion_concern',
                'description': f'Route through high congestion areas (max: {max_congestion:.2f}) with {route_length} regions',
                'severity': 'low', 
                'resolution_needed': False
            })
        
        return conflicts
    
    def _call_regional_llm_observer(self, vehicle_id: str, regional_context: Dict[str, Any],
                                  pioneer_result: Dict[str, Any], feasibility_score: float,
                                  efficiency_score: float, fairness_score: float,
                                  improvements: List[Dict], conflicts: List[Dict],
                                  current_time: float) -> Dict[str, Any]:
        """Call Regional LLM for Observer feedback."""
        try:
            # Create regional observation text
            observation = self._create_observer_observation(
                vehicle_id, regional_context, pioneer_result,
                feasibility_score, efficiency_score, fairness_score,
                improvements, conflicts
            )
            
            # Use Regional LLM for evaluation - with logging
            answer_options = "Accept/Reject/Modify"
            
            # Log LLM call start
            call_id = self.logger.log_llm_call_start(
                "ObserverFeedback", vehicle_id, len(observation)
            )
            
            try:
                decisions = self.regional_llm.hybrid_decision_making_pipeline(
                    [observation], [f'"{answer_options}"']
                )
                
                if decisions and len(decisions) > 0:
                    decision = decisions[0]
                    answer = decision.get('answer', 'accept')
                    # Safe handling of potentially None answer
                    answer_str = str(answer).lower() if answer is not None else 'accept'
                    acceptance = 'accept' in answer_str
                    
                    reasoning = decision.get('summary', 'Regional evaluation completed')
                    
                    # Log successful LLM call
                    self.logger.log_llm_call_end(
                        call_id, True, f"Decision: {answer_str}. Reasoning: {reasoning}",
                        len(observation)
                    )
                    
                    return {
                        'acceptance': acceptance,
                        'reasoning': reasoning,
                        'refined_routes': [],
                        'decision_response': decision
                    }
                else:
                    # Log failed LLM call
                    self.logger.log_llm_call_end(
                        call_id, False, "LLM returned empty response",
                        len(observation), "Empty decisions list"
                    )
                    
                    # Default acceptance
                    return {
                        'acceptance': True,
                        'reasoning': 'Default acceptance (LLM call failed)',
                        'refined_routes': []
                    }
                    
            except Exception as llm_error:
                # Log failed LLM call
                self.logger.log_llm_call_end(
                    call_id, False, "LLM call exception",
                    len(observation), str(llm_error)
                )
                
                # Default acceptance on error
                return {
                    'acceptance': True,
                    'reasoning': f'Default acceptance (LLM error: {llm_error})',
                    'refined_routes': []
                }
            
        except Exception as e:
            self.logger.log_error(f"CORY_REGIONAL_LLM_ERROR: {e}")
            return {
                'acceptance': True,  # Default to acceptance on error
                'reasoning': f'Error in regional LLM call: {e}',
                'refined_routes': []
            }
    
    def _create_observer_observation(self, vehicle_id: str, regional_context: Dict[str, Any],
                                   pioneer_result: Dict[str, Any], feasibility_score: float,
                                   efficiency_score: float, fairness_score: float,
                                   improvements: List[Dict], conflicts: List[Dict]) -> str:
        """Create observation text for Observer LLM evaluation."""
        observation = f"""OBSERVER REGIONAL EVALUATION:
        
        Vehicle: {vehicle_id}
        Pioneer Selected Route: {regional_context['pioneer_selected_route']}
        Pioneer Reasoning: {pioneer_result.get('reasoning', 'Not provided')}
        Pioneer Confidence: {pioneer_result.get('confidence', 0.5):.3f}
        
        ROUTE ASSESSMENT SCORES:
        Feasibility: {feasibility_score:.3f} {'(PASS)' if feasibility_score > 0.7 else '(FAIL)'}
        Efficiency: {efficiency_score:.3f} {'(PASS)' if efficiency_score > 0.6 else '(FAIL)'}
        Fairness: {fairness_score:.3f} {'(PASS)' if fairness_score > 0.8 else '(FAIL)'}
        
        REGIONAL DETAILS:
        """
        
        for region_id, details in regional_context['route_regional_details'].items():
            observation += f"Region {region_id}: Congestion {details['congestion_level']:.3f}\n"
        
        if improvements:
            observation += f"\nIMPROVEMENT SUGGESTIONS ({len(improvements)}):\n"
            for imp in improvements:
                observation += f"- {imp['type'].upper()}: {imp['suggestion']}\n"
        
        if conflicts:
            observation += f"\nCONFLICTS IDENTIFIED ({len(conflicts)}):\n"
            for conflict in conflicts:
                observation += f"- {conflict['type'].upper()}: {conflict['description']}\n"
        
        observation += "\n\nDECISION GUIDELINES:\n"
        observation += "- Accept: When all scores are PASS (feasibility > 0.7, efficiency > 0.6, fairness > 0.8)\n"
        observation += "- Accept: When most scores are PASS and conflicts are minor or resolvable\n"
        observation += "- Reject: Only when critical failures exist (feasibility FAIL or severe fairness violations)\n"
        observation += "- Modify: When improvements can significantly enhance the route\n"
        observation += "\nPlease evaluate and provide acceptance decision (Accept/Reject/Modify)."
        
        return observation
    
    def _j1_judge_evaluation(self, vehicle_id: str, pioneer_result: Dict[str, Any], 
                           observer_result: Dict[str, Any], current_time: float) -> Dict[str, Any]:
        """
        J1-Judge Quality Evaluation - Core component of CORY framework.
        
        Following CLAUDE.md: J1-Judge是整个CORY框架中的关键组件，负责评估Pioneer和Observer之间合作的质量。
        
        评估维度:
        - 一致性分数: Pioneer决策和Observer反馈之间的协调程度
        - 改进分数: Observer提出的改进建议数量和质量
        - 冲突解决分数: Observer识别和解决冲突的能力
        
        Returns:
            Quality evaluation with consistency, improvement, conflict resolution scores
        """
        try:
            self.logger.log_info(f"CORY_J1_JUDGE: Starting quality evaluation for {vehicle_id}")
            
            # [Consistency Score Calculation] - 一致性分数衡量Pioneer决策和Observer反馈之间的协调程度
            consistency_score = self._calculate_consistency_score(
                pioneer_result, observer_result
            )
            
            # [Improvement Score Calculation] - 改进分数基于Observer提出的改进建议数量和质量
            improvement_score = self._calculate_improvement_score(
                observer_result.get('improvements', [])
            )
            
            # [Conflict Resolution Score] - 冲突解决分数衡量Observer识别和解决冲突的能力
            conflict_resolution_score = self._calculate_conflict_resolution_score(
                observer_result.get('conflicts', [])
            )
            
            # [Comprehensive Quality Score] - 综合质量分数通过加权求和得到
            # 权重分配: 一致性(0.4) + 改进能力(0.3) + 冲突解决(0.3)
            cooperation_quality = (
                0.4 * consistency_score + 
                0.3 * improvement_score + 
                0.3 * conflict_resolution_score
            )
            
            # [Quality Assessment Details]
            quality_details = {
                'consistency_analysis': self._analyze_consistency(pioneer_result, observer_result),
                'improvement_analysis': self._analyze_improvements(observer_result.get('improvements', [])),
                'conflict_analysis': self._analyze_conflicts(observer_result.get('conflicts', []))
            }
            
            j1_judge_result = {
                'cooperation_quality': cooperation_quality,
                'consistency_score': consistency_score,
                'improvement_score': improvement_score,
                'conflict_resolution_score': conflict_resolution_score,
                'quality_details': quality_details,
                'evaluation_timestamp': current_time,
                'quality_level': self._determine_quality_level(cooperation_quality)
            }
            
            self.logger.log_info(f"CORY_J1_JUDGE_SUCCESS: Quality evaluation completed for {vehicle_id} - "
                               f"Overall Quality: {cooperation_quality:.3f} ({j1_judge_result['quality_level']}), "
                               f"Consistency: {consistency_score:.3f}, Improvement: {improvement_score:.3f}, "
                               f"Conflict Resolution: {conflict_resolution_score:.3f}")
            
            return j1_judge_result
            
        except Exception as e:
            self.logger.log_error(f"CORY_J1_JUDGE_ERROR: Quality evaluation failed for {vehicle_id}: {e}")
            return {
                'cooperation_quality': 0.5,  # Default medium quality on error
                'consistency_score': 0.5,
                'improvement_score': 0.5,
                'conflict_resolution_score': 0.5,
                'error': str(e),
                'evaluation_timestamp': current_time,
                'quality_level': 'MEDIUM'
            }
    
    def _calculate_consistency_score(self, pioneer_result: Dict[str, Any], observer_result: Dict[str, Any]) -> float:
        """Calculate consistency score between Pioneer and Observer."""
        # Check Observer's acceptance of Pioneer decision
        acceptance = observer_result.get('acceptance', True)
        
        if acceptance:
            # Complete acceptance
            return 0.9
        else:
            # Check if there are minor adjustments suggested
            improvements = observer_result.get('improvements', [])
            minor_improvements = [imp for imp in improvements if imp.get('priority') in ['low', 'medium']]
            major_improvements = [imp for imp in improvements if imp.get('priority') == 'high']
            
            if len(major_improvements) == 0 and len(minor_improvements) > 0:
                # Minor adjustments only
                return 0.7
            elif len(major_improvements) > 0:
                # Major revisions needed
                return 0.4
            else:
                # Rejection without clear improvements
                return 0.2
    
    def _calculate_improvement_score(self, improvements: List[Dict[str, Any]]) -> float:
        """Calculate improvement score based on suggestions."""
        if not improvements:
            return 0.3  # Base score for no suggestions
        
        # Score based on number and quality of improvements
        # Formula: min(0.3 + 0.1 * improvement_count, 1.0)
        improvement_count = len(improvements)
        improvement_score = min(0.3 + 0.1 * improvement_count, 1.0)
        
        # Adjust based on priority of improvements
        high_priority_count = len([imp for imp in improvements if imp.get('priority') == 'high'])
        if high_priority_count > 0:
            improvement_score = min(improvement_score + 0.1 * high_priority_count, 1.0)
        
        return improvement_score
    
    def _calculate_conflict_resolution_score(self, conflicts: List[Dict[str, Any]]) -> float:
        """Calculate conflict resolution score."""
        if not conflicts:
            # No conflicts identified - could be good or indicate poor analysis
            return 0.8
        
        # Score based on conflict identification and resolution
        resolved_conflicts = [c for c in conflicts if c.get('resolution_needed', False)]
        
        if len(resolved_conflicts) == 0:
            # All conflicts resolved
            return 1.0
        else:
            # Partial resolution
            total_conflicts = len(conflicts)
            resolved_count = len(conflicts) - len(resolved_conflicts)
            return max(0.0, resolved_count / total_conflicts)
    
    def _analyze_consistency(self, pioneer_result: Dict[str, Any], observer_result: Dict[str, Any]) -> str:
        """Analyze consistency between Pioneer and Observer decisions."""
        acceptance = observer_result.get('acceptance', True)
        if acceptance:
            return "High consistency - Observer accepts Pioneer decision"
        else:
            improvements = observer_result.get('improvements', [])
            return f"Low consistency - Observer suggests {len(improvements)} improvements"
    
    def _analyze_improvements(self, improvements: List[Dict[str, Any]]) -> str:
        """Analyze improvement suggestions."""
        if not improvements:
            return "No improvement suggestions provided"
        
        high_priority = len([imp for imp in improvements if imp.get('priority') == 'high'])
        medium_priority = len([imp for imp in improvements if imp.get('priority') == 'medium'])
        low_priority = len([imp for imp in improvements if imp.get('priority') == 'low'])
        
        return f"Improvements suggested: {high_priority} high, {medium_priority} medium, {low_priority} low priority"
    
    def _analyze_conflicts(self, conflicts: List[Dict[str, Any]]) -> str:
        """Analyze conflict identification and resolution."""
        if not conflicts:
            return "No conflicts identified"
        
        resolved = len([c for c in conflicts if not c.get('resolution_needed', True)])
        unresolved = len(conflicts) - resolved
        
        return f"Conflicts: {len(conflicts)} total, {resolved} resolved, {unresolved} unresolved"
    
    def _determine_quality_level(self, cooperation_quality: float) -> str:
        """Determine quality level based on cooperation quality score."""
        if cooperation_quality >= 0.8:
            return "HIGH"
        elif cooperation_quality >= 0.6:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _synthesize_final_decision(self, vehicle_id: str, pioneer_result: Dict[str, Any],
                                 observer_result: Dict[str, Any], j1_judge_result: Dict[str, Any],
                                 current_time: float) -> Dict[str, Any]:
        """
        Synthesize final decision based on Pioneer-Observer cooperation and J1-Judge evaluation.
        
        Following CLAUDE.md: 在获得合作质量分数后，系统通过_synthesize_final_decision函数生成最终决策。
        
        决策策略:
        - 高质量合作(>=0.7): 优先采用Observer优化的决策
        - 低质量合作(<0.7): 谨慎处理Observer建议，主要使用Pioneer决策
        
        Returns:
            Final synthesized decision with route, quality, and metadata
        """
        try:
            self.logger.log_info(f"CORY_SYNTHESIZE: Starting final decision synthesis for {vehicle_id}")
            
            cooperation_quality = j1_judge_result.get('cooperation_quality', 0.5)
            
            # [High Quality Cooperation] - cooperation_quality >= 0.7
            if cooperation_quality >= 0.7:
                self.logger.log_info(f"CORY_SYNTHESIZE: High quality cooperation detected ({cooperation_quality:.3f})")
                
                # Check if Observer provided refined routes
                refined_routes = observer_result.get('refined_routes', [])
                if refined_routes and len(refined_routes) > 0:
                    final_route = refined_routes[0]  # Use first refined route
                    decision_rationale = "Observer-optimized route selected (high cooperation quality)"
                else:
                    final_route = pioneer_result['selected_route']
                    decision_rationale = "Pioneer route confirmed by Observer (high cooperation quality)"
                
                # Include Observer's improvements and reasoning
                synthesis_details = {
                    'synthesis_type': 'high_quality_cooperation',
                    'observer_improvements_applied': len(observer_result.get('improvements', [])),
                    'conflicts_resolved': len([c for c in observer_result.get('conflicts', []) if not c.get('resolution_needed', True)]),
                    'observer_acceptance': observer_result.get('acceptance', True)
                }
                
            # [Low Quality Cooperation] - cooperation_quality < 0.7
            else:
                self.logger.log_info(f"CORY_SYNTHESIZE: Low quality cooperation detected ({cooperation_quality:.3f})")
                
                # Use Pioneer's decision as primary, but record Observer feedback
                final_route = pioneer_result['selected_route']
                decision_rationale = "Pioneer route maintained (low cooperation quality - Observer feedback recorded for reference)"
                
                synthesis_details = {
                    'synthesis_type': 'low_quality_cooperation',
                    'observer_concerns': len(observer_result.get('improvements', [])),
                    'unresolved_conflicts': len([c for c in observer_result.get('conflicts', []) if c.get('resolution_needed', True)]),
                    'fallback_to_pioneer': True
                }
            
            # [Decision Confidence] - 基于合作质量传递置信度
            final_confidence = cooperation_quality
            
            # [Compile Final Result]
            final_result = {
                'final_route': final_route,
                'cooperation_quality': cooperation_quality,
                'final_confidence': final_confidence,
                'decision_rationale': decision_rationale,
                'synthesis_details': synthesis_details,
                'pioneer_decision': pioneer_result,
                'observer_feedback': observer_result,
                'j1_judge_evaluation': j1_judge_result,
                'synthesis_timestamp': current_time,
                'cory_framework_version': '1.0'
            }
            
            self.logger.log_info(f"CORY_SYNTHESIZE_SUCCESS: Final decision synthesized for {vehicle_id} - "
                               f"Route: {final_route}, Quality: {cooperation_quality:.3f}, "
                               f"Confidence: {final_confidence:.3f}, Type: {synthesis_details['synthesis_type']}")
            
            return final_result
            
        except Exception as e:
            self.logger.log_error(f"CORY_SYNTHESIZE_ERROR: Final decision synthesis failed for {vehicle_id}: {e}")
            # Fallback to Pioneer decision
            return {
                'final_route': pioneer_result.get('selected_route', []),
                'cooperation_quality': 0.1,
                'final_confidence': 0.1,
                'decision_rationale': f'Error in synthesis - fallback to Pioneer: {e}',
                'synthesis_details': {'synthesis_type': 'error_fallback'},
                'pioneer_decision': pioneer_result,
                'observer_feedback': observer_result,
                'j1_judge_evaluation': j1_judge_result,
                'synthesis_timestamp': current_time,
                'error': str(e)
            }
    
    def _supplement_region_connections_emergency(self, start_region: int, dest_region: int):
        """Emergency supplement of region connections when pathfinding fails."""
        try:
            # Get all edges in the start and destination regions
            start_edges = [edge_id for edge_id, region in self.edge_to_region.items() if region == start_region]
            dest_edges = [edge_id for edge_id, region in self.edge_to_region.items() if region == dest_region]
            
            supplemented = 0
            
            # Check connections from start region edges using TRACI
            for edge_id in start_edges:
                if edge_id.startswith(':'):
                    continue
                    
                try:
                    # Get the junction at the end of this edge
                    to_junction = traci.edge.getToJunction(edge_id)
                    if to_junction:
                        # Get all outgoing edges from this junction
                        outgoing_edges = traci.junction.getOutgoingEdges(to_junction)
                        
                        for target_edge in outgoing_edges:
                            if target_edge.startswith(':'):
                                continue
                                
                            target_region = self.edge_to_region.get(target_edge)
                            if target_region is None or target_region == start_region:
                                continue
                            
                            # Add connection to region graph
                            if not self.traffic_agent.region_graph.has_edge(start_region, target_region):
                                self.traffic_agent.region_graph.add_edge(start_region, target_region, weight=1.0, edges=[])
                                supplemented += 1
                                self.logger.log_info(f"EMERGENCY_SUPPLEMENT: Added connection {start_region} -> {target_region}")
                                
                except Exception:
                    continue
            
            # Also check for paths through intermediate regions
            all_regions = list(range(self.num_regions))
            for intermediate_region in all_regions:
                if intermediate_region == start_region or intermediate_region == dest_region:
                    continue
                    
                # Check if we can connect start -> intermediate -> dest
                if (self.traffic_agent.region_graph.has_edge(start_region, intermediate_region) and 
                    self.traffic_agent.region_graph.has_edge(intermediate_region, dest_region)):
                    # Path already exists through this intermediate
                    continue
                    
                # Try to find real network connections
                intermediate_edges = [edge_id for edge_id, region in self.edge_to_region.items() if region == intermediate_region]
                
                # Check start -> intermediate using TRACI
                if not self.traffic_agent.region_graph.has_edge(start_region, intermediate_region):
                    for start_edge in start_edges:
                        if start_edge.startswith(':'):
                            continue
                        try:
                            to_junction = traci.edge.getToJunction(start_edge)
                            if to_junction:
                                outgoing = traci.junction.getOutgoingEdges(to_junction)
                                for out_edge in outgoing:
                                    if out_edge in intermediate_edges and not out_edge.startswith(':'):
                                        self.traffic_agent.region_graph.add_edge(start_region, intermediate_region, weight=1.0, edges=[])
                                        supplemented += 1
                                        self.logger.log_info(f"EMERGENCY_SUPPLEMENT: Added intermediate connection {start_region} -> {intermediate_region}")
                                        break
                        except:
                            continue
                
                # Check intermediate -> dest using TRACI
                if not self.traffic_agent.region_graph.has_edge(intermediate_region, dest_region):
                    for int_edge in intermediate_edges:
                        if int_edge.startswith(':'):
                            continue
                        try:
                            to_junction = traci.edge.getToJunction(int_edge)
                            if to_junction:
                                outgoing = traci.junction.getOutgoingEdges(to_junction)
                                for out_edge in outgoing:
                                    if out_edge in dest_edges and not out_edge.startswith(':'):
                                        self.traffic_agent.region_graph.add_edge(intermediate_region, dest_region, weight=1.0, edges=[])
                                        supplemented += 1
                                        self.logger.log_info(f"EMERGENCY_SUPPLEMENT: Added intermediate connection {intermediate_region} -> {dest_region}")
                                        break
                        except:
                            continue
            
            self.logger.log_info(f"EMERGENCY_SUPPLEMENT: Added {supplemented} emergency connections for {start_region} -> {dest_region}")
            
        except Exception as e:
            self.logger.log_error(f"EMERGENCY_SUPPLEMENT: Failed: {e}")
    
    def _evaluate_macro_route_candidate(self, route: List[int], current_time: float) -> float:
        """
        Evaluate macro route candidate based on reachability, connectivity, congestion, distance.
        """
        try:
            score = 100.0  # Base score
            
            # Factor 1: Distance (shorter routes preferred)
            distance_penalty = (len(route) - 2) * 10  # Penalty for longer routes
            score -= distance_penalty
            
            # Factor 2: Congestion levels in target regions
            congestion_penalty = 0
            current_state = self.traffic_agent.global_state_history[-1] if self.traffic_agent.global_state_history else None
            
            if current_state:
                for region_id in route[1:]:  # Skip start region
                    region_congestion = current_state.regional_congestion.get(region_id, 0)
                    congestion_penalty += region_congestion * 5  # 5 points penalty per congestion level
            
            score -= congestion_penalty
            
            # Factor 3: Boundary edge congestion
            boundary_penalty = 0
            for i in range(len(route) - 1):
                from_region = route[i]
                to_region = route[i + 1]
                
                # Find boundary edges between these regions
                boundary_edges = self.traffic_agent.region_connections.get((from_region, to_region), [])
                if boundary_edges and current_state:
                    avg_boundary_congestion = sum(
                        current_state.boundary_congestion.get(edge, 0) for edge in boundary_edges
                    ) / len(boundary_edges)
                    boundary_penalty += avg_boundary_congestion * 3  # 3 points penalty per boundary congestion level
            
            score -= boundary_penalty
            
            # Factor 4: Current planned usage (avoid overloading regions)
            usage_penalty = 0
            for region_id in route[1:]:
                planned_vehicles = self.region_vehicle_plans.get(region_id, 0)
                usage_penalty += planned_vehicles * 2  # 2 points penalty per planned vehicle
            
            score -= usage_penalty
            
            # Factor 5: Connectivity bonus (well-connected regions)
            connectivity_bonus = 0
            for region_id in route[1:]:
                # Count outgoing connections from this region
                outgoing_connections = len([
                    edge for edge in self.traffic_agent.region_graph.edges()
                    if edge[0] == region_id
                ])
                connectivity_bonus += outgoing_connections * 1  # 1 point bonus per connection
            
            score += connectivity_bonus
            
            return max(0, score)  # Ensure non-negative score
            
        except Exception as e:
            self.logger.log_error(f"ROUTE_EVALUATION: Failed to evaluate route {route}: {e}")
            return 0.0
    
    def _llm_select_macro_route(self, vehicle_id: str, start_region: int, dest_region: int, 
                               candidates: List[List[int]], current_time: float) -> Optional[List[int]]:
        """
        Async-enabled LLM macro route selection with caching.
        """
        return self._async_llm_call_macro_route(vehicle_id, start_region, dest_region, candidates, current_time)
    
    def _llm_select_macro_route_sync(self, vehicle_id: str, start_region: int, dest_region: int, 
                               candidates: List[List[int]], current_time: float) -> Optional[List[int]]:
        """
        Use LLM to select optimal macro route from candidates.
        """
        try:
            # Prepare data for LLM decision
            route_descriptions = []
            route_data = []
            
            for i, route in enumerate(candidates):
                # Create detailed description for this route
                route_desc = self._create_macro_route_description(route, current_time)
                route_descriptions.append(route_desc)
                route_data.append(route)
            
            # Create observation text for LLM
            observation_text = self._create_macro_planning_observation(
                vehicle_id, start_region, dest_region, route_descriptions, current_time
            )
            
            # Create answer options
            answer_options = "/".join([str(route) for route in candidates])
            
            # Use LLM for decision making
            call_id = self.logger.log_llm_call_start(
                "MacroPlanning", vehicle_id, len(observation_text)
            )
            
            try:
                # Use enhanced LLM method if available
                if hasattr(self.llm_agent, 'macro_route_planning'):
                    # Prepare structured data for enhanced LLM
                    global_state = self._get_current_global_state()
                    route_requests = [{
                        'vehicle_id': vehicle_id,
                        'start_region': start_region,
                        'end_region': dest_region,
                        'possible_routes': candidates,
                        'route_urgency': 'normal',
                        'special_requirements': None
                    }]
                    
                    regional_conditions = self._get_regional_conditions()
                    boundary_analysis = self._get_boundary_analysis()
                    flow_predictions = {'time_horizon': 1800, 'regional_trend': 'stable'}
                    coordination_needs = {'load_balancing_required': True}
                    
                    region_routes = {}
                    for route in candidates:
                        route_key = f"{start_region}-{dest_region}"
                        region_routes[route_key] = {
                            'available_routes': candidates,
                            'recommended_route': route,
                            'route_quality': 'optimal'
                        }
                    
                    llm_result = self.llm_agent.macro_route_planning(
                        global_state=global_state,
                        route_requests=route_requests,
                        regional_conditions=regional_conditions,
                        boundary_analysis=boundary_analysis,
                        flow_predictions=flow_predictions,
                        coordination_needs=coordination_needs,
                        region_routes=region_routes
                    )
                    
                    # Extract selected route from LLM result
                    macro_routes = llm_result.get('macro_routes', [])
                    if macro_routes:
                        selected_route = macro_routes[0].get('planned_route', candidates[0])
                        reasoning = macro_routes[0].get('reasoning', 'LLM macro planning decision')
                    else:
                        selected_route = candidates[0]
                        reasoning = 'Fallback to first candidate'
                        
                else:
                    # Use basic LLM decision making if available
                    if self.llm_agent and hasattr(self.llm_agent, 'hybrid_decision_making_pipeline'):
                        decisions = self.llm_agent.hybrid_decision_making_pipeline(
                            [observation_text], [f'"{answer_options}"']
                        )
                        
                        if decisions and decisions[0]['answer']:
                            # Parse the selected route
                            selected_route = self._parse_macro_route_answer(decisions[0]['answer'], candidates)
                            reasoning = decisions[0].get('summary', 'LLM selection decision')
                        else:
                            selected_route = candidates[0]  # Fallback to first candidate
                            reasoning = 'Fallback decision - LLM failed'
                    else:
                        selected_route = candidates[0]  # Fallback when llm_agent is None
                        reasoning = 'Fallback decision - LLM agent not available'
                
                self.logger.log_llm_call_end(
                    call_id, True, f"Selected route: {selected_route}. Reasoning: {reasoning}", 
                    len(observation_text)
                )
                
                return selected_route
                
            except Exception as llm_error:
                self.logger.log_llm_call_end(
                    call_id, False, "LLM macro route selection failed", 
                    len(observation_text), str(llm_error)
                )
                
                # Fallback to heuristic selection (best scored candidate)
                return candidates[0] if candidates else None
                
        except Exception as e:
            self.logger.log_error(f"LLM_MACRO_SELECT: Failed for {vehicle_id}: {e}")
            return candidates[0] if candidates else None
    
    def _create_macro_route_description(self, route: List[int], current_time: float) -> str:
        """Create detailed description for a macro route."""
        try:
            description_parts = []
            description_parts.append(f"Route: {' -> '.join([f'Region{r}' for r in route])}")
            description_parts.append(f"Length: {len(route)} regions")
            
            # Add congestion information
            current_state = self.traffic_agent.global_state_history[-1] if self.traffic_agent.global_state_history else None
            if current_state:
                congestion_levels = []
                for region_id in route[1:]:  # Skip start region
                    congestion = current_state.regional_congestion.get(region_id, 0)
                    congestion_levels.append(f"R{region_id}:{congestion:.1f}")
                description_parts.append(f"Congestion: {', '.join(congestion_levels)}")
            
            # Add planned vehicle count
            planned_counts = []
            for region_id in route[1:]:
                count = self.region_vehicle_plans.get(region_id, 0)
                planned_counts.append(f"R{region_id}:{count}")
            description_parts.append(f"Planned vehicles: {', '.join(planned_counts)}")
            
            return " | ".join(description_parts)
            
        except Exception as e:
            return f"Route: {route} | Error: {e}"
    
    def _create_macro_planning_observation(self, vehicle_id: str, start_region: int, dest_region: int,
                                         route_descriptions: List[str], current_time: float) -> str:
        """Create observation text for LLM macro planning."""
        observation_parts = []
        
        observation_parts.append(f"MACRO ROUTE PLANNING FOR VEHICLE {vehicle_id}")
        observation_parts.append(f"Origin: Region {start_region} -> Destination: Region {dest_region}")
        observation_parts.append(f"Current time: {current_time:.1f}s")
        observation_parts.append("")
        
        # System overview
        current_state = self.traffic_agent.global_state_history[-1] if self.traffic_agent.global_state_history else None
        if current_state:
            observation_parts.append("SYSTEM STATUS:")
            observation_parts.append(f"Total vehicles: {current_state.total_vehicles}")
            observation_parts.append(f"System average travel time: {current_state.avg_travel_time:.1f}s")
            observation_parts.append("")
        
        # Route candidates
        observation_parts.append("AVAILABLE MACRO ROUTE CANDIDATES:")
        for i, desc in enumerate(route_descriptions):
            observation_parts.append(f"Option {i+1}: {desc}")
        observation_parts.append("")
        
        # Global congestion overview
        if current_state:
            observation_parts.append("REGIONAL CONGESTION LEVELS:")
            for region_id, congestion in current_state.regional_congestion.items():
                vehicle_count = self.region_vehicle_plans.get(region_id, 0)
                observation_parts.append(f"Region {region_id}: Congestion={congestion:.1f}, Planned={vehicle_count}")
            observation_parts.append("")
        
        # Boundary edge status
        observation_parts.append("BOUNDARY EDGE STATUS:")
        if current_state:
            for edge, congestion in list(current_state.boundary_congestion.items())[:10]:  # Show top 10
                planned = self.boundary_vehicle_plans.get(edge, 0)
                observation_parts.append(f"Edge {edge}: Congestion={congestion:.1f}, Planned={planned}")
        observation_parts.append("")
        
        observation_parts.append("OBJECTIVE: Select the optimal macro route to minimize total travel time")
        observation_parts.append("while avoiding overloading any single region or boundary edge.")
        
        return "\n".join(observation_parts)
    
    def _parse_macro_route_answer(self, answer: str, candidates: List[List[int]]) -> List[int]:
        """Parse LLM answer to extract selected macro route."""
        try:
            # Clean the answer
            answer = answer.strip().strip('"\'')
            
            # Try to find exact match in candidates
            for candidate in candidates:
                candidate_str = str(candidate)
                if candidate_str in answer or answer in candidate_str:
                    return candidate
            
            # Try to parse as list
            try:
                import ast
                parsed = ast.literal_eval(answer)
                if isinstance(parsed, list) and all(isinstance(x, int) for x in parsed):
                    return parsed
            except:
                pass
            
            # Fallback to first candidate
            return candidates[0] if candidates else []
            
        except Exception as e:
            self.logger.log_error(f"PARSE_MACRO_ANSWER: Failed to parse '{answer}': {e}")
            return candidates[0] if candidates else []
    
    def _broadcast_vehicle_macro_plan(self, vehicle_id: str, macro_route: List[int], current_time: float):
        """Broadcast vehicle macro plan to update communication system."""
        try:
            # Update region vehicle plans
            for region_id in macro_route[1:]:  # Skip start region
                if region_id not in self.region_vehicle_plans:
                    self.region_vehicle_plans[region_id] = 0
                self.region_vehicle_plans[region_id] += 1
            
            # Update boundary vehicle plans
            for i in range(len(macro_route) - 1):
                from_region = macro_route[i]
                to_region = macro_route[i + 1]
                
                # Find boundary edges for this transition
                boundary_edges = self.traffic_agent.region_connections.get((from_region, to_region), [])
                for edge in boundary_edges:
                    if edge not in self.boundary_vehicle_plans:
                        self.boundary_vehicle_plans[edge] = 0
                    self.boundary_vehicle_plans[edge] += 1
            
            # Create broadcast message
            broadcast_msg = {
                'type': 'MACRO_PLAN_UPDATE',
                'vehicle_id': vehicle_id,
                'macro_route': macro_route,
                'affected_regions': macro_route[1:],
                'timestamp': current_time,
                'message': f"Vehicle {vehicle_id} planning route through regions {macro_route}"
            }
            
            self.broadcast_messages.append(broadcast_msg)
            self.communication_log.append(broadcast_msg)
            
            self.logger.log_info(f"BROADCAST: Vehicle {vehicle_id} macro plan broadcasted to system")
            
        except Exception as e:
            self.logger.log_error(f"BROADCAST: Failed to broadcast macro plan for {vehicle_id}: {e}")
    
    def _log_vehicle_decision(self, vehicle_id: str, decision_type: str, decision_data, current_time: float):
        """Log real-time vehicle decision to console and metrics."""
        try:
            # Get current vehicle metrics
            travel_time = current_time - self.vehicle_start_times.get(vehicle_id, current_time)
            current_edge = traci.vehicle.getRoadID(vehicle_id) if vehicle_id in traci.vehicle.getIDList() else "unknown"
            
            try:
                vehicle_speed = traci.vehicle.getSpeed(vehicle_id) if vehicle_id in traci.vehicle.getIDList() else 0.0
            except:
                vehicle_speed = 0.0
            
            # Update vehicle travel metrics
            self.vehicle_travel_metrics[vehicle_id] = {
                'travel_time': travel_time,
                'current_edge': current_edge,
                'average_speed': vehicle_speed,
                'last_decision': decision_type,
                'last_decision_time': current_time,
                'decision_data': str(decision_data)
            }
            
            # Real-time console output
            print(f"[{current_time:.1f}s] VEHICLE_DECISION: {vehicle_id}")
            print(f"  Type: {decision_type}")
            print(f"  Decision: {decision_data}")
            print(f"  Travel Time: {travel_time:.1f}s")
            print(f"  Current Edge: {current_edge}")
            print(f"  Speed: {vehicle_speed:.1f} m/s")
            print(f"  ATT (Avg Travel Time): {self._calculate_current_att():.1f}s")
            print("---")
            
            # Log to structured log
            self.logger.log_vehicle_status(vehicle_id, current_edge, "planning", 
                                         self.vehicle_regions.get(vehicle_id, -1), 
                                         travel_time, current_time)
            
        except Exception as e:
            self.logger.log_error(f"LOG_VEHICLE_DECISION: Failed for {vehicle_id}: {e}")
    
    def _calculate_current_att(self) -> float:
        """Calculate current Average Travel Time (ATT) using unified method."""
        try:
            if not self.vehicle_travel_metrics:
                return 0.0
            
            # Use unified travel time calculation method
            current_sim_time = traci.simulation.getTime()
            total_travel_time = 0.0
            valid_vehicles = 0
            
            for veh_id, metrics in self.vehicle_travel_metrics.items():
                if veh_id in self.vehicle_start_times:
                    # Unified ATT calculation: current_sim_time - actual_departure_time
                    travel_time = current_sim_time - self.vehicle_start_times[veh_id]
                    total_travel_time += travel_time
                    valid_vehicles += 1
            
            return total_travel_time / valid_vehicles if valid_vehicles > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _get_current_global_state(self) -> Dict:
        """Get current global state for LLM."""
        current_state = self.traffic_agent.global_state_history[-1] if self.traffic_agent.global_state_history else None
        
        if current_state:
            return {
                'current_time': current_state.timestamp,
                'total_vehicles': current_state.total_vehicles,
                'regional_congestion': current_state.regional_congestion,
                'boundary_congestion': current_state.boundary_congestion,
                'avg_travel_time': current_state.avg_travel_time
            }
        else:
            return {
                'current_time': 0.0,
                'total_vehicles': 0,
                'regional_congestion': {},
                'boundary_congestion': {},
                'avg_travel_time': 0.0
            }
    
    def _get_regional_conditions(self) -> Dict:
        """Get regional conditions for LLM."""
        conditions = {}
        current_state = self.traffic_agent.global_state_history[-1] if self.traffic_agent.global_state_history else None
        
        for region_id in range(self.num_regions):
            congestion = current_state.regional_congestion.get(region_id, 0) if current_state else 0
            planned_vehicles = self.region_vehicle_plans.get(region_id, 0)
            
            conditions[region_id] = {
                'congestion_level': congestion,
                'capacity_utilization': min(1.0, congestion / 5.0),
                'vehicle_count': planned_vehicles,
                'status': 'congested' if congestion > 3.0 else 'normal'
            }
        
        return conditions
    
    def _get_boundary_analysis(self) -> Dict:
        """Get boundary analysis for LLM."""
        analysis = {}
        current_state = self.traffic_agent.global_state_history[-1] if self.traffic_agent.global_state_history else None
        
        for boundary_info in self.boundary_edges:
            edge_id = boundary_info['edge_id']
            congestion = current_state.boundary_congestion.get(edge_id, 0) if current_state else 0
            planned_vehicles = self.boundary_vehicle_plans.get(edge_id, 0)
            
            analysis[edge_id] = {
                'from_region': boundary_info['from_region'],
                'to_region': boundary_info['to_region'],
                'congestion_level': congestion,
                'utilization': min(1.0, congestion / 5.0),
                'capacity_remaining': max(0, 1.0 - congestion / 5.0),
                'planned_vehicles': planned_vehicles,
                'predicted_flow': 'stable'
            }
        
        return analysis
    
    def update_road_information(self, current_time: float):
        """Update road information only for edges with vehicles - highly optimized for performance."""
        start_time = time.time()
        
        try:
            # Get all vehicles in simulation to identify active edges
            all_vehicle_ids = traci.vehicle.getIDList()
            active_edges = set()
            
            # Collect edges that currently have vehicles
            for veh_id in all_vehicle_ids:
                try:
                    edge_id = traci.vehicle.getRoadID(veh_id)
                    if edge_id and not edge_id.startswith(':') and edge_id in self.road_info:
                        active_edges.add(edge_id)
                except:
                    continue
            
            # Always include boundary edges for inter-regional coordination
            boundary_edge_ids = {info['edge_id'] for info in self.boundary_edges if info['edge_id'] in self.road_info}
            active_edges.update(boundary_edge_ids)
            
            total_active_edges = len(active_edges)
            if total_active_edges == 0:
                return  # No active edges to update
            
            # Only log if this is a significant change from previous update
            if not hasattr(self, '_last_active_edge_count') or abs(total_active_edges - getattr(self, '_last_active_edge_count', 0)) > 10:
                self.logger.log_info(f"ROAD_UPDATE: Processing {total_active_edges} active edges (optimized)")
                self._last_active_edge_count = total_active_edges
            
            processed_edges = 0
            failed_edges = 0
            
            # Process active edges in optimized batches
            active_edges_list = list(active_edges)
            batch_size = 30  # Smaller batches for active edges only
            
            for i in range(0, len(active_edges_list), batch_size):
                batch_edges = active_edges_list[i:i + batch_size]
                
                try:
                    # Query batch of active edges
                    edges_info = get_multiple_edges_info(batch_edges)
                    
                    for edge_id in batch_edges:
                        try:
                            if edge_id in edges_info:
                                _, vehicle_num, vehicle_speed, vehicle_length, road_len = edges_info[edge_id]
                            else:
                                # Use default values if query failed
                                vehicle_num, vehicle_speed, vehicle_length, road_len = 0, 0.0, 0.0, 100.0
                            
                            # Calculate traffic metrics
                            stored_road_len = self.road_info[edge_id]["road_len"]
                            speed_limit = self.road_info[edge_id]["speed_limit"]
                            effective_road_len = road_len if road_len > 0 else stored_road_len
                            
                            if effective_road_len > 0:
                                occupancy_rate = vehicle_num / (effective_road_len / 8.0)
                                occupancy_rate = min(1.0, max(0.0, occupancy_rate))
                            else:
                                occupancy_rate = 0.0
                            
                            alpha = occupancy_rate * 0.08 + 0.02
                            min_eta = effective_road_len / speed_limit if speed_limit > 0 else effective_road_len / 13.89
                            eta = min_eta * (1 + alpha * vehicle_num)
                            congestion_level = get_congestion_level(occupancy_rate)
                            
                            # Update road info
                            self.road_info[edge_id].update({
                                "vehicle_num": max(0, vehicle_num),
                                "vehicle_speed": max(0.0, vehicle_speed),
                                "vehicle_length": max(0.0, vehicle_length),
                                "congestion_level": congestion_level,
                                "occupancy_rate": occupancy_rate,
                                "min_eta": min_eta,
                                "eta": eta,
                                "last_update": current_time
                            })
                            
                            processed_edges += 1
                            
                            # Only log boundary edges with actual vehicles (reduce noise)
                            is_boundary = any(b.get('edge_id') == edge_id for b in self.boundary_edges)
                            if is_boundary and vehicle_num > 0:
                                self.logger.log_info(f"BOUNDARY_ACTIVE: {edge_id} -> vehicles:{vehicle_num}, speed:{vehicle_speed:.1f}, congestion:{congestion_level}")
                            
                        except Exception as e:
                            failed_edges += 1
                            if failed_edges <= 3:  # Only log first few errors to avoid spam
                                self.logger.log_error(f"EDGE_ERROR: {edge_id} -> {str(e)[:50]}")
                            
                            # Preserve existing data
                            if edge_id in self.road_info:
                                self.road_info[edge_id]["last_update"] = current_time
                                
                except Exception as e:
                    failed_edges += len(batch_edges)
                    self.logger.log_error(f"BATCH_ERROR: {str(e)[:50]}")
            
            total_time = (time.time() - start_time) * 1000
            success_rate = (processed_edges / total_active_edges * 100) if total_active_edges > 0 else 0
            
            # Only log completion for significant updates or failures
            if total_time > 1000 or failed_edges > 0 or processed_edges > 50:
                self.logger.log_info(f"ROAD_UPDATE_COMPLETE: {processed_edges}/{total_active_edges} active edges updated "
                                   f"({success_rate:.1f}% success) in {total_time:.1f}ms")
            
            # Store performance metrics
            if not hasattr(self, 'road_update_stats'):
                self.road_update_stats = []
            
            self.road_update_stats.append({
                'timestamp': current_time,
                'total_time_ms': total_time,
                'processed_edges': processed_edges,
                'failed_edges': failed_edges,
                'success_rate': success_rate,
                'total_active_edges': total_active_edges
            })
            
            # Keep only recent stats (last 50 updates)
            if len(self.road_update_stats) > 50:
                self.road_update_stats = self.road_update_stats[-50:]
                
        except Exception as e:
            self.logger.log_error(f"ROAD_UPDATE_CRITICAL_ERROR: {e}")
            return
    
    def update_vehicle_positions_and_regions(self, current_time: float):
        """Update vehicle positions and region assignments only - no decision making."""
        tracking_start_time = time.time()
        
        try:
            # Get all vehicles in simulation
            all_vehicle_ids = traci.vehicle.getIDList()
            autonomous_in_sim = [veh_id for veh_id in all_vehicle_ids if veh_id in self.autonomous_vehicles]
            
            self.logger.log_info(f"VEHICLE_POSITION_UPDATE_START: {len(autonomous_in_sim)}/{len(all_vehicle_ids)} autonomous vehicles active")
            
            # Batch vehicle information gathering for efficiency
            vehicles_processed = 0
            vehicles_failed = 0
            region_changes = 0
            new_vehicles = 0
            
            # Initialize tracking for decision stage
            if not hasattr(self, 'pending_decisions'):
                self.pending_decisions = {
                    'new_vehicles': [],
                    'region_changes': [],
                    'stuck_vehicles': []
                }
            # Don't clear previous pending decisions here - they will be processed and cleared in process_vehicle_decisions
            
            # Track vehicles entering and exiting the simulation
            for veh_id in autonomous_in_sim:
                try:
                    # Record start time for new vehicles (no planning yet)
                    if veh_id not in self.vehicle_start_times:
                        # Get accurate departure time from SUMO
                        try:
                            actual_depart_time = traci.vehicle.getDeparture(veh_id)
                            self.vehicle_start_times[veh_id] = actual_depart_time
                            new_vehicles += 1
                            self.logger.log_info(f"NEW_VEHICLE: {veh_id} started at time {actual_depart_time:.1f}")
                            
                            # Queue for decision making (with deduplication)
                            if veh_id not in self.pending_decisions['new_vehicles']:
                                self.pending_decisions['new_vehicles'].append(veh_id)
                        except Exception as e:
                            # Fallback to current simulation time if getDeparture() fails
                            self.vehicle_start_times[veh_id] = current_time
                            new_vehicles += 1
                            self.logger.log_info(f"NEW_VEHICLE: {veh_id} started at time {current_time:.1f} (fallback)")
                            self.logger.log_warning(f"VEHICLE_DEPART_TIME: Could not get departure time for {veh_id}: {e}")
                            
                            # Queue for decision making (with deduplication)
                            if veh_id not in self.pending_decisions['new_vehicles']:
                                self.pending_decisions['new_vehicles'].append(veh_id)
                    
                    # Batch TraCI calls for this vehicle
                    try:
                        current_edge = traci.vehicle.getRoadID(veh_id)
                        route = traci.vehicle.getRoute(veh_id)
                        
                        # Skip vehicles without valid road assignment
                        if not current_edge or current_edge.startswith(':'):
                            continue  # Skip junction vehicles
                            
                    except Exception as traci_error:
                        self.logger.log_warning(f"VEHICLE_TRACI_ERROR: {veh_id} -> {traci_error}")
                        vehicles_failed += 1
                        continue
                    
                    # Update vehicle region (no planning yet)
                    if current_edge in self.edge_to_region:
                        current_region = self.edge_to_region[current_edge]
                        
                        # Check if vehicle moved to a new region
                        if veh_id not in self.vehicle_regions:
                            self.vehicle_regions[veh_id] = current_region
                            self.logger.log_info(f"VEHICLE_REGION_INIT: {veh_id} assigned to region {current_region}")
                            
                            # Queue for regional planning
                            self.pending_decisions['region_changes'].append({
                                'vehicle_id': veh_id,
                                'type': 'init',
                                'region': current_region
                            })
                            
                        elif self.vehicle_regions[veh_id] != current_region:
                            # Vehicle changed regions - record for decision making
                            old_region = self.vehicle_regions[veh_id]
                            self.vehicle_regions[veh_id] = current_region
                            region_changes += 1
                            
                            self.logger.log_info(f"VEHICLE_REGION_CHANGE: {veh_id} moved from region {old_region} to {current_region}")
                            
                            # Queue for region change planning
                            self.pending_decisions['region_changes'].append({
                                'vehicle_id': veh_id,
                                'type': 'change',
                                'old_region': old_region,
                                'new_region': current_region
                            })
                        
                        # Real-time logging and metrics update
                        destination = route[-1] if route else "unknown"
                        travel_time = current_time - self.vehicle_start_times[veh_id]
                        
                        # Get additional vehicle metrics
                        try:
                            vehicle_speed = traci.vehicle.getSpeed(veh_id)
                            vehicle_position = traci.vehicle.getPosition(veh_id)
                            route_progress = traci.vehicle.getRouteIndex(veh_id)
                            
                            # Update real-time vehicle metrics for console output
                            self.vehicle_travel_metrics[veh_id] = {
                                'travel_time': travel_time,
                                'current_edge': current_edge,
                                'current_region': current_region,
                                'average_speed': vehicle_speed,
                                'destination': destination,
                                'route_progress': route_progress,
                                'last_update': current_time
                            }
                            
                            self.logger.log_vehicle_status(
                                veh_id, current_edge, destination, current_region,
                                travel_time, current_time
                            )
                            
                            # Real-time console output for vehicle state changes
                            if hasattr(self, '_last_console_update') and current_time - getattr(self, '_last_console_update', 0) > 10:
                                self._print_real_time_system_status(current_time)
                                self._last_console_update = current_time
                            
                            # Check for stuck vehicles (queue for decision making)
                            # Use shorter threshold for vehicles without macro plans
                            has_macro_plan = veh_id in self.vehicle_current_plans
                            stuck_threshold = 180 if not has_macro_plan else 300  # 3 minutes vs 5 minutes
                            
                            if vehicle_speed < 0.1 and travel_time > stuck_threshold:
                                self.logger.log_warning(f"VEHICLE_STUCK: {veh_id} stopped on {current_edge} for {travel_time:.1f}s (macro_plan={has_macro_plan})")
                                
                                # Queue for stuck vehicle replanning (with deduplication)
                                if veh_id not in self.pending_decisions['stuck_vehicles']:
                                    self.pending_decisions['stuck_vehicles'].append(veh_id)
                            
                        except Exception as metrics_error:
                            self.logger.log_warning(f"VEHICLE_METRICS_ERROR: {veh_id} -> {metrics_error}")
                        
                        vehicles_processed += 1
                        
                    else:
                        self.logger.log_warning(f"VEHICLE_EDGE_UNKNOWN: {veh_id} on edge {current_edge} not in region mapping")
                        
                except Exception as vehicle_error:
                    vehicles_failed += 1
                    self.logger.log_error(f"VEHICLE_PROCESSING_ERROR: {veh_id} -> {vehicle_error}")
                    continue
            
            # Check for completed vehicles with enhanced logging
            completed_this_step = 0
            try:
                arrived_vehicles = traci.simulation.getArrivedIDList()
                for veh_id in arrived_vehicles:
                    if veh_id in self.vehicle_start_times and veh_id not in self.vehicle_end_times:
                        self.vehicle_end_times[veh_id] = current_time
                        self.completed_vehicles += 1
                        completed_this_step += 1
                        
                        # Log completion with detailed metrics and cleanup
                        travel_time = current_time - self.vehicle_start_times[veh_id]
                        self.logger.log_vehicle_completion(
                            veh_id, self.vehicle_start_times[veh_id], current_time, travel_time
                        )
                        
                        # RL Training Data Collection - Collect data BEFORE cleanup
                        if self.rl_data_collection_enabled:
                            self.logger.log_info(f"RL_DATA_COLLECTION_TRIGGER: Vehicle {veh_id} completed, collecting training data")
                            self._collect_rl_training_data(veh_id, travel_time, current_time)
                        else:
                            self.logger.log_info(f"RL_DATA_COLLECTION_DISABLED: Vehicle {veh_id} completed but RL data collection is disabled")
                        
                        # Clean up vehicle plans and broadcast completion
                        self._cleanup_completed_vehicle_plans(veh_id, current_time)
                        
                        # Real-time console output for completion
                        print(f"[{current_time:.1f}s] VEHICLE_COMPLETED: {veh_id}")
                        print(f"  Total Travel Time: {travel_time:.1f}s")
                        print(f"  System ATT: {self._calculate_current_att():.1f}s")
                        print(f"  Completed: {self.completed_vehicles}/{len(self.autonomous_vehicles)}")
                        print("---")
                        
                        self.logger.log_info(f"VEHICLE_COMPLETED: {veh_id} finished in {travel_time:.1f}s")
                        
                        # Clean up tracking data
                        if veh_id in self.vehicle_regions:
                            del self.vehicle_regions[veh_id]
                        if veh_id in self.vehicle_current_plans:
                            del self.vehicle_current_plans[veh_id]
                        if veh_id in self.vehicle_regional_plans:
                            del self.vehicle_regional_plans[veh_id]
                        if veh_id in self.vehicle_travel_metrics:
                            del self.vehicle_travel_metrics[veh_id]
                            
            except Exception as completion_error:
                self.logger.log_error(f"VEHICLE_COMPLETION_ERROR: {completion_error}")
            
            # Performance summary
            tracking_time = (time.time() - tracking_start_time) * 1000
            active_vehicles = len(self.vehicle_regions)
            
            self.logger.log_info(f"VEHICLE_POSITION_UPDATE_COMPLETE: processed:{vehicles_processed}, failed:{vehicles_failed}, "
                               f"new:{new_vehicles}, region_changes:{region_changes}, completed:{completed_this_step}, "
                               f"active:{active_vehicles}, time:{tracking_time:.1f}ms")
            
            # Performance warnings
            if tracking_time > 2000:  # > 2 seconds
                self.logger.log_warning(f"VEHICLE_POSITION_UPDATE_SLOW: Update took {tracking_time:.1f}ms")
            
            if vehicles_failed > vehicles_processed * 0.1:  # > 10% failure rate
                self.logger.log_warning(f"VEHICLE_POSITION_UPDATE_HIGH_FAILURE: {vehicles_failed}/{vehicles_processed + vehicles_failed} vehicles failed")
                
        except Exception as e:
            self.logger.log_error(f"VEHICLE_POSITION_UPDATE_CRITICAL_ERROR: {e}")
            raise
    
    def process_vehicle_decisions(self, current_time: float):
        """Process all vehicle decisions based on updated state information."""
        decision_start_time = time.time()
        
        try:
            if not hasattr(self, 'pending_decisions') or not self.pending_decisions:
                return  # No decisions to process
            
            # Atomically extract pending decisions to avoid accumulation
            new_vehicles = self.pending_decisions.get('new_vehicles', []).copy()
            region_changes = self.pending_decisions.get('region_changes', []).copy()
            stuck_vehicles = self.pending_decisions.get('stuck_vehicles', []).copy()
            
            # Clear pending decisions immediately to avoid accumulation
            self.pending_decisions = {
                'new_vehicles': [],
                'region_changes': [],
                'stuck_vehicles': []
            }
            
            total_decisions = len(new_vehicles) + len(region_changes) + len(stuck_vehicles)
            
            if total_decisions == 0:
                return
                
            decisions_processed = 0
            decisions_failed = 0
            
            # Process new vehicle births
            for veh_id in new_vehicles:
                try:
                    self.handle_vehicle_birth_macro_planning(veh_id, current_time)
                    decisions_processed += 1
                except Exception as e:
                    self.logger.log_error(f"VEHICLE_DECISION_ERROR: Failed to process new vehicle {veh_id}: {e}")
                    decisions_failed += 1
            
            # Process region changes
            for region_change in region_changes:
                try:
                    veh_id = region_change['vehicle_id']
                    
                    if region_change['type'] == 'init':
                        # Initial regional planning
                        self.handle_vehicle_regional_planning(veh_id, region_change['region'], current_time)
                    elif region_change['type'] == 'change':
                        # Region change replanning
                        self.handle_vehicle_region_change_replanning(
                            veh_id, region_change['old_region'], region_change['new_region'], current_time
                        )
                        self.handle_vehicle_regional_planning(veh_id, region_change['new_region'], current_time)
                    
                    decisions_processed += 1
                except Exception as e:
                    self.logger.log_error(f"VEHICLE_DECISION_ERROR: Failed to process region change for {veh_id}: {e}")
                    decisions_failed += 1
            
            # Process stuck vehicles
            for veh_id in stuck_vehicles:
                try:
                    self._handle_stuck_vehicle_replanning(veh_id, current_time)
                    decisions_processed += 1
                except Exception as e:
                    self.logger.log_error(f"VEHICLE_DECISION_ERROR: Failed to process stuck vehicle {veh_id}: {e}")
                    decisions_failed += 1
            
            # Performance summary
            decision_time = (time.time() - decision_start_time) * 1000
            
            self.logger.log_info(f"VEHICLE_DECISIONS_COMPLETE: total:{total_decisions}, processed:{decisions_processed}, "
                               f"failed:{decisions_failed}, time:{decision_time:.1f}ms")
            
            # Performance warnings
            if decision_time > 5000:  # > 5 seconds
                self.logger.log_warning(f"VEHICLE_DECISIONS_SLOW: Processing took {decision_time:.1f}ms")
            
            if decisions_failed > 0:
                self.logger.log_warning(f"VEHICLE_DECISIONS_FAILURES: {decisions_failed} decisions failed")
                
        except Exception as e:
            self.logger.log_error(f"VEHICLE_DECISIONS_CRITICAL_ERROR: {e}")
            # Don't re-raise, just log the error to prevent simulation crash
    
    def coordinate_regional_agents(self, current_time: float):
        """Coordinate regional agents with batch asynchronous regional planning."""
        try:
            # Update vehicle status for all regional agents first
            for region_id, agent in self.regional_agents.items():
                try:
                    agent.update_vehicle_status(current_time)
                except Exception as e:
                    self.logger.log_error(f"REGIONAL_COORD: Failed to update status for region {region_id}: {e}")
            
            # Check if any regions need batch regional planning
            regions_needing_planning = self._identify_regions_needing_planning(current_time)
            
            if regions_needing_planning:
                self.logger.log_info(f"REGIONAL_COORD: Initiating batch planning for {len(regions_needing_planning)} regions")
                
                # Execute batch asynchronous regional planning
                self._execute_batch_regional_planning(regions_needing_planning, current_time)
            
            self.logger.log_info(f"REGIONAL_COORD: Coordination completed at {current_time:.1f}s")
            
        except Exception as e:
            self.logger.log_error(f"REGIONAL_COORD: Coordination failed: {e}")
    
    def update_traffic_agent(self, current_time: float):
        """Update Traffic Agent with global state - now event-driven LLM system."""
        try:
            # Always update global traffic state for LLM decision context
            self.traffic_agent.update_global_traffic_state(current_time)
            
            # Collect regional congestion reports for LLM decisions
            regional_report = self.traffic_agent.collect_regional_congestion_report(current_time)
            
            # Store the report for use in LLM decisions
            if not hasattr(self, 'latest_regional_report'):
                self.latest_regional_report = {}
            self.latest_regional_report = regional_report
            
            # No batch macro planning - all planning is now event-driven via LLM:
            # - Vehicle birth events trigger handle_vehicle_birth_macro_planning()
            # - Region change events trigger handle_vehicle_region_change_replanning()
            # - All decisions go through LLM hybrid_decision_making_pipeline()
            
            self.logger.log_info(f"TRAFFIC_AGENT: Updated global state and regional report at {current_time:.1f}s")
            
        except Exception as e:
            self.logger.log_error(f"TRAFFIC_AGENT: Update failed: {e}")
    
    def update_prediction_engine(self, current_time: float):
        """Update prediction engine with latest observations."""
        if current_time - self.last_prediction_update_time >= self.prediction_update_interval:
            try:
                # Update observations
                self.prediction_engine.update_observations(self.road_info, current_time)
                
                # Train models periodically
                self.prediction_engine.train_models(current_time)
                
                self.last_prediction_update_time = current_time
                
            except Exception as e:
                self.logger.log_error(f"Prediction Engine update failed: {e}")
    
    def log_system_performance(self, current_time: float):
        """Log overall system performance including ATT metrics."""
        try:
            # Collect metrics from all agents
            regional_metrics = {}
            for region_id, agent in self.regional_agents.items():
                regional_metrics[region_id] = agent.get_performance_metrics()
            
            traffic_metrics = self.traffic_agent.get_performance_metrics()
            prediction_metrics = self.prediction_engine.get_performance_metrics()
            
            # Calculate current ATT for performance logging
            current_att = self._calculate_current_att()
            active_vehicles = len(self.vehicle_travel_metrics)
            completed_vehicles = self.completed_vehicles
            total_vehicles = len(self.autonomous_vehicles)
            
            # Log ATT metrics to system log
            self.logger.log_info(f"SYSTEM_ATT_METRICS: current_att={current_att:.2f}s, "
                               f"active_vehicles={active_vehicles}, completed_vehicles={completed_vehicles}, "
                               f"total_vehicles={total_vehicles}, completion_rate={completed_vehicles/total_vehicles*100:.1f}%")
            
            # Log async LLM performance
            pending_calls = len(self.pending_llm_calls)
            
            self.logger.log_info(f"ASYNC_LLM_METRICS: pending_calls={pending_calls}, "
                               f"macro_calls={self.llm_call_stats['macro_calls']}, "
                               f"regional_calls={self.llm_call_stats['regional_calls']}, "
                               f"async_calls={self.llm_call_stats['async_calls']}, "
                               f"time_saved={self.llm_call_stats['total_time_saved']:.1f}s, "
                               f"temp_decisions={len(self.temp_decisions)}")
            
            # Log comprehensive performance data
            self.logger.log_system_performance(
                regional_metrics, traffic_metrics, prediction_metrics, current_time
            )
            
        except Exception as e:
            self.logger.log_error(f"Performance logging failed: {e}")
    
    def validate_system_stability(self) -> bool:
        """
        Comprehensive system validation to ensure all components work correctly.
        
        Returns:
            bool: True if all validations pass, False otherwise
        """
        validation_start = time.time()
        self.logger.log_info("SYSTEM_VALIDATION_START: Running comprehensive system checks")
        
        validations_passed = 0
        validations_total = 8
        
        try:
            # 1. Validate TraCI connection
            try:
                sim_time = traci.simulation.getTime()
                edge_count = len(self.edges)  # Use cached edge list instead of querying
                self.logger.log_info(f"VALIDATION_TRACI: ✓ Connected - SimTime:{sim_time}, Edges:{edge_count}")
                validations_passed += 1
            except Exception as e:
                self.logger.log_error(f"VALIDATION_TRACI: ✗ Failed - {e}")
            
            # 2. Validate region data integrity
            try:
                if self.num_regions > 0 and len(self.boundary_edges) > 0 and len(self.edge_to_region) > 0:
                    self.logger.log_info(f"VALIDATION_REGIONS: ✓ {self.num_regions} regions, {len(self.boundary_edges)} boundary edges")
                    validations_passed += 1
                else:
                    self.logger.log_error("VALIDATION_REGIONS: ✗ Missing or invalid region data")
            except Exception as e:
                self.logger.log_error(f"VALIDATION_REGIONS: ✗ Error - {e}")
            
            # 3. Validate road network data
            try:
                if len(self.road_info) > 0 and self.road_network.number_of_nodes() > 0:
                    self.logger.log_info(f"VALIDATION_NETWORK: ✓ {len(self.road_info)} edges, {self.road_network.number_of_nodes()} nodes")
                    validations_passed += 1
                else:
                    self.logger.log_error("VALIDATION_NETWORK: ✗ Missing or invalid network data")
            except Exception as e:
                self.logger.log_error(f"VALIDATION_NETWORK: ✗ Error - {e}")
            
            # 4. Validate agent initialization
            try:
                if (hasattr(self, 'regional_agents') and len(self.regional_agents) == self.num_regions and
                    hasattr(self, 'traffic_agent') and hasattr(self, 'prediction_engine')):
                    self.logger.log_info(f"VALIDATION_AGENTS: ✓ {len(self.regional_agents)} regional agents, traffic agent, prediction engine")
                    validations_passed += 1
                else:
                    self.logger.log_error("VALIDATION_AGENTS: ✗ Missing or incomplete agent initialization")
            except Exception as e:
                self.logger.log_error(f"VALIDATION_AGENTS: ✗ Error - {e}")
            
            # 5. Test edge information queries (performance critical) - 批量优化
            try:
                test_edges = list(self.edges)[:5]  # Test first 5 edges
                query_start = time.time()
                
                # 使用批量查询而不是逐个查询
                edges_info = get_multiple_edges_info(test_edges)
                
                query_time = (time.time() - query_start) * 1000
                avg_time_per_edge = query_time / len(test_edges)
                
                if avg_time_per_edge < 50:  # Less than 50ms per edge
                    self.logger.log_info(f"VALIDATION_EDGE_QUERIES: ✓ {avg_time_per_edge:.1f}ms per edge (excellent)")
                    validations_passed += 1
                elif avg_time_per_edge < 200:
                    self.logger.log_info(f"VALIDATION_EDGE_QUERIES: ✓ {avg_time_per_edge:.1f}ms per edge (acceptable)")
                    validations_passed += 1
                else:
                    self.logger.log_warning(f"VALIDATION_EDGE_QUERIES: ⚠ {avg_time_per_edge:.1f}ms per edge (slow)")
                    validations_passed += 1  # Still pass but with warning
                    
            except Exception as e:
                self.logger.log_error(f"VALIDATION_EDGE_QUERIES: ✗ Error - {e}")
            
            # 6. Test logging system functionality
            try:
                # Test LLM call logging
                test_call_id = self.logger.log_llm_call_start("TestAgent", "test_0", 100, "validation", "Test input")
                self.logger.log_llm_call_end(test_call_id, True, "Test decision output", 100)
                
                # Test vehicle logging
                self.logger.log_vehicle_status("test_vehicle", "test_edge", "dest_edge", 0, 10.0, time.time())
                
                self.logger.log_info("VALIDATION_LOGGING: ✓ All logging functions working correctly")
                validations_passed += 1
            except Exception as e:
                self.logger.log_error(f"VALIDATION_LOGGING: ✗ Error - {e}")
            
            # 7. Test autonomous vehicle selection
            try:
                if len(self.autonomous_vehicles) > 0:
                    percentage = len(self.autonomous_vehicles) / self.total_vehicles * 100
                    self.logger.log_info(f"VALIDATION_AUTONOMOUS: ✓ {len(self.autonomous_vehicles)} vehicles ({percentage:.1f}%)")
                    validations_passed += 1
                else:
                    self.logger.log_error("VALIDATION_AUTONOMOUS: ✗ No autonomous vehicles selected")
            except Exception as e:
                self.logger.log_error(f"VALIDATION_AUTONOMOUS: ✗ Error - {e}")
            
            # 8. Test batch processing capability
            try:
                batch_test_start = time.time()
                test_edges = list(self.edges)[:20]  # Test batch of 20 edges
                
                # 使用真正的批量处理
                batch_results = []
                try:
                    edges_info = get_multiple_edges_info(test_edges)
                    batch_results = list(edges_info.values())
                except Exception as e:
                    self.logger.log_warning(f"BATCH_TEST_ERROR: {e}")
                        
                batch_time = (time.time() - batch_test_start) * 1000
                success_rate = len(batch_results) / len(test_edges)
                
                if success_rate >= 0.8 and batch_time < 2000:  # 80% success, <2s
                    self.logger.log_info(f"VALIDATION_BATCH: ✓ {success_rate:.1%} success rate, {batch_time:.1f}ms total")
                    validations_passed += 1
                else:
                    self.logger.log_warning(f"VALIDATION_BATCH: ⚠ {success_rate:.1%} success rate, {batch_time:.1f}ms total")
                    validations_passed += 1  # Pass with warning
                    
            except Exception as e:
                self.logger.log_error(f"VALIDATION_BATCH: ✗ Error - {e}")
            
            # Final validation summary
            validation_time = (time.time() - validation_start) * 1000
            success_rate = validations_passed / validations_total
            
            if success_rate >= 0.9:  # 90% pass rate
                self.logger.log_info(f"SYSTEM_VALIDATION_SUCCESS: {validations_passed}/{validations_total} checks passed "
                                   f"({success_rate:.1%}) in {validation_time:.1f}ms")
                self.logger.log_info("SYSTEM_READY: All critical components validated, performance optimized")
                return True
            else:
                self.logger.log_error(f"SYSTEM_VALIDATION_FAILED: Only {validations_passed}/{validations_total} checks passed "
                                    f"({success_rate:.1%}) in {validation_time:.1f}ms")
                return False
                
        except Exception as e:
            self.logger.log_error(f"SYSTEM_VALIDATION_CRITICAL_ERROR: {e}")
            return False
    
    def run_simulation(self) -> Tuple[float, int]:
        """
        Run the complete multi-agent simulation.
        
        Returns:
            Tuple of (average_travel_time, completed_vehicles)
        """
        try:
            self.initialize_simulation()
            
            # Run system validation before starting main simulation
            if not self.validate_system_stability():
                self.logger.log_error("SIMULATION_ABORTED: System validation failed")
                raise RuntimeError("System validation failed - cannot proceed with simulation")
            
            self.logger.log_info("SIMULATION_VALIDATED: System ready for multi-agent simulation")
            
            # Initialize step tracking
            step = 0.0
            self.logger.log_info(f"SIMULATION_START: Beginning simulation with {self.max_steps} steps (step_size: {self.step_size})")
            
            self.logger.log_info("Starting multi-agent simulation")
            
            while step < self.max_steps:
                # Advance simulation first to ensure consistent timing
                traci.simulationStep(step)
                current_time = traci.simulation.getTime()
                
                # Update road information
                self.update_road_information(current_time)
                
                # PHASE 1: Update all vehicle positions and states
                self.update_vehicle_positions_and_regions(current_time)
                
                # Update prediction engine based on new positions
                self.update_prediction_engine(current_time)
                
                # Check for and apply latest LoRA adapters (every 10 simulation steps)
                if int(step) % 10 == 0:
                    self._check_and_apply_latest_adapters()
                
                # Update traffic agent global state based on new positions
                self.update_traffic_agent(current_time)
                
                # Update regional agent states before making decisions
                self.coordinate_regional_agents(current_time)
                
                # PHASE 2: Make all decisions based on updated states
                self.process_vehicle_decisions(current_time)
                
                # PHASE 3: Process completed async LLM calls
                self._process_completed_llm_calls(current_time)
                
                # Process any pending broadcast messages
                self._process_broadcast_messages(current_time)
                
                # Increment step for next iteration
                step += self.step_size
                
                # Display progress
                self.logger.display_progress(current_time)
                
                # Log performance periodically
                if int(current_time) % 300 == 0:  # Every 5 minutes
                    self.log_system_performance(current_time)
            
            # Calculate final results using unified method
            if self.vehicle_end_times:
                total_travel_time = sum(
                    self.vehicle_end_times[veh_id] - self.vehicle_start_times.get(veh_id, 0)
                    for veh_id in self.vehicle_end_times
                    if veh_id in self.vehicle_start_times
                )
                valid_completed = len([veh_id for veh_id in self.vehicle_end_times if veh_id in self.vehicle_start_times])
                average_travel_time = total_travel_time / valid_completed if valid_completed > 0 else 0.0
            else:
                average_travel_time = 0.0
            
            self.logger.log_info(f"Simulation completed: Average travel time: {average_travel_time:.2f}s, "
                               f"Completed vehicles: {self.completed_vehicles}/{self.total_vehicles}")
            
            return average_travel_time, self.completed_vehicles
            
        except Exception as e:
            self.logger.log_error(f"Simulation failed: {e}")
            raise
        finally:
            # Clean up
            try:
                traci.close()
            except:
                pass
            
            # Close logger and generate reports
            self.logger.close_session()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
    
    def handle_vehicle_region_change_replanning(self, vehicle_id: str, old_region: int, new_region: int, current_time: float):
        """
        Handle LLM-based macro route replanning when vehicle reaches new region.
        
        This implements the user requirement: when vehicle arrives in new region,
        provide original macro route and new candidates to traffic agent for LLM decision.
        """
        try:
            self.logger.log_info(f"REGION_CHANGE_REPLAN: {vehicle_id} from region {old_region} to {new_region}")
            
            # Get vehicle's destination
            route = traci.vehicle.getRoute(vehicle_id)
            if not route:
                return
            
            dest_edge = route[-1]
            dest_region = self.edge_to_region.get(dest_edge, new_region)
            
            # If already at destination region, no replanning needed
            if new_region == dest_region:
                self.logger.log_info(f"REGION_CHANGE_REPLAN: {vehicle_id} reached destination region {dest_region}")
                return
            
            # Get original macro route
            original_macro_route = None
            if vehicle_id in self.vehicle_current_plans:
                original_macro_route = self.vehicle_current_plans[vehicle_id]['macro_route']
            
            # Generate new macro route candidates from current region to destination
            new_macro_candidates = self._generate_macro_route_candidates(
                new_region, dest_region, current_time
            )
            
            if not new_macro_candidates:
                self.logger.log_warning(f"REGION_CHANGE_REPLAN: No new candidates for {vehicle_id}")
                return
            
            # Use LLM to decide: keep original route or select new route
            selected_route = self._llm_replan_macro_route(
                vehicle_id, original_macro_route, new_macro_candidates, 
                old_region, new_region, dest_region, current_time
            )
            
            if selected_route:
                # Update vehicle's macro plan
                self.vehicle_current_plans[vehicle_id] = {
                    'macro_route': selected_route,
                    'current_region_index': selected_route.index(new_region) if new_region in selected_route else 0,
                    'creation_time': current_time,
                    'last_update': current_time,
                    'replanned': True,
                    'original_route': original_macro_route
                }
                
                # Broadcast the updated plan
                self._broadcast_vehicle_macro_plan_update(vehicle_id, original_macro_route, selected_route, current_time)
                
                # Real-time log output
                self._log_vehicle_decision(vehicle_id, "MACRO_REPLANNING", 
                                         f"Original: {original_macro_route} -> New: {selected_route}", current_time)
                
                self.logger.log_info(f"REGION_CHANGE_REPLAN: {vehicle_id} updated route to {selected_route}")
            
        except Exception as e:
            self.logger.log_error(f"REGION_CHANGE_REPLAN: Failed for {vehicle_id}: {e}")
    
    def handle_vehicle_regional_planning(self, vehicle_id: str, region_id: int, current_time: float):
        """Handle regional planning for vehicle within a region using LLM."""
        try:
            # Get vehicle's macro route to determine next boundary target
            if vehicle_id not in self.vehicle_current_plans:
                self.logger.log_warning(f"REGIONAL_PLANNING: No macro plan for {vehicle_id}")
                return
            
            macro_route = self.vehicle_current_plans[vehicle_id]['macro_route']
            current_region_index = self.vehicle_current_plans[vehicle_id].get('current_region_index', 0)
            
            # Determine target boundary edge to reach next region
            if current_region_index + 1 < len(macro_route):
                next_region = macro_route[current_region_index + 1]
                
                # Get regional agent for this region
                if region_id in self.regional_agents:
                    regional_agent = self.regional_agents[region_id]
                    
                    # Get vehicle's current edge for planning
                    try:
                        current_edge = traci.vehicle.getRoadID(vehicle_id)
                        if not current_edge or current_edge.startswith(':'):
                            self.logger.log_warning(f"REGIONAL_PLANNING: Vehicle {vehicle_id} on invalid edge: {current_edge}")
                            return
                    except Exception as traci_error:
                        self.logger.log_error(f"REGIONAL_PLANNING: TraCI error for vehicle {vehicle_id}: {traci_error}")
                        return
                    
                    # Get boundary candidates and route candidates through regional agent
                    try:
                        boundary_candidates = regional_agent._get_boundary_candidates_to_region(next_region)
                        if not boundary_candidates:
                            # Fallback to any outgoing boundary
                            boundary_candidates = regional_agent.outgoing_boundaries[:3] if regional_agent.outgoing_boundaries else []
                        
                        if boundary_candidates:
                            route_candidates = regional_agent._generate_regional_route_candidates(
                                current_edge, boundary_candidates, current_time
                            )
                            
                            if route_candidates:
                                # Use async LLM call for regional route selection
                                regional_plan = self._async_llm_call_regional_route(
                                    vehicle_id, current_edge, route_candidates, next_region, region_id, current_time
                                )
                            else:
                                # Fallback: synchronous call
                                regional_plan = regional_agent.make_regional_route_planning(
                                    vehicle_id, next_region, current_time
                                )
                        else:
                            self.logger.log_warning(f"REGIONAL_PLANNING: No boundary candidates for {vehicle_id}")
                            return
                            
                    except Exception as planning_error:
                        self.logger.log_warning(f"REGIONAL_PLANNING: Async planning failed for {vehicle_id}, using fallback: {planning_error}")
                        # Fallback to synchronous call
                        regional_plan = regional_agent.make_regional_route_planning(
                            vehicle_id, next_region, current_time
                        )
                    
                    if regional_plan:
                        # Validate regional plan has required keys
                        if 'boundary_edge' not in regional_plan or 'route' not in regional_plan:
                            self.logger.log_error(f"REGIONAL_PLANNING: Invalid plan structure for {vehicle_id}: {regional_plan.keys()}")
                            return
                        
                        # Store and execute regional plan
                        self.vehicle_regional_plans[vehicle_id] = {
                            'region_id': region_id,
                            'target_region': next_region,
                            'boundary_edge': regional_plan['boundary_edge'],
                            'route': regional_plan['route'],
                            'creation_time': current_time,
                            'travel_time': regional_plan.get('travel_time', 0),
                            'reasoning': regional_plan.get('reasoning', 'Regional route planning')
                        }
                        
                        # Execute the regional route using SUMO's setRoute with safety checks
                        if regional_plan['route'] and len(regional_plan['route']) > 0:
                            try:
                                # Ensure safe route setting
                                current_edge = traci.vehicle.getRoadID(vehicle_id)
                                safe_route = self._create_safe_route(current_edge, regional_plan['route'])
                                if safe_route:
                                    traci.vehicle.setRoute(vehicle_id, safe_route)
                                    self.logger.log_info(f"REGIONAL_PLANNING: Set safe route for {vehicle_id}")
                                else:
                                    self.logger.log_warning(f"REGIONAL_PLANNING: Cannot create safe route for {vehicle_id}")
                            except Exception as route_error:
                                self.logger.log_error(f"REGIONAL_PLANNING: Failed to set route for {vehicle_id}: {route_error}")
                                return
                        
                        # Real-time log output
                        self._log_vehicle_decision(vehicle_id, "REGIONAL_PLANNING", 
                                                 f"Target: {regional_plan['boundary_edge']}", current_time)
                        
                        self.logger.log_info(f"REGIONAL_PLANNING: {vehicle_id} planned route in region {region_id}")
            
        except Exception as e:
            self.logger.log_error(f"REGIONAL_PLANNING: Failed for {vehicle_id}: {e}")
    
    def _broadcast_vehicle_macro_plan_update(self, vehicle_id: str, original_route: Optional[List[int]], 
                                           new_route: List[int], current_time: float):
        """Broadcast macro plan update to communication system."""
        try:
            # Clean up old plan counts
            if original_route:
                for region_id in original_route[1:]:
                    if region_id in self.region_vehicle_plans:
                        self.region_vehicle_plans[region_id] = max(0, self.region_vehicle_plans[region_id] - 1)
            
            # Update with new plan
            self._broadcast_vehicle_macro_plan(vehicle_id, new_route, current_time)
            
            # Create update broadcast message
            update_msg = {
                'type': 'MACRO_PLAN_REPLAN',
                'vehicle_id': vehicle_id,
                'original_route': original_route,
                'new_route': new_route,
                'timestamp': current_time,
                'message': f"Vehicle {vehicle_id} replanned: {original_route} -> {new_route}"
            }
            
            self.broadcast_messages.append(update_msg)
            self.communication_log.append(update_msg)
            
        except Exception as e:
            self.logger.log_error(f"BROADCAST_UPDATE: Failed for {vehicle_id}: {e}")
    
    def _collect_rl_training_data(self, vehicle_id: str, travel_time: float, completion_time: float):
        """
        Collect comprehensive RL training data for completed vehicles.
        
        This function gathers all necessary data for MAGRPO training, including:
        - CORY decision data (Pioneer, Observer, J1-Judge)
        - Performance metrics and rewards
        - State and action information
        """
        try:
            self.logger.log_info(f"RL_DATA_COLLECTION_START: Starting data collection for {vehicle_id}, "
                               f"travel_time={travel_time:.1f}s, training_queue={'available' if self.training_queue else 'None'}")
            
            rl_training_data = {}
            
            # Basic vehicle information
            rl_training_data['vehicle_id'] = vehicle_id
            rl_training_data['start_time'] = self.vehicle_start_times.get(vehicle_id, completion_time)
            rl_training_data['completion_time'] = completion_time
            rl_training_data['travel_time'] = travel_time
            
            # CORY Decision Data - Extract from vehicle_current_plans
            macro_plan = self.vehicle_current_plans.get(vehicle_id, {})
            if macro_plan:
                rl_training_data['macro_route'] = macro_plan.get('macro_route', [])
                rl_training_data['cooperation_quality'] = macro_plan.get('cooperation_quality', 0.0)
                rl_training_data['cory_decision_type'] = macro_plan.get('cory_decision_type', 'unknown')
                rl_training_data['pioneer_decision'] = macro_plan.get('pioneer_decision', {})
                rl_training_data['observer_feedback'] = macro_plan.get('observer_feedback', {})
                rl_training_data['j1_judge_evaluation'] = macro_plan.get('j1_judge_evaluation', {})
                rl_training_data['state_context'] = macro_plan.get('state_context', {})
            
            # Regional Planning Data
            regional_plan = self.vehicle_regional_plans.get(vehicle_id, {})
            if regional_plan:
                rl_training_data['regional_route'] = regional_plan.get('route', [])
                rl_training_data['target_region'] = regional_plan.get('target_region', -1)
                rl_training_data['boundary_edge'] = regional_plan.get('boundary_edge', '')
                rl_training_data['regional_reasoning'] = regional_plan.get('reasoning', '')
            
            # Travel Metrics
            travel_metrics = self.vehicle_travel_metrics.get(vehicle_id, {})
            rl_training_data['travel_metrics'] = travel_metrics
            
            # Calculate normalized rewards for RL training
            rewards = self._calculate_normalized_rewards(vehicle_id, travel_time, completion_time)
            rl_training_data['rewards'] = rewards
            
            # Performance indicators for GRPO grouping
            rl_training_data['vehicle_region_start'] = None
            rl_training_data['vehicle_region_dest'] = None
            if macro_plan.get('macro_route'):
                route = macro_plan['macro_route']
                rl_training_data['vehicle_region_start'] = route[0] if route else None
                rl_training_data['vehicle_region_dest'] = route[-1] if route else None
            
            # System state at completion
            rl_training_data['system_att'] = self._calculate_current_att()
            rl_training_data['system_vehicle_count'] = len(self.vehicle_travel_metrics)
            rl_training_data['completion_order'] = self.completed_vehicles
            
            # Store completion time for ATT calculation
            self.completed_vehicle_times[vehicle_id] = travel_time
            
            # Send to training manager if queue is available
            if self.training_queue is not None:
                try:
                    self.training_queue.put(rl_training_data, block=False)
                    self.logger.log_info(f"RL_DATA_SENT: Training data for {vehicle_id} sent to training manager")
                except Exception as queue_error:
                    self.logger.log_warning(f"RL_DATA_QUEUE_ERROR: Failed to send training data for {vehicle_id}: {queue_error}")
            else:
                # Training mode not enabled - this is normal for non-training runs
                self.logger.log_info(f"RL_DATA_OFFLINE: Training disabled, storing data locally for {vehicle_id}")
                
                # Store training data locally for potential future analysis
                if not hasattr(self, '_offline_training_data'):
                    self._offline_training_data = []
                
                self._offline_training_data.append(rl_training_data)
                
                # Keep only recent data to avoid memory issues
                if len(self._offline_training_data) > 1000:
                    self._offline_training_data = self._offline_training_data[-500:]  # Keep last 500 records
                
        except Exception as e:
            self.logger.log_error(f"RL_DATA_COLLECTION_ERROR: Failed to collect training data for {vehicle_id}: {e}")
    
    def _calculate_normalized_rewards(self, vehicle_id: str, travel_time: float, completion_time: float) -> Dict[str, float]:
        """
        Calculate normalized rewards for RL training following CLAUDE.md specifications.
        
        Returns separate rewards for Traffic LLM and Regional LLM based on their roles:
        - Traffic LLM: ATT improvement + cooperation quality  
        - Regional LLM: Regional efficiency + individual protection + cooperation quality
        """
        try:
            rewards = {}
            
            # Get cooperation quality from CORY framework first
            macro_plan = self.vehicle_current_plans.get(vehicle_id, {})
            cooperation_quality = macro_plan.get('cooperation_quality', 0.5)
            
            # Adaptive expected travel time based on current system state
            expected_time = self._get_adaptive_expected_time(cooperation_quality)
            
            # ATT improvement calculation (for Traffic LLM)
            att_improvement = max(0, (expected_time - travel_time) / expected_time)
            att_reward = min(att_improvement * 2.0, 1.0)  # Cap at 1.0, multiply by 2.0 for sensitivity
            
            # Regional efficiency calculation (for Regional LLM)
            regional_metrics = self._calculate_regional_efficiency(vehicle_id, travel_time)
            efficiency_reward = min(regional_metrics * 2.0, 1.0)
            
            # Individual protection with adaptive fairness threshold
            fairness_threshold = self._get_adaptive_fairness_threshold(cooperation_quality)
            if travel_time <= fairness_threshold:
                individual_protection = 1.0
            else:
                excess_ratio = (travel_time - fairness_threshold) / fairness_threshold
                individual_protection = max(0.0, 1.0 - excess_ratio ** 1.2)  # Non-linear penalty
            
            # Traffic LLM rewards (Pioneer role)
            rewards['traffic_llm'] = {
                'att_reward': att_reward,
                'cooperation_reward': cooperation_quality,
                'total_reward': 0.6 * att_reward + 0.4 * cooperation_quality  # Weights from CLAUDE.md
            }
            
            # Regional LLM rewards (Observer role)  
            rewards['regional_llm'] = {
                'efficiency_reward': efficiency_reward,
                'individual_protection_reward': individual_protection,
                'cooperation_reward': cooperation_quality,
                'total_reward': 0.5 * efficiency_reward + 0.2 * individual_protection + 0.3 * cooperation_quality
            }
            
            self.logger.log_info(f"RL_REWARDS: {vehicle_id} -> Traffic:{rewards['traffic_llm']['total_reward']:.3f}, "
                               f"Regional:{rewards['regional_llm']['total_reward']:.3f}")
            
            return rewards
            
        except Exception as e:
            self.logger.log_error(f"RL_REWARD_CALCULATION_ERROR: {vehicle_id} -> {e}")
            # Return default rewards on error
            return {
                'traffic_llm': {'att_reward': 0.5, 'cooperation_reward': 0.5, 'total_reward': 0.5},
                'regional_llm': {'efficiency_reward': 0.5, 'individual_protection_reward': 0.5, 
                               'cooperation_reward': 0.5, 'total_reward': 0.5}
            }
    
    def _calculate_regional_efficiency(self, vehicle_id: str, travel_time: float) -> float:
        """
        Calculate regional efficiency based on vehicle's performance in different regions.
        
        Following CLAUDE.md: Regional efficiency considers vehicle's contribution to 
        regional traffic flow and its adherence to regional routing decisions.
        """
        try:
            efficiency_score = 0.5  # Default baseline
            
            # Get vehicle's macro route
            macro_plan = self.vehicle_current_plans.get(vehicle_id, {})
            macro_route = macro_plan.get('macro_route', [])
            
            if not macro_route:
                return efficiency_score
            
            # Calculate efficiency based on route length and travel time
            route_length = len(macro_route)
            
            # Get adaptive baseline time for this route
            cooperation_quality = macro_plan.get('cooperation_quality', 0.5)
            baseline_time = self._get_route_baseline_time(route_length, cooperation_quality)
            
            if route_length <= 2:  # Direct route
                # Direct routes get efficiency bonus
                time_efficiency = min(1.0, baseline_time / max(travel_time, baseline_time * 0.5))
                efficiency_score = time_efficiency * 1.2  # Bonus for direct routes
            else:
                # Multi-region routes: evaluate against adaptive expectations
                time_efficiency = min(1.0, baseline_time / max(travel_time, baseline_time * 0.5))
                efficiency_score = time_efficiency
            
            # Consider cooperation quality impact on regional efficiency
            cooperation_quality = macro_plan.get('cooperation_quality', 0.5)
            if cooperation_quality > 0.7:  # High quality cooperation
                efficiency_score *= 1.1  # Bonus for good cooperation
            elif cooperation_quality < 0.3:  # Poor cooperation
                efficiency_score *= 0.9  # Penalty for poor cooperation
            
            # Cap efficiency score at 1.0
            efficiency_score = min(efficiency_score, 1.0)
            
            return efficiency_score
            
        except Exception as e:
            self.logger.log_error(f"REGIONAL_EFFICIENCY_ERROR: {vehicle_id} -> {e}")
            return 0.5  # Default efficiency on error

    def _cleanup_completed_vehicle_plans(self, vehicle_id: str, current_time: float):
        """Clean up plans when vehicle completes journey."""
        try:
            # Clean up macro route plans
            if vehicle_id in self.vehicle_current_plans:
                macro_route = self.vehicle_current_plans[vehicle_id]['macro_route']
                for region_id in macro_route[1:]:
                    if region_id in self.region_vehicle_plans:
                        self.region_vehicle_plans[region_id] = max(0, self.region_vehicle_plans[region_id] - 1)
                
                # Clean up boundary plans
                for i in range(len(macro_route) - 1):
                    from_region = macro_route[i]
                    to_region = macro_route[i + 1]
                    boundary_edges = self.traffic_agent.region_connections.get((from_region, to_region), [])
                    for edge in boundary_edges:
                        if edge in self.boundary_vehicle_plans:
                            self.boundary_vehicle_plans[edge] = max(0, self.boundary_vehicle_plans[edge] - 1)
            
            # Broadcast completion
            completion_msg = {
                'type': 'VEHICLE_COMPLETION',
                'vehicle_id': vehicle_id,
                'timestamp': current_time,
                'message': f"Vehicle {vehicle_id} completed journey"
            }
            
            self.broadcast_messages.append(completion_msg)
            self.communication_log.append(completion_msg)
            
        except Exception as e:
            self.logger.log_error(f"CLEANUP_COMPLETION: Failed for {vehicle_id}: {e}")
    
    def _print_real_time_system_status(self, current_time: float):
        """Print real-time system status to console."""
        try:
            print(f"\n=== SYSTEM STATUS [{current_time:.1f}s] ===")
            print(f"Active Vehicles: {len(self.vehicle_travel_metrics)}")
            print(f"Completed: {self.completed_vehicles}/{len(self.autonomous_vehicles)}")
            print(f"Current ATT: {self._calculate_current_att():.1f}s")
            
            # Regional status
            print("Regional Vehicle Distribution:")
            region_counts = {}
            for vehicle_id, region_id in self.vehicle_regions.items():
                region_counts[region_id] = region_counts.get(region_id, 0) + 1
            
            for region_id in sorted(region_counts.keys()):
                count = region_counts[region_id]
                planned = self.region_vehicle_plans.get(region_id, 0)
                print(f"  Region {region_id}: {count} active, {planned} planned")
            
            # Recent decisions
            recent_decisions = len([v for v in self.vehicle_travel_metrics.values() 
                                  if current_time - v.get('last_update', 0) < 30])
            print(f"Recent decisions (30s): {recent_decisions}")
            print("=" * 50)
            
        except Exception as e:
            self.logger.log_error(f"REAL_TIME_STATUS: Failed to print status: {e}")
    
    def _process_broadcast_messages(self, current_time: float):
        """Process pending broadcast messages from the communication system."""
        try:
            if not hasattr(self, 'broadcast_messages') or not self.broadcast_messages:
                return
            
            # Process recent broadcast messages
            recent_messages = []
            for msg in self.broadcast_messages[-10:]:  # Process last 10 messages
                if current_time - msg['timestamp'] < 30:  # Messages from last 30 seconds
                    recent_messages.append(msg)
            
            if recent_messages:
                self.logger.log_info(f"BROADCAST: Processing {len(recent_messages)} recent messages")
                
                # Log important broadcast messages to console
                for msg in recent_messages[-3:]:  # Show last 3 messages
                    print(f"[{msg['timestamp']:.1f}s] {msg['type']}: {msg['message']}")
            
            # Clean old messages to prevent memory buildup
            if len(self.broadcast_messages) > 100:
                self.broadcast_messages = self.broadcast_messages[-50:]
                
        except Exception as e:
            self.logger.log_error(f"BROADCAST: Failed to process messages: {e}")
    
    def _llm_replan_macro_route(self, vehicle_id: str, original_route: Optional[List[int]], 
                               new_candidates: List[List[int]], old_region: int, new_region: int,
                               dest_region: int, current_time: float) -> Optional[List[int]]:
        """Use LLM to decide between original macro route and new candidates."""
        try:
            # Prepare all options for LLM
            all_options = []
            
            # Add original route if valid (and can continue from new region)
            if original_route and new_region in original_route:
                # Extract remaining part of original route
                try:
                    region_index = original_route.index(new_region)
                    remaining_original = original_route[region_index:]
                    if len(remaining_original) > 1:  # More than just current region
                        all_options.append(remaining_original)
                except ValueError:
                    pass  # new_region not in original route
            
            # Add new candidates
            for candidate in new_candidates:
                if candidate not in all_options:
                    all_options.append(candidate)
            
            if not all_options:
                return new_candidates[0] if new_candidates else None
            
            # Create observation for LLM replanning decision
            observation_text = self._create_replanning_observation(
                vehicle_id, original_route, all_options, old_region, new_region, dest_region, current_time
            )
            
            # Create answer options
            answer_options = "/".join([str(option) for option in all_options])
            
            # Use LLM for replanning decision
            call_id = self.logger.log_llm_call_start(
                "MacroReplanning", vehicle_id, len(observation_text)
            )
            
            try:
                # Use enhanced LLM if available
                if hasattr(self.llm_agent, 'macro_route_planning'):
                    # Use structured replanning approach
                    global_state = self._get_current_global_state()
                    route_requests = [{
                        'vehicle_id': vehicle_id,
                        'start_region': new_region,
                        'end_region': dest_region,
                        'possible_routes': all_options,
                        'route_urgency': 'normal',
                        'original_route': original_route,
                        'replanning_reason': f'Reached region {new_region} from {old_region}'
                    }]
                    
                    # Enhanced context for replanning
                    regional_conditions = self._get_regional_conditions()
                    boundary_analysis = self._get_boundary_analysis()
                    
                    llm_result = self.llm_agent.macro_route_planning(
                        global_state=global_state,
                        route_requests=route_requests,
                        regional_conditions=regional_conditions,
                        boundary_analysis=boundary_analysis,
                        flow_predictions={'replanning_context': True},
                        coordination_needs={'replanning_optimization': True},
                        region_routes={}
                    )
                    
                    # Extract selected route
                    macro_routes = llm_result.get('macro_routes', [])
                    if macro_routes:
                        selected_route = macro_routes[0].get('planned_route', all_options[0])
                        reasoning = macro_routes[0].get('reasoning', 'LLM replanning decision')
                    else:
                        selected_route = all_options[0]
                        reasoning = 'Fallback to first option'
                        
                else:
                    # Basic LLM decision making if available
                    if self.llm_agent and hasattr(self.llm_agent, 'hybrid_decision_making_pipeline'):
                        decisions = self.llm_agent.hybrid_decision_making_pipeline(
                            [observation_text], [f'"{answer_options}"']
                        )
                        
                        if decisions and decisions[0]['answer']:
                            selected_route = self._parse_macro_route_answer(decisions[0]['answer'], all_options)
                            reasoning = decisions[0].get('summary', 'LLM replanning decision')
                        else:
                            selected_route = all_options[0]
                            reasoning = 'Fallback decision - LLM failed'
                    else:
                        selected_route = all_options[0]  # Fallback when llm_agent is None
                        reasoning = 'Fallback decision - LLM agent not available'
                
                self.logger.log_llm_call_end(
                    call_id, True, f"Replanned route: {selected_route}. Reasoning: {reasoning}",
                    len(observation_text)
                )
                
                return selected_route
                
            except Exception as llm_error:
                self.logger.log_llm_call_end(
                    call_id, False, "LLM replanning failed",
                    len(observation_text), str(llm_error)
                )
                
                # Fallback: prefer original route if valid, otherwise first new candidate
                if original_route and new_region in original_route:
                    region_index = original_route.index(new_region)
                    return original_route[region_index:]
                else:
                    return all_options[0] if all_options else None
                    
        except Exception as e:
            self.logger.log_error(f"LLM_REPLAN: Failed for {vehicle_id}: {e}")
            return new_candidates[0] if new_candidates else None
    
    def _create_replanning_observation(self, vehicle_id: str, original_route: Optional[List[int]],
                                     options: List[List[int]], old_region: int, new_region: int,
                                     dest_region: int, current_time: float) -> str:
        """Create observation text for LLM replanning decision."""
        observation_parts = []
        
        observation_parts.append(f"MACRO ROUTE REPLANNING FOR VEHICLE {vehicle_id}")
        observation_parts.append(f"Vehicle moved from Region {old_region} to Region {new_region}")
        observation_parts.append(f"Final destination: Region {dest_region}")
        observation_parts.append(f"Current time: {current_time:.1f}s")
        observation_parts.append("")
        
        # Show original route
        if original_route:
            observation_parts.append(f"ORIGINAL MACRO ROUTE: {original_route}")
            # Calculate performance of original route so far
            travel_time = current_time - self.vehicle_start_times.get(vehicle_id, current_time)
            observation_parts.append(f"Travel time so far: {travel_time:.1f}s")
        else:
            observation_parts.append("ORIGINAL MACRO ROUTE: None (first planning)")
        observation_parts.append("")
        
        # Show available options
        observation_parts.append("AVAILABLE OPTIONS:")
        for i, option in enumerate(options):
            option_desc = self._create_macro_route_description(option, current_time)
            is_original = (original_route and new_region in original_route and 
                          option == original_route[original_route.index(new_region):])
            marker = " (CONTINUE ORIGINAL)" if is_original else " (NEW ROUTE)"
            observation_parts.append(f"Option {i+1}: {option_desc}{marker}")
        observation_parts.append("")
        
        # Current system state
        current_state = self.traffic_agent.global_state_history[-1] if self.traffic_agent.global_state_history else None
        if current_state:
            observation_parts.append("CURRENT SYSTEM STATE:")
            observation_parts.append(f"Total vehicles: {current_state.total_vehicles}")
            observation_parts.append(f"Average travel time: {current_state.avg_travel_time:.1f}s")
            observation_parts.append("")
            
            observation_parts.append("REGIONAL CONGESTION:")
            for region_id in [new_region, dest_region]:
                if region_id in current_state.regional_congestion:
                    congestion = current_state.regional_congestion[region_id]
                    planned = self.region_vehicle_plans.get(region_id, 0)
                    observation_parts.append(f"Region {region_id}: Congestion={congestion:.1f}, Planned={planned}")
        
        observation_parts.append("")
        observation_parts.append("OBJECTIVE: Choose the best route to minimize travel time while")
        observation_parts.append("avoiding congestion and balancing system load. Consider whether to")
        observation_parts.append("continue the original plan or switch to a potentially better route.")
        
        return "\n".join(observation_parts)
    
    def _identify_regions_needing_planning(self, current_time: float) -> List[int]:
        """Identify regions that need batch regional planning based on vehicle requests."""
        regions_needing_planning = []
        
        try:
            # Check each region for vehicles needing regional planning
            for region_id, regional_agent in self.regional_agents.items():
                region_needs_planning = False
                
                # Get vehicles in this region that need regional planning
                vehicles_in_region = []
                for vehicle_id, vehicle_region in self.vehicle_regions.items():
                    if vehicle_region == region_id and vehicle_id in self.vehicle_current_plans:
                        # Check if vehicle has macro plan but no regional plan yet
                        if vehicle_id not in self.vehicle_regional_plans:
                            vehicles_in_region.append(vehicle_id)
                            region_needs_planning = True
                        # Or if regional plan is outdated
                        elif current_time - self.vehicle_regional_plans[vehicle_id].get('creation_time', 0) > 600:
                            vehicles_in_region.append(vehicle_id)
                            region_needs_planning = True
                
                # Also check for vehicles that recently changed regions
                recent_arrivals = []
                for vehicle_id in vehicles_in_region:
                    if vehicle_id in self.vehicle_current_plans:
                        last_update = self.vehicle_current_plans[vehicle_id].get('last_update', 0)
                        if current_time - last_update < 30:  # Recent update within 30 seconds
                            recent_arrivals.append(vehicle_id)
                
                # Region needs planning if has vehicles needing plans or recent arrivals
                if region_needs_planning or recent_arrivals:
                    self.logger.log_info(f"BATCH_PLANNING: Region {region_id} needs planning - "
                                       f"{len(vehicles_in_region)} vehicles need plans, "
                                       f"{len(recent_arrivals)} recent arrivals")
                    regions_needing_planning.append(region_id)
            
            return regions_needing_planning
            
        except Exception as e:
            self.logger.log_error(f"BATCH_PLANNING: Failed to identify regions needing planning: {e}")
            return []
    
    def _execute_batch_regional_planning(self, regions_needing_planning: List[int], current_time: float):
        """Execute batch asynchronous regional planning across multiple regions."""
        try:
            # Create futures for asynchronous execution
            import concurrent.futures
            planning_futures = {}
            
            self.logger.log_info(f"BATCH_PLANNING: Starting asynchronous planning for {len(regions_needing_planning)} regions")
            
            # Submit planning tasks to thread pool for asynchronous execution
            for region_id in regions_needing_planning:
                if region_id in self.regional_agents:
                    # Collect all vehicles in this region needing planning
                    vehicles_needing_planning = []
                    for vehicle_id, vehicle_region in self.vehicle_regions.items():
                        if (vehicle_region == region_id and 
                            vehicle_id in self.vehicle_current_plans and 
                            vehicle_id not in self.vehicle_regional_plans):
                            vehicles_needing_planning.append(vehicle_id)
                    
                    if vehicles_needing_planning:
                        # Submit planning task for this region
                        future = self.executor.submit(
                            self._execute_regional_planning_for_vehicles,
                            region_id, vehicles_needing_planning, current_time
                        )
                        planning_futures[region_id] = {
                            'future': future,
                            'vehicles': vehicles_needing_planning,
                            'start_time': current_time
                        }
            
            # Wait for all planning to complete with timeout
            completed_regions = []
            failed_regions = []
            
            for region_id, planning_data in planning_futures.items():
                try:
                    # Wait for completion with timeout
                    result = planning_data['future'].result(timeout=3000)  # 30 second timeout per region
                    
                    if result:
                        completed_regions.append(region_id)
                        planning_time = current_time - planning_data['start_time']
                        self.logger.log_info(f"BATCH_PLANNING: Region {region_id} completed "
                                           f"{len(planning_data['vehicles'])} vehicle plans in {planning_time:.1f}s")
                    else:
                        failed_regions.append(region_id)
                        self.logger.log_warning(f"BATCH_PLANNING: Region {region_id} planning failed")
                        
                except concurrent.futures.TimeoutError:
                    failed_regions.append(region_id)
                    self.logger.log_warning(f"BATCH_PLANNING: Region {region_id} planning timed out")
                    
                except Exception as region_error:
                    failed_regions.append(region_id)
                    self.logger.log_error(f"BATCH_PLANNING: Region {region_id} planning error: {region_error}")
            
            # Summary logging
            total_regions = len(regions_needing_planning)
            success_rate = len(completed_regions) / total_regions if total_regions > 0 else 0
            
            self.logger.log_info(f"BATCH_PLANNING_COMPLETE: {len(completed_regions)}/{total_regions} regions completed "
                               f"({success_rate:.1%} success rate)")
            
            if failed_regions:
                self.logger.log_warning(f"BATCH_PLANNING: Failed regions: {failed_regions}")
            
        except Exception as e:
            self.logger.log_error(f"BATCH_PLANNING: Critical error in batch regional planning: {e}")
    
    def _execute_regional_planning_for_vehicles(self, region_id: int, vehicle_ids: List[str], current_time: float) -> bool:
        """Execute regional planning for multiple vehicles in one region - designed for single LLM conversation."""
        try:
            regional_agent = self.regional_agents[region_id]
            
            self.logger.log_info(f"REGIONAL_BATCH: Planning for {len(vehicle_ids)} vehicles in region {region_id}")
            
            # Process each vehicle's regional planning needs
            successful_plans = 0
            
            for vehicle_id in vehicle_ids:
                try:
                    # Get target region from macro plan
                    if vehicle_id not in self.vehicle_current_plans:
                        continue
                    
                    macro_route = self.vehicle_current_plans[vehicle_id]['macro_route']
                    current_region_index = self.vehicle_current_plans[vehicle_id].get('current_region_index', 0)
                    
                    # Determine next region from macro route
                    if current_region_index + 1 < len(macro_route):
                        target_region = macro_route[current_region_index + 1]
                        
                        # Execute regional planning for this vehicle
                        regional_plan = regional_agent.make_regional_route_planning(
                            vehicle_id, target_region, current_time
                        )
                        
                        if regional_plan and 'boundary_edge' in regional_plan and 'route' in regional_plan:
                            # Store successful regional plan
                            self.vehicle_regional_plans[vehicle_id] = {
                                'region_id': region_id,
                                'target_region': target_region,
                                'boundary_edge': regional_plan['boundary_edge'],
                                'route': regional_plan['route'],
                                'creation_time': current_time,
                                'travel_time': regional_plan.get('travel_time', 0),
                                'reasoning': regional_plan.get('reasoning', 'Batch regional planning')
                            }
                            
                            # Apply route to vehicle in SUMO with safety checks
                            if regional_plan['route'] and len(regional_plan['route']) > 0:
                                try:
                                    # Ensure safe route setting
                                    current_edge = traci.vehicle.getRoadID(vehicle_id)
                                    safe_route = self._create_safe_route(current_edge, regional_plan['route'])
                                    if safe_route:
                                        traci.vehicle.setRoute(vehicle_id, safe_route)
                                        successful_plans += 1
                                        
                                        self.logger.log_info(f"REGIONAL_BATCH: {vehicle_id} assigned safe route to "
                                                           f"{regional_plan['boundary_edge']} "
                                                           f"(travel_time: {regional_plan.get('travel_time', 'unknown')}s)")
                                    else:
                                        self.logger.log_warning(f"REGIONAL_BATCH: Cannot create safe route for {vehicle_id}")
                                except Exception as route_error:
                                    self.logger.log_error(f"REGIONAL_BATCH: Failed to set route for {vehicle_id}: {route_error}")
                        else:
                            self.logger.log_warning(f"REGIONAL_BATCH: Invalid plan structure for {vehicle_id}")
                    else:
                        # Vehicle reached final region
                        self.logger.log_info(f"REGIONAL_BATCH: {vehicle_id} reached final region {region_id}")
                        successful_plans += 1
                        
                except Exception as vehicle_error:
                    self.logger.log_error(f"REGIONAL_BATCH: Failed planning for {vehicle_id}: {vehicle_error}")
                    continue
            
            success_rate = successful_plans / len(vehicle_ids) if vehicle_ids else 0
            self.logger.log_info(f"REGIONAL_BATCH_COMPLETE: Region {region_id} completed {successful_plans}/{len(vehicle_ids)} "
                               f"vehicle plans ({success_rate:.1%} success)")
            
            return success_rate >= 0.5  # Consider successful if at least 50% of vehicles got plans
            
        except Exception as e:
            self.logger.log_error(f"REGIONAL_BATCH: Critical error for region {region_id}: {e}")
            return False
    
    def _create_replanning_observation(self, vehicle_id: str, original_route: Optional[List[int]],
                                     options: List[List[int]], old_region: int, new_region: int,
                                     dest_region: int, current_time: float) -> str:
        """Create observation text for LLM replanning decision."""
        observation_parts = []
        
        observation_parts.append(f"MACRO ROUTE REPLANNING FOR VEHICLE {vehicle_id}")
        observation_parts.append(f"Vehicle moved from Region {old_region} to Region {new_region}")
        observation_parts.append(f"Final destination: Region {dest_region}")
        observation_parts.append(f"Current time: {current_time:.1f}s")
        observation_parts.append("")
        
        # Show original route
        if original_route:
            observation_parts.append(f"ORIGINAL MACRO ROUTE: {original_route}")
            # Calculate performance of original route so far
            travel_time = current_time - self.vehicle_start_times.get(vehicle_id, current_time)
            observation_parts.append(f"Travel time so far: {travel_time:.1f}s")
        else:
            observation_parts.append("ORIGINAL MACRO ROUTE: None (first planning)")
        observation_parts.append("")
        
        # Show available options
        observation_parts.append("AVAILABLE OPTIONS:")
        for i, option in enumerate(options):
            option_desc = self._create_macro_route_description(option, current_time)
            is_original = (original_route and new_region in original_route and 
                          option == original_route[original_route.index(new_region):])
            marker = " (CONTINUE ORIGINAL)" if is_original else " (NEW ROUTE)"
            observation_parts.append(f"Option {i+1}: {option_desc}{marker}")
        observation_parts.append("")
        
        # Current system state
        current_state = self.traffic_agent.global_state_history[-1] if self.traffic_agent.global_state_history else None
        if current_state:
            observation_parts.append("CURRENT SYSTEM STATE:")
            observation_parts.append(f"Total vehicles: {current_state.total_vehicles}")
            observation_parts.append(f"Average travel time: {current_state.avg_travel_time:.1f}s")
            observation_parts.append("")
            
            observation_parts.append("REGIONAL CONGESTION:")
            for region_id in [new_region, dest_region]:
                if region_id in current_state.regional_congestion:
                    congestion = current_state.regional_congestion[region_id]
                    planned = self.region_vehicle_plans.get(region_id, 0)
                    observation_parts.append(f"Region {region_id}: Congestion={congestion:.1f}, Planned={planned}")
        
        observation_parts.append("")
        observation_parts.append("OBJECTIVE: Choose the best route to minimize travel time while")
        observation_parts.append("avoiding congestion and balancing system load. Consider whether to")
        observation_parts.append("continue the original plan or switch to a potentially better route.")
        
        return "\n".join(observation_parts)
    


# ===== GLOBAL ACCESS FUNCTIONS FOR PROGRESSIVE TRAINING =====

def get_global_llm_manager():
    """Get the currently registered global LLM manager."""
    global _global_llm_manager_registry
    return _global_llm_manager_registry.get("current")

def get_llm_manager_by_key(key: str):
    """Get LLM manager by specific key."""
    global _global_llm_manager_registry
    return _global_llm_manager_registry.get(key)

def list_registered_llm_managers():
    """List all registered LLM managers."""
    global _global_llm_manager_registry
    return list(_global_llm_manager_registry.keys())

def clear_llm_manager_registry():
    """Clear the global LLM manager registry."""
    global _global_llm_manager_registry
    _global_llm_manager_registry.clear()

