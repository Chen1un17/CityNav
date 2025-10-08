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
import signal
import queue
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Set, Optional, Tuple, Any
import networkx as nx
import traci
import numpy as np
import matplotlib
# Set backend for server environments without display
import os
if os.environ.get('DISPLAY') is None or os.environ.get('SSH_CLIENT') is not None:
    matplotlib.use('Agg')  # Headless mode
else:
    try:
        import tkinter
        matplotlib.use('TkAgg')
    except ImportError:
        matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

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
                 log_dir: str = "logs", task_info=None, use_local_llm: bool = True, training_queue=None,
                 use_time_sliced_training: bool = True, start_time: float = 0, av_ratio: float = 0.02,
                 traffic_lora_path: str = None, regional_lora_path: str = None):
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
            use_time_sliced_training: Whether to use time-sliced training
            start_time: Simulation start time in seconds (default: 0)
            av_ratio: Ratio of autonomous vehicles from first route file (default: 0.02)
            traffic_lora_path: Path to Traffic LLM LoRA adapter (default: None)
            regional_lora_path: Path to Regional LLM LoRA adapter (default: None)
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
        self.start_time = start_time
        self.av_ratio = av_ratio
        self.traffic_lora_path = traffic_lora_path
        self.regional_lora_path = regional_lora_path
        
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
        
        # Time-sliced Training Configuration
        # Enable only if training queue provided and explicitly requested
        self.enable_time_sliced_training = (training_queue is not None) and bool(use_time_sliced_training)
        self.training_threshold = {'traffic': 8, 'regional': 12}  # Training trigger thresholds
        self.training_data_buffer = {'traffic': [], 'regional': []}  # Buffer for training data
        self.is_training_active = False  # Flag to indicate training is in progress
        self.simulation_paused = False  # Flag to indicate simulation is paused for training
        self.training_session_count = 0  # Counter for training sessions
        # 仅在确有新LoRA时才恢复：默认不在时间分片训练时释放推理模型，避免空重载
        self.release_models_during_time_sliced_training = False
        self.models_released = False
        
        # Circuit breaker for LLM call management
        self.llm_failure_counts = {}  # region_id -> (failure_count, last_failure_time)
        self.llm_circuit_breaker_threshold = 3  # failures before circuit breaker activates
        self.llm_circuit_breaker_timeout = 300  # 5 minutes recovery time
        
        # Decision queue management - limit to 2 concurrent LLM calls
        self.decision_queue = queue.Queue()
        self.active_decisions = {}  # decision_id -> future
        self.max_concurrent_decisions = 2
        self.decision_lock = threading.Lock()
        self.regional_planning_results = {}  # vehicle_id -> regional planning result
        
        # Pending routes for vehicles on junction edges
        self.pending_routes = {}  # vehicle_id -> route_info (to apply when vehicle exits junction)
        
        
        # Hot-reload mechanism for LoRA adapters
        self.current_lora_adapters = {
            'traffic': None,  # Current Traffic LLM LoRA adapter name
            'regional': None  # Current Regional LLM LoRA adapter name
        }
        self.lora_update_lock = threading.Lock()  # Thread-safe adapter updates
        self.training_lock = threading.Lock()     # Time-sliced training synchronization
        
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
        load_static_road_data(
            data_dir=self.region_data_dir,
            road_info_file=self.road_info_file,
            adjacency_file=self.adjacency_file
        )
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
        
        # Vehicle metrics tracking for completed vehicles
        self.vehicle_waiting_times_last = {}  # Last waiting time snapshot for active vehicles
        self.vehicle_delay_times_last = {}    # Last delay/time loss snapshot for active vehicles
        self.vehicle_waiting_times_final = {} # Final waiting time for completed vehicles
        self.vehicle_delay_times_final = {}   # Final delay/time loss for completed vehicles
        
        # Communication and broadcasting system - enhanced for LLM coordination
        self.boundary_vehicle_plans = {}  # Track vehicles planning to reach each boundary
        self.region_vehicle_plans = {}    # Track vehicles planning to reach each region
        self.communication_log = []       # Log all communication events
        self.broadcast_messages = []       # Queue for broadcast messages
        # Global macro guidance cache (timestamp-based)
        self.global_macro_guidance = {
            'data': None,           # last guidance dict
            'expire_at': 0.0        # sim time when expires
        }
        self.vehicle_plan_updates = {}     # Track real-time vehicle plan updates
        
        # Real-time tracking for LLM decisions
        self.vehicle_current_plans = {}    # Current macro routes for each vehicle
        self.vehicle_regional_plans = {}   # Current regional routes for each vehicle
        self.vehicle_travel_metrics = {}   # Real-time travel metrics per vehicle
        
        # Threading for parallel processing - match decision queue limit
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="decision_worker")
        # 并行宏观/区域规划已回退，移除专用执行器与每区限流
        
        # Step-wise traffic state caching for optimization
        self._step_traffic_state_cache = {
            'current_step': -1,
            'regional_congestion': None,
            'boundary_flows': None,
            'global_state': None,
            'traffic_predictions': None,
            'cache_timestamp': -1
        }
        
        # Idle/movement tracking for improved STUCK detection
        self.vehicle_last_movement_time: Dict[str, float] = {}
        self.vehicle_last_edge: Dict[str, str] = {}
        self.vehicle_edge_entry_time: Dict[str, float] = {}
        self.vehicle_last_lane_position: Dict[str, float] = {}
        
        # Lane-exit hamper monitoring cache (for hotspot-aware routing)
        self.exit_hamper_counts: Dict[str, int] = {}
        self._hamper_samples: int = 0
        self.hotspot_avoid_edges: Set[str] = set()
        
        # Stuck-event tracking for RL penalty and analytics
        self.vehicle_stuck_events: Dict[str, int] = {}
        self.currently_stuck_vehicles: Set[str] = set()
        # Stuck zone management: active blocked edges and cooldown control
        self.vehicle_stuck_since: Dict[str, float] = {}
        self.vehicle_stuck_zone_edges: Dict[str, Set[str]] = {}
        self.blocked_edges_active: Set[str] = set()
        self.cooldown_window_sec: float = 720.0
        self.cooldown_zones: List[Dict[str, Any]] = []  # {edges:set, expire_at:float, assignments:int}
        self.cooldown_edges: Set[str] = set()
        self.max_zone_assignments_per_window: int = 30
        
        # Parameters for congestion-aware emergency routing
        self.congestion_weight_alpha: float = 1.2  # occupancy influence
        self.congestion_weight_beta: float = 0.8   # hamper rate influence
        self.hotspot_edge_penalty: float = 5.0     # strong penalty for hotspot edges
        
        # Async LLM calling and caching system
        self._initialize_async_llm_system()
        
        # Real-time monitoring state
        self.current_step = 0
        self.current_sim_time = 0.0
        self.active_autonomous_vehicles = 0
        self.att_calculation = 0.0
        self.system_throughput = 0.0
        
        # Visualization state variables
        self._initialize_visualization_system()
        
        # Feature flags from environment
        try:
            self.disable_global_guidance = bool(int(os.environ.get('DISABLE_GLOBAL_GUIDANCE', '0')))
        except Exception:
            self.disable_global_guidance = False

    def _initialize_visualization_system(self):
        """Initialize visualization components for real-time monitoring."""
        # Visualization update interval (every 3600 steps)
        self.vis_update_interval = 3600
        self.total_time_slots = 86400 // self.vis_update_interval  # 24 time slots
        self.current_time_slot = 0
        
        # Regional congestion data (75 regions x time_slots)
        self.congestion_matrix = np.zeros((75, self.total_time_slots))
        self.region_labels = [f"Region {i}" for i in range(75)]
        self.time_labels = [f"T{i}" for i in range(self.total_time_slots)]
        
        # Metrics for line plots
        self.att_history = []
        self.throughput_history = []
        self.co2_history = []
        self.time_history = []
        
        # Matplotlib figure setup
        self.fig = None
        self.axes = None
        self.heatmap_im = None
        self.line_plots = {}
        
        # Initialize plots
        self._setup_visualization_plots()
        
    def _setup_visualization_plots(self):
        """Setup matplotlib plots for real-time visualization."""
        plt.ion()  # Enable interactive mode
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(2, 3, figure=self.fig, height_ratios=[2, 1])
        
        # Heatmap for regional congestion
        self.axes = {
            'heatmap': self.fig.add_subplot(gs[0, :]),
            'att': self.fig.add_subplot(gs[1, 0]),
            'throughput': self.fig.add_subplot(gs[1, 1]),
            'co2': self.fig.add_subplot(gs[1, 2])
        }
        
        # Initialize heatmap
        self.heatmap_im = self.axes['heatmap'].imshow(
            self.congestion_matrix, 
            cmap='hot', 
            aspect='auto',
            vmin=0, vmax=1
        )
        self.axes['heatmap'].set_title('Regional Traffic Congestion Heatmap')
        self.axes['heatmap'].set_xlabel('Time Slots')
        self.axes['heatmap'].set_ylabel('Regions')
        self.axes['heatmap'].set_yticks(range(0, 75, 5))
        self.axes['heatmap'].set_yticklabels([f"R{i}" for i in range(0, 75, 5)])
        plt.colorbar(self.heatmap_im, ax=self.axes['heatmap'])
        
        # Initialize line plots
        self.line_plots['att'], = self.axes['att'].plot([], [], 'b-', linewidth=2)
        self.axes['att'].set_title('Average Travel Time (ATT)')
        self.axes['att'].set_xlabel('Time Slots')
        self.axes['att'].set_ylabel('ATT (seconds)')
        self.axes['att'].grid(True)
        
        self.line_plots['throughput'], = self.axes['throughput'].plot([], [], 'g-', linewidth=2)
        self.axes['throughput'].set_title('System Throughput')
        self.axes['throughput'].set_xlabel('Time Slots')
        self.axes['throughput'].set_ylabel('Vehicles/hour')
        self.axes['throughput'].grid(True)
        
        self.line_plots['co2'], = self.axes['co2'].plot([], [], 'r-', linewidth=2)
        self.axes['co2'].set_title('Average CO2 Emission')
        self.axes['co2'].set_xlabel('Time Slots')
        self.axes['co2'].set_ylabel('CO2 Volume')
        self.axes['co2'].grid(True)
        
        plt.tight_layout()
        plt.show(block=False)
        
    def _update_visualization(self, current_time):
        """Update visualization plots with current data."""
        try:
            # Calculate current time slot
            time_slot = min(int(current_time // self.vis_update_interval), self.total_time_slots - 1)
            self.current_time_slot = time_slot
            
            # Update congestion matrix
            self._update_congestion_matrix(time_slot)
            
            # Calculate current metrics
            current_att = self._calculate_current_att()
            current_throughput = self._calculate_current_throughput()
            current_co2 = self._calculate_current_co2()
            
            # Update history
            if len(self.att_history) <= time_slot:
                self.att_history.extend([0] * (time_slot - len(self.att_history) + 1))
                self.throughput_history.extend([0] * (time_slot - len(self.throughput_history) + 1))
                self.co2_history.extend([0] * (time_slot - len(self.co2_history) + 1))
                self.time_history.extend(list(range(len(self.time_history), time_slot + 1)))
            
            self.att_history[time_slot] = current_att
            self.throughput_history[time_slot] = current_throughput
            self.co2_history[time_slot] = current_co2
            
            # Update plots
            self._refresh_plots()
            
            self.logger.log_info(f"VISUALIZATION: Updated plots at time slot {time_slot}")
            
        except Exception as e:
            self.logger.log_error(f"VISUALIZATION_ERROR: {e}")
    
    def _update_congestion_matrix(self, time_slot):
        """Update congestion matrix with current regional data."""
        for region_id in range(75):
            region_edges = [edge for edge, reg in self.edge_to_region.items() if reg == region_id]
            if region_edges:
                total_congestion = 0
                valid_edges = 0
                
                for edge in region_edges[:10]:  # Sample first 10 edges for performance
                    try:
                        occupancy = traci.edge.getLastStepOccupancy(edge)
                        total_congestion += occupancy
                        valid_edges += 1
                    except:
                        continue
                
                avg_congestion = total_congestion / max(valid_edges, 1)
                self.congestion_matrix[region_id, time_slot] = avg_congestion
    
    def _calculate_current_att(self):
        """Calculate current average travel time."""
        if self.vehicle_end_times and self.vehicle_start_times:
            completed_times = []
            for veh_id in self.vehicle_end_times:
                if veh_id in self.vehicle_start_times:
                    travel_time = self.vehicle_end_times[veh_id] - self.vehicle_start_times[veh_id]
                    completed_times.append(travel_time)
            
            if completed_times:
                return sum(completed_times) / len(completed_times)
        return 0
    
    def _calculate_current_throughput(self):
        """Calculate current system throughput."""
        return len(self.vehicle_end_times) / max(self.current_sim_time / 3600, 0.1)  # vehicles per hour
    
    def _calculate_current_co2(self):
        """Calculate current average CO2 emission."""
        total_co2 = 0
        vehicle_count = 0
        
        try:
            for veh_id in traci.vehicle.getIDList():
                try:
                    co2_emission = traci.vehicle.getCO2Emission(veh_id)
                    total_co2 += co2_emission
                    vehicle_count += 1
                except:
                    continue
            
            return total_co2 / max(vehicle_count, 1)
        except:
            return 0
    
    def _refresh_plots(self):
        """Refresh all visualization plots."""
        try:
            # Update heatmap
            self.heatmap_im.set_array(self.congestion_matrix)
            
            # Update line plots
            valid_indices = list(range(len(self.att_history)))
            
            if valid_indices:
                self.line_plots['att'].set_data(valid_indices, self.att_history)
                self.axes['att'].relim()
                self.axes['att'].autoscale_view()
                
                self.line_plots['throughput'].set_data(valid_indices, self.throughput_history)
                self.axes['throughput'].relim()
                self.axes['throughput'].autoscale_view()
                
                self.line_plots['co2'].set_data(valid_indices, self.co2_history)
                self.axes['co2'].relim()
                self.axes['co2'].autoscale_view()
            
            # Refresh display
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception as e:
            self.logger.log_error(f"PLOT_REFRESH_ERROR: {e}")
    
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
    
    @contextmanager
    def timeout_context(self, seconds):
        """Context manager for LLM call timeouts."""
        def signal_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {seconds} seconds")
        
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
    
    def _enqueue_decision(self, decision_type: str, decision_data: Dict):
        """Add a decision request to the queue."""
        decision_id = f"{decision_type}_{time.time()}"
        decision_item = {
            'id': decision_id,
            'type': decision_type,
            'data': decision_data,
            'timestamp': time.time()
        }
        self.decision_queue.put(decision_item)
        
        # Debug logging
        with self.decision_lock:
            active_count = len(self.active_decisions)
            queue_size = self.decision_queue.qsize()
        self.logger.log_info(f"DECISION_QUEUE: Enqueued {decision_id} ({decision_type}) - active: {active_count}, queued: {queue_size}")
        
        return decision_id
        
    def _process_decision_queue(self):
        """Process decisions from queue with concurrent limit of 2."""
        # Clean up completed decisions first
        completed_decisions = []
        
        with self.decision_lock:
            for decision_id, future in list(self.active_decisions.items()):
                if future.done():
                    completed_decisions.append(decision_id)
                    # Get result to handle any exceptions
                    try:
                        result = future.result()
                        self.logger.log_info(f"DECISION_QUEUE: Decision {decision_id} completed successfully")
                    except Exception as e:
                        self.logger.log_error(f"DECISION_QUEUE: Decision {decision_id} failed: {e}")
            
            # Remove completed decisions
            for decision_id in completed_decisions:
                if decision_id in self.active_decisions:
                    del self.active_decisions[decision_id]
        
        # Start new decisions if under limit and queue has items
        while True:
            with self.decision_lock:
                active_count = len(self.active_decisions)
                queue_empty = self.decision_queue.empty()
                
            # Break if at limit or no items in queue
            if active_count >= self.max_concurrent_decisions or queue_empty:
                break
                
            # Try to get next decision item
            try:
                decision_item = self.decision_queue.get_nowait()
                
                # Submit to executor
                future = self.executor.submit(self._execute_single_decision, decision_item)
                
                with self.decision_lock:
                    self.active_decisions[decision_item['id']] = future
                    
                self.logger.log_info(f"DECISION_QUEUE: Started decision {decision_item['id']}, active: {len(self.active_decisions)}")
                
            except queue.Empty:
                break
            except Exception as e:
                self.logger.log_error(f"DECISION_QUEUE: Failed to start decision: {e}")
                break
                    
    def _execute_single_decision(self, decision_item: Dict):
        """Execute a single decision (macro or regional planning)."""
        decision_type = decision_item['type']
        decision_data = decision_item['data']
        
        try:
            if decision_type == 'macro':
                return self._execute_single_macro_decision(decision_data)
            elif decision_type == 'regional':
                return self._execute_single_regional_decision(decision_data)
            else:
                self.logger.log_error(f"DECISION_QUEUE: Unknown decision type {decision_type}")
                return None
        except Exception as e:
            self.logger.log_error(f"DECISION_QUEUE: Failed to execute {decision_type} decision: {e}")
            return None
            
    def _execute_single_macro_decision(self, decision_data: Dict):
        """Execute single vehicle macro planning."""
        vehicle_id = decision_data['vehicle_id']
        start_region = decision_data['start_region']
        end_region = decision_data['end_region']
        current_time = decision_data['current_time']
        coordination_data = decision_data.get('coordination_data')
        
        route = self.traffic_agent.plan_single_macro_route(
            vehicle_id, start_region, end_region, current_time, coordination_data
        )
        
        if route:
            with self.decision_lock:
                self.upcoming_vehicle_cache[vehicle_id] = {
                    'macro_route': route,
                    'planned_at': current_time
                }
            
        return route
        
    def _execute_single_regional_decision(self, decision_data: Dict):
        """Execute single vehicle regional planning."""
        vehicle_id = decision_data['vehicle_id']
        region_id = decision_data['region_id']
        target_region = decision_data['target_region']
        current_time = decision_data['current_time']
        vehicle_info = decision_data.get('vehicle_info')
        
        regional_agent = self.regional_agents.get(region_id)
        if not regional_agent:
            return None
        
        try:
            # For pre-planning vehicles, use modified approach with vehicle_info
            if vehicle_info and 'start_edge' in vehicle_info:
                # Pre-planning: use vehicle_info data instead of TraCI
                start_edge = vehicle_info['start_edge']
                
                # Generate candidate routes
                boundary_candidates = regional_agent._get_boundary_candidates_to_region(target_region)
                if not boundary_candidates:
                    boundary_candidates = regional_agent.outgoing_boundaries[:3]
                    
                route_candidates = regional_agent._generate_regional_route_candidates(
                    start_edge, boundary_candidates, current_time
                )
                
                if route_candidates:
                    # Use LLM to select best route
                    result = regional_agent._llm_select_regional_route(
                        vehicle_id, start_edge, route_candidates, target_region, current_time
                    )
                    
                    # Store result for later application
                    if result:
                        with self.decision_lock:
                            if not hasattr(self, 'regional_planning_results'):
                                self.regional_planning_results = {}
                            self.regional_planning_results[vehicle_id] = result
                            
                    return result
                else:
                    self.logger.log_warning(f"REGIONAL_PLANNING: No route candidates for {vehicle_id}")
                    return None
                    
            else:
                # Real-time planning: use existing method with TraCI
                result = regional_agent.make_regional_route_planning(vehicle_id, target_region, current_time)
                
                # Store result for later application
                if result:
                    with self.decision_lock:
                        if not hasattr(self, 'regional_planning_results'):
                            self.regional_planning_results = {}
                        self.regional_planning_results[vehicle_id] = result
                        
                return result
                
        except Exception as e:
            self.logger.log_error(f"REGIONAL_PLANNING: Failed for {vehicle_id}: {e}")
            return None
        
    def _wait_for_decisions(self, timeout: float = None):
        """Wait for all queued decisions to complete."""
        start_time = time.time()
        
        if timeout is None:
            # No timeout - wait until all decisions complete
            while not self.decision_queue.empty() or self.active_decisions:
                self._process_decision_queue()
                
                # Debug logging
                active_count = len(self.active_decisions)
                queue_size = self.decision_queue.qsize()
                elapsed = time.time() - start_time
                
                if elapsed > 5 and int(elapsed) % 60 == 0:  # Log every 5 seconds
                    self.logger.log_info(f"DECISION_QUEUE: Waiting - active: {active_count}, queued: {queue_size}, elapsed: {elapsed:.1f}s")
                    
                    # Log active decision details for debugging
                    with self.decision_lock:
                        for decision_id, future in self.active_decisions.items():
                            done_status = future.done()
                            self.logger.log_info(f"  - {decision_id}: done={done_status}")
                
                time.sleep(0.1)
        else:
            # With timeout (legacy behavior)
            while (not self.decision_queue.empty() or self.active_decisions) and (time.time() - start_time) < timeout:
                self._process_decision_queue()
                time.sleep(0.1)
                
            # Force cleanup any remaining incomplete decisions after timeout
            if self.active_decisions:
                incomplete_count = len(self.active_decisions)
                self.logger.log_warning(f"DECISION_QUEUE: {incomplete_count} decisions still pending after {timeout}s timeout")
                
                # Force cleanup to prevent hanging
                with self.decision_lock:
                    for decision_id, future in list(self.active_decisions.items()):
                        try:
                            if not future.done():
                                future.cancel()
                            else:
                                future.result()  # Get result to handle exceptions
                        except Exception as e:
                            self.logger.log_error(f"DECISION_QUEUE: Force cleanup of {decision_id}: {e}")
                        finally:
                            del self.active_decisions[decision_id]
    
    def _cleanup_gpu_memory(self):
        """定期清理GPU内存和KV cache"""
        if hasattr(self, 'llm_manager') and self.llm_manager:
            try:
                import torch
                torch.cuda.empty_cache()
                
                # 清理LLM的KV cache
                if hasattr(self.traffic_llm, 'model'):
                    if hasattr(self.traffic_llm.model, 'clear_cache'):
                        self.traffic_llm.model.clear_cache()
                
                if hasattr(self.regional_llm, 'model'):
                    if hasattr(self.regional_llm.model, 'clear_cache'):
                        self.regional_llm.model.clear_cache()
                        
                self.logger.log_info("GPU_CLEANUP: Memory and KV cache cleared")
            except Exception as e:
                self.logger.log_error(f"GPU_CLEANUP: Failed to clear memory: {e}")
    
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
                                
                                # If vehicle is on junction edge, store route for later application
                                if current_edge.startswith(':'):
                                    self.pending_routes[vehicle_id] = {
                                        'route': route,
                                        'boundary_edge': result.get('boundary_edge'),
                                        'travel_time': result.get('travel_time', 0),
                                        'reasoning': result.get('reasoning', 'Regional planning'),
                                        'creation_time': time.time()
                                    }
                                    # Store in regional plans for tracking
                                    self.vehicle_regional_plans[vehicle_id] = {
                                        'region_id': self.vehicle_regions.get(vehicle_id, -1),
                                        'target_region': result.get('target_region', -1),
                                        'boundary_edge': result.get('boundary_edge'),
                                        'route': route,
                                        'creation_time': time.time(),
                                        'travel_time': result.get('travel_time', 0),
                                        'reasoning': f"Pending: {result.get('reasoning', 'Regional planning')}"
                                    }
                                    self.logger.log_info(f"LLM_UPDATE: Stored route for {vehicle_id} (on junction, will apply when exits)")
                                else:
                                    # Normal route application
                                    safe_route = self._create_safe_route(current_edge, route)
                                    if safe_route:
                                        # Apply via unified setter with stuck/cooldown checks
                                        try:
                                            self._set_route_and_register(vehicle_id, safe_route)
                                            self.logger.log_info(f"LLM_UPDATE: Applied regional route for {vehicle_id}")
                                        except Exception:
                                            traci.vehicle.setRoute(vehicle_id, safe_route)
                                            self.logger.log_warning(f"LLM_UPDATE_FALLBACK: Direct setRoute due to unified setter failure")
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
                    latest_traffic = max(traffic_adapters, key=lambda x: self._parse_adapter_step(x))
                    adapter_name = os.path.basename(latest_traffic)
                    
                    if self.current_lora_adapters['traffic'] != adapter_name:
                        self.current_lora_adapters['traffic'] = adapter_name
                        self.logger.log_info(f"LORA_UPDATE_TRAFFIC: 发现新Traffic适配器 {adapter_name}")
                        # 立即加载到本地推理模型
                        try:
                            loaded = self.llm_manager.load_lora_adapter_direct('traffic', latest_traffic)
                            if loaded:
                                self.logger.log_info(f"LORA_HOT_RELOAD_TRAFFIC: 已加载 {adapter_name}")
                                # Emit event to training manager
                                try:
                                    if self.training_queue is not None:
                                        self.training_queue.put({
                                            'message_type': 'lora_hot_reload',
                                            'llm_type': 'traffic',
                                            'adapter_name': adapter_name,
                                            'adapter_path': latest_traffic,
                                            'source': 'env_polling'
                                        }, block=False)
                                except Exception:
                                    pass
                            else:
                                self.logger.log_warning(f"LORA_HOT_RELOAD_TRAFFIC_FAILED: {adapter_name}")
                        except Exception as load_err:
                            self.logger.log_error(f"LORA_HOT_RELOAD_TRAFFIC_ERROR: {load_err}")
                
                # Check Regional LLM adapters  
                regional_adapters = glob.glob(os.path.join(adapter_sync_dir, 'regional_adapter_step_*'))
                if regional_adapters:
                    latest_regional = max(regional_adapters, key=lambda x: self._parse_adapter_step(x))
                    adapter_name = os.path.basename(latest_regional)
                    
                    if self.current_lora_adapters['regional'] != adapter_name:
                        self.current_lora_adapters['regional'] = adapter_name
                        self.logger.log_info(f"LORA_UPDATE_REGIONAL: 发现新Regional适配器 {adapter_name}")
                        # 立即加载到本地推理模型
                        try:
                            loaded = self.llm_manager.load_lora_adapter_direct('regional', latest_regional)
                            if loaded:
                                self.logger.log_info(f"LORA_HOT_RELOAD_REGIONAL: 已加载 {adapter_name}")
                                # Emit event to training manager
                                try:
                                    if self.training_queue is not None:
                                        self.training_queue.put({
                                            'message_type': 'lora_hot_reload',
                                            'llm_type': 'regional',
                                            'adapter_name': adapter_name,
                                            'adapter_path': latest_regional,
                                            'source': 'env_polling'
                                        }, block=False)
                                except Exception:
                                    pass
                            else:
                                self.logger.log_warning(f"LORA_HOT_RELOAD_REGIONAL_FAILED: {adapter_name}")
                        except Exception as load_err:
                            self.logger.log_error(f"LORA_HOT_RELOAD_REGIONAL_ERROR: {load_err}")
                        
        except Exception as e:
            self.logger.log_error(f"LORA_UPDATE_ERROR: {e}")
    
    def _parse_adapter_step(self, adapter_path):
        """Parse step number from adapter path robustly.
        Expected basename like 'traffic_adapter_step_123'. Returns -1 if unknown.
        """
        try:
            import os
            import re
            base = os.path.basename(adapter_path)
            m = re.search(r"_step_(\d+)$", base)
            if m:
                return int(m.group(1))
            # Fallback: last underscore token
            token = base.split('_')[-1]
            return int(token) if token.isdigit() else -1
        except Exception:
            return -1

    def _start_lora_watchdog(self):
        """Start a background watcher to hot-reload adapters immediately when generated."""
        try:
            import threading
            if getattr(self, '_lora_watch_thread', None) and getattr(self, '_lora_watch_thread').is_alive():
                return
            self._lora_watch_stop = threading.Event()
            self._lora_watch_thread = threading.Thread(target=self._lora_watch_loop, name="LoRAWatchdog", daemon=True)
            self._lora_watch_thread.start()
            self.logger.log_info("LORA_WATCHDOG: started")
        except Exception as e:
            try:
                self.logger.log_warning(f"LORA_WATCHDOG_START_FAILED: {e}")
            except Exception:
                pass

    def _stop_lora_watchdog(self):
        """Stop the LoRA watcher thread if running."""
        try:
            if getattr(self, '_lora_watch_stop', None):
                self._lora_watch_stop.set()
            if getattr(self, '_lora_watch_thread', None):
                self._lora_watch_thread.join(timeout=2.0)
                self._lora_watch_thread = None
                self.logger.log_info("LORA_WATCHDOG: stopped")
        except Exception:
            pass

    def _lora_watch_loop(self):
        """Polling loop to detect and apply newest adapters ASAP."""
        import os
        import time
        import glob
        poll_interval_sec = float(os.environ.get("LORA_WATCHDOG_INTERVAL", "2.0"))
        while not getattr(self, '_lora_watch_stop', None) or not self._lora_watch_stop.is_set():
            try:
                if not hasattr(self, 'llm_manager') or not self.llm_manager:
                    time.sleep(poll_interval_sec)
                    continue
                log_dir = getattr(self, 'log_dir', 'logs')
                adapter_sync_dir = os.path.join(log_dir, 'lora_adapters')
                if not os.path.exists(adapter_sync_dir):
                    time.sleep(poll_interval_sec)
                    continue
                with self.lora_update_lock:
                    # traffic
                    traffic_adapters = glob.glob(os.path.join(adapter_sync_dir, 'traffic_adapter_step_*'))
                    if traffic_adapters:
                        latest_traffic = max(traffic_adapters, key=lambda x: self._parse_adapter_step(x))
                        traffic_name = os.path.basename(latest_traffic)
                        if self.current_lora_adapters.get('traffic') != traffic_name:
                            try:
                                if self.llm_manager.load_lora_adapter_direct('traffic', latest_traffic):
                                    self.current_lora_adapters['traffic'] = traffic_name
                                    self.logger.log_info(f"LORA_HOT_RELOAD_TRAFFIC: 已加载 {traffic_name}")
                                    # Emit event to training manager
                                    try:
                                        if self.training_queue is not None:
                                            self.training_queue.put({
                                                'message_type': 'lora_hot_reload',
                                                'llm_type': 'traffic',
                                                'adapter_name': traffic_name,
                                                'adapter_path': latest_traffic,
                                                'source': 'env_watchdog'
                                            }, block=False)
                                    except Exception:
                                        pass
                            except Exception as e:
                                self.logger.log_error(f"LORA_HOT_RELOAD_TRAFFIC_ERROR: {e}")
                    # regional
                    regional_adapters = glob.glob(os.path.join(adapter_sync_dir, 'regional_adapter_step_*'))
                    if regional_adapters:
                        latest_regional = max(regional_adapters, key=lambda x: self._parse_adapter_step(x))
                        regional_name = os.path.basename(latest_regional)
                        if self.current_lora_adapters.get('regional') != regional_name:
                            try:
                                if self.llm_manager.load_lora_adapter_direct('regional', latest_regional):
                                    self.current_lora_adapters['regional'] = regional_name
                                    self.logger.log_info(f"LORA_HOT_RELOAD_REGIONAL: 已加载 {regional_name}")
                                    # Emit event to training manager
                                    try:
                                        if self.training_queue is not None:
                                            self.training_queue.put({
                                                'message_type': 'lora_hot_reload',
                                                'llm_type': 'regional',
                                                'adapter_name': regional_name,
                                                'adapter_path': latest_regional,
                                                'source': 'env_watchdog'
                                            }, block=False)
                                    except Exception:
                                        pass
                            except Exception as e:
                                self.logger.log_error(f"LORA_HOT_RELOAD_REGIONAL_ERROR: {e}")
            except Exception as loop_err:
                try:
                    self.logger.log_error(f"LORA_WATCHDOG_ERROR: {loop_err}")
                except Exception:
                    pass
            finally:
                time.sleep(poll_interval_sec)

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
                
                # Load pre-trained LoRA adapters if provided
                if self.traffic_lora_path or self.regional_lora_path:
                    print("\n=== 加载预训练LoRA适配器 ===")
                    if self.traffic_lora_path:
                        print(f"正在加载Traffic LoRA: {self.traffic_lora_path}")
                        if os.path.exists(self.traffic_lora_path):
                            success = self.llm_manager.load_lora_adapter_direct('traffic', self.traffic_lora_path)
                            if success:
                                print(f"[成功] Traffic LoRA已加载")
                            else:
                                print(f"[警告] Traffic LoRA加载失败")
                        else:
                            print(f"[错误] Traffic LoRA路径不存在: {self.traffic_lora_path}")
                    
                    if self.regional_lora_path:
                        print(f"正在加载Regional LoRA: {self.regional_lora_path}")
                        if os.path.exists(self.regional_lora_path):
                            success = self.llm_manager.load_lora_adapter_direct('regional', self.regional_lora_path)
                            if success:
                                print(f"[成功] Regional LoRA已加载")
                            else:
                                print(f"[警告] Regional LoRA加载失败")
                        else:
                            print(f"[错误] Regional LoRA路径不存在: {self.regional_lora_path}")
                    print()
                
                # Register LLM manager globally for training manager access
                self._register_llm_manager_globally()

                # Align log directory with training manager for adapter sync
                try:
                    self.log_dir = f"logs/training_{self.location}"
                except Exception:
                    self.log_dir = "logs"
                
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

        # 启动LoRA热重载后台监听线程（初始化LLM完成后立即启动）
        try:
            self._start_lora_watchdog()
        except Exception:
            pass
    
    def _initialize_agents_with_existing_llms(self):
        """使用已初始化的LLM创建智能体实例"""
        try:
            # Initialize prediction engine
            edge_list = list(self.road_info.keys())
            self.prediction_engine = PredictionEngine(edge_list, self.logger)
            
            # Initialize Traffic Agent with appropriate LLM (already loaded)
            # 原始LLM仅在本地共享LLM模式下可用；API/单一LLM模式下为None
            raw_traffic_llm = None
            if self.use_local_llm and getattr(self, 'llm_manager', None):
                try:
                    raw_traffic_llm = self.llm_manager.get_traffic_llm_raw()
                except Exception:
                    raw_traffic_llm = None
            self.traffic_agent = TrafficAgent(
                boundary_edges=self.boundary_edges,
                edge_to_region=self.edge_to_region,
                road_info=self.road_info,
                num_regions=self.num_regions,
                llm_agent=self.traffic_llm,  # 使用已初始化的Traffic LLM
                logger=self.logger,
                prediction_engine=self.prediction_engine,
                raw_llm_agent=raw_traffic_llm  # 原始LLM用于macro planning（仅本地LLM模式）
            )
            # 为Traffic Agent提供环境引用，便于获取全局宏观指导
            try:
                self.traffic_agent.parent_env = self
            except Exception:
                pass
            
            # Initialize Regional Agents with shared Regional LLM (already loaded)
            self.regional_agents = {}
            raw_regional_llm = None
            if self.use_local_llm and getattr(self, 'llm_manager', None):
                try:
                    raw_regional_llm = self.llm_manager.get_regional_llm_raw()
                except Exception:
                    raw_regional_llm = None
            for region_id in range(self.num_regions):
                regional_agent = RegionalAgent(
                    region_id=region_id,
                    boundary_edges=self.boundary_edges,
                    edge_to_region=self.edge_to_region,
                    road_info=self.road_info,
                    road_network=self.road_network,
                    llm_agent=self.regional_llm,  # 所有区域共享已初始化的LLM
                    raw_llm_agent=raw_regional_llm,  # 注入原始Regional LLM（仅本地LLM模式）
                    logger=self.logger,
                    prediction_engine=self.prediction_engine
                )
                # Set parent environment reference for data access
                regional_agent.parent_env = self
                # Inject global guidance accessor into regional agent if supported later
                try:
                    regional_agent.get_global_macro_guidance = self._get_current_global_macro_guidance
                except Exception:
                    pass
                # Bind route validation helper for safe route setting in agent
                try:
                    regional_agent._validate_route_setting_ref = self._validate_route_setting
                except Exception:
                    pass
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

    def _get_current_global_macro_guidance(self) -> Optional[Dict[str, Any]]:
        """Return current valid global macro guidance dict or None."""
        try:
            data = self.global_macro_guidance.get('data') if hasattr(self, 'global_macro_guidance') else None
            expire_at = float(self.global_macro_guidance.get('expire_at', 0.0)) if hasattr(self, 'global_macro_guidance') else 0.0
            now = float(traci.simulation.getTime()) if traci else 0.0
            if data and now < expire_at:
                return data
            return None
        except Exception:
            return None

    def _build_hotspots_snapshot(self, current_time: float) -> Dict[str, Any]:
        """Construct hotspots data including stuck edges and lane exit hamper edges (compact)."""
        try:
            hotspots = {
                'stuck_edges': [],
                'hamper_edges': []
            }
            # Extract from regional agents' blacklists
            stuck_edges = set()
            try:
                for r_id, agent in getattr(self, 'regional_agents', {}).items():
                    if hasattr(agent, 'stuck_edge_blacklist') and isinstance(agent.stuck_edge_blacklist, set):
                        for e in list(agent.stuck_edge_blacklist)[:20]:
                            stuck_edges.add(e)
            except Exception:
                pass
            hotspots['stuck_edges'] = list(stuck_edges)[:20]
            # lane-exit hamper recent log cannot be directly read; fallback empty
            return hotspots
        except Exception:
            return {'stuck_edges': [], 'hamper_edges': []}

    def _build_flow_targets_snapshot(self, current_time: float) -> Dict[str, Any]:
        """Summarize near-term flow targets for regions/boundaries from caches (compact)."""
        try:
            # Count planned incoming to regions from vehicle_current_plans
            region_incoming = {}
            try:
                for vid, plan in getattr(self, 'vehicle_current_plans', {}).items():
                    try:
                        macro_route = plan.get('macro_route') or []
                        for rid in macro_route[1:]:
                            region_incoming[rid] = region_incoming.get(rid, 0) + 1
                    except Exception:
                        continue
            except Exception:
                pass
            # Boundary planned counts from boundary_vehicle_plans if available
            boundary_incoming = {}
            try:
                for be, vlist in getattr(self, 'boundary_vehicle_plans', {}).items():
                    try:
                        boundary_incoming[be] = len(vlist) if isinstance(vlist, (list, set)) else int(vlist)
                    except Exception:
                        continue
            except Exception:
                pass
            return {
                'planned_region_incoming': region_incoming,
                'planned_boundary_incoming': boundary_incoming
            }
        except Exception:
            return {'planned_region_incoming': {}, 'planned_boundary_incoming': {}}
    
    def initialize_simulation(self):
        """Initialize SUMO simulation."""
        try:
            # Start SUMO with teleport timeout and optional start time
            sumo_cmd = ["sumo", "-c", self.sumo_config_file, "--no-warnings", 
                       "--ignore-route-errors", "--time-to-teleport", "600"]
            
            # Add start time if specified
            if self.start_time > 0:
                sumo_cmd.extend(["--begin", str(self.start_time)])
                self.logger.log_info(f"Starting simulation at time {self.start_time}s")
            
            traci.start(sumo_cmd)
            
            # Parse only the first route file for autonomous vehicle selection
            # This ensures only taxi vehicles from NewYork_od_0.1.rou.alt.xml can be autonomous
            # The second file NYC_routes_0.1_20250830_111509.alt.xml is loaded as environment traffic only
            first_route_vehicles = parse_rou_file(self.route_file)
            
            # Filter vehicles by start_time - only select from vehicles that will actually depart
            if self.start_time > 0:
                # Only consider vehicles with depart time >= start_time
                valid_vehicles = [(veh_id, start_edge, end_edge, depart_time) 
                                  for veh_id, start_edge, end_edge, depart_time in first_route_vehicles 
                                  if depart_time >= self.start_time]
                self.logger.log_info(f"Filtered vehicles by start_time {self.start_time}s: "
                                   f"{len(first_route_vehicles)} total -> {len(valid_vehicles)} valid")
                first_route_vehicle_ids = [veh_id for veh_id, _, _, _ in valid_vehicles]
            else:
                first_route_vehicle_ids = [veh_id for veh_id, _, _, _ in first_route_vehicles]
            
            # Get total vehicles from SUMO simulation (includes both route files)
            # We need to wait a bit for SUMO to load all vehicles
            time.sleep(0.1)  # Small delay to ensure all vehicles are loaded
            
            # Count total vehicles that will be in simulation
            self.total_vehicles = len(first_route_vehicles)  # Base count from first file
            
            # Try to get additional vehicles from the simulation
            try:
                # Get all vehicle IDs that SUMO knows about (including from both files)
                # Parse environment route file directly instead of querying SUMO
                import os
                import glob
                
                # Find other route files in the same directory
                route_dir = os.path.dirname(self.route_file)
                self.logger.log_info(f"Searching for additional route files in: {route_dir}")
                
                route_files = []
                route_files.extend(glob.glob(os.path.join(route_dir, "*.rou.xml")))
                route_files.extend(glob.glob(os.path.join(route_dir, "*.rou.alt.xml")))
                route_files = [f for f in route_files if f != self.route_file]
                
                self.logger.log_info(f"Found additional route files: {route_files}")
                
                environment_vehicle_count = 0
                if route_files:
                    # Parse the second route file
                    second_route_file = route_files[0]  # Take the first additional route file
                    self.logger.log_info(f"Parsing second route file: {second_route_file}")
                    
                    second_route_vehicles = parse_rou_file(second_route_file)
                    environment_vehicle_count = len(second_route_vehicles)
                    
                    self.logger.log_info(f"Parsed {environment_vehicle_count} vehicles from second route file")
                else:
                    self.logger.log_warning("No additional route files found for environment vehicles")
                
                # Set total vehicles (both files)
                self.total_vehicles = len(first_route_vehicle_ids) + environment_vehicle_count
                
                
                self.logger.log_info(f"Vehicle loading: {len(first_route_vehicle_ids)} from primary route file, "
                                   f"{environment_vehicle_count} from environment route file, "
                                   f"{self.total_vehicles} total vehicles")
                
            except Exception as count_error:
                import traceback
                self.logger.log_warning(f"Could not count environment vehicles: {count_error}")
                self.logger.log_warning(f"Exception traceback: {traceback.format_exc()}")
                # Fall back to just the first route file count
                self.total_vehicles = len(first_route_vehicles)
            
            # Select autonomous vehicles based on av_ratio - ONLY from the first route file
            import random
            num_autonomous = int(self.av_ratio * len(first_route_vehicle_ids))
            self.autonomous_vehicles = set(random.sample(first_route_vehicle_ids, num_autonomous))
            self.logger.log_info(f"Selected {num_autonomous} vehicles ({self.av_ratio*100:.1f}%) from first route file as autonomous")
            
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
        Enhanced with simplified coordination framework:
        1. State space construction
        2. Action space generation 
        3. Regional status collection for coordination
        4. Traffic Agent macro planning with coordination data
        """
        try:
            # Ensure this is an autonomous vehicle
            if vehicle_id not in self.autonomous_vehicles:
                self.logger.log_warning(f"MACRO_PLANNING_SKIP: {vehicle_id} is not an autonomous vehicle, skipping macro planning")
                return
                
            self.logger.log_info(f"COORDINATION_VEHICLE_BIRTH: Processing coordination-based macro planning for autonomous vehicle {vehicle_id}")
            
            # [Phase 1: State Space Construction] - Following CLAUDE.md specifications
            state_context = self._construct_state_space(vehicle_id, current_time)
            if not state_context:
                self.logger.log_warning(f"CORY_STATE_CONSTRUCTION: Failed to construct state space for autonomous vehicle {vehicle_id}")
                return
            
            start_region = state_context['start_region']
            dest_region = state_context['dest_region']
            
            self.logger.log_info(f"CORY_STATE_SPACE: Autonomous vehicle {vehicle_id} state constructed - "
                               f"Start: Region {start_region}, Dest: Region {dest_region}, "
                               f"Current traffic: {state_context['regional_congestion']}, "
                               f"Predictions available: {len(state_context['traffic_predictions'])} time windows")
            
            # Handle single-region case with CORY logging
            if start_region == dest_region:
                self.logger.log_info(f"CORY_SINGLE_REGION: Autonomous vehicle {vehicle_id} intra-region route in Region {dest_region}")
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
                self.logger.log_info(f"CORY_SINGLE_REGION: Autonomous vehicle {vehicle_id} assigned single-region macro route [{start_region}]")
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
            
            # [Phase 3: Simplified Coordination-Based Macro Planning]
            # Collect regional status reports for coordination
            active_autonomous_vehicles = set(self.vehicle_current_plans.keys())
            regional_status_reports = self._collect_regional_status_reports(active_autonomous_vehicles)
            
            # Use simplified macro planning with coordination data
            macro_result = self._simplified_macro_planning(
                vehicle_id, state_context, macro_candidates, current_time, regional_status_reports
            )
            
            selected_macro_route = macro_result.get('final_route')
            coordination_used = macro_result.get('coordination_used', False)
            decision_type = macro_result.get('decision_type', 'unknown')
            
            self.logger.log_info(f"COORDINATION_DECISION: Vehicle {vehicle_id} macro planning completed - "
                               f"Route: {selected_macro_route}, Coordination used: {coordination_used}, "
                               f"Decision type: {decision_type}")
            
            if selected_macro_route:
                # Store macro route with coordination metadata
                self.vehicle_current_plans[vehicle_id] = {
                    'macro_route': selected_macro_route,
                    'current_region_index': 0,
                    'creation_time': current_time,
                    'last_update': current_time,
                    'decision_type': decision_type,
                    'coordination_used': coordination_used,
                    'macro_route_obj': macro_result.get('macro_route_obj'),
                    'state_context': state_context,  # Store for RL training
                    'regional_status_reports': regional_status_reports  # Store coordination data for analysis
                }
                
                # Update communication system
                self._broadcast_vehicle_macro_plan(vehicle_id, selected_macro_route, current_time)
                self._log_vehicle_decision(vehicle_id, "COORDINATION_MACRO_PLANNING", selected_macro_route, current_time)
                
                self.logger.log_info(f"COORDINATION_SUCCESS: {vehicle_id} assigned macro route {selected_macro_route} "
                                   f"(Coordination: {'Used' if coordination_used else 'Not used'})")
            else:
                self.logger.log_error(f"COORDINATION_FAILURE: Failed macro planning for {vehicle_id}")
                
        except Exception as e:
            self.logger.log_error(f"COORDINATION_ERROR: Coordination-based macro planning failed for {vehicle_id}: {e}")
            import traceback
            self.logger.log_error(f"COORDINATION_TRACEBACK: {traceback.format_exc()}")
            # Fallback to original system if coordination fails
            self._fallback_original_planning(vehicle_id, current_time)
    
    def _simplified_macro_planning(self, vehicle_id: str, state_context: Dict[str, Any], 
                                 macro_candidates: List[List[int]], current_time: float,
                                 regional_status_reports: Dict) -> Dict[str, Any]:
        """
        Simplified macro planning using Traffic Agent with coordination data.
        Replaces the complex CORY cooperative decision mechanism.
        
        Args:
            vehicle_id: Vehicle identifier  
            state_context: Current traffic state context
            macro_candidates: Candidate macro routes
            current_time: Current simulation time
            regional_status_reports: Regional coordination data
            
        Returns:
            Dictionary containing final route and coordination info
        """
        try:
            self.logger.log_info(f"MACRO_PLANNING: Starting simplified macro planning for {vehicle_id}")
            
            # Prepare batch macro planning request 
            # Use Traffic Agent's single vehicle macro planning
            coordination_data = self.traffic_agent._analyze_coordination_data(regional_status_reports)
            macro_route = self.traffic_agent.plan_single_macro_route(
                vehicle_id=vehicle_id,
                start_region=state_context['start_region'], 
                end_region=state_context['dest_region'],
                current_time=current_time,
                coordination_data=coordination_data
            )
            
            if macro_route:
                selected_route = macro_route.region_sequence
                self.logger.log_info(f"MACRO_PLANNING: Traffic Agent selected route {selected_route} for {vehicle_id}")
                
                return {
                    'final_route': selected_route,
                    'coordination_used': True,
                    'macro_route_obj': macro_route,
                    'decision_type': 'coordinated_macro_planning'
                }
            else:
                # Fallback to first candidate if Traffic Agent fails
                fallback_route = macro_candidates[0] if macro_candidates else None
                self.logger.log_warning(f"MACRO_PLANNING: Traffic Agent failed, using fallback route {fallback_route} for {vehicle_id}")
                
                return {
                    'final_route': fallback_route,
                    'coordination_used': False,
                    'decision_type': 'fallback_candidate'
                }
                
        except Exception as e:
            self.logger.log_error(f"MACRO_PLANNING: Simplified macro planning failed for {vehicle_id}: {e}")
            fallback_route = macro_candidates[0] if macro_candidates else None
            return {
                'final_route': fallback_route,
                'coordination_used': False,
                'decision_type': 'error_fallback',
                'error': str(e)
            }
    
    def _collect_regional_status_reports(self, active_autonomous_vehicles: set) -> Dict:
        """
        Collect status reports from all regional agents for coordination.
        
        Args:
            active_autonomous_vehicles: Set of currently active autonomous vehicle IDs
            
        Returns:
            Dictionary mapping region_id to regional status report
        """
        try:
            regional_status_reports = {}
            
            for region_id, regional_agent in self.regional_agents.items():
                try:
                    status_report = regional_agent.report_region_status(active_autonomous_vehicles)
                    regional_status_reports[region_id] = status_report
                except Exception as e:
                    self.logger.log_warning(f"COORDINATION: Failed to get status report from region {region_id}: {e}")
                    # Provide empty report as fallback
                    regional_status_reports[region_id] = {
                        'active_vehicles': {},
                        'capacity_status': {'current_load': 0, 'congestion_warning': False}
                    }
            
            self.logger.log_info(f"COORDINATION: Collected status reports from {len(regional_status_reports)} regions")
            return regional_status_reports
            
        except Exception as e:
            self.logger.log_error(f"COORDINATION: Failed to collect regional status reports: {e}")
            return {}
    
    def _create_safe_route(self, current_edge: str, target_route: List[str]) -> Optional[List[str]]:
        """
        Create a safe route that includes the vehicle's current edge as the first element.
        Enhanced to handle junction edges by waiting for vehicle to exit junction.
        
        Args:
            current_edge: Vehicle's current edge ID
            target_route: Desired route
            
        Returns:
            Safe route with current edge as first element, or None if impossible
        """
        try:
            # Handle junction edges (internal edges) - indicate special handling needed
            if current_edge.startswith(':'):
                self.logger.log_info(f"SAFE_ROUTE: Vehicle on junction edge {current_edge}, route will be applied when exits")
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

    def _compute_stuck_zone_edges(self, center_edge: str) -> Set[str]:
        """Compute stuck zone as center edge plus 1-hop and 2-hop successors."""
        try:
            zone: Set[str] = set()
            if not center_edge or center_edge.startswith(':'):
                return zone
            zone.add(center_edge)
            try:
                one_hop = list(self.road_network.successors(center_edge))
            except Exception:
                one_hop = []
            zone.update(one_hop)
            two_hop: Set[str] = set()
            for e in one_hop:
                try:
                    for n2 in self.road_network.successors(e):
                        two_hop.add(n2)
                except Exception:
                    continue
            zone.update(two_hop)
            return zone
        except Exception:
            return set()

    def _activate_stuck_zone_for_vehicle(self, vehicle_id: str, center_edge: str, current_time: float) -> None:
        """Activate blocked zone for a stuck vehicle after threshold."""
        try:
            zone = self._compute_stuck_zone_edges(center_edge)
            if not zone:
                return
            prev = self.vehicle_stuck_zone_edges.get(vehicle_id)
            if prev:
                self.blocked_edges_active.update(prev)
            self.vehicle_stuck_zone_edges[vehicle_id] = set(zone)
            self.blocked_edges_active.update(zone)
            try:
                for r_id, agent in getattr(self, 'regional_agents', {}).items():
                    if hasattr(agent, 'update_stuck_edge_blacklist'):
                        agent.update_stuck_edge_blacklist(set(zone), current_time)
            except Exception:
                pass
            self.logger.log_warning(f"STUCK_ZONE_ACTIVATED: {vehicle_id} at {center_edge}, edges={len(zone)}")
        except Exception as e:
            self.logger.log_warning(f"STUCK_ZONE_ACTIVATE_ERROR: {vehicle_id} -> {e}")

    def _deactivate_stuck_zone_for_vehicle(self, vehicle_id: str, current_time: float) -> None:
        """Deactivate blocked zone when vehicle recovers; convert to cooldown."""
        try:
            zone = self.vehicle_stuck_zone_edges.get(vehicle_id)
            if not zone:
                return
            for e in zone:
                if e in self.blocked_edges_active:
                    self.blocked_edges_active.discard(e)
            expire_at = current_time + self.cooldown_window_sec
            self.cooldown_zones.append({'edges': set(zone), 'expire_at': expire_at, 'assignments': 0})
            self.cooldown_edges.update(zone)
            try:
                del self.vehicle_stuck_zone_edges[vehicle_id]
            except Exception:
                pass
            self.logger.log_info(f"STUCK_ZONE_COOLDOWN: {vehicle_id} until {expire_at:.1f}")
        except Exception as e:
            self.logger.log_warning(f"STUCK_ZONE_DEACTIVATE_ERROR: {vehicle_id} -> {e}")

    def _cleanup_cooldown_zones(self, current_time: float) -> None:
        """Cleanup expired cooldown zones."""
        try:
            if not self.cooldown_zones:
                return
            remaining = []
            removed_edges: Set[str] = set()
            for z in self.cooldown_zones:
                if current_time >= float(z.get('expire_at', 0)):
                    removed_edges.update(z.get('edges', set()))
                else:
                    remaining.append(z)
            self.cooldown_zones = remaining
            if removed_edges:
                for e in removed_edges:
                    if e in self.cooldown_edges:
                        self.cooldown_edges.discard(e)
                self.logger.log_info(f"COOLDOWN_CLEANUP: removed {len(removed_edges)} edges")
        except Exception:
            pass

    def is_edge_blocked(self, edge_id: str, current_time: float = 0.0) -> bool:
        try:
            return bool(edge_id in self.blocked_edges_active)
        except Exception:
            return False

    def is_route_blocked(self, route: List[str], current_time: float = 0.0) -> bool:
        try:
            if not route:
                return False
            for e in route:
                if e in self.blocked_edges_active:
                    return True
            return False
        except Exception:
            return False

    def is_route_allowed_under_cooldown(self, route: List[str], current_time: float) -> bool:
        # Cooling throttling disabled: always allow. Stuck edges handled by is_route_blocked.
        return True

    def _register_route_assignment_to_cooldown(self, route: List[str], current_time: float) -> None:
        # Cooling throttling disabled: no-op.
        return

    def _set_route_and_register(self, vehicle_id: str, route: List[str]) -> None:
        """Unified route set with blocked/cooldown checks and assignment registration."""
        try:
            if not route:
                return
            now = float(traci.simulation.getTime()) if traci else 0.0
            if self.is_route_blocked(route, now):
                self.logger.log_warning(f"ROUTE_BLOCKED: {vehicle_id} intersects active stuck zone")
                return
            if not self.is_route_allowed_under_cooldown(route, now):
                self.logger.log_warning(f"ROUTE_COOLDOWN_THROTTLED: {vehicle_id} rejected by cooldown cap")
                return
            traci.vehicle.setRoute(vehicle_id, route)
            self._register_route_assignment_to_cooldown(route, now)
        except Exception as e:
            try:
                self.logger.log_warning(f"SET_ROUTE_REGISTER_FAIL: {vehicle_id} -> {e}")
            except Exception:
                pass

    def _get_or_create_step_traffic_state(self, current_time: float) -> Dict[str, Any]:
        """
        Get or create step-wise cached traffic state data.
        Only compute once per simulation step for all vehicles.
        """
        try:
            current_step = int(current_time)
            cache = self._step_traffic_state_cache
            
            # Check if cache is valid for current step
            if (cache['current_step'] == current_step and 
                cache['regional_congestion'] is not None and
                abs(cache['cache_timestamp'] - current_time) < 2.0):  # Allow 2s tolerance
                
                self.logger.log_info(f"CORY_STATE_CACHE_HIT: Using cached traffic state for step {current_step}")
                return {
                    'regional_congestion': cache['regional_congestion'].copy(),
                    'boundary_flows': cache['boundary_flows'].copy(),
                    'global_state': cache['global_state'].copy(),
                    'traffic_predictions': cache['traffic_predictions'].copy()
                }
            
            # Cache miss - compute new traffic state
            self.logger.log_info(f"CORY_STATE_CACHE_MISS: Computing new traffic state for step {current_step} (prev: {cache['current_step']})")
            
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
            
            # [Traffic Predictions] - Multi-horizon predictions (computationally expensive)
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
            
            # Update cache
            cache['current_step'] = current_step
            cache['regional_congestion'] = regional_congestion.copy()
            cache['boundary_flows'] = boundary_flows.copy()
            cache['global_state'] = global_state.copy()
            cache['traffic_predictions'] = traffic_predictions.copy()
            cache['cache_timestamp'] = current_time
            
            return {
                'regional_congestion': regional_congestion,
                'boundary_flows': boundary_flows,
                'global_state': global_state,
                'traffic_predictions': traffic_predictions
            }
            
        except Exception as e:
            self.logger.log_error(f"CORY_TRAFFIC_STATE_ERROR: Failed to get/create traffic state: {e}")
            # Return empty state on error
            return {
                'regional_congestion': {},
                'boundary_flows': {},
                'global_state': {'current_time': current_time, 'total_vehicles': 0, 'completed_vehicles': 0, 'autonomous_vehicles_active': 0, 'current_avg_travel_time': 0},
                'traffic_predictions': {}
            }

    def _infer_region_from_neighbors(self, edge_id: str) -> int:
        """Infer a plausible region for an edge using neighbors when mapping is missing."""
        try:
            rid = self.edge_to_region.get(edge_id, -1)
            if rid != -1:
                return rid
            # Try successors then predecessors in road_network
            if hasattr(self, 'road_network') and self.road_network is not None:
                try:
                    for succ in list(self.road_network.successors(edge_id))[:5]:
                        r = self.edge_to_region.get(succ, -1)
                        if r != -1:
                            return r
                except Exception:
                    pass
                try:
                    for pred in list(self.road_network.predecessors(edge_id))[:5]:
                        r = self.edge_to_region.get(pred, -1)
                        if r != -1:
                            return r
                except Exception:
                    pass
            return -1
        except Exception:
            return -1
    
    def _construct_state_space(self, vehicle_id: str, current_time: float) -> Dict[str, Any]:
        """
        Construct comprehensive state space for CORY cooperative decision making.
        Following CLAUDE.md Phase 1: Decision Need Generation & Environment State Construction.
        
        OPTIMIZED: Now uses step-wise cached traffic state to avoid redundant calculations.
        
        Returns:
            Complete state context including:
            - Vehicle individual state (position, route, region mapping)
            - Traffic state perception (regional congestion, boundary flows) - CACHED
            - System global state (vehicle counts, ATT, predictions) - CACHED
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
            
            # Map edges to regions (apply neighbor-based fallback when missing)
            start_region = self.edge_to_region.get(start_edge)
            dest_region = self.edge_to_region.get(dest_edge)

            if start_region is None:
                inferred = self._infer_region_from_neighbors(start_edge)
                start_region = inferred if inferred != -1 else None

            if dest_region is None:
                inferred = self._infer_region_from_neighbors(dest_edge)
                dest_region = inferred if inferred != -1 else None

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
            
            # [OPTIMIZED] Get cached traffic state for this step
            cached_traffic_state = self._get_or_create_step_traffic_state(current_time)
            
            # Compile comprehensive state context using cached data
            state_context = {
                'vehicle_state': vehicle_state,
                'start_region': start_region,
                'dest_region': dest_region,
                'regional_congestion': cached_traffic_state['regional_congestion'],
                'boundary_flows': cached_traffic_state['boundary_flows'],
                'global_state': cached_traffic_state['global_state'],
                'traffic_predictions': cached_traffic_state['traffic_predictions'],
                'state_construction_time': current_time
            }
            
            self.logger.log_info(f"CORY_STATE_SUCCESS: Constructed state space for {vehicle_id} - "
                               f"Regions: {start_region}->{dest_region}, "
                               f"Global vehicles: {cached_traffic_state['global_state']['total_vehicles']}, "
                               f"Regional congestion range: [{min(cached_traffic_state['regional_congestion'].values()):.3f}, {max(cached_traffic_state['regional_congestion'].values()):.3f}] (cached)")
            
            return state_context
            
        except Exception as e:
            self.logger.log_error(f"CORY_STATE_ERROR: State space construction failed for {vehicle_id}: {e}")
            return None
    
    def _clear_step_traffic_state_cache_if_needed(self, current_time: float) -> None:
        """
        Clear step traffic state cache when moving to a new simulation step.
        This ensures fresh traffic data calculation for each new step.
        """
        current_step = int(current_time)
        cache = self._step_traffic_state_cache
        
        # Clear cache if we've moved to a new step or if cache is stale
        if (cache['current_step'] != current_step or 
            cache['current_step'] == -1 or
            abs(cache['cache_timestamp'] - current_time) > 5.0):  # 5s staleness threshold
            
            if cache['current_step'] != -1:  # Not first run
                self.logger.log_info(f"CORY_CACHE_CLEAR: Cleared traffic state cache for new step {current_step} (prev: {cache['current_step']})")
            
            # Reset all cached values
            cache['current_step'] = -1
            cache['regional_congestion'] = None
            cache['boundary_flows'] = None
            cache['global_state'] = None
            cache['traffic_predictions'] = None
            cache['cache_timestamp'] = -1
    
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
        
        Pioneer 单体决策（已精简流程，无 Observer 环节）:
        - Traffic LLM 负责宏观战略决策与全局优化
        
        Returns:
            Dict containing:
            - final_route: Selected macro route
            - cooperation_quality: Quality score of cooperation
            - pioneer_decision: Pioneer's original decision
            - observer_feedback: []  # 已移除
            - j1_judge_evaluation: {}  # 已移除
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
            
            # 直接采用 Pioneer 决策（无 Observer/J1 流程）
            final_result = {
                'final_route': pioneer_result.get('selected_route'),
                'cooperation_quality': 1.0,
                'pioneer_decision': pioneer_result,
                'observer_feedback': [],
                'j1_judge_evaluation': {},
            }
            
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
            
            # Use Traffic LLM for Pioneer decision
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
        """Handle replanning for stuck autonomous vehicles with improved safety checks and route validation."""
        try:
            # Ensure this is an autonomous vehicle
            if vehicle_id not in self.autonomous_vehicles:
                self.logger.log_warning(f"STUCK_REPLAN_SKIP: {vehicle_id} is not an autonomous vehicle, skipping replanning")
                return
                
            # Check if vehicle still exists in simulation
            if vehicle_id not in traci.vehicle.getIDList():
                self.logger.log_warning(f"STUCK_REPLAN: Autonomous vehicle {vehicle_id} no longer exists, skipping replanning")
                return
            
            # Check if vehicle already being replanned recently to avoid spam
            if hasattr(self, '_vehicle_replan_times'):
                last_replan = self._vehicle_replan_times.get(vehicle_id, 0)
                if current_time - last_replan < 120:  # Avoid replanning same vehicle within 2 minutes
                    self.logger.log_info(f"STUCK_REPLAN: Autonomous vehicle {vehicle_id} was replanned recently, skipping")
                    return
            else:
                self._vehicle_replan_times = {}
            
            # Get vehicle's current position and route info
            try:
                current_edge = traci.vehicle.getRoadID(vehicle_id)
                original_route = traci.vehicle.getRoute(vehicle_id)
                speed = traci.vehicle.getSpeed(vehicle_id)
                
                self.logger.log_info(f"STUCK_REPLAN: Initiating replanning for stuck autonomous vehicle {vehicle_id} - "
                                   f"current_edge: {current_edge}, speed: {speed:.2f}, original_route_length: {len(original_route)}")
                
                # Skip if vehicle is on junction (internal edge)
                if current_edge.startswith(':'):
                    self.logger.log_warning(f"STUCK_REPLAN: Autonomous vehicle {vehicle_id} on junction edge {current_edge}, waiting for exit")
                    return
                    
            except Exception as pos_error:
                self.logger.log_error(f"STUCK_REPLAN: Cannot get position info for autonomous vehicle {vehicle_id}: {pos_error}")
                return
            
            # Check if this is an emergency route vehicle
            is_emergency_route = False
            original_dest_region = None
            if vehicle_id in self.vehicle_current_plans:
                plan = self.vehicle_current_plans[vehicle_id]
                is_emergency_route = plan.get('emergency_route', False)
                original_dest_region = plan.get('original_dest_region')
                
                if is_emergency_route:
                    self.logger.log_info(f"STUCK_REPLAN: Autonomous vehicle {vehicle_id} is on emergency route, attempting to find path to original destination {original_dest_region}")
            
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
            
            # Try to create emergency route first (congestion-aware variant)
            try:
                dest_edge = original_route[-1] if original_route else None
                if dest_edge and current_edge != dest_edge:
                    # Generate multiple candidate routes (k-shortest approximation via local expansions)
                    base = traci.simulation.findRoute(fromEdge=current_edge, toEdge=dest_edge)
                    candidates: List[List[str]] = []
                    if base and hasattr(base, 'edges') and len(base.edges) > 1:
                        candidates.append(list(base.edges))
                    # Try small detours: neighbors of current_edge next hops
                    try:
                        next_edges = []
                        try:
                            route_idx = traci.vehicle.getRouteIndex(vehicle_id)
                            vr = traci.vehicle.getRoute(vehicle_id)
                            if route_idx + 1 < len(vr):
                                next_edges.append(vr[route_idx + 1])
                        except Exception:
                            next_edges = []
                        # Also add graph successors if available
                        try:
                            for succ in list(self.road_network.successors(current_edge))[:5]:
                                next_edges.append(succ)
                        except Exception:
                            pass
                        # Build detour candidates by forcing initial hop
                        for ne in next_edges[:5]:
                            try:
                                pre = traci.simulation.findRoute(fromEdge=current_edge, toEdge=ne)
                                post = traci.simulation.findRoute(fromEdge=ne, toEdge=dest_edge)
                                if pre and hasattr(pre, 'edges') and post and hasattr(post, 'edges'):
                                    route_edges = list(pre.edges)[:-1] + list(post.edges)
                                    if len(route_edges) > 1:
                                        candidates.append(route_edges)
                            except Exception:
                                continue
                    except Exception:
                        pass

                    # Score candidates with congestion-aware cost
                    def edge_cost(e: str) -> float:
                        try:
                            dist = float(self.road_info.get(e, {}).get('length', 50.0))
                        except Exception:
                            dist = 50.0
                        try:
                            occ = float(traci.edge.getLastStepOccupancy(e))
                        except Exception:
                            occ = 0.0
                        hamper_c = float(self.exit_hamper_counts.get(e, 0))
                        hamper_r = (hamper_c / float(max(self._hamper_samples, 1))) if hamper_c > 0 else 0.0
                        penalty = 0.0
                        if e in self.hotspot_avoid_edges:
                            penalty += self.hotspot_edge_penalty
                        return max(1.0, dist) * (1.0 + self.congestion_weight_alpha * occ + self.congestion_weight_beta * hamper_r) + penalty

                    def route_cost(path: List[str]) -> float:
                        return sum(edge_cost(e) for e in path)

                    if candidates:
                        candidates = [c for c in candidates if c and c[0] == current_edge]
                        if not candidates and base and hasattr(base, 'edges'):
                            candidates = [list(base.edges)]
                        best = None
                        best_cost = float('inf')
                        for cand in candidates[:8]:
                            cost = route_cost(cand)
                            if cost < best_cost:
                                best, best_cost = cand, cost
                        if best and len(best) > 1:
                            # Ensure safe head with current edge and avoid hotspot as first hop
                            try:
                                safe_best = self._create_safe_route(current_edge, best)
                            except Exception:
                                safe_best = None
                            target_apply = safe_best or best
                            # If first connection after current is hotspot, try trimming/alternate
                            if len(target_apply) > 1 and target_apply[1] in self.hotspot_avoid_edges:
                                # Try to replace with alternative neighbor not in hotspots
                                try:
                                    alts = []
                                    for succ in list(self.road_network.successors(current_edge))[:6]:
                                        if succ in self.hotspot_avoid_edges:
                                            continue
                                        pre = traci.simulation.findRoute(current_edge, succ)
                                        post = traci.simulation.findRoute(succ, dest_edge)
                                        if pre and hasattr(pre, 'edges') and post and hasattr(post, 'edges'):
                                            candidate = list(pre.edges)[:-1] + list(post.edges)
                                            if candidate and candidate[0] == current_edge and len(candidate) > 1:
                                                alts.append(candidate)
                                    if alts:
                                        alt = min(alts, key=lambda r: route_cost(r))
                                        target_apply = alt
                                except Exception:
                                    pass
                            if self._validate_route_setting(vehicle_id, target_apply):
                                try:
                                    self._set_route_and_register(vehicle_id, target_apply)
                                except Exception:
                                    traci.vehicle.setRoute(vehicle_id, target_apply)
                            else:
                                try:
                                    self._set_route_and_register(vehicle_id, best)
                                except Exception:
                                    traci.vehicle.setRoute(vehicle_id, best)

                            self.logger.log_info(
                                f"STUCK_REPLAN: Applied congestion-aware emergency route for {vehicle_id} - len:{len(target_apply)}, cost:{best_cost:.1f}"
                            )

                            # Update vehicle plans with emergency route info
                            current_region = self.edge_to_region.get(current_edge, -1)
                            dest_region = self.edge_to_region.get(dest_edge, -1)
                            if dest_region == -1:
                                # Fallback to nearest known region by neighbors to keep coordination available
                                try:
                                    dest_region = self._infer_region_from_neighbors(dest_edge)
                                except Exception:
                                    dest_region = -1
                            self.vehicle_current_plans[vehicle_id] = {
                                'macro_route': [current_region, dest_region] if current_region != -1 and dest_region != -1 else [current_region if current_region != -1 else 0, dest_region if dest_region != -1 else 0],
                                'detailed_route': target_apply,
                                'emergency_route': True,
                                'original_dest_region': dest_region,
                                'replan_time': current_time
                            }

                            return
                        
                    else:
                        self.logger.log_warning(f"STUCK_REPLAN: No alternative route found from {current_edge} to {dest_edge}")
                        try:
                            if not hasattr(self, '_monitor_counters'):
                                self._monitor_counters = {}
                            self._monitor_counters['route_find_fail'] = self._monitor_counters.get('route_find_fail', 0) + 1
                        except Exception:
                            pass
                        
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
                
                # Build raw-only regional conditions for the specific decision
                regional_conditions = {}
                for region_id, congestion in decision_input['regional_congestion'].items():
                    regional_conditions[region_id] = {
                        'region_id': region_id,
                        'congestion_level': congestion
                    }
                
                llm_result = self.traffic_llm.macro_route_planning(
                    global_state=global_state,
                    route_requests=route_requests,
                    regional_conditions=regional_conditions,
                    boundary_analysis={},  # Raw-only path keeps this empty here
                    flow_predictions={'time_horizon_s': 900},
                    coordination_needs={},
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
        """Create English observation for macro (Pioneer) LLM decision.
        Keep only raw facts; no hand-crafted scores. Require numeric-only reply.
        """
        try:
            # Basic context
            v_id = str(decision_input['vehicle_id'])[-4:]
            s_reg = decision_input['start_region']
            d_reg = decision_input['dest_region']
            time_s = decision_input['current_time']

            parts = []
            parts.append(f"Task: Choose ONE macro route option for vehicle V{v_id}. Reply with the option number only (1..K).")
            parts.append(f"OD: R{s_reg} -> R{d_reg}. Time: {time_s:.0f}s.")

            # Optional batch/jam context (raw facts only)
            recent_first_hops = []
            jam_edges = []
            if macro_context:
                recent_first_hops = list(macro_context.get('recent_first_hops', []))
                jam_edges = list(macro_context.get('jam_edges', []))

            if recent_first_hops:
                # show top-3 counts as raw facts
                from collections import Counter
                hop_counts = Counter(recent_first_hops)
                common = ", ".join([f"{k}:{v}" for k, v in hop_counts.most_common(3)])
                parts.append(f"Batch first-hop usage (top-3): {common}.")
            if jam_edges:
                parts.append(f"Jammed/closed edges (avoid): {jam_edges}.")

            # Regional congestion raw facts (only relevant regions)
            reg_cong = decision_input.get('regional_congestion', {})
            relevant_regions = set([s_reg, d_reg])
            for route in decision_input['route_candidates']:
                relevant_regions.update(route)
            cong_str = " ".join([f"R{r}:{reg_cong.get(r, 0.0):.2f}" for r in sorted(relevant_regions)])
            if cong_str:
                parts.append(f"Regional congestion (raw): {cong_str}.")

            # Candidates: show route and first hop only (no scores)
            parts.append("Candidates:")
            for i, route in enumerate(decision_input['route_candidates']):
                first_hop = route[1] if isinstance(route, list) and len(route) > 1 else d_reg
                parts.append(f"  {i+1}) route={route}, first_hop=R{first_hop}")

            # Global raw state
            glb = decision_input.get('global_state', {})
            total_v = glb.get('total_vehicles', 0)
            att = glb.get('current_avg_travel_time', 0.0)
            parts.append(f"Global (raw): vehicles={total_v}, ATT={att:.0f}s.")

            # Simple rules (text only, LLM decides trade-offs)
            parts.append("Rules: Prefer options that diversify the first hop vs. current batch; avoid jammed/closed edges if any.")
            parts.append("Answer: provide the option number only (e.g., 2). No extra text.")

            return "\n".join(parts)
        except Exception:
            # Fallback to previous minimal prompt if anything fails
            v_id = str(decision_input['vehicle_id'])[-4:]
            s_reg = decision_input['start_region']
            d_reg = decision_input['dest_region']
            time_s = decision_input['current_time']
            observation = f"V{v_id} R{s_reg}->R{d_reg} T{time_s:.0f}s\n\nOPTS:"
            for i, route in enumerate(decision_input['route_candidates']):
                observation += f" {i+1}:{route}"
            observation += "\n\nDECISION FORMAT: Reply with the option number only (1, 2, ...)."
            return observation
    
    def _generate_macro_route_candidates_original(self, start_region: int, dest_region: int, current_time: float) -> List[List[int]]:
        """Original macro route candidate generation for fallback."""
        return self._generate_macro_route_candidates(start_region, dest_region, current_time)
    
    def _evaluate_region_transition_availability(self, from_region: int, to_region: int, current_time: float) -> Dict[str, Any]:
        """评估区域间切边可用性：
        - 是否存在区域有向边
        - 是否有通过封锁/冷却检查的可用切边
        - 拥堵是否低于阈值（soft cutoff）
        返回：flags + 该对区域之间最小拥堵值
        """
        result = {
            'has_edge': False,
            'has_passable': False,   # 通过 blocked + cooldown 检查
            'has_viable': False,     # 通过且拥堵低于阈值
            'min_congestion': None,
            'edges_evaluated': 0
        }
        try:
            G = self.traffic_agent.region_graph
            if not (G and G.has_edge(from_region, to_region)):
                return result
            result['has_edge'] = True
            edge_ids = G[from_region][to_region].get('edges', [])
            # Switch to continuous congestion_score; cutoff now in [0,1] (default 0.6)
            cutoff = float(getattr(self, 'macro_boundary_congestion_cutoff', 0.6))
            best_cong = None
            for eid in edge_ids:
                result['edges_evaluated'] += 1
                try:
                    # 用单边路由代理做 blocked/cooldown 检查
                    if self.is_route_blocked([eid], current_time):
                        continue
                    if not self.is_route_allowed_under_cooldown([eid], current_time):
                        continue
                    result['has_passable'] = True
                    cong = float(self.road_info.get(eid, {}).get('congestion_score', self.road_info.get(eid, {}).get('occupancy_rate', self.road_info.get(eid, {}).get('congestion_level', 0.0))))
                    if best_cong is None or cong < best_cong:
                        best_cong = cong
                    if cong <= cutoff:
                        result['has_viable'] = True
                except Exception:
                    continue
            result['min_congestion'] = best_cong if best_cong is not None else float('inf')
            return result
        except Exception:
            return result

    def _filter_macro_candidates_by_boundary_availability(self, candidates: List[List[int]], current_time: float) -> List[List[int]]:
        """按区域切边可用性与拥堵阈值过滤宏观路径：
        策略：
          - 优先保留每个跳跃(from->to)都“可行(viable)”的路径（过 blocked/cooldown 且拥堵<=阈值）
          - 若没有“viable”路径，则保留“passable”（只过 blocked/cooldown，不看拥堵）的路径
          - 都没有则返回空（上层会回退到未过滤候选或其它机制）
        """
        try:
            if not candidates:
                return []
            good: List[List[int]] = []
            ok: List[List[int]] = []
            for route in candidates:
                try:
                    if not route or len(route) == 1:
                        good.append(route)
                        continue
                    all_passable = True
                    all_viable = True
                    for i in range(len(route) - 1):
                        from_r = route[i]
                        to_r = route[i + 1]
                        ev = self._evaluate_region_transition_availability(from_r, to_r, current_time)
                        if not ev.get('has_edge'):
                            all_passable = False
                            all_viable = False
                            break
                        if not ev.get('has_passable'):
                            all_passable = False
                            all_viable = False
                            break
                        if not ev.get('has_viable'):
                            all_viable = False
                    if all_viable:
                        good.append(route)
                    elif all_passable:
                        ok.append(route)
                except Exception:
                    # 跳过异常路径
                    continue
            if good:
                return good[:5]
            if ok:
                return ok[:5]
            return []
        except Exception:
            return []

    def _generate_macro_route_candidates(self, start_region: int, dest_region: int, current_time: float, state_context: Dict[str, Any] = None) -> List[List[int]]:
        """
        Enhanced macro route candidate generation for CORY framework.
        Considers reachability, connectivity, congestion, distance, and state context.
        
        Args:
            state_context: Enhanced state information from CORY state construction
        """
        try:
            # Local imports to avoid global changes
            from itertools import islice
            import random
            try:
                # Prefer diverse, structurally different simple paths (no manual scoring)
                G = self.traffic_agent.region_graph
                max_k = 6  # generate slightly more, we'll trim to 5 later

                # Collect unique candidates using different generators
                seen = set()  # for tuple(path)
                cand_list = []

                # 1) Direct edge (if exists)
                if G.has_edge(start_region, dest_region):
                    direct = [start_region, dest_region]
                    tup = tuple(direct)
                    if tup not in seen:
                        seen.add(tup)
                        cand_list.append(direct)

                # 2) K shortest simple paths (unweighted)
                try:
                    for path in islice(nx.shortest_simple_paths(G, start_region, dest_region, weight=None), 0, max_k):
                        tup = tuple(path)
                        if tup not in seen:
                            seen.add(tup)
                            cand_list.append(path)
                except (nx.NetworkXNoPath, nx.NetworkXException):
                    # fallback silently
                    pass

                # 3) Edge-disjoint paths (adds routes with low overlap)
                try:
                    from networkx.algorithms.connectivity import disjoint_paths as nxdp
                    disjoint_gen = nxdp.edge_disjoint_paths(G, start_region, dest_region)
                    for idx, path in enumerate(disjoint_gen):
                        if idx >= 2:
                            break
                        tup = tuple(path)
                        if tup not in seen:
                            seen.add(tup)
                            cand_list.append(path)
                except Exception:
                    pass

                # 4) Ensure first-hop diversity (keep at most one per first hop first)
                first_hop_buckets = {}
                diversified = []
                for path in cand_list:
                    first_hop = path[1] if isinstance(path, list) and len(path) > 1 else dest_region
                    if first_hop not in first_hop_buckets:
                        first_hop_buckets[first_hop] = path
                        diversified.append(path)

                # 5) Fill with remaining unique simple paths (fallback) up to limit
                if len(diversified) < 6:
                    try:
                        extra_paths = list(nx.all_simple_paths(G, start_region, dest_region, cutoff=6))
                        random.shuffle(extra_paths)
                        for path in extra_paths:
                            tup = tuple(path)
                            if tup in seen:
                                continue
                            seen.add(tup)
                            diversified.append(path)
                            if len(diversified) >= 6:
                                break
                    except nx.NetworkXNoPath:
                        pass

                candidates = diversified
            except Exception as gen_err:
                # Emergency supplement as before
                try:
                    self._supplement_region_connections_emergency(start_region, dest_region)
                except Exception:
                    pass
                try:
                    candidates = list(nx.all_simple_paths(
                        self.traffic_agent.region_graph,
                        start_region,
                        dest_region,
                        cutoff=5
                    ))[:6]
                except Exception:
                    self.logger.log_error(f"MACRO_CANDIDATES: generation failed: {gen_err}")
                    return []

            # Evaluate side-effects only (no sorting by score). Keep up to 5.
            top_candidates = []
            for candidate in candidates:
                try:
                    _ = self._evaluate_macro_route_candidate(candidate, current_time)
                except Exception:
                    pass
                top_candidates.append(candidate)
                if len(top_candidates) >= 5:
                    break
            
            # Enhanced logging with state context if available
            if state_context:
                regional_congestion = state_context.get('regional_congestion', {})
                avg_congestion = sum(regional_congestion.get(r, 0) for route in top_candidates for r in route) / max(sum(len(route) for route in top_candidates), 1)
                self.logger.log_info(f"CORY_MACRO_CANDIDATES: Generated {len(top_candidates)} candidates for regions {start_region}->{dest_region}: {top_candidates} "
                                   f"(Avg route congestion: {avg_congestion:.3f})")
            else:
                self.logger.log_info(f"MACRO_CANDIDATES: Generated {len(top_candidates)} diversified candidates for {start_region}->{dest_region}")
            
            # 在宏观层面先行过滤：区域切边是否可用（未被封锁/未超限流）且拥堵低于阈值
            try:
                filtered_candidates = self._filter_macro_candidates_by_boundary_availability(top_candidates, current_time)
                if filtered_candidates:
                    if len(filtered_candidates) != len(top_candidates):
                        self.logger.log_info(f"MACRO_CANDIDATES_FILTERED: {len(filtered_candidates)}/{len(top_candidates)} pass boundary availability checks")
                    return filtered_candidates
            except Exception as _ferr:
                self.logger.log_warning(f"MACRO_CANDIDATES_FILTER_ERROR: {_ferr}")
            
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
            
            # [Regional LLM Feedback Call]
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
            
            # Combine observation with answer options so LLM input is complete
            complete_llm_input = f"{observation}\n\nOPTIONS: {answer_options}"
            
            # Log LLM call start
            call_id = self.logger.log_llm_call_start(
                "ObserverFeedback", vehicle_id, len(complete_llm_input),
                "decision", "", complete_llm_input
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
    
    def _evaluate_macro_route_candidate(self, route: List[int], current_time: float) -> Dict[str, Any]:
        """
        收集候选路径的原始度量（不返回综合分）。
        返回: dict(route_len, region_cong_series, boundary_cong_series, planned_usage_series, connectivity_series)
        """
        try:
            current_state = self.traffic_agent.global_state_history[-1] if self.traffic_agent.global_state_history else None
            region_cong_series = []
            boundary_cong_series = []
            planned_usage_series = []
            connectivity_series = []
            for idx, region_id in enumerate(route):
                cong = 0.0
                if current_state:
                    cong = float(current_state.regional_congestion.get(region_id, 0.0))
                region_cong_series.append((region_id, cong))
                if idx > 0:
                    planned_usage_series.append((region_id, int(self.region_vehicle_plans.get(region_id, 0))))
                out_conn = len([edge for edge in self.traffic_agent.region_graph.edges() if edge[0] == region_id])
                connectivity_series.append((region_id, int(out_conn)))
                if idx > 0:
                    prev_region = route[idx-1]
                    edges = self.traffic_agent.region_connections.get((prev_region, region_id), [])
                    if edges and current_state:
                        avg_b = sum(current_state.boundary_congestion.get(e, 0.0) for e in edges) / len(edges)
                    else:
                        avg_b = 0.0
                    boundary_cong_series.append((f"{prev_region}->{region_id}", float(avg_b)))
            return {
                'route_len': int(len(route)),
                'region_cong_series': region_cong_series,
                'boundary_cong_series': boundary_cong_series,
                'planned_usage_series': planned_usage_series,
                'connectivity_series': connectivity_series
            }
            
        except Exception as e:
            self.logger.log_error(f"ROUTE_EVALUATION: Failed to evaluate route {route}: {e}")
            return {
                'route_len': int(len(route)),
                'region_cong_series': [],
                'boundary_cong_series': [],
                'planned_usage_series': [],
                'connectivity_series': []
            }
    
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
            
            # Combine observation with answer options so LLM input is complete
            complete_llm_input = f"{observation_text}\n\nROUTE OPTIONS: {answer_options}"
            
            # Use LLM for decision making
            call_id = self.logger.log_llm_call_start(
                "MacroPlanning", vehicle_id, len(complete_llm_input),
                "decision", "", complete_llm_input
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
                            # Unified continuous congestion score in [0,1]
                            try:
                                speed_fraction = 0.0
                                if speed_limit > 0:
                                    speed_fraction = max(0.0, min(1.0, vehicle_speed / speed_limit))
                                speed_component = 1.0 - speed_fraction  # 0 at free-flow, ->1 as speed drops
                                occ_component = max(0.0, min(1.0, occupancy_rate))
                                congestion_score = 0.7 * speed_component + 0.3 * occ_component
                            except Exception:
                                congestion_score = max(0.0, min(1.0, occupancy_rate))
                            
                            # Update road info
                            self.road_info[edge_id].update({
                                "vehicle_num": max(0, vehicle_num),
                                "vehicle_speed": max(0.0, vehicle_speed),
                                "vehicle_length": max(0.0, vehicle_length),
                                "congestion_level": congestion_level,
                                "congestion_score": float(congestion_score),
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

    def _monitor_lane_exit_hamper(self, current_time: float) -> None:
        """监控接近路口车辆的 bestLanes 可用性，推断出口抢占困难热区，并维护热点缓存。"""
        try:
            veh_ids = traci.vehicle.getIDList()
            hamper_counts = {}
            sample_count = 0
            for vid in veh_ids:
                try:
                    edge = traci.vehicle.getRoadID(vid)
                    if not edge or edge.startswith(':'):
                        continue
                    lanes = traci.edge.getLaneNumber(edge)
                    if lanes <= 0:
                        continue
                    pos = traci.vehicle.getLanePosition(vid)
                    edge_len = traci.lane.getLength(f"{edge}_0") if lanes > 0 else 0
                    dist_to_end = max(0.0, edge_len - pos)
                    if dist_to_end > 200.0:
                        continue
                    sample_count += 1
                    best = None
                    try:
                        best = traci.vehicle.getBestLanes(vid)
                    except Exception:
                        best = None
                    if not best:
                        hamper_counts[edge] = hamper_counts.get(edge, 0) + 1
                        continue
                    # Heuristic: if none continuation matches next edge, count hamper
                    route = traci.vehicle.getRoute(vid)
                    next_edge = None
                    if route and edge in route:
                        idx = route.index(edge)
                        if idx + 1 < len(route):
                            next_edge = route[idx + 1]
                    ok = False
                    if next_edge is not None:
                        for info in best:
                            try:
                                allows = bool(info[2])
                                nLane = info[3] if len(info) > 3 else None
                            except Exception:
                                allows, nLane = False, None
                            if allows and isinstance(nLane, str) and nLane.startswith(next_edge + "_"):
                                ok = True
                                break
                    if not ok:
                        hamper_counts[edge] = hamper_counts.get(edge, 0) + 1
                except Exception:
                    pass
            if hamper_counts:
                top = sorted(hamper_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                self.logger.log_info(f"MONITOR_EXIT_HAMPER: samples={sample_count}, top_edges={top}")
                # Update rolling counters (decay others slightly)
                try:
                    # Exponential decay to forget old hotspots slowly
                    for e in list(self.exit_hamper_counts.keys()):
                        self.exit_hamper_counts[e] = int(self.exit_hamper_counts[e] * 0.9)
                        if self.exit_hamper_counts[e] <= 0:
                            del self.exit_hamper_counts[e]
                    for e, c in hamper_counts.items():
                        self.exit_hamper_counts[e] = self.exit_hamper_counts.get(e, 0) + int(c)
                    self._hamper_samples = max(sample_count, 1)
                    # Refresh hotspot avoid set based on threshold
                    hotspot_edges = [e for e, c in sorted(self.exit_hamper_counts.items(), key=lambda x: x[1], reverse=True)[:10]]
                    self.hotspot_avoid_edges = set(hotspot_edges)
                    # Update global macro guidance to avoid these edges for a short horizon
                    try:
                        self.global_macro_guidance['data'] = {
                            'avoid_edges': hotspot_edges,
                            'message': 'Avoid hotspot edges due to lane-exit hamper'
                        }
                        self.global_macro_guidance['expire_at'] = float(current_time + 300.0)
                    except Exception:
                        pass
                except Exception:
                    pass
        except Exception:
            pass

    def _monitor_connectivity_consistency(self, current_time: float) -> None:
        """抽样校验 adjacency 与 SUMO 连接一致性，记录路由找不到的边对。"""
        try:
            if not hasattr(self, 'road_network') or self.road_network is None:
                return
            edges = list(self.road_network.nodes())
            issues = []
            for i in range(0, len(edges), max(1, len(edges)//50)):
                a = edges[i]
                # pick one successor in adjacency if any
                try:
                    succ = list(self.road_network.successors(a))
                    if not succ:
                        continue
                    b = succ[0]
                    try:
                        res = traci.simulation.findRoute(a, b)
                        if not (res and hasattr(res, 'edges')):
                            issues.append((a, b))
                    except Exception:
                        issues.append((a, b))
                except Exception:
                    continue
            if issues:
                self.logger.log_warning(f"MONITOR_CONNECTIVITY_MISMATCH: samples={len(issues)} e.g. {issues[:3]}")
        except Exception:
            pass
    
    def update_vehicle_positions_and_regions(self, current_time: float):
        """Update vehicle positions and region assignments only - no decision making."""
        tracking_start_time = time.time()
        
        try:
            # Get all vehicles in simulation
            all_vehicle_ids = traci.vehicle.getIDList()
            autonomous_in_sim = [veh_id for veh_id in all_vehicle_ids if veh_id in self.autonomous_vehicles]
            
            self.logger.log_info(f"VEHICLE_POSITION_UPDATE_START: {len(autonomous_in_sim)}/{len(all_vehicle_ids)} autonomous vehicles active")
            
            # Process pending routes first - apply routes for vehicles that exited junctions
            pending_applied = 0
            pending_removed = []
            for vehicle_id, route_info in list(self.pending_routes.items()):
                if vehicle_id in autonomous_in_sim:
                    try:
                        current_edge = traci.vehicle.getRoadID(vehicle_id)
                        # If vehicle exited junction, apply the route
                        if not current_edge.startswith(':'):
                            safe_route = self._create_safe_route(current_edge, route_info['route'])
                            if safe_route:
                                try:
                                    self._set_route_and_register(vehicle_id, safe_route)
                                except Exception:
                                    traci.vehicle.setRoute(vehicle_id, safe_route)
                                self.logger.log_info(f"PENDING_ROUTE: Applied route for {vehicle_id} to {route_info['boundary_edge']}")
                                pending_applied += 1
                            else:
                                self.logger.log_warning(f"PENDING_ROUTE: Failed to apply route for {vehicle_id}")
                            pending_removed.append(vehicle_id)
                    except Exception as e:
                        self.logger.log_error(f"PENDING_ROUTE: Error processing {vehicle_id}: {e}")
                        pending_removed.append(vehicle_id)
                else:
                    # Vehicle no longer in simulation
                    pending_removed.append(vehicle_id)
            
            # Clean up processed pending routes
            for vehicle_id in pending_removed:
                self.pending_routes.pop(vehicle_id, None)
            
            if pending_applied > 0:
                self.logger.log_info(f"PENDING_ROUTES: Applied {pending_applied} routes, {len(self.pending_routes)} still pending")
            
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
                    # Record start time for new autonomous vehicles only (no planning yet)
                    if veh_id not in self.vehicle_start_times:
                        # Only process autonomous vehicles
                        if veh_id not in self.autonomous_vehicles:
                            continue
                            
                        # Get accurate departure time from SUMO
                        try:
                            actual_depart_time = traci.vehicle.getDeparture(veh_id)
                            self.vehicle_start_times[veh_id] = actual_depart_time
                            new_vehicles += 1
                            self.logger.log_info(f"NEW_AUTONOMOUS_VEHICLE: {veh_id} started at time {actual_depart_time:.1f}")
                            
                            # Queue for decision making (with deduplication)
                            if veh_id not in self.pending_decisions['new_vehicles']:
                                self.pending_decisions['new_vehicles'].append(veh_id)
                        except Exception as e:
                            # Fallback to current simulation time if getDeparture() fails
                            self.vehicle_start_times[veh_id] = current_time
                            new_vehicles += 1
                            self.logger.log_info(f"NEW_AUTONOMOUS_VEHICLE: {veh_id} started at time {current_time:.1f} (fallback)")
                            self.logger.log_warning(f"VEHICLE_DEPART_TIME: Could not get departure time for autonomous vehicle {veh_id}: {e}")
                            
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
                            # Only track autonomous vehicles for region initialization and planning
                            if veh_id not in self.autonomous_vehicles:
                                continue
                                
                            self.vehicle_regions[veh_id] = current_region
                            self.logger.log_info(f"AUTONOMOUS_VEHICLE_REGION_INIT: {veh_id} assigned to region {current_region}")
                            
                            # Queue for regional planning
                            self.pending_decisions['region_changes'].append({
                                'vehicle_id': veh_id,
                                'type': 'init',
                                'region': current_region
                            })
                            
                        if self.vehicle_regions[veh_id] != current_region:
                            # Vehicle changed regions - record for decision making (autonomous only)
                            if veh_id not in self.autonomous_vehicles:
                                # Update region tracking for non-autonomous but don't queue for planning
                                self.vehicle_regions[veh_id] = current_region
                                continue
                                
                            old_region = self.vehicle_regions[veh_id]
                            self.vehicle_regions[veh_id] = current_region
                            region_changes += 1
                            
                            self.logger.log_info(f"AUTONOMOUS_VEHICLE_REGION_CHANGE: {veh_id} moved from region {old_region} to {current_region}")
                            
                            # Queue for region change planning
                            self.pending_decisions['region_changes'].append({
                                'vehicle_id': veh_id,
                                'type': 'change',
                                'old_region': old_region,
                                'new_region': current_region
                            })
                        
                        # Real-time logging and metrics update - autonomous vehicles only
                        if veh_id in self.autonomous_vehicles:
                            destination = route[-1] if route else "unknown"
                            travel_time = current_time - self.vehicle_start_times[veh_id]
                            
                            # Get additional vehicle metrics
                            try:
                                vehicle_speed = traci.vehicle.getSpeed(veh_id)
                                vehicle_position = traci.vehicle.getPosition(veh_id)
                                route_progress = traci.vehicle.getRouteIndex(veh_id)
                                lane_position = traci.vehicle.getLanePosition(veh_id)
                                
                                # Initialize movement tracking for new vehicles
                                if veh_id not in self.vehicle_last_movement_time:
                                    self.vehicle_last_movement_time[veh_id] = current_time
                                if veh_id not in self.vehicle_last_edge:
                                    self.vehicle_last_edge[veh_id] = current_edge
                                if veh_id not in self.vehicle_edge_entry_time:
                                    self.vehicle_edge_entry_time[veh_id] = current_time
                                if veh_id not in self.vehicle_last_lane_position:
                                    self.vehicle_last_lane_position[veh_id] = lane_position
                                
                                # Update movement tracking based on edge change or lane progress
                                last_edge = self.vehicle_last_edge.get(veh_id, current_edge)
                                last_lane_pos = self.vehicle_last_lane_position.get(veh_id, lane_position)
                                if current_edge != last_edge:
                                    self.vehicle_last_movement_time[veh_id] = current_time
                                    self.vehicle_last_edge[veh_id] = current_edge
                                    self.vehicle_edge_entry_time[veh_id] = current_time
                                else:
                                    # Consider movement if lane position increased sufficiently or speed above threshold
                                    if (lane_position > last_lane_pos + 0.5) or (vehicle_speed > 0.3):
                                        self.vehicle_last_movement_time[veh_id] = current_time
                                        self.vehicle_last_lane_position[veh_id] = lane_position
                                
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
                                
                                # Check for stuck vehicles (queue for decision making) - autonomous only
                                # New rule: idle_duration threshold AND congestion/hamper abnormal
                                if veh_id in self.autonomous_vehicles:
                                    has_macro_plan = veh_id in self.vehicle_current_plans
                                    idle_threshold = 180.0 if not has_macro_plan else 300.0
                                    last_move = self.vehicle_last_movement_time.get(veh_id, current_time)
                                    idle_duration = max(0.0, current_time - last_move)
                                    # Real-time edge occupancy
                                    try:
                                        edge_occupancy = float(traci.edge.getLastStepOccupancy(current_edge))
                                    except Exception:
                                        edge_occupancy = 0.0
                                    # Lane-exit hamper rate (normalized)
                                    hamper_count = float(self.exit_hamper_counts.get(current_edge, 0))
                                    hamper_rate = (hamper_count / float(max(self._hamper_samples, 1))) if hamper_count > 0 else 0.0
                                    # Anomaly if occupancy high or hamper high
                                    occupancy_anomaly = edge_occupancy >= 0.6
                                    hamper_anomaly = hamper_rate >= 0.1
                                    if idle_duration > idle_threshold and (occupancy_anomaly or hamper_anomaly):
                                        # Track stuck event transitions (count once per stuck episode)
                                        if veh_id not in self.currently_stuck_vehicles:
                                            self.vehicle_stuck_events[veh_id] = self.vehicle_stuck_events.get(veh_id, 0) + 1
                                            self.currently_stuck_vehicles.add(veh_id)
                                        self.logger.log_warning(
                                            f"AUTONOMOUS_VEHICLE_STUCK: {veh_id} idle {idle_duration:.1f}s on {current_edge} (occ={edge_occupancy:.2f}, hamper={hamper_rate:.2f}, macro_plan={has_macro_plan})"
                                        )
                                        # If stuck duration >= 720s, activate stuck zone (center + 1/2-hop)
                                        if idle_duration >= self.cooldown_window_sec:
                                            self._activate_stuck_zone_for_vehicle(veh_id, current_edge, current_time)
                                    else:
                                        # Clear stuck state when condition relieved
                                        if veh_id in self.currently_stuck_vehicles:
                                            self.currently_stuck_vehicles.discard(veh_id)
                                        # If had active zone, end zone and move into cooldown
                                        if veh_id in self.vehicle_stuck_zone_edges:
                                            self._deactivate_stuck_zone_for_vehicle(veh_id, current_time)
                                
                            except Exception as metrics_error:
                                self.logger.log_warning(f"AUTONOMOUS_VEHICLE_METRICS_ERROR: {veh_id} -> {metrics_error}")
                        
                        vehicles_processed += 1
                        
                    else:
                        self.logger.log_warning(f"VEHICLE_EDGE_UNKNOWN: {veh_id} on edge {current_edge} not in region mapping")
                    
                    # Update waiting time and delay time for autonomous vehicles
                    if veh_id in self.autonomous_vehicles:
                        try:
                            waiting_time = traci.vehicle.getAccumulatedWaitingTime(veh_id)
                            self.vehicle_waiting_times_last[veh_id] = float(waiting_time)
                        except Exception:
                            pass
                        try:
                            delay_time = traci.vehicle.getTimeLoss(veh_id)
                            self.vehicle_delay_times_last[veh_id] = float(delay_time)
                        except Exception:
                            pass
                        
                except Exception as vehicle_error:
                    vehicles_failed += 1
                    self.logger.log_error(f"VEHICLE_PROCESSING_ERROR: {veh_id} -> {vehicle_error}")
                    continue
            
            # Check for completed vehicles with enhanced logging - autonomous only
            completed_this_step = 0
            try:
                arrived_vehicles = traci.simulation.getArrivedIDList()
                for veh_id in arrived_vehicles:
                    # Only track autonomous vehicles for completion metrics
                    if (veh_id in self.autonomous_vehicles and 
                        veh_id in self.vehicle_start_times and 
                        veh_id not in self.vehicle_end_times):
                        
                        self.vehicle_end_times[veh_id] = current_time
                        self.completed_vehicles += 1
                        completed_this_step += 1
                        
                        # Save final waiting time and delay time for completed vehicles
                        if veh_id in self.vehicle_waiting_times_last:
                            self.vehicle_waiting_times_final[veh_id] = self.vehicle_waiting_times_last[veh_id]
                        if veh_id in self.vehicle_delay_times_last:
                            self.vehicle_delay_times_final[veh_id] = self.vehicle_delay_times_last[veh_id]
                        
                        # Log completion with detailed metrics and cleanup
                        travel_time = current_time - self.vehicle_start_times[veh_id]
                        self.logger.log_vehicle_completion(
                            veh_id, self.vehicle_start_times[veh_id], current_time, travel_time
                        )
                        
                        # Record final dwell in last region BEFORE cleanup
                        try:
                            self._record_region_dwell_on_completion(veh_id, current_time)
                        except Exception as dwell_fin_err:
                            self.logger.log_warning(f"DWELL_FINAL_WARN: {veh_id} record failed: {dwell_fin_err}")

                        # RL Training Data Collection - Collect data BEFORE cleanup
                        if self.rl_data_collection_enabled:
                            self.logger.log_info(f"RL_DATA_COLLECTION_TRIGGER: Autonomous vehicle {veh_id} completed, collecting training data")
                            self._collect_rl_training_data(veh_id, travel_time, current_time)
                        else:
                            self.logger.log_info(f"RL_DATA_COLLECTION_DISABLED: Autonomous vehicle {veh_id} completed but RL data collection is disabled")
                        
                        # Clean up vehicle plans and broadcast completion
                        self._cleanup_completed_vehicle_plans(veh_id, current_time)
                        
                        # Real-time console output for completion
                        final_waiting = self.vehicle_waiting_times_final.get(veh_id, 0.0)
                        final_delay = self.vehicle_delay_times_final.get(veh_id, 0.0)
                        
                        print(f"[{current_time:.1f}s] AUTONOMOUS_VEHICLE_COMPLETED: {veh_id}")
                        print(f"  Total Travel Time: {travel_time:.1f}s")
                        print(f"  Waiting Time: {final_waiting:.1f}s")
                        print(f"  Delay/Time Loss: {final_delay:.1f}s")
                        print(f"  System ATT: {self._calculate_current_att():.1f}s")
                        print(f"  Completed: {self.completed_vehicles}/{len(self.autonomous_vehicles)}")
                        print("---")
                        
                        self.logger.log_info(f"AUTONOMOUS_VEHICLE_COMPLETED: {veh_id} finished in {travel_time:.1f}s")
                        
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
                self.logger.log_error(f"AUTONOMOUS_VEHICLE_COMPLETION_ERROR: {completion_error}")
            
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
            
            # Process new vehicle births - autonomous only
            for veh_id in new_vehicles:
                try:
                    # Only process autonomous vehicles for decision making
                    if veh_id not in self.autonomous_vehicles:
                        continue
                        
                    self.handle_vehicle_birth_macro_planning(veh_id, current_time)
                    decisions_processed += 1
                except Exception as e:
                    self.logger.log_error(f"VEHICLE_DECISION_ERROR: Failed to process new autonomous vehicle {veh_id}: {e}")
                    decisions_failed += 1
            
            # Process region changes - autonomous only
            for region_change in region_changes:
                try:
                    veh_id = region_change['vehicle_id']
                    
                    # Only process autonomous vehicles for decision making
                    if veh_id not in self.autonomous_vehicles:
                        continue
                    
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
                    self.logger.log_error(f"VEHICLE_DECISION_ERROR: Failed to process region change for autonomous vehicle {veh_id}: {e}")
                    decisions_failed += 1
            
            # Process stuck vehicles - replan disabled, zones handled inline during scanning
            if stuck_vehicles:
                self.logger.log_info(f"STUCK_REPLAN_DISABLED: {len(stuck_vehicles)} stuck vehicles observed; applying zone-based avoidance")
            
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
    
    def process_midstep_decisions(self, current_time: float):
        """Process only region changes and stuck vehicles to avoid duplication with pre-step planning."""
        try:
            if not hasattr(self, 'pending_decisions') or not self.pending_decisions:
                return
            region_changes = self.pending_decisions.get('region_changes', []).copy()
            stuck_vehicles = self.pending_decisions.get('stuck_vehicles', []).copy()
            # Clear only the processed queues, preserve new_vehicles for pre-step pipeline
            try:
                self.pending_decisions['region_changes'] = []
                self.pending_decisions['stuck_vehicles'] = []
            except Exception:
                pass
            # Region changes
            for region_change in region_changes:
                try:
                    veh_id = region_change['vehicle_id']
                    if veh_id not in self.autonomous_vehicles:
                        continue
                    if region_change.get('type') == 'init':
                        self.handle_vehicle_regional_planning(veh_id, region_change.get('region', self.vehicle_regions.get(veh_id, -1)), current_time)
                    elif region_change.get('type') == 'change':
                        self.handle_vehicle_region_change_replanning(
                            veh_id, region_change.get('old_region', self.vehicle_regions.get(veh_id, -1)), region_change.get('new_region', self.vehicle_regions.get(veh_id, -1)), current_time
                        )
                        self.handle_vehicle_regional_planning(veh_id, region_change.get('new_region', self.vehicle_regions.get(veh_id, -1)), current_time)
                except Exception as e:
                    self.logger.log_error(f"MIDSTEP_DECISION_ERROR: Region change for {veh_id}: {e}")
            # Stuck vehicles: no-op (zone avoidance already applied)
        except Exception as e:
            self.logger.log_error(f"MIDSTEP_DECISIONS_CRITICAL_ERROR: {e}")

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
            
            # Compute additional performance metrics for logging
            # Alat: Average latency (use current ATT)
            # AWai: Average accumulated waiting time (use completed vehicles)
            # Adis: Average remaining driving distance to destination (meters)
            # ADet: Average delay/timeLoss (use completed vehicles)
            try:
                from typing import List
                
                # Calculate AWai and ADet from COMPLETED vehicles (not active ones)
                completed_waiting_times = list(self.vehicle_waiting_times_final.values())
                completed_delay_times = list(self.vehicle_delay_times_final.values())
                
                # For active vehicles, still calculate remaining distance
                active_ids: List[str] = [vid for vid in self.vehicle_travel_metrics.keys() if vid in self.autonomous_vehicles]
                remain_distances = []
                for vid in active_ids:
                    # Remaining driving distance to destination
                    try:
                        route = traci.vehicle.getRoute(vid)
                        if route:
                            dest_edge = route[-1]
                            try:
                                dest_len = traci.edge.getLength(dest_edge)
                            except Exception:
                                dest_len = 0.0
                            # Use end position of destination edge
                            rem = traci.vehicle.getDrivingDistance(vid, dest_edge, max(0.0, dest_len - 0.1))
                            remain_distances.append(float(rem))
                    except Exception:
                        pass
                
                alat = float(current_att)
                # Use completed vehicles' final values for AWai and ADet
                awai = (sum(completed_waiting_times) / len(completed_waiting_times)) if completed_waiting_times else 0.0
                adis = (sum(remain_distances) / len(remain_distances)) if remain_distances else 0.0
                adet = (sum(completed_delay_times) / len(completed_delay_times)) if completed_delay_times else 0.0
                
                extra_metrics = {
                    'Alat': alat,
                    'AWai': awai,
                    'Adis': adis,
                    'ADet': adet,
                    'completed_vehicles_count': len(completed_waiting_times),
                    'active_vehicles_count': len(active_ids)
                }
            except Exception as e:
                self.logger.log_error(f"Metrics calculation error: {e}")
                extra_metrics = {
                    'Alat': float(current_att),
                    'AWai': 0.0,
                    'Adis': 0.0,
                    'ADet': 0.0
                }
            
            # Log comprehensive performance data (backward-compatible with older AgentLogger signatures)
            try:
                self.logger.log_system_performance(
                    regional_metrics, traffic_metrics, prediction_metrics, current_time, extra_metrics
                )
            except TypeError:
                # Fallback: older versions may not accept extra_metrics
                self.logger.log_system_performance(
                    regional_metrics, traffic_metrics, prediction_metrics, current_time
                )
            
        except Exception as e:
            self.logger.log_error(f"Performance logging failed: {e}")
    
    def _periodic_cleanup(self, current_time: float):
        """历史数据清理机制 - 每30分钟清理一次"""
        try:
            self.logger.log_info(f"PERIODIC_CLEANUP: Starting cleanup at {current_time}s")
            
            # 清理旧的历史状态
            if hasattr(self, 'traffic_agent') and self.traffic_agent:
                if hasattr(self.traffic_agent, 'global_state_history'):
                    self.traffic_agent.global_state_history.clear()
                    self.logger.log_info("CLEANUP: Cleared traffic agent global state history")
            
            # 清理区域智能体的历史决策
            if hasattr(self, 'regional_agents') and self.regional_agents:
                for region_id, agent in self.regional_agents.items():
                    if hasattr(agent, 'recent_decisions'):
                        agent.recent_decisions.clear()
                    if hasattr(agent, 'route_effectiveness'):
                        agent.route_effectiveness.clear()
                self.logger.log_info(f"CLEANUP: Cleared {len(self.regional_agents)} regional agents' history")
            
            # 清理老旧的车辆追踪数据(保留最近的1000个)
            if hasattr(self, 'vehicle_travel_metrics') and len(self.vehicle_travel_metrics) > 1000:
                old_count = len(self.vehicle_travel_metrics)
                # 按时间排序，保留最新的1000个
                sorted_vehicles = sorted(self.vehicle_travel_metrics.items(), 
                                        key=lambda x: x[1].get('last_update', 0), reverse=True)
                self.vehicle_travel_metrics = dict(sorted_vehicles[:1000])
                self.logger.log_info(f"CLEANUP: Reduced vehicle metrics from {old_count} to {len(self.vehicle_travel_metrics)}")
            
            # 清理完成的LLM调用的缓存
            if hasattr(self, 'completed_llm_calls') and len(self.completed_llm_calls) > 500:
                old_count = len(self.completed_llm_calls)
                self.completed_llm_calls = self.completed_llm_calls[-500:]  # 保留最近500个
                self.logger.log_info(f"CLEANUP: Reduced completed LLM calls from {old_count} to {len(self.completed_llm_calls)}")
            
            self.logger.log_info(f"PERIODIC_CLEANUP: Completed cleanup at {current_time}s")
            
        except Exception as e:
            self.logger.log_error(f"PERIODIC_CLEANUP: Failed at {current_time}s: {e}")
    
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
                test_call_id = self.logger.log_llm_call_start("TestAgent", "test_0", 100, "validation", "Test input", "Test input context for validation")
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
        Run the complete multi-agent simulation with synchronized decision-making.
        
        New architecture:
        1. Pre-step pause: Collect upcoming autonomous vehicles
        2. Synchronized planning: Traffic agent + Regional agent planning before step
        3. Step execution: Run simulation for one step_size
        4. Regional completion wait: Ensure all regional planning is complete
        
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
            
            # Initialize synchronized decision tracking
            # Get actual SUMO simulation time (respects --begin parameter)
            step = float(traci.simulation.getTime())
            initial_step = step
            # Calculate end time: start_time + max_steps (duration)
            end_time = initial_step + self.max_steps
            self.logger.log_info(f"SIMULATION_STEP_INIT: Starting from step {step:.1f}s (SUMO time)")
            self.logger.log_info(f"SIMULATION_DURATION: Will run for {self.max_steps}s (until {end_time:.1f}s)")
            self._init_synchronized_decision_tracking()
            # Initialize monitoring counters
            try:
                self._monitor_counters = {'route_find_fail': 0}
            except Exception:
                self._monitor_counters = {}
            
            self.logger.log_info(f"Starting multi-agent simulation")
            self.logger.log_info(f"SIMULATION_START: Beginning simulation with {self.max_steps}s duration (step_size: {self.step_size})")
            
            while step < end_time:
                simulation_cycle_start = time.time()
                next_step_time = step + self.step_size
                
                # ===== PHASE 1: PRE-STEP PAUSE AND PLANNING =====
                # 0.0 Global macro guidance at the start of timestamp
                try:
                    # Refresh road and global state minimally for guidance freshness
                    if not self.disable_global_guidance:
                        self.update_road_information(step)
                        self.traffic_agent.update_global_traffic_state(step)
                        regional_report = self.traffic_agent.collect_regional_congestion_report(step)
                        # Build hotspots and flow targets from existing monitors/queues
                        hotspots = self._build_hotspots_snapshot(step)
                        flow_targets = self._build_flow_targets_snapshot(step)
                        # Only call LLM if expired
                        if (not self.global_macro_guidance.get('data')) or (step >= float(self.global_macro_guidance.get('expire_at', 0.0))):
                            try:
                                if hasattr(self.llm_manager, 'get_traffic_llm_raw') and self.llm_manager.get_traffic_llm_raw():
                                    raw_traffic_llm = self.llm_manager.get_traffic_llm_raw()
                                else:
                                    raw_traffic_llm = self.traffic_llm
                                guidance = raw_traffic_llm.global_macro_guidance(
                                    global_state={
                                        'current_time': float(step),
                                        'total_vehicles': len(traci.vehicle.getIDList()) if traci else 0
                                    },
                                    regional_report=regional_report,
                                    hotspots=hotspots,
                                    flow_targets=flow_targets
                                )
                                ttl = int(guidance.get('ttl', 60))
                                self.global_macro_guidance['data'] = guidance
                                self.global_macro_guidance['expire_at'] = float(step + max(1, ttl))
                                # Broadcast as a system message
                                try:
                                    msg = {
                                        'type': 'global_macro_guidance',
                                        'time': float(step),
                                        'guidance': guidance
                                    }
                                    self.broadcast_messages.append(msg)
                                    self.communication_log.append(msg)
                                    self.logger.log_info(f"GLOBAL_GUIDANCE: {guidance.get('message', 'ok')} (ttl={ttl}s)")
                                except Exception:
                                    pass
                            except Exception as g_err:
                                self.logger.log_warning(f"GLOBAL_GUIDANCE_ERROR: {g_err}")
                except Exception as pre_g_err:
                    self.logger.log_warning(f"GLOBAL_GUIDANCE_PREP_ERROR: {pre_g_err}")
                # 1.1 Collect upcoming autonomous vehicles for next step
                upcoming_vehicles = self._collect_upcoming_autonomous_vehicles(step, next_step_time)
                
                # Pre-step planning for decision making
                if upcoming_vehicles or int(step) % 180 == 0:  # Log every 3 minutes or when there are vehicles
                    self.logger.log_info(f"SYNC_PHASE1_START: Pre-step planning for step {step:.1f} -> {next_step_time:.1f}")
                    # Opportunistically apply completed async LLM results at phase start
                    try:
                        self._process_completed_llm_calls(step)
                    except Exception:
                        pass
                
                # 1.2 Wait for any pending regional decisions from previous cycle
                self._wait_for_pending_regional_decisions()
                
                # 1.3 Synchronized planning: Traffic Agent + Regional Agents
                if upcoming_vehicles:
                    self._execute_synchronized_planning(upcoming_vehicles, step)
                
                if upcoming_vehicles or int(step) % 180 == 0:
                    self.logger.log_info(f"SYNC_PHASE1_COMPLETE: Planning completed, starting step execution")
                
                # ===== PHASE 2: STEP EXECUTION =====
                step_execution_start = time.time()
                
                # Advance simulation to next step
                traci.simulationStep(next_step_time)
                current_time = traci.simulation.getTime()
                
                # Apply decisions and update states after jump
                self.update_road_information(current_time)
                self.update_vehicle_positions_and_regions(current_time)
                
                # Monitoring: insertion backlog, collisions, and accumulated route-find failures
                try:
                    try:
                        pending = traci.simulation.getPendingVehicles()
                        if pending:
                            top_edges = []
                            try:
                                edge_ids = traci.edge.getIDList()
                                counts = []
                                for e in edge_ids:
                                    try:
                                        ev = traci.edge.getPendingVehicles(e)
                                        c = len(ev) if ev else 0
                                        if c > 0:
                                            counts.append((e, c))
                                    except Exception:
                                        pass
                                counts.sort(key=lambda x: x[1], reverse=True)
                                top_edges = counts[:5]
                            except Exception:
                                pass
                            self.logger.log_info(f"MONITOR_PENDING: total={len(pending)}, top_edges={top_edges}")
                    except Exception as _pend_err:
                        self.logger.log_warning(f"MONITOR_PENDING_ERROR: {_pend_err}")
                    try:
                        collisions = traci.simulation.getCollisions()
                        if collisions:
                            self.logger.log_warning(f"MONITOR_COLLISIONS: count={len(collisions)}")
                    except Exception as _col_err:
                        self.logger.log_warning(f"MONITOR_COLLISIONS_ERROR: {_col_err}")
                    try:
                        fails = getattr(self, '_monitor_counters', {}).get('route_find_fail', 0)
                        if fails:
                            self.logger.log_info(f"MONITOR_ROUTE_FIND_FAIL_ACCUM: {fails}")
                    except Exception:
                        pass
                    # Lane exit hamper (bestLanes) monitoring
                    try:
                        self._monitor_lane_exit_hamper(current_time)
                    except Exception as _bl_err:
                        self.logger.log_warning(f"MONITOR_BESTLANES_ERROR: {_bl_err}")
                    # Periodic connectivity consistency sampling
                    try:
                        if int(current_time) % 600 == 0:
                            self._monitor_connectivity_consistency(current_time)
                    except Exception as _conn_err:
                        self.logger.log_warning(f"MONITOR_CONNECTIVITY_ERROR: {_conn_err}")
                except Exception:
                    pass

                # Process decisions immediately after step (avoid queuing delays)
                try:
                    # Keep pre-step pipeline authoritative for new vehicles
                    if hasattr(self, 'pending_decisions') and isinstance(self.pending_decisions, dict):
                        try:
                            self.pending_decisions['new_vehicles'] = []
                        except Exception:
                            pass
                    # Process all pending decisions (stuck + region changes)
                    self.process_vehicle_decisions(current_time)
                except Exception as _dec_err:
                    self.logger.log_error(f"POSTSTEP_DECISIONS_ERROR: {_dec_err}")
                
                # Apply pre-planned decisions for newly spawned vehicles
                self._apply_preplanned_decisions_for_new_vehicles(current_time)
                
                # Update prediction engine and agents
                self.update_prediction_engine(current_time)
                self.update_traffic_agent(current_time)
                
                step_execution_time = (time.time() - step_execution_start) * 1000
                
                # ===== PHASE 3: REGIONAL COMPLETION CHECK =====
                # Ensure all regional agents have completed their planning
                self._ensure_regional_planning_completion()
                
                # Increment step for next iteration
                step = next_step_time
                
                # ===== PHASE 4: PERIODIC MAINTENANCE =====
                # Update visualization every 3600 steps
                if int(step) % self.vis_update_interval == 0:
                    self._update_visualization(step)
                
                # Before progress display, process any newly completed async results
                try:
                    self._process_completed_llm_calls(step)
                except Exception:
                    pass
                
                # Display progress (use relative step for progress calculation)
                relative_step = step - initial_step
                self.logger.display_progress(step, current_step=relative_step)
                
                # Log performance periodically
                self.log_system_performance(step)
                
                # Clean up periodically
                if int(step) % 1800 == 0:  # Every 30 minutes
                    self._periodic_cleanup(step)
                    self.cleanup_synchronized_decision_data(step)
                
                # Performance monitoring
                total_cycle_time = (time.time() - simulation_cycle_start) * 1000
                if total_cycle_time > 10000:  # > 10 seconds warning
                    self.logger.log_warning(f"SYNC_SLOW_CYCLE: Step {step:.1f} took {total_cycle_time:.1f}ms "
                                          f"(execution: {step_execution_time:.1f}ms)")
                
                # Check for and apply latest LoRA adapters (every 10 simulation steps)
                if int(step) % 10 == 0:
                    self._check_and_apply_latest_adapters()
                
                # Clean GPU memory and cache periodically
                if int(step) % 10 == 0:
                    self._cleanup_gpu_memory()
            
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
            
            # Close visualization
            if self.fig:
                plt.close(self.fig)
            
            # Shutdown executor
            self.executor.shutdown(wait=True)

            # 停止LoRA监听线程
            try:
                self._stop_lora_watchdog()
            except Exception:
                pass
    
    def _collect_upcoming_autonomous_vehicles(self, current_step: float, next_step: float) -> Dict[str, Dict]:
        """
        Collect autonomous vehicles that will start in the upcoming step interval.
        
        Args:
            current_step: Current simulation step
            next_step: Next simulation step (current_step + step_size)
            
        Returns:
            Dictionary mapping vehicle_id to vehicle info for upcoming autonomous vehicles
        """
        upcoming_vehicles = {}
        
        try:
            # Parse route file to find vehicles starting in the next step interval
            if not hasattr(self, 'route_data') or self.route_data is None:
                # Load route data if not already loaded
                self.route_data = parse_rou_file(self.route_file)
            
            # Find vehicles that will depart in the upcoming step interval
            for trip_data in self.route_data:
                if len(trip_data) >= 4:  # Ensure tuple has at least 4 elements (id, start, end, depart)
                    try:
                        vehicle_id, start_edge, end_edge, depart_time = trip_data[:4]
                        # self.logger.log_info(f"ROUTE_PARSE: {vehicle_id} start_edge={start_edge} end_edge={end_edge} depart_time={depart_time}")
                    except ValueError as e:
                        self.logger.log_warning(f"Failed to unpack trip_data {trip_data}: {e}")
                        continue
                    
                    # Only process vehicles that are marked as autonomous (2% of total)
                    # AND will depart in the upcoming step interval
                    if (vehicle_id in self.autonomous_vehicles and 
                        start_edge and end_edge and start_edge != end_edge and
                        current_step <= depart_time < next_step):
                        
                        # Determine start and end regions
                        start_region = self.edge_to_region.get(start_edge, -1)
                        end_region = self.edge_to_region.get(end_edge, -1)
                        
                        if start_region != -1 and end_region != -1:
                            upcoming_vehicles[vehicle_id] = {
                                'depart_time': depart_time,  # Use actual depart time from XML
                                'start_edge': start_edge,
                                'end_edge': end_edge,
                                'start_region': start_region,
                                'end_region': end_region,
                                'route': [start_edge, end_edge],  # Basic route with start/end
                                'requires_macro_planning': start_region != end_region,
                                'requires_regional_planning': True
                            }
            
            if upcoming_vehicles:
                self.logger.log_info(f"UPCOMING_VEHICLES: Found {len(upcoming_vehicles)} autonomous vehicles "
                                   f"available for planning at step {current_step:.1f}")
                
                # Log summary of vehicles by region
                by_region = {}
                for veh_id, veh_info in upcoming_vehicles.items():
                    start_region = veh_info['start_region']
                    by_region.setdefault(start_region, []).append(veh_id)
                
                region_summary = ", ".join([f"R{rid}:{len(vehs)}" for rid, vehs in sorted(by_region.items())])
                self.logger.log_info(f"UPCOMING_VEHICLES_BY_REGION: {region_summary}")
            
            return upcoming_vehicles
            
        except Exception as e:
            self.logger.log_error(f"UPCOMING_VEHICLES_ERROR: Failed to collect upcoming vehicles: {e}")
            return {}
    
    def _wait_for_pending_regional_decisions(self):
        """Wait for any pending regional decisions from the previous cycle."""
        if not self.pending_regional_decisions:
            return
        
        wait_start = time.time()
        max_wait_time = 30.0  # Maximum 30 seconds wait
        
        self.logger.log_info(f"REGIONAL_WAIT: Waiting for {len(self.pending_regional_decisions)} pending regional decisions")
        
        while self.pending_regional_decisions and (time.time() - wait_start) < max_wait_time:
            # Check and process completed regional decisions
            completed_regions = []
            
            for region_id, decision_future in self.pending_regional_decisions.items():
                if decision_future.done():
                    try:
                        # Process the completed decision
                        result = decision_future.result()
                        self._apply_regional_planning_result(region_id, result)
                        completed_regions.append(region_id)
                    except Exception as e:
                        self.logger.log_error(f"REGIONAL_WAIT_ERROR: Failed to process region {region_id} decision: {e}")
                        completed_regions.append(region_id)  # Remove failed decision
            
            # Remove completed decisions
            for region_id in completed_regions:
                del self.pending_regional_decisions[region_id]
            
            if self.pending_regional_decisions:
                time.sleep(0.1)  # Brief wait before checking again
        
        # Handle timeout
        if self.pending_regional_decisions:
            timeout_regions = list(self.pending_regional_decisions.keys())
            self.logger.log_warning(f"REGIONAL_WAIT_TIMEOUT: {len(timeout_regions)} regions did not complete: {timeout_regions}")
            
            # Clear timed out decisions to prevent blocking
            self.pending_regional_decisions.clear()
        
        wait_time = (time.time() - wait_start) * 1000
        if wait_time > 100:  # > 100ms
            self.logger.log_info(f"REGIONAL_WAIT_COMPLETE: Wait took {wait_time:.1f}ms")
    
    def _execute_synchronized_planning(self, upcoming_vehicles: Dict[str, Dict], current_step: float):
        """
        Execute synchronized planning for traffic agent and regional agents.
        
        Args:
            upcoming_vehicles: Dictionary of upcoming autonomous vehicles
            current_step: Current simulation step time
        """
        planning_start = time.time()
        
        try:
            # Phase 1: Traffic Agent Macro Planning（串行）
            vehicles_needing_macro_planning = {
                veh_id: veh_info for veh_id, veh_info in upcoming_vehicles.items()
                if veh_info['requires_macro_planning']
            }

            macro_plans = {}
            if vehicles_needing_macro_planning:
                self.logger.log_info(f"SYNC_MACRO_PLANNING: Planning for {len(vehicles_needing_macro_planning)} cross-region vehicles")
                try:
                    self.update_road_information(current_step)
                    self.traffic_agent.update_global_traffic_state(current_step)
                    self.latest_regional_report = self.traffic_agent.collect_regional_congestion_report(current_step)
                except Exception as _prep_err:
                    self.logger.log_warning(f"SYNC_MACRO_PLANNING_PREP: Failed to refresh context: {_prep_err}")

                for veh_id, veh_info in vehicles_needing_macro_planning.items():
                    self.logger.log_info(f"SERIAL_MACRO: Processing {veh_id} from region {veh_info['start_region']} to {veh_info['end_region']}")
                    route = self.traffic_agent.plan_single_macro_route(
                        vehicle_id=veh_id,
                        start_region=veh_info['start_region'],
                        end_region=veh_info['end_region'],
                        current_time=current_step,
                        coordination_data=getattr(self, 'latest_regional_report', None)
                    )
                    if route:
                        macro_plans[veh_id] = route
                        if not hasattr(self, 'upcoming_vehicle_cache'):
                            self.upcoming_vehicle_cache = {}
                        self.upcoming_vehicle_cache[veh_id] = {
                            'macro_route': route,
                            'planned_at': current_step
                        }
                        self.logger.log_info(f"SERIAL_MACRO: Completed {veh_id} -> route {route.region_sequence}")
                    else:
                        self.logger.log_warning(f"SERIAL_MACRO: Failed to plan route for {veh_id}")
                self.logger.log_info(f"SERIAL_MACRO: Completed planning for {len(macro_plans)}/{len(vehicles_needing_macro_planning)} vehicles")

            # Phase 2: Regional Agent Single-Vehicle Planning（串行）
            completed_regional_plans = {}
            if not hasattr(self, 'regional_planning_results'):
                self.regional_planning_results = {}

            for veh_id, veh_info in upcoming_vehicles.items():
                start_region = veh_info['start_region']
                target_region = veh_info['end_region']
                start_edge = veh_info.get('start_edge')
                self.logger.log_info(f"SERIAL_REGIONAL: Processing {veh_id} start_edge={start_edge} R{start_region}->R{target_region}")
                if start_region in self.regional_agents:
                    regional_agent = self.regional_agents[start_region]
                    try:
                        if start_edge:
                            if start_region == target_region:
                                end_edge = veh_info.get('end_edge')
                                if end_edge:
                                    result = self._offline_intra_region_plan(regional_agent, start_edge, end_edge, current_step)
                                    if result:
                                        self.regional_planning_results[veh_id] = result
                                        if start_region not in completed_regional_plans:
                                            completed_regional_plans[start_region] = {
                                                'status': 'completed',
                                                'region_id': start_region,
                                                'vehicle_count': 0,
                                                'successful_vehicles': []
                                            }
                                        completed_regional_plans[start_region]['vehicle_count'] += 1
                                        completed_regional_plans[start_region]['successful_vehicles'].append(veh_id)
                                        self.logger.log_info(f"SERIAL_REGIONAL: Completed {veh_id} -> destination {end_edge}")
                                    else:
                                        self.logger.log_warning(f"SERIAL_REGIONAL: Failed to plan intra-region route for {veh_id}")
                                else:
                                    self.logger.log_warning(f"SERIAL_REGIONAL: No end edge for {veh_id}")
                            else:
                                boundary_candidates = regional_agent._get_boundary_candidates_to_region(target_region)
                                if not boundary_candidates:
                                    boundary_candidates = regional_agent.outgoing_boundaries[:3]
                                route_candidates = regional_agent._generate_regional_route_candidates(start_edge, boundary_candidates, current_step)
                                if route_candidates:
                                    result = regional_agent._llm_select_regional_route(veh_id, start_edge, route_candidates, target_region, current_step)
                                    if result:
                                        self.regional_planning_results[veh_id] = result
                                        if start_region not in completed_regional_plans:
                                            completed_regional_plans[start_region] = {
                                                'status': 'completed',
                                                'region_id': start_region,
                                                'vehicle_count': 0,
                                                'successful_vehicles': []
                                            }
                                        completed_regional_plans[start_region]['vehicle_count'] += 1
                                        completed_regional_plans[start_region]['successful_vehicles'].append(veh_id)
                                        self.logger.log_info(f"SERIAL_REGIONAL: Completed {veh_id} -> boundary {result.get('boundary_edge', 'unknown')}")
                                    else:
                                        self.logger.log_warning(f"SERIAL_REGIONAL: Failed to plan route for {veh_id}")
                                else:
                                    self.logger.log_warning(f"SERIAL_REGIONAL: No route candidates for {veh_id}")
                        else:
                            self.logger.log_warning(f"SERIAL_REGIONAL: No start edge for {veh_id}")
                    except Exception as e:
                        self.logger.log_error(f"SERIAL_REGIONAL: Error planning for {veh_id}: {e}")
                else:
                    self.logger.log_warning(f"SERIAL_REGIONAL: No regional agent for region {start_region}")

            self.completed_regional_plans = completed_regional_plans

            planning_time = (time.time() - planning_start) * 1000
            self.logger.log_info(f"SYNC_PLANNING_COMPLETE: Planned for {len(upcoming_vehicles)} vehicles in {planning_time:.1f}ms "
                               f"(macro: {len(macro_plans)}, regional: {len(completed_regional_plans)} regions)")

        except Exception as e:
            self.logger.log_error(f"SYNC_PLANNING_ERROR: Synchronized planning failed: {e}")
    
    def _execute_simulation_step_range(self, start_step: float, end_step: float):
        """
        DEPRECATED: This method is no longer used in jump-based architecture.
        
        The simulation now uses direct jumping in run_simulation() instead of 
        step-by-step execution to avoid autonomous vehicle count inconsistencies.
        
        Args:
            start_step: Starting step time  
            end_step: Ending step time
        """
        self.logger.log_warning(f"DEPRECATED_METHOD: _execute_simulation_step_range called but no longer used in jump-based architecture")
        # This method is now a placeholder - actual execution happens in run_simulation()
    
    def _ensure_regional_planning_completion(self):
        """Ensure all regional planning has been completed before proceeding."""
        if not hasattr(self, 'pending_regional_decisions') or not self.pending_regional_decisions:
            return
        
        # This is a final check - if there are still pending decisions, wait briefly
        self._wait_for_pending_regional_decisions()

    def _offline_intra_region_plan(self, regional_agent, start_edge: str, destination_edge: str, current_time: float) -> Optional[Dict]:
        """Create an intra-region route without TraCI for pre-spawn vehicles.
        
        Plans from start_edge to destination_edge using the regional agent's
        network graph. This avoids TraCI calls before the vehicle exists in SUMO.
        """
        try:
            if not start_edge or not destination_edge:
                return None
            # Ensure destination belongs to the agent's region
            try:
                if (destination_edge not in regional_agent.edge_to_region or
                    regional_agent.edge_to_region[destination_edge] != regional_agent.region_id):
                    return None
            except Exception:
                return None
            # Attempt to plan via regional network (no TraCI)
            route = None
            try:
                route = regional_agent._plan_route(start_edge, destination_edge)
            except Exception:
                route = None
            if route and len(route) > 0:
                return {
                    'boundary_edge': destination_edge,
                    'route': list(route),
                    'travel_time': 0,
                    'reasoning': f'Offline intra-region preplanning at {current_time:.1f}s'
                }
            return None
        except Exception as e:
            try:
                self.logger.log_warning(f"OFFLINE_INTRA_PLAN: Failed for {start_edge}->{destination_edge}: {e}")
            except Exception:
                pass
            return None
    
    def _apply_preplanned_decisions_for_new_vehicles(self, current_time: float):
        """Apply pre-planned decisions for newly spawned autonomous vehicles only."""
        try:
            # Initialize idempotency tracker
            if not hasattr(self, 'last_route_apply_time'):
                self.last_route_apply_time = {}
            # Get newly spawned vehicles in this simulation step
            current_vehicles = set(traci.vehicle.getIDList())
            
            if not hasattr(self, 'previous_vehicles'):
                self.previous_vehicles = set()
            
            newly_spawned = current_vehicles - self.previous_vehicles
            self.previous_vehicles = current_vehicles
            
            if not newly_spawned:
                return
            
            # Filter to only autonomous vehicles - this is the key fix
            autonomous_newly_spawned = [v for v in newly_spawned if v in self.autonomous_vehicles]
            
            if not autonomous_newly_spawned:
                return
            
            applied_decisions = 0
            fallback_decisions = 0
            
            for vehicle_id in autonomous_newly_spawned:
                try:
                    # Check if we have pre-planned decisions for this autonomous vehicle
                    decision_applied = False

                    # If regional planning already applied, avoid duplicate route application
                    try:
                        if vehicle_id in getattr(self, 'vehicle_regional_plans', {}):
                            decision_applied = True
                    except Exception:
                        pass
                    
                    # Ensure same-region vehicles have a minimal macro plan to avoid macro_plan=False
                    if vehicle_id not in self.vehicle_current_plans:
                        try:
                            current_edge = traci.vehicle.getRoadID(vehicle_id)
                            current_region = self.edge_to_region.get(current_edge, -1)
                            if current_region != -1:
                                self.vehicle_current_plans[vehicle_id] = {
                                    'macro_route': [current_region],
                                    'current_region_index': 0,
                                    'creation_time': current_time,
                                    'last_update': current_time
                                }
                        except Exception:
                            pass
                    
                    # Apply macro route if available
                    if vehicle_id in self.upcoming_vehicle_cache:
                        cache_data = self.upcoming_vehicle_cache[vehicle_id]
                        macro_route = cache_data.get('macro_route')
                        
                        if macro_route:
                            # Apply macro route planning
                            self._apply_macro_route_to_vehicle(vehicle_id, macro_route, current_time)
                            decision_applied = True
                    
                    # Apply regional route if available from single-vehicle planning results
                    if not decision_applied and hasattr(self, 'regional_planning_results'):
                        if vehicle_id in self.regional_planning_results:
                            result = self.regional_planning_results[vehicle_id]
                            if result and 'boundary_edge' in result and 'route' in result:
                                # Safe apply: create safe route and validate before setting
                                try:
                                    current_edge = traci.vehicle.getRoadID(vehicle_id)
                                    route = result.get('route', [])
                                    safe_route = self._create_safe_route(current_edge, route)
                                except Exception:
                                    safe_route = None
                                applied_route = None
                                if safe_route and self._validate_route_setting(vehicle_id, safe_route):
                                    try:
                                        self._set_route_and_register(vehicle_id, safe_route)
                                    except Exception:
                                        traci.vehicle.setRoute(vehicle_id, safe_route)
                                    applied_route = list(safe_route)
                                else:
                                    self._apply_regional_route_to_vehicle(vehicle_id, result, current_time)
                                    applied_route = list(result.get('route', []))
                                # Attempt lane preselection via agent
                                try:
                                    region_id = self.edge_to_region.get(current_edge, -1)
                                    agent = self.regional_agents.get(region_id)
                                    if agent and hasattr(agent, '_ensure_exit_lane_preselection'):
                                        agent._ensure_exit_lane_preselection(vehicle_id, result.get('route', []))
                                except Exception:
                                    pass
                                # Record application to prevent duplicates
                                try:
                                    self.last_route_apply_time[vehicle_id] = current_time
                                    self.vehicle_regional_plans[vehicle_id] = {
                                        'region_id': self.edge_to_region.get(current_edge, -1),
                                        'target_region': self.edge_to_region.get(result.get('boundary_edge', ''), -1),
                                        'boundary_edge': result.get('boundary_edge'),
                                        'route': applied_route or list(result.get('route', [])),
                                        'creation_time': current_time,
                                        'travel_time': result.get('travel_time', 0),
                                        'reasoning': result.get('reasoning', 'Preplanned application at spawn')
                                    }
                                except Exception:
                                    pass
                                decision_applied = True
                    
                    if decision_applied:
                        applied_decisions += 1
                    else:
                        # Fallback: use original vehicle route or trigger emergency planning
                        self._apply_fallback_route_for_vehicle(vehicle_id, current_time)
                        fallback_decisions += 1
                        
                except Exception as e:
                    self.logger.log_error(f"APPLY_PREPLANNED_ERROR: Failed to apply decisions for autonomous vehicle {vehicle_id}: {e}")
                    fallback_decisions += 1
            
            if applied_decisions > 0 or fallback_decisions > 0:
                self.logger.log_info(f"PREPLANNED_DECISIONS: Applied {applied_decisions} pre-planned, {fallback_decisions} fallback decisions for autonomous vehicles")
            
            # Clean up applied cache entries
            for vehicle_id in autonomous_newly_spawned:
                if vehicle_id in self.upcoming_vehicle_cache:
                    del self.upcoming_vehicle_cache[vehicle_id]
                if hasattr(self, 'regional_planning_results') and vehicle_id in self.regional_planning_results:
                    del self.regional_planning_results[vehicle_id]
            
            # Clear pending new vehicle decisions to avoid accumulation (handled by Phase 1 planning)
            if hasattr(self, 'pending_decisions') and isinstance(self.pending_decisions, dict):
                try:
                    self.pending_decisions['new_vehicles'] = []
                except Exception:
                    pass
                    
        except Exception as e:
            self.logger.log_error(f"APPLY_PREPLANNED_CRITICAL_ERROR: {e}")
    
    def _handle_region_changes_during_execution(self, current_time: float):
        """
        Handle vehicles that enter new regions during step execution.
        Uses candidate route mechanism for immediate navigation.
        """
        try:
            if not hasattr(self, 'vehicle_regions_last_check'):
                self.vehicle_regions_last_check = {}
            
            region_changes = []
            current_vehicles = traci.vehicle.getIDList()
            
            for vehicle_id in current_vehicles:
                try:
                    current_edge = traci.vehicle.getRoadID(vehicle_id)
                    current_region = self.edge_to_region.get(current_edge, -1)
                    
                    if current_region == -1:
                        continue  # Invalid region
                    
                    # Check if region changed
                    previous_region = self.vehicle_regions_last_check.get(vehicle_id, current_region)
                    
                    if previous_region != current_region:
                        region_changes.append({
                            'vehicle_id': vehicle_id,
                            'old_region': previous_region,
                            'new_region': current_region,
                            'current_edge': current_edge
                        })
                        
                    # Update region tracking
                    self.vehicle_regions_last_check[vehicle_id] = current_region
                    
                except Exception:
                    continue
            
            # Handle region changes with candidate route mechanism
            for change in region_changes:
                self._handle_region_change_with_candidate_routes(change, current_time)
                
            if region_changes:
                self.logger.log_info(f"REGION_CHANGES_HANDLED: {len(region_changes)} vehicles changed regions during execution")
                
        except Exception as e:
            self.logger.log_error(f"REGION_CHANGES_ERROR: {e}")
    
    def _handle_region_change_with_candidate_routes(self, change: Dict, current_time: float):
        """
        Handle region change using candidate routes mechanism.
        Vehicle uses first candidate route while regional planning is in progress.
        """
        try:
            vehicle_id = change['vehicle_id']
            new_region = change['new_region']
            current_edge = change['current_edge']
            
            self.logger.log_info(f"REGION_CHANGE_CANDIDATE: {vehicle_id} entered region {new_region}, applying candidate route")
            
            # Get vehicle's destination
            route = traci.vehicle.getRoute(vehicle_id)
            if not route:
                return
                
            dest_edge = route[-1]
            dest_region = self.edge_to_region.get(dest_edge, new_region)
            
            if new_region == dest_region:
                return  # Already at destination region
            
            # Record dwell time for the region the vehicle just left (autonomous only)
            try:
                if vehicle_id in self.autonomous_vehicles and 'old_region' in change:
                    old_region = change['old_region']
                    # Use last known time in region from vehicle_travel_metrics if present
                    # Fallback to step size as minimal dwell
                    last_update = self.vehicle_travel_metrics.get(vehicle_id, {}).get('last_update', current_time)
                    # Estimate dwell as time since last decision update within tolerance
                    dwell_time = max(0.0, current_time - last_update)
                    # Avoid recording zero-length
                    if dwell_time <= 0.0:
                        dwell_time = self.step_size
                    self.traffic_agent.record_region_dwell_time(old_region, dwell_time)
            except Exception as dwell_err:
                self.logger.log_warning(f"DWELL_RECORD_WARN: {vehicle_id} region change dwell record failed: {dwell_err}")

            # Generate candidate routes for immediate navigation
            regional_agent = self.regional_agents.get(new_region)
            if not regional_agent:
                return
            
            # Get boundary candidates for this vehicle's target region
            boundary_candidates = []
            for boundary_info in self.boundary_edges:
                if (boundary_info['from_region'] == new_region and 
                    boundary_info['to_region'] == dest_region):
                    boundary_candidates.append(boundary_info['edge_id'])
            
            if not boundary_candidates:
                # Use any outgoing boundary from this region
                for boundary_info in self.boundary_edges:
                    if boundary_info['from_region'] == new_region:
                        boundary_candidates.append(boundary_info['edge_id'])
                        break
            
            if boundary_candidates:
                # Use the first boundary as immediate candidate route
                target_boundary = boundary_candidates[0]
                
                # Generate route to first candidate boundary
                try:
                    route_result = traci.simulation.findRoute(current_edge, target_boundary)
                    if route_result and route_result.edges:
                        candidate_route = list(route_result.edges)
                        
                        # Apply candidate route immediately
                        try:
                            self._set_route_and_register(vehicle_id, candidate_route)
                        except Exception:
                            traci.vehicle.setRoute(vehicle_id, candidate_route)
                        
                        self.logger.log_info(f"CANDIDATE_ROUTE_APPLIED: {vehicle_id} using candidate route to {target_boundary}")
                        
                        # Schedule regional planning for this vehicle asynchronously
                        self._schedule_async_regional_planning(vehicle_id, new_region, dest_region, current_time)
                        
                except Exception as route_error:
                    self.logger.log_warning(f"CANDIDATE_ROUTE_FAILED: {vehicle_id} route generation failed: {route_error}")
                    
        except Exception as e:
            self.logger.log_error(f"REGION_CHANGE_CANDIDATE_ERROR: {e}")
    
    def _schedule_async_regional_planning(self, vehicle_id: str, current_region: int, target_region: int, current_time: float):
        """Schedule asynchronous regional planning for a vehicle that changed regions."""
        try:
            # Submit async regional planning task
            future = self.executor.submit(
                self._async_regional_planning_for_vehicle,
                vehicle_id, current_region, target_region, current_time
            )
            
            # Store as pending decision
            if not hasattr(self, 'async_regional_futures'):
                self.async_regional_futures = {}
                
            self.async_regional_futures[vehicle_id] = {
                'future': future,
                'region': current_region,
                'target_region': target_region,
                'scheduled_at': current_time
            }
            
            self.logger.log_info(f"ASYNC_REGIONAL_SCHEDULED: {vehicle_id} regional planning scheduled for R{current_region}->R{target_region}")
            
        except Exception as e:
            self.logger.log_error(f"ASYNC_REGIONAL_SCHEDULE_ERROR: {e}")
    
    def _async_regional_planning_for_vehicle(self, vehicle_id: str, current_region: int, target_region: int, current_time: float) -> Dict:
        """Execute asynchronous regional planning for a single vehicle."""
        try:
            # Skip cross-region planning when current and target regions are the same
            if current_region == target_region:
                return {
                    'status': 'skipped_same_region',
                    'vehicle_id': vehicle_id,
                    'region': current_region
                }

            regional_agent = self.regional_agents.get(current_region)
            if not regional_agent:
                return {'status': 'no_agent', 'vehicle_id': vehicle_id}
            
            # Execute single-vehicle regional planning
            planning_result = regional_agent.make_regional_route_planning(vehicle_id, target_region, current_time)
            
            if planning_result:
                return {
                    'status': 'completed',
                    'vehicle_id': vehicle_id,
                    'region': current_region,
                    'target_region': target_region,
                    'planning_result': planning_result
                }
            else:
                return {'status': 'failed', 'vehicle_id': vehicle_id}
                
        except Exception as e:
            self.logger.log_error(f"ASYNC_REGIONAL_PLANNING_ERROR: {vehicle_id}: {e}")
            return {'status': 'error', 'vehicle_id': vehicle_id, 'error': str(e)}
    
    def _apply_regional_planning_result(self, region_id: int, result: Dict):
        """Apply the result of regional planning."""
        try:
            if result.get('status') != 'completed':
                return
            
            planning_results = result.get('planning_results', [])
            if not planning_results:
                return
                
            applied_count = 0
            for planning_result in planning_results:
                if planning_result and 'boundary_edge' in planning_result:
                    # This result should be applied when the corresponding vehicle spawns
                    # For now, we store it for application during vehicle spawning
                    applied_count += 1
            
            if applied_count > 0:
                self.logger.log_info(f"REGIONAL_PLANNING_STORED: Region {region_id} stored {applied_count} planning results")
                
        except Exception as e:
            self.logger.log_error(f"APPLY_REGIONAL_RESULT_ERROR: Region {region_id}: {e}")
    
    def _apply_macro_route_to_vehicle(self, vehicle_id: str, macro_route, current_time: float):
        """Apply macro route planning to a vehicle."""
        try:
            if not macro_route or not hasattr(macro_route, 'boundary_edges'):
                return
                
            # Store macro route in vehicle plans
            if not hasattr(self, 'vehicle_current_plans'):
                self.vehicle_current_plans = {}
                
            self.vehicle_current_plans[vehicle_id] = {
                'macro_route': macro_route.region_sequence,
                'planned_at': current_time,
                'status': 'active'
            }
            
            self.logger.log_info(f"MACRO_ROUTE_APPLIED: {vehicle_id} assigned macro route through regions {macro_route.region_sequence}")
            
        except Exception as e:
            self.logger.log_error(f"APPLY_MACRO_ROUTE_ERROR: {vehicle_id}: {e}")
    
    def _apply_regional_route_to_vehicle(self, vehicle_id: str, regional_result: Dict, current_time: float):
        """Apply regional route planning to a vehicle using LLM-selected boundary edge."""
        try:
            if not regional_result or 'boundary_edge' not in regional_result:
                return
                
            # Get LLM's decision: the target boundary edge
            target_boundary_edge = regional_result['boundary_edge']
            if not target_boundary_edge:
                return
                
            # Always compute route from current position to LLM-selected boundary edge
            try:
                current_edge = traci.vehicle.getRoadID(vehicle_id)
                
                # Use SUMO's findRoute to get optimal path from current position to LLM target
                route_result = traci.simulation.findRoute(current_edge, target_boundary_edge)
                
                if route_result and route_result.edges:
                    route = list(route_result.edges)
                    # Use safe route creation and validation before applying
                    try:
                        safe_route = self._create_safe_route(current_edge, route)
                    except Exception:
                        safe_route = None
                    if safe_route and self._validate_route_setting(vehicle_id, safe_route):
                        try:
                            self._set_route_and_register(vehicle_id, safe_route)
                        except Exception:
                            traci.vehicle.setRoute(vehicle_id, safe_route)
                        self.logger.log_info(f"LLM_ROUTE_APPLIED_SAFE: {vehicle_id} from {current_edge} to {target_boundary_edge}")
                    else:
                        try:
                            try:
                                self._set_route_and_register(vehicle_id, route)
                            except Exception:
                                traci.vehicle.setRoute(vehicle_id, route)
                            self.logger.log_warning(f"LLM_ROUTE_APPLIED_RAW: {vehicle_id} (safe validation failed), applying raw route")
                        except Exception:
                            self.logger.log_warning(f"LLM_ROUTE_SET_FAIL: {vehicle_id} route application failed")
                else:
                    # Fallback: use original planned route if available
                    original_route = regional_result.get('route', [])
                    if original_route:
                        try:
                            if self._validate_route_setting(vehicle_id, original_route):
                                try:
                                    self._set_route_and_register(vehicle_id, original_route)
                                except Exception:
                                    traci.vehicle.setRoute(vehicle_id, original_route)
                                self.logger.log_warning(f"LLM_ROUTE_FALLBACK_SAFE: {vehicle_id} using validated original route to {target_boundary_edge}")
                            else:
                                self.logger.log_warning(f"LLM_ROUTE_FALLBACK_REJECT: {vehicle_id} original route invalid")
                        except Exception:
                            self.logger.log_warning(f"LLM_ROUTE_FALLBACK_ERROR: {vehicle_id} original route apply failed")
                        
            except Exception as e:
                # Vehicle not in simulation yet, use original planned route
                original_route = regional_result.get('route', [])
                if original_route:
                    try:
                        self._set_route_and_register(vehicle_id, original_route)
                    except Exception:
                        traci.vehicle.setRoute(vehicle_id, original_route)
                    self.logger.log_info(f"LLM_ROUTE_PREPLANNED: {vehicle_id} using pre-planned route to {target_boundary_edge}")
                else:
                    raise e
            
        except Exception as e:
            self.logger.log_error(f"APPLY_LLM_ROUTE_ERROR: {vehicle_id}: {e}")
    
    def _apply_fallback_route_for_vehicle(self, vehicle_id: str, current_time: float):
        """Apply fallback route for vehicle when no pre-planned decision is available."""
        try:
            # Get vehicle's original route
            route = traci.vehicle.getRoute(vehicle_id)
            if route:
                # Keep original route - this is the simplest fallback
                self.logger.log_info(f"FALLBACK_ROUTE: {vehicle_id} using original route")
            else:
                # Emergency: generate a basic route
                self.logger.log_warning(f"FALLBACK_EMERGENCY: {vehicle_id} has no route, needs emergency planning")
                
        except Exception as e:
            self.logger.log_error(f"APPLY_FALLBACK_ROUTE_ERROR: {vehicle_id}: {e}")
    
    def _get_vehicle_region(self, vehicle_id: str) -> int:
        """Get the current region of a vehicle."""
        try:
            current_edge = traci.vehicle.getRoadID(vehicle_id)
            return self.edge_to_region.get(current_edge, -1)
        except Exception:
            return -1

    def _record_region_dwell_on_completion(self, vehicle_id: str, current_time: float):
        """Record final region dwell time for autonomous vehicle upon completion."""
        try:
            if vehicle_id not in self.autonomous_vehicles:
                return
            # Use last update time stored in travel metrics as the entry time proxy
            last_update = self.vehicle_travel_metrics.get(vehicle_id, {}).get('last_update', current_time)
            dwell_time = max(0.0, current_time - last_update)
            if dwell_time <= 0.0:
                dwell_time = self.step_size
            # Determine last known region
            region_id = self.vehicle_regions.get(vehicle_id, -1)
            if region_id != -1:
                self.traffic_agent.record_region_dwell_time(region_id, dwell_time)
        except Exception as e:
            self.logger.log_warning(f"DWELL_COMPLETE_WARN: {vehicle_id} record failed: {e}")
    
    def _process_async_regional_decisions(self, current_time: float):
        """
        Process completed asynchronous regional decisions and apply optimal routes.
        
        This method completes the candidate route mechanism by applying optimal
        routes when they become available.
        """
        if not hasattr(self, 'async_regional_futures'):
            return
        
        completed_futures = []
        
        for vehicle_id, future_data in self.async_regional_futures.items():
            future = future_data['future']
            
            if future.done():
                try:
                    result = future.result()
                    
                    if result.get('status') == 'completed':
                        # Apply optimal route to vehicle using candidate route mechanism
                        planning_result = result.get('planning_result')
                        
                        if planning_result and vehicle_id in traci.vehicle.getIDList():
                            region_id = result['region']
                            regional_agent = self.regional_agents.get(region_id)
                            
                            if regional_agent:
                                success = regional_agent.apply_optimal_route_decision(
                                    vehicle_id, planning_result, current_time
                                )
                                
                                if success:
                                    self.logger.log_info(f"ASYNC_OPTIMAL_APPLIED: {vehicle_id} transitioned "
                                                       f"from candidate to optimal route in region {region_id}")
                                else:
                                    self.logger.log_warning(f"ASYNC_OPTIMAL_FAILED: {vehicle_id} failed "
                                                          f"to apply optimal route in region {region_id}")
                    
                    completed_futures.append(vehicle_id)
                    
                except Exception as e:
                    self.logger.log_error(f"ASYNC_DECISION_PROCESS_ERROR: {vehicle_id}: {e}")
                    completed_futures.append(vehicle_id)
        
        # Clean up completed futures
        for vehicle_id in completed_futures:
            del self.async_regional_futures[vehicle_id]
        
        if completed_futures:
            self.logger.log_info(f"ASYNC_DECISIONS_COMPLETE: Processed {len(completed_futures)} async decisions")
    
    def _init_synchronized_decision_tracking(self):
        """Initialize tracking structures for synchronized decision-making."""
        if not hasattr(self, 'pending_regional_decisions'):
            self.pending_regional_decisions = {}
        
        if not hasattr(self, 'upcoming_vehicle_cache'):
            self.upcoming_vehicle_cache = {}
        
        if not hasattr(self, 'async_regional_futures'):
            self.async_regional_futures = {}
        
        if not hasattr(self, 'completed_regional_plans'):
            self.completed_regional_plans = {}
        
        if not hasattr(self, 'vehicle_regions_last_check'):
            self.vehicle_regions_last_check = {}
        
        if not hasattr(self, 'previous_vehicles'):
            self.previous_vehicles = set()
        
        self.logger.log_info("SYNC_INIT: Initialized synchronized decision tracking structures")
    
    def cleanup_synchronized_decision_data(self, current_time: float):
        """
        Clean up synchronized decision data periodically to prevent memory buildup.
        """
        try:
            cleanup_threshold = current_time - 1800  # Clean data older than 30 minutes
            
            # Clean up upcoming vehicle cache
            if hasattr(self, 'upcoming_vehicle_cache'):
                expired_vehicles = [
                    veh_id for veh_id, data in self.upcoming_vehicle_cache.items()
                    if data.get('planned_at', 0) < cleanup_threshold
                ]
                for veh_id in expired_vehicles:
                    del self.upcoming_vehicle_cache[veh_id]
            
            # Clean up async regional futures for non-existent vehicles
            if hasattr(self, 'async_regional_futures'):
                current_vehicles = set(traci.vehicle.getIDList())
                expired_futures = [
                    veh_id for veh_id in self.async_regional_futures.keys()
                    if veh_id not in current_vehicles
                ]
                for veh_id in expired_futures:
                    # Cancel the future if possible
                    try:
                        future_data = self.async_regional_futures[veh_id]
                        future_data['future'].cancel()
                    except Exception:
                        pass
                    del self.async_regional_futures[veh_id]
            
            # Clean up vehicle region tracking for non-existent vehicles
            if hasattr(self, 'vehicle_regions_last_check'):
                current_vehicles = set(traci.vehicle.getIDList())
                expired_vehicles = [
                    veh_id for veh_id in self.vehicle_regions_last_check.keys()
                    if veh_id not in current_vehicles
                ]
                for veh_id in expired_vehicles:
                    del self.vehicle_regions_last_check[veh_id]
            
            # Clean up completed regional plans older than threshold
            if hasattr(self, 'completed_regional_plans'):
                self.completed_regional_plans.clear()  # Clean all after use
            
            cleaned_items = len(expired_vehicles) if 'expired_vehicles' in locals() else 0
            if cleaned_items > 0:
                self.logger.log_info(f"SYNC_CLEANUP: Cleaned {cleaned_items} expired decision tracking entries")
                
        except Exception as e:
            self.logger.log_error(f"SYNC_CLEANUP_ERROR: {e}")
    
    def get_synchronization_status(self) -> Dict:
        """
        Get status information about the synchronized decision system.
        
        Returns:
            Dictionary containing system status information
        """
        try:
            status = {
                'pending_regional_decisions': len(getattr(self, 'pending_regional_decisions', {})),
                'upcoming_vehicle_cache': len(getattr(self, 'upcoming_vehicle_cache', {})),
                'async_regional_futures': len(getattr(self, 'async_regional_futures', {})),
                'completed_regional_plans': len(getattr(self, 'completed_regional_plans', {})),
                'tracked_vehicles': len(getattr(self, 'vehicle_regions_last_check', {}))
            }
            
            # Add details about pending operations
            if hasattr(self, 'async_regional_futures'):
                active_regions = set()
                for future_data in self.async_regional_futures.values():
                    active_regions.add(future_data.get('region', -1))
                status['active_async_regions'] = list(active_regions)
            
            return status
            
        except Exception as e:
            self.logger.log_error(f"SYNC_STATUS_ERROR: {e}")
            return {'error': str(e)}
    
    def handle_vehicle_region_change_replanning(self, vehicle_id: str, old_region: int, new_region: int, current_time: float):
        """
        Handle LLM-based macro route replanning when autonomous vehicle reaches new region.
        
        This implements the user requirement: when vehicle arrives in new region,
        provide original macro route and new candidates to traffic agent for LLM decision.
        """
        try:
            # Ensure this is an autonomous vehicle
            if vehicle_id not in self.autonomous_vehicles:
                self.logger.log_warning(f"REGION_CHANGE_REPLAN_SKIP: {vehicle_id} is not an autonomous vehicle, skipping replanning")
                return
                
            self.logger.log_info(f"REGION_CHANGE_REPLAN: Autonomous vehicle {vehicle_id} from region {old_region} to {new_region}")
            
            # Get vehicle's destination
            route = traci.vehicle.getRoute(vehicle_id)
            if not route:
                return
            
            dest_edge = route[-1]
            dest_region = self.edge_to_region.get(dest_edge, new_region)
            
            # If already at destination region, no replanning needed
            if new_region == dest_region:
                self.logger.log_info(f"REGION_CHANGE_REPLAN: Autonomous vehicle {vehicle_id} reached destination region {dest_region}")
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
                self.logger.log_warning(f"REGION_CHANGE_REPLAN: No new candidates for autonomous vehicle {vehicle_id}")
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
        """Handle regional planning for autonomous vehicle within a region using LLM."""
        try:
            # Ensure this is an autonomous vehicle
            if vehicle_id not in self.autonomous_vehicles:
                self.logger.log_warning(f"REGIONAL_PLANNING_SKIP: {vehicle_id} is not an autonomous vehicle, skipping regional planning")
                return
                
            # Promote macro plan from upcoming cache if not yet written
            if vehicle_id not in self.vehicle_current_plans:
                try:
                    # Some macro decisions are produced asynchronously into upcoming cache
                    cached = getattr(self, 'upcoming_vehicle_cache', {}).get(vehicle_id)
                    if cached and 'macro_route' in cached and cached['macro_route']:
                        # Ensure we store a region sequence list in current plans
                        cached_macro = cached['macro_route']
                        macro_seq = None
                        try:
                            # Prefer attribute access without importing the class
                            if hasattr(cached_macro, 'region_sequence'):
                                macro_seq = list(cached_macro.region_sequence)
                            elif isinstance(cached_macro, list):
                                macro_seq = list(cached_macro)
                        except Exception:
                            macro_seq = None

                        if macro_seq:
                            self.vehicle_current_plans[vehicle_id] = {
                                'macro_route': macro_seq,
                                'current_region_index': 0,
                                'creation_time': current_time,
                                'last_update': current_time,
                                'replanned': False
                            }
                except Exception:
                    pass
            
            # Get or synthesize vehicle's macro route
            if vehicle_id not in self.vehicle_current_plans:
                # Attempt to synthesize single-region macro plan if destination is within current region
                try:
                    route = traci.vehicle.getRoute(vehicle_id)
                    if route and len(route) > 0:
                        dest_edge = route[-1]
                        dest_region = self.edge_to_region.get(dest_edge, -1)
                        if dest_region == region_id and dest_region != -1:
                            self.vehicle_current_plans[vehicle_id] = {
                                'macro_route': [region_id],
                                'current_region_index': 0,
                                'creation_time': current_time,
                                'last_update': current_time,
                                'single_region': True,
                                'cory_decision_type': 'synthesized_single_region'
                            }
                        else:
                            # For cross-region without macro plan, generate macro plan now
                            self.handle_vehicle_birth_macro_planning(vehicle_id, current_time)
                except Exception:
                    pass
            if vehicle_id not in self.vehicle_current_plans:
                self.logger.log_warning(f"REGIONAL_PLANNING: No macro plan for autonomous vehicle {vehicle_id}")
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
                            self.logger.log_warning(f"REGIONAL_PLANNING: Autonomous vehicle {vehicle_id} on invalid edge: {current_edge}")
                            return
                    except Exception as traci_error:
                        self.logger.log_error(f"REGIONAL_PLANNING: TraCI error for autonomous vehicle {vehicle_id}: {traci_error}")
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
                                # Synchronous LLM call for regional route selection (no heuristic fallback)
                                try:
                                    with self.timeout_context(300):
                                        # Direct synchronous selection using RegionalAgent, waiting for final result
                                        regional_plan = regional_agent.make_regional_route_planning(
                                            vehicle_id, next_region, current_time
                                        )
                                except TimeoutError:
                                    self.logger.log_error(f"REGIONAL_PLANNING: LLM call timed out for {vehicle_id}")
                                    regional_plan = None
                                    # Trigger GPU cleanup after timeout
                                    self._cleanup_gpu_memory()
                            else:
                                # No route candidates available
                                regional_plan = None
                        else:
                            self.logger.log_warning(f"REGIONAL_PLANNING: No boundary candidates for {vehicle_id}")
                            return
                            
                    except Exception as planning_error:
                        self.logger.log_warning(f"REGIONAL_PLANNING: Async planning failed for {vehicle_id}, using fallback: {planning_error}")
                        # Fallback to synchronous call with timeout
                        try:
                            with self.timeout_context(300):
                                regional_plan = regional_agent.make_regional_route_planning(
                                    vehicle_id, next_region, current_time
                                )
                        except TimeoutError:
                            self.logger.log_error(f"REGIONAL_PLANNING: Fallback LLM call timed out for {vehicle_id}")
                            regional_plan = None
                            # Trigger GPU cleanup after timeout
                            self._cleanup_gpu_memory()
                    
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
                                    try:
                                        self._set_route_and_register(vehicle_id, safe_route)
                                    except Exception:
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
            else:
                try:
                    # Intra-region planning via RegionalAgent to destination edge (LLM-based)
                    route = traci.vehicle.getRoute(vehicle_id)
                    if not route:
                        self.logger.log_warning(f"REGIONAL_PLANNING_INTRA: No original route for autonomous vehicle {vehicle_id}")
                        return
                    dest_edge = route[-1]
                    dest_region = self.edge_to_region.get(dest_edge, -1)
                    if dest_region != region_id:
                        self.logger.log_warning(f"REGIONAL_PLANNING_INTRA: Destination edge {dest_edge} not in region {region_id} for {vehicle_id}")
                        return
                    # Get regional agent
                    regional_agent = self.regional_agents.get(region_id)
                    if not regional_agent:
                        self.logger.log_warning(f"REGIONAL_PLANNING_INTRA: No regional agent for region {region_id}")
                        return
                    # Execute LLM-based intra-region destination planning (with timeout)
                    try:
                        with self.timeout_context(300):
                            regional_plan = regional_agent.make_intra_region_destination_planning(
                                vehicle_id, dest_edge, current_time
                            )
                    except TimeoutError:
                        self.logger.log_error(f"REGIONAL_PLANNING_INTRA: LLM call timed out for {vehicle_id}")
                        regional_plan = None
                        self._cleanup_gpu_memory()
                    # Apply plan
                    if regional_plan and 'route' in regional_plan and regional_plan['route']:
                        try:
                            current_edge = traci.vehicle.getRoadID(vehicle_id)
                            safe_route = None
                            try:
                                safe_route = self._create_safe_route(current_edge, regional_plan['route'])
                            except Exception:
                                safe_route = None
                            if safe_route and self._validate_route_setting(vehicle_id, safe_route):
                                try:
                                    self._set_route_and_register(vehicle_id, safe_route)
                                except Exception:
                                    traci.vehicle.setRoute(vehicle_id, safe_route)
                                self.logger.log_info(f"REGIONAL_PLANNING_INTRA_SAFE: {vehicle_id} -> {dest_edge}")
                            else:
                                try:
                                    self._set_route_and_register(vehicle_id, regional_plan['route'])
                                except Exception:
                                    traci.vehicle.setRoute(vehicle_id, regional_plan['route'])
                                self.logger.log_warning(f"REGIONAL_PLANNING_INTRA_RAW: {vehicle_id} route applied without safe validation")
                        except Exception as set_err:
                            self.logger.log_error(f"REGIONAL_PLANNING_INTRA_SET_FAIL: {vehicle_id}: {set_err}")
                            return
                        # Optional lane preselection via regional agent
                        try:
                            if hasattr(regional_agent, '_ensure_exit_lane_preselection'):
                                regional_agent._ensure_exit_lane_preselection(vehicle_id, regional_plan['route'])
                        except Exception:
                            pass
                        # Store planning record
                        try:
                            self.vehicle_regional_plans[vehicle_id] = {
                                'region_id': region_id,
                                'target_region': region_id,
                                'boundary_edge': dest_edge,
                                'route': list(regional_plan['route']),
                                'creation_time': current_time,
                                'travel_time': regional_plan.get('travel_time', 0),
                                'reasoning': regional_plan.get('reasoning', 'Intra-region route to destination')
                            }
                        except Exception:
                            pass
                        # Log decision
                        self._log_vehicle_decision(vehicle_id, "REGIONAL_PLANNING_INTRA", f"Dest: {dest_edge}", current_time)
                        self.logger.log_info(f"REGIONAL_PLANNING: {vehicle_id} intra-region route planned to destination in region {region_id}")
                    else:
                        self.logger.log_warning(f"REGIONAL_PLANNING_INTRA: No valid plan to {dest_edge} for {vehicle_id}")
                except Exception as intra_err:
                    self.logger.log_error(f"REGIONAL_PLANNING_INTRA_ERROR: {vehicle_id}: {intra_err}")
            
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
            
            # Handle case where macro_plan might be a MacroRoute object instead of dict
            if hasattr(macro_plan, 'region_sequence'):
                # Convert MacroRoute object to dict format
                macro_plan = {
                    'macro_route': macro_plan.region_sequence,
                    'cooperation_quality': 0.5,
                    'cory_decision_type': 'converted_from_macro_route',
                    'pioneer_decision': {},
                    'observer_feedback': {},
                    'j1_judge_evaluation': {},
                    'state_context': {}
                }
            
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
            
            # Determine agent type based on the data content
            agent_type = self._determine_agent_type(rl_training_data)
            
            # Send to training manager if queue is available
            if self.training_queue is not None:
                if self.enable_time_sliced_training:
                    # Time-sliced training: add to buffer and check threshold
                    self.training_data_buffer[agent_type].append(rl_training_data)
                    self.logger.log_info(f"RL_DATA_BUFFERED: Added {vehicle_id} data to {agent_type} buffer "
                                       f"(size: {len(self.training_data_buffer[agent_type])}/{self.training_threshold[agent_type]})")
                    
                    # Check if training threshold is reached
                    if len(self.training_data_buffer[agent_type]) >= self.training_threshold[agent_type]:
                        self._trigger_time_sliced_training(agent_type)
                else:
                    # Original mode: send directly to training queue
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
        Calculate ATT-only rewards for RL training.
        
        Reward definition:
          - Base (ATT-only): att_reward = E / (E + T), where E is adaptive expected time
            derived from global congestion and route complexity; T is actual travel time.
          - Stuck penalty: per-event deduction applied cumulatively.
        
        Returns the same ATT-only reward for both Traffic and Regional LLMs.
        """
        try:
            # Derive route length from macro plan if available
            macro_plan = self.vehicle_current_plans.get(vehicle_id, {})
            macro_route_seq = []
            try:
                if hasattr(macro_plan, 'region_sequence'):
                    macro_route_seq = list(macro_plan.region_sequence or [])
                elif isinstance(macro_plan, dict):
                    route_val = macro_plan.get('macro_route', [])
                    if hasattr(route_val, 'region_sequence'):
                        macro_route_seq = list(route_val.region_sequence or [])
                    elif isinstance(route_val, (list, tuple)):
                        macro_route_seq = list(route_val)
                elif isinstance(macro_plan, (list, tuple)):
                    macro_route_seq = list(macro_plan)
            except Exception:
                macro_route_seq = []
            route_length = len(macro_route_seq) if macro_route_seq else 1

            # Adaptive expected time: fuse global congestion and route complexity (neutral cooperation)
            expected_time = float(self._get_route_baseline_time(route_length, 0.5))

            # ATT-only reward in (0, 1]: smooth, scale-free, fair across distances/conditions
            E = max(1e-6, expected_time)
            T = max(1e-6, float(travel_time))
            att_reward = E / (E + T)

            # Apply per-stuck-event penalty (accumulative)
            try:
                stuck_events = int(self.vehicle_stuck_events.get(vehicle_id, 0))
            except Exception:
                stuck_events = 0
            penalty_per_event = float(getattr(self, 'stuck_penalty_per_event', 0.15))
            total_penalty = min(0.95, penalty_per_event * max(0, stuck_events))
            att_reward = max(0.0, att_reward - total_penalty)

            rewards: Dict[str, Any] = {
                'traffic_llm': {
                    'att_reward': float(att_reward),
                    'total_reward': float(att_reward)
                },
                'regional_llm': {
                    'att_reward': float(att_reward),
                    'total_reward': float(att_reward)
                }
            }
            if stuck_events > 0:
                rewards['stuck_penalty'] = {
                    'events': stuck_events,
                    'penalty_per_event': penalty_per_event,
                    'total_penalty': total_penalty
                }

            self.logger.log_info(
                f"RL_REWARDS: {vehicle_id} -> Traffic:{rewards['traffic_llm']['total_reward']:.3f}, "
                f"Regional:{rewards['regional_llm']['total_reward']:.3f}"
            )

            return rewards

        except Exception as e:
            self.logger.log_error(f"RL_REWARD_CALCULATION_ERROR: {vehicle_id} -> {e}")
            # Return default rewards on error (ATT-only neutral)
            return {
                'traffic_llm': {'att_reward': 0.5, 'total_reward': 0.5},
                'regional_llm': {'att_reward': 0.5, 'total_reward': 0.5}
            }
    
    def _calculate_regional_efficiency(self, vehicle_id: str, travel_time: float) -> float:
        """
        Calculate regional efficiency based on vehicle's performance in different regions.
        
        Following CLAUDE.md: Regional efficiency considers vehicle's contribution to 
        regional traffic flow and its adherence to regional routing decisions.
        """
        try:
            efficiency_score = 0.5  # Default baseline
            
            # Get vehicle's macro route robustly across representations
            macro_plan = self.vehicle_current_plans.get(vehicle_id, {})
            macro_route = []
            try:
                if hasattr(macro_plan, 'region_sequence'):
                    macro_route = list(macro_plan.region_sequence or [])
                    macro_plan = {
                        'macro_route': macro_route,
                        'cooperation_quality': 0.5,
                        'cory_decision_type': 'converted_from_macro_route'
                    }
                elif isinstance(macro_plan, dict):
                    route_val = macro_plan.get('macro_route', [])
                    if hasattr(route_val, 'region_sequence'):
                        macro_route = list(route_val.region_sequence or [])
                        macro_plan['macro_route'] = macro_route
                    elif isinstance(route_val, (list, tuple)):
                        macro_route = list(route_val)
                    else:
                        macro_route = []
                elif isinstance(macro_plan, (list, tuple)):
                    macro_route = list(macro_plan)
                    macro_plan = {'macro_route': macro_route}
                else:
                    macro_route = []
                    macro_plan = {'macro_route': []}
            except Exception:
                macro_route = []
                macro_plan = {'macro_route': []}
            
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
                vehicle_plan = self.vehicle_current_plans[vehicle_id]
                
                # Robustly extract macro_route sequence regardless of representation
                if hasattr(vehicle_plan, 'region_sequence'):
                    macro_route = list(vehicle_plan.region_sequence or [])
                elif isinstance(vehicle_plan, dict):
                    route_val = vehicle_plan.get('macro_route', [])
                    if hasattr(route_val, 'region_sequence'):
                        macro_route = list(route_val.region_sequence or [])
                    elif isinstance(route_val, (list, tuple)):
                        macro_route = list(route_val)
                    else:
                        macro_route = []
                elif isinstance(vehicle_plan, (list, tuple)):
                    macro_route = list(vehicle_plan)
                else:
                    macro_route = []
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
            
            # Time-sliced training status
            if self.enable_time_sliced_training:
                print(f"Time-sliced Training Status:")
                print(f"  Training Active: {self.is_training_active}")
                print(f"  Simulation Paused: {self.simulation_paused}")
                print(f"  Training Sessions: {self.training_session_count}")
                print(f"  Buffer Sizes: Traffic={len(self.training_data_buffer['traffic'])}, Regional={len(self.training_data_buffer['regional'])}")
            
            print("=" * 50)
            
        except Exception as e:
            self.logger.log_error(f"REAL_TIME_STATUS: Failed to print status: {e}")
    
    # ===== TIME-SLICED TRAINING METHODS =====
    
    def _determine_agent_type(self, rl_training_data: dict) -> str:
        """
        Determine which agent type (traffic or regional) the training data belongs to
        
        Args:
            rl_training_data: Training data dictionary
            
        Returns:
            str: 'traffic' or 'regional'
        """
        # Check if this data contains macro-level planning information
        if (rl_training_data.get('macro_route') or 
            rl_training_data.get('cory_decision_type') != 'unknown' or
            rl_training_data.get('pioneer_decision')):
            return 'traffic'
        else:
            return 'regional'
    
    def _trigger_time_sliced_training(self, agent_type: str) -> bool:
        """
        触发时间分片训练
        
        Args:
            agent_type: 触发训练的智能体类型
        
        Returns:
            bool: 是否成功触发训练
        """
        if self.is_training_active or self.simulation_paused:
            return False  # Training already in progress
            
        with self.training_lock:
            if self.is_training_active:  # Double check
                return False
                
            self.logger.log_info(f"TIME_SLICED_TRAINING_TRIGGER: Starting {agent_type} training with {len(self.training_data_buffer[agent_type])} samples")
            
            try:
                # Step 1: Pause simulation and save state
                self.simulation_paused = True
                self.is_training_active = True
                
                # Step 2: Send training data to queue
                training_batch = {
                    'session_id': f"timesliced_{self.training_session_count}_{agent_type}",
                    'agent_type': agent_type,
                    'data': self.training_data_buffer[agent_type].copy(),
                    'trigger_time': time.time(),
                    'trigger_step': self.current_step,
                    'time_sliced_training': True  # Flag to indicate this is time-sliced training
                }
                
                self.training_queue.put(training_batch)
                
                # Step 3: Release inference models（可选，默认跳过以避免无LoRA时空重载）
                if hasattr(self, 'llm_manager') and self.llm_manager:
                    if getattr(self, 'release_models_during_time_sliced_training', False):
                        self.logger.log_info("TIME_SLICED_TRAINING: Releasing inference models for training...")
                        success = self.llm_manager.release_inference_models()
                        self.models_released = bool(success)
                        if not success:
                            self.logger.log_warning("TIME_SLICED_TRAINING: Inference model release failed, continuing with training")
                    else:
                        self.models_released = False
                        self.logger.log_info("TIME_SLICED_TRAINING: Skip releasing inference models (no-adapter mode)")
                
                # Clear the buffer for this agent type
                self.training_data_buffer[agent_type].clear()
                self.training_session_count += 1
                
                self.logger.log_info(f"TIME_SLICED_TRAINING: Training triggered successfully, session ID: {training_batch['session_id']}")
                return True
                
            except Exception as e:
                self.logger.log_error(f"TIME_SLICED_TRAINING_ERROR: Failed to trigger training for {agent_type}: {e}")
                self.simulation_paused = False
                self.is_training_active = False
                return False
    
    def _resume_after_training(self, new_adapters: dict = None) -> bool:
        """
        训练完成后恢复仿真
        
        Args:
            new_adapters: 新的适配器路径字典
        
        Returns:
            bool: 是否成功恢复
        """
        with self.training_lock:
            if not self.is_training_active:
                return True  # Already resumed
                
            try:
                self.logger.log_info("TIME_SLICED_TRAINING: Resuming simulation after training completion")
                
                # Step 1: Restore inference models only if确有新LoRA，或之前确实释放过
                if hasattr(self, 'llm_manager') and self.llm_manager:
                    has_new_adapters = bool(new_adapters) and isinstance(new_adapters, dict) and any(new_adapters.values())
                    if has_new_adapters or getattr(self, 'models_released', False):
                        success = self.llm_manager.restore_inference_models(new_adapters if has_new_adapters else None)
                        if not success:
                            self.logger.log_error("TIME_SLICED_TRAINING: Failed to restore inference models")
                            return False
                    else:
                        self.logger.log_info("TIME_SLICED_TRAINING: No new adapters and models not released, skip restore")
                
                # Step 2: Resume simulation
                self.simulation_paused = False
                self.is_training_active = False
                self.models_released = False
                
                self.logger.log_info("TIME_SLICED_TRAINING: Simulation resumed successfully")
                return True
                
            except Exception as e:
                self.logger.log_error(f"TIME_SLICED_TRAINING_RESUME_ERROR: Failed to resume simulation: {e}")
                return False
    
    def notify_training_complete(self, session_id: str, new_adapters: dict = None) -> bool:
        """
        通知环境训练已完成
        
        Args:
            session_id: 训练会话ID
            new_adapters: 新的适配器路径字典
        
        Returns:
            bool: 是否成功处理通知
        """
        try:
            self.logger.log_info(f"TIME_SLICED_TRAINING_COMPLETE: Received training completion notification for session {session_id}")
            if new_adapters:
                self.logger.log_info(f"TIME_SLICED_TRAINING_COMPLETE: New adapters available: {new_adapters}")
            
            return self._resume_after_training(new_adapters)
            
        except Exception as e:
            self.logger.log_error(f"TIME_SLICED_TRAINING_NOTIFY_ERROR: Failed to process training completion notification: {e}")
            return False
    
    def get_time_sliced_training_status(self) -> dict:
        """
        获取时间分片训练状态
        
        Returns:
            dict: 训练状态信息
        """
        return {
            'enabled': self.enable_time_sliced_training,
            'is_training_active': self.is_training_active,
            'simulation_paused': self.simulation_paused,
            'training_session_count': self.training_session_count,
            'training_thresholds': self.training_threshold,
            'buffer_sizes': {
                'traffic': len(self.training_data_buffer['traffic']),
                'regional': len(self.training_data_buffer['regional'])
            }
        }
    
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
            
            # Combine observation with answer options so LLM input is complete
            complete_llm_input = f"{observation_text}\n\nREPLANNING OPTIONS: {answer_options}"
            
            # Use LLM for replanning decision
            call_id = self.logger.log_llm_call_start(
                "MacroReplanning", vehicle_id, len(complete_llm_input),
                "decision", "", complete_llm_input
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
                        # Or if regional plan is outdated (increased from 600s to 2400s for stability)
                        elif current_time - self.vehicle_regional_plans[vehicle_id].get('creation_time', 0) > 2400:
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
    
    def _get_system_load_level(self) -> float:
        """
        Get current system load level for performance monitoring.
        """
        try:
            # Calculate load based on active LLM calls, queued vehicles, etc.
            total_vehicles = len(self.vehicle_current_plans)
            vehicles_needing_plans = sum(1 for v_id in self.vehicle_current_plans 
                                       if v_id not in self.vehicle_regional_plans)
            
            # Load factor: 0.0 (no load) to 1.0+ (overload)
            load_factor = vehicles_needing_plans / max(1, total_vehicles * 0.1)  # 10% planning at once is normal
            
            return min(load_factor, 2.0)  # Cap at 2.0 for extreme overload
            
        except Exception:
            return 0.5  # Default moderate load
    
    def _execute_batch_regional_planning(self, regions_needing_planning: List[int], current_time: float):
        """Execute serial regional planning across multiple regions - one region at a time."""
        try:
            self.logger.log_info(f"SERIAL_PLANNING: Starting serial planning for {len(regions_needing_planning)} regions")
            
            completed_regions = []
            failed_regions = []
            
            # Process regions serially, one at a time
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
                        try:
                            start_time = current_time
                            # Execute planning for this region (serial, per-vehicle LLM calls)
                            result = self._execute_regional_planning_for_vehicles(
                                region_id, vehicles_needing_planning, current_time
                            )
                            
                            if result:
                                completed_regions.append(region_id)
                                planning_time = current_time - start_time
                                self.logger.log_info(f"SERIAL_PLANNING: Region {region_id} completed "
                                                   f"{len(vehicles_needing_planning)} vehicle plans in {planning_time:.1f}s")
                            else:
                                failed_regions.append(region_id)
                                self.logger.log_warning(f"SERIAL_PLANNING: Region {region_id} planning failed")
                                
                        except Exception as region_error:
                            failed_regions.append(region_id)
                            self.logger.log_error(f"SERIAL_PLANNING: Region {region_id} planning error: {region_error}")
            
            # Summary logging
            total_regions = len(regions_needing_planning)
            success_rate = len(completed_regions) / total_regions if total_regions > 0 else 0
            
            self.logger.log_info(f"SERIAL_PLANNING_COMPLETE: {len(completed_regions)}/{total_regions} regions completed "
                               f"({success_rate:.1%} success rate)")
            
            if failed_regions:
                self.logger.log_warning(f"SERIAL_PLANNING: Failed regions: {failed_regions}")
            
        except Exception as e:
            self.logger.log_error(f"SERIAL_PLANNING: Critical error in serial regional planning: {e}")
    
    def _handle_same_region_planning(self, regional_agent, vehicle_id: str, region_id: int, current_time: float) -> Dict:
        """
        Handle same-region planning without LLM calls - direct route to destination.
        """
        try:
            current_edge = traci.vehicle.getRoadID(vehicle_id)
            route = traci.vehicle.getRoute(vehicle_id)
            
            if not route or len(route) == 0:
                return None
            
            # Get destination edge (last edge in original route)
            destination_edge = route[-1]
            
            # Use SUMO's built-in routing for same-region travel
            try:
                same_region_route = traci.simulation.findRoute(current_edge, destination_edge)
                if same_region_route and same_region_route.edges:
                    travel_time = same_region_route.travelTime if hasattr(same_region_route, 'travelTime') else 300
                    
                    self.logger.log_info(f"SAME_REGION_PLANNING: Direct route for {vehicle_id} in region {region_id}, travel_time: {travel_time:.1f}s")
                    
                    return {
                        'boundary_edge': destination_edge,  # destination is the "boundary"
                        'route': same_region_route.edges,
                        'travel_time': travel_time,
                        'reasoning': f'Same-region direct routing to destination (Region {region_id})'
                    }
            except Exception as route_error:
                self.logger.log_warning(f"SAME_REGION_PLANNING: SUMO routing failed for {vehicle_id}: {route_error}")
            
            # Fallback: use existing route if available
            if len(route) > 0:
                estimated_time = len(route) * 30  # rough estimate: 30s per edge
                return {
                    'boundary_edge': route[-1],
                    'route': route,
                    'travel_time': estimated_time,
                    'reasoning': f'Same-region fallback routing (Region {region_id})'
                }
                
            return None
            
        except Exception as e:
            self.logger.log_error(f"SAME_REGION_PLANNING: Failed for {vehicle_id}: {e}")
            return None
    
    def _should_allow_llm_call(self, region_id: int, current_time: float) -> bool:
        """
        Circuit breaker logic to prevent LLM call overload.
        """
        if region_id not in self.llm_failure_counts:
            return True
        
        failure_count, last_failure_time = self.llm_failure_counts[region_id]
        
        # If circuit breaker is active
        if failure_count >= self.llm_circuit_breaker_threshold:
            # Check if recovery time has passed
            if current_time - last_failure_time > self.llm_circuit_breaker_timeout:
                # Reset circuit breaker
                self.llm_failure_counts[region_id] = (0, current_time)
                self.logger.log_info(f"CIRCUIT_BREAKER: Reset for region {region_id}")
                return True
            else:
                # Still in circuit breaker state
                return False
        
        return True
    
    def _record_llm_failure(self, region_id: int, current_time: float):
        """
        Record LLM failure for circuit breaker tracking.
        """
        if region_id not in self.llm_failure_counts:
            self.llm_failure_counts[region_id] = (1, current_time)
        else:
            failure_count, _ = self.llm_failure_counts[region_id]
            self.llm_failure_counts[region_id] = (failure_count + 1, current_time)
            
            if failure_count + 1 >= self.llm_circuit_breaker_threshold:
                self.logger.log_warning(f"CIRCUIT_BREAKER: Activated for region {region_id} after {failure_count + 1} failures")
    
    def _get_fallback_regional_plan(self, regional_agent, vehicle_id: str, target_region: int, current_time: float) -> Dict:
        """
        Get fallback regional plan without LLM call.
        """
        try:
            # Use regional agent's boundary candidates directly
            boundary_candidates = regional_agent._get_boundary_candidates_to_region(target_region)
            if not boundary_candidates:
                boundary_candidates = regional_agent.outgoing_boundaries[:1] if regional_agent.outgoing_boundaries else []
            
            if boundary_candidates:
                # Select first boundary candidate
                selected_boundary = boundary_candidates[0]
                current_edge = traci.vehicle.getRoadID(vehicle_id)
                
                # Generate simple route to boundary
                try:
                    route_result = traci.simulation.findRoute(current_edge, selected_boundary)
                    if route_result and route_result.edges:
                        travel_time = route_result.travelTime if hasattr(route_result, 'travelTime') else 600
                        
                        return {
                            'boundary_edge': selected_boundary,
                            'route': route_result.edges,
                            'travel_time': travel_time,
                            'reasoning': f'Fallback routing to boundary {selected_boundary} (Circuit breaker active)'
                        }
                except Exception:
                    pass
            
            return None
            
        except Exception as e:
            self.logger.log_error(f"FALLBACK_REGIONAL_PLAN: Failed for {vehicle_id}: {e}")
            return None
    
    def _execute_regional_planning_for_vehicles(self, region_id: int, vehicle_ids: List[str], current_time: float) -> bool:
        """Execute efficient regional planning inspired by traffic agent's success strategy."""
        try:
            regional_agent = self.regional_agents[region_id]
            
            self.logger.log_info(f"EFFICIENT_REGIONAL: Planning for {len(vehicle_ids)} vehicles in region {region_id}")
            
            # Handle empty vehicle list
            if not vehicle_ids:
                self.logger.log_info(f"EFFICIENT_REGIONAL: No vehicles to process in region {region_id}")
                return True
            
            successful_plans = 0
            failed_plans = 0
            
            # Separate same-region and inter-region vehicles for optimal processing
            same_region_vehicles = []
            inter_region_vehicles = []
            
            for vehicle_id in vehicle_ids:
                try:
                    if vehicle_id not in self.vehicle_current_plans:
                        failed_plans += 1
                        continue
                        
                    macro_route = self.vehicle_current_plans[vehicle_id]['macro_route']
                    current_region_index = self.vehicle_current_plans[vehicle_id].get('current_region_index', 0)
                    
                    if current_region_index + 1 >= len(macro_route):
                        failed_plans += 1
                        continue
                    
                    target_region = macro_route[current_region_index + 1]
                    
                    # Categorize vehicles by planning complexity
                    if region_id == target_region:
                        same_region_vehicles.append((vehicle_id, target_region))
                    else:
                        inter_region_vehicles.append((vehicle_id, target_region))
                        
                except Exception as e:
                    self.logger.log_error(f"EFFICIENT_REGIONAL: Failed to categorize {vehicle_id}: {e}")
                    failed_plans += 1
            
            # Phase 1: Process same-region vehicles efficiently (no LLM needed)
            for vehicle_id, target_region in same_region_vehicles:
                regional_plan = self._handle_same_region_planning(
                    regional_agent, vehicle_id, region_id, current_time
                )
                if self._apply_regional_plan(vehicle_id, region_id, target_region, regional_plan, current_time):
                    successful_plans += 1
                else:
                    failed_plans += 1
            
            # Phase 2: Process inter-region vehicles using traffic agent's efficient batch strategy
            if inter_region_vehicles:
                if self._should_allow_llm_call(region_id, current_time):
                    # Use efficient batching strategy from traffic agent
                    batch_successful, batch_failed = self._execute_efficient_inter_region_planning(
                        regional_agent, region_id, inter_region_vehicles, current_time
                    )
                    successful_plans += batch_successful
                    failed_plans += batch_failed
                else:
                    # Circuit breaker active - use fallback for all inter-region vehicles
                    self.logger.log_warning(f"CIRCUIT_BREAKER: Using fallback for {len(inter_region_vehicles)} vehicles in region {region_id}")
                    for vehicle_id, target_region in inter_region_vehicles:
                        regional_plan = self._get_fallback_regional_plan(
                            regional_agent, vehicle_id, target_region, current_time
                        )
                        if self._apply_regional_plan(vehicle_id, region_id, target_region, regional_plan, current_time):
                            successful_plans += 1
                        else:
                            failed_plans += 1
            
            # Summary and performance feedback
            total_vehicles = len(vehicle_ids)
            success_rate = successful_plans / total_vehicles if total_vehicles > 0 else 0
            
            self.logger.log_info(f"EFFICIENT_REGIONAL_COMPLETE: Region {region_id} processed {successful_plans}/{total_vehicles} "
                               f"vehicle plans ({success_rate:.1%} success), {failed_plans} failed "
                               f"(Same-region: {len(same_region_vehicles)}, Inter-region: {len(inter_region_vehicles)})")
            
            # Update circuit breaker state based on results
            if success_rate < 0.3:  # Very poor performance
                self._record_llm_failure(region_id, current_time)
            elif success_rate > 0.8:  # Good performance - help recovery
                self._record_llm_success(region_id, current_time)
            
            # Always return True to ensure all vehicles are processed
            return True
            
        except Exception as e:
            self.logger.log_error(f"EFFICIENT_REGIONAL: Critical error for region {region_id}: {e}")
            return False
    
    def _execute_efficient_inter_region_planning(self, regional_agent, region_id: int, 
                                               vehicle_targets: List[Tuple[str, int]], 
                                               current_time: float) -> Tuple[int, int]:
        """Execute inter-region planning using traffic agent's efficient batch strategy."""
        try:
            if not vehicle_targets:
                return 0, 0
            
            successful_plans = 0
            failed_plans = 0
            
            # Use single-vehicle planning for each vehicle 
            for vehicle_id, target_region in vehicle_targets:
                try:
                    # Use single-vehicle regional planning
                    regional_plan = regional_agent.make_regional_route_planning(
                        vehicle_id, target_region, current_time
                    )
                    
                    if self._apply_regional_plan(vehicle_id, region_id, target_region, regional_plan, current_time):
                        successful_plans += 1
                    else:
                        failed_plans += 1
                        
                except Exception as e:
                    self.logger.log_warning(f"EFFICIENT_INTER_REGION: Single vehicle planning failed for {vehicle_id}: {e}")
                    self._record_llm_failure(region_id, current_time)
                    # Use fallback for failed vehicle
                    regional_plan = self._get_fallback_regional_plan(
                        regional_agent, vehicle_id, target_region, current_time
                    )
                    if self._apply_regional_plan(vehicle_id, region_id, target_region, regional_plan, current_time):
                        successful_plans += 1
                    else:
                        failed_plans += 1
            
            return successful_plans, failed_plans
            
        except Exception as e:
            self.logger.log_error(f"EFFICIENT_INTER_REGION: Failed for region {region_id}: {e}")
            return 0, len(vehicle_targets)
    
    def _apply_regional_plan(self, vehicle_id: str, region_id: int, target_region: int, 
                           regional_plan: Dict, current_time: float) -> bool:
        """Apply regional plan to vehicle and store in system."""
        try:
            if not regional_plan or 'boundary_edge' not in regional_plan or 'route' not in regional_plan:
                self.logger.log_warning(f"APPLY_PLAN: Invalid plan for {vehicle_id}")
                return False
            
            # Store successful regional plan
            self.vehicle_regional_plans[vehicle_id] = {
                'region_id': region_id,
                'target_region': target_region,
                'boundary_edge': regional_plan['boundary_edge'],
                'route': regional_plan['route'],
                'creation_time': current_time,
                'travel_time': regional_plan.get('travel_time', 0),
                'reasoning': regional_plan.get('reasoning', 'Efficient regional planning')
            }
            
            # Apply route to vehicle in SUMO with safety checks
            if regional_plan['route'] and len(regional_plan['route']) > 0:
                try:
                    # Ensure safe route setting
                    current_edge = traci.vehicle.getRoadID(vehicle_id)
                    safe_route = self._create_safe_route(current_edge, regional_plan['route'])
                    # If vehicle is on junction edge, store route for later application
                    if current_edge.startswith(':'):
                        self.pending_routes[vehicle_id] = {
                            'route': regional_plan['route'],
                            'boundary_edge': regional_plan['boundary_edge'],
                            'travel_time': regional_plan.get('travel_time', 0),
                            'reasoning': regional_plan.get('reasoning', 'Regional planning'),
                            'creation_time': current_time
                        }
                        self.logger.log_info(f"APPLY_PLAN: Stored route for {vehicle_id} (on junction, will apply when exits)")
                        return True  # Consider successful - route will be applied later
                    else:
                        # Normal route application
                        if safe_route:
                            try:
                                self._set_route_and_register(vehicle_id, safe_route)
                            except Exception:
                                traci.vehicle.setRoute(vehicle_id, safe_route)
                            self.logger.log_info(f"APPLY_PLAN: {vehicle_id} assigned route to "
                                               f"{regional_plan['boundary_edge']} "
                                               f"(travel_time: {regional_plan.get('travel_time', 'unknown')}s)")
                            return True
                        else:
                            self.logger.log_warning(f"APPLY_PLAN: Cannot create safe route for {vehicle_id}")
                            return False
                except Exception as route_error:
                    self.logger.log_error(f"APPLY_PLAN: Failed to set route for {vehicle_id}: {route_error}")
                    return False
            else:
                self.logger.log_warning(f"APPLY_PLAN: Empty route for {vehicle_id}")
                return False
                
        except Exception as e:
            self.logger.log_error(f"APPLY_PLAN: Failed for {vehicle_id}: {e}")
            return False
    
    def _record_llm_success(self, region_id: int, current_time: float):
        """Record LLM success to help circuit breaker recovery."""
        if region_id in self.llm_failure_counts:
            failure_count, _ = self.llm_failure_counts[region_id]
            if failure_count > 0:
                # Reduce failure count on success
                new_count = max(0, failure_count - 1)
                self.llm_failure_counts[region_id] = (new_count, current_time)
                if new_count == 0:
                    self.logger.log_info(f"CIRCUIT_BREAKER: Recovered for region {region_id}")
    
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
    """Get the currently registered global LLM manager - 增强版本."""
    global _global_llm_manager_registry
    
    # 方法1: 从全局注册表获取
    manager = _global_llm_manager_registry.get("current")
    if manager:
        return manager
    
    # 方法2: 尝试从language_model模块的全局变量获取
    try:
        import utils.language_model as lm_module
        if hasattr(lm_module, '_local_llm_manager_instance'):
            fallback_manager = getattr(lm_module, '_local_llm_manager_instance')
            if fallback_manager:
                print(f"[DEBUG] 从language_model模块获取到LLM管理器")
                # 同时注册到全局表中
                _global_llm_manager_registry["current"] = fallback_manager
                return fallback_manager
    except ImportError:
        pass
    
    # 方法3: 检查sys.modules中是否有language_model
    try:
        import sys
        if 'utils.language_model' in sys.modules:
            lm_module = sys.modules['utils.language_model']
            if hasattr(lm_module, '_local_llm_manager_instance'):
                fallback_manager = getattr(lm_module, '_local_llm_manager_instance')
                if fallback_manager:
                    print(f"[DEBUG] 从sys.modules获取到LLM管理器")
                    _global_llm_manager_registry["current"] = fallback_manager
                    return fallback_manager
    except Exception as e:
        print(f"[DEBUG] sys.modules访问失败: {e}")
    
    print(f"[DEBUG] 未找到LLM管理器，当前注册表键: {list(_global_llm_manager_registry.keys())}")
    return None

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

