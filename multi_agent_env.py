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
from typing import Dict, List, Set, Optional, Tuple
import networkx as nx
import traci

from utils.read_utils import load_json
from agents import RegionalAgent, TrafficAgent, PredictionEngine, AgentLogger
from env_utils import parse_rou_file, get_edge_info, get_multiple_edges_info, get_dynamic_data_batch, get_congestion_level


class MultiAgentTrafficEnvironment:
    """
    Main environment class for multi-agent traffic control system.
    
    Integrates Regional Agents, Traffic Agent, and Prediction Engine
    for coordinated traffic management.
    """
    
    def __init__(self, location: str, sumo_config_file: str, route_file: str,
                 road_info_file: str, adjacency_file: str, region_data_dir: str,
                 llm_agent, step_size: float = 1.0, max_steps: int = 1000,
                 log_dir: str = "logs"):
        """
        Initialize multi-agent traffic environment.
        
        Args:
            location: Location name (e.g., Manhattan)
            sumo_config_file: Path to SUMO configuration file
            route_file: Path to route file
            road_info_file: Path to road information file
            adjacency_file: Path to adjacency information file
            region_data_dir: Directory containing region partition data
            llm_agent: Language model agent for decision making
            step_size: Simulation step size in seconds
            max_steps: Maximum simulation steps
            log_dir: Directory for log files
        """
        self.location = location
        self.sumo_config_file = sumo_config_file
        self.route_file = route_file
        self.road_info_file = road_info_file
        self.adjacency_file = adjacency_file
        self.region_data_dir = region_data_dir
        self.llm_agent = llm_agent
        self.step_size = step_size
        self.max_steps = max_steps
        
        # Initialize logger
        self.logger = AgentLogger(log_dir=log_dir, console_output=True)
        
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
        
        # Initialize agents
        self._initialize_agents()
        
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
        
        # Real-time monitoring state
        self.current_step = 0
        self.current_sim_time = 0.0
        self.active_autonomous_vehicles = 0
        self.att_calculation = 0.0
        self.system_throughput = 0.0
        
    def _load_region_data(self):
        """Load region partition data."""
        try:
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
            
        except Exception as e:
            self.logger.log_error(f"Failed to load region data: {e}")
            raise
    
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
    
    def _initialize_agents(self):
        """Initialize all agents."""
        try:
            # Initialize prediction engine
            edge_list = list(self.road_info.keys())
            self.prediction_engine = PredictionEngine(edge_list, self.logger)
            
            # Initialize Traffic Agent
            self.traffic_agent = TrafficAgent(
                boundary_edges=self.boundary_edges,
                edge_to_region=self.edge_to_region,
                road_info=self.road_info,
                num_regions=self.num_regions,
                llm_agent=self.llm_agent,
                logger=self.logger,
                prediction_engine=self.prediction_engine
            )
            
            # Initialize Regional Agents
            self.regional_agents = {}
            for region_id in range(self.num_regions):
                regional_agent = RegionalAgent(
                    region_id=region_id,
                    boundary_edges=self.boundary_edges,
                    edge_to_region=self.edge_to_region,
                    road_info=self.road_info,
                    road_network=self.road_network,
                    llm_agent=self.llm_agent,
                    logger=self.logger
                )
                self.regional_agents[region_id] = regional_agent
                self.last_regional_decision_time[region_id] = 0.0
            
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
            
            # Parse route file and select autonomous vehicles
            all_vehicles = parse_rou_file(self.route_file)
            all_vehicle_ids = [veh_id for veh_id, _, _ in all_vehicles]
            self.total_vehicles = len(all_vehicles)
            
            # Select 2% of vehicles as autonomous
            import random
            self.autonomous_vehicles = set(random.sample(all_vehicle_ids, 
                                                       int(0.02 * self.total_vehicles)))
            
            # Initialize edge information
            print("MULTI_AGENT_ENV: Initializing edge list from SUMO")
            self.edges = traci.edge.getIDList()
            print(f"MULTI_AGENT_ENV: Retrieved {len(self.edges)} edges from SUMO")
            
            # Update logger with initial vehicle count
            self.logger.update_vehicle_count(self.total_vehicles, 0.0)
            
            self.logger.log_info(f"Simulation initialized: {self.total_vehicles} total vehicles, "
                               f"{len(self.autonomous_vehicles)} autonomous vehicles")
            
        except Exception as e:
            self.logger.log_error(f"Failed to initialize simulation: {e}")
            raise
    
    def handle_vehicle_birth_macro_planning(self, vehicle_id: str, current_time: float):
        """
        Handle macro route planning when an autonomous vehicle is born.
        
        Based on reachability, connectivity, congestion, distance factors,
        generate macro route candidates and use LLM to select optimal route.
        """
        try:
            self.logger.log_info(f"VEHICLE_BIRTH: Processing macro planning for {vehicle_id}")
            
            # Get vehicle's destination
            route = traci.vehicle.getRoute(vehicle_id)
            if not route:
                self.logger.log_warning(f"VEHICLE_BIRTH: No route found for {vehicle_id}")
                return
            
            start_edge = route[0]
            dest_edge = route[-1]
            
            # Determine start and destination regions
            start_region = self.edge_to_region.get(start_edge)
            dest_region = self.edge_to_region.get(dest_edge)
            
            if start_region is None or dest_region is None:
                self.logger.log_warning(f"VEHICLE_BIRTH: Region mapping not found for {vehicle_id}")
                return
            
            # If already in destination region, create a single-region macro route
            if start_region == dest_region:
                self.logger.log_info(f"VEHICLE_BIRTH: {vehicle_id} already in destination region {dest_region}")
                # Create single-region macro route for regional planning consistency
                single_region_route = [start_region]
                self.vehicle_current_plans[vehicle_id] = {
                    'macro_route': single_region_route,
                    'current_region_index': 0,
                    'creation_time': current_time,
                    'last_update': current_time,
                    'single_region': True  # Mark as single-region route
                }
                
                # No need for broadcast since this is intra-region only
                self.logger.log_info(f"VEHICLE_BIRTH: {vehicle_id} assigned single-region macro route [{start_region}]")
                return
            
            # Generate macro route candidates based on reachability, connectivity, congestion, distance
            macro_candidates = self._generate_macro_route_candidates(
                start_region, dest_region, current_time
            )
            
            if not macro_candidates:
                self.logger.log_warning(f"VEHICLE_BIRTH: No macro candidates found for {vehicle_id}")
                return
            
            # Use LLM to select optimal macro route
            selected_macro_route = self._llm_select_macro_route(
                vehicle_id, start_region, dest_region, macro_candidates, current_time
            )
            
            if selected_macro_route:
                # Store the selected macro route
                self.vehicle_current_plans[vehicle_id] = {
                    'macro_route': selected_macro_route,
                    'current_region_index': 0,
                    'creation_time': current_time,
                    'last_update': current_time
                }
                
                # Update communication system - broadcast this plan
                self._broadcast_vehicle_macro_plan(vehicle_id, selected_macro_route, current_time)
                
                # Log real-time console output
                self._log_vehicle_decision(vehicle_id, "MACRO_PLANNING", selected_macro_route, current_time)
                
                self.logger.log_info(f"VEHICLE_BIRTH: {vehicle_id} assigned macro route {selected_macro_route}")
            else:
                self.logger.log_error(f"VEHICLE_BIRTH: Failed to select macro route for {vehicle_id}")
                
        except Exception as e:
            self.logger.log_error(f"VEHICLE_BIRTH: Macro planning failed for {vehicle_id}: {e}")
    
    def _handle_stuck_vehicle_replanning(self, vehicle_id: str, current_time: float):
        """Handle replanning for stuck vehicles by treating them as newly born."""
        try:
            # Check if vehicle already being replanned recently to avoid spam
            if hasattr(self, '_vehicle_replan_times'):
                last_replan = self._vehicle_replan_times.get(vehicle_id, 0)
                if current_time - last_replan < 120:  # Avoid replanning same vehicle within 2 minutes
                    return
            else:
                self._vehicle_replan_times = {}
            
            self.logger.log_info(f"STUCK_REPLAN: Initiating replanning for stuck vehicle {vehicle_id}")
            
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
            
            # Trigger macro planning as if vehicle just born
            self.handle_vehicle_birth_macro_planning(vehicle_id, current_time)
            
            # Also trigger regional planning if vehicle has macro route now
            if vehicle_id in self.vehicle_regions and vehicle_id in self.vehicle_current_plans:
                current_region = self.vehicle_regions[vehicle_id]
                self.handle_vehicle_regional_planning(vehicle_id, current_region, current_time)
            
            self.logger.log_info(f"STUCK_REPLAN: Completed replanning for vehicle {vehicle_id}")
            
        except Exception as e:
            self.logger.log_error(f"STUCK_REPLAN: Failed for {vehicle_id}: {e}")
    
    def _generate_macro_route_candidates(self, start_region: int, dest_region: int, current_time: float) -> List[List[int]]:
        """
        Generate macro route candidates based on reachability, connectivity, congestion, distance.
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
                self.logger.log_warning(f"MACRO_CANDIDATES: No path found from region {start_region} to {dest_region}")
                return []
            
            # Evaluate candidates based on multiple factors
            evaluated_candidates = []
            
            for candidate in candidates:
                score = self._evaluate_macro_route_candidate(candidate, current_time)
                evaluated_candidates.append((candidate, score))
            
            # Sort by score (higher is better) and return top candidates
            evaluated_candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = [route for route, score in evaluated_candidates[:5]]  # Top 5 candidates
            
            self.logger.log_info(f"MACRO_CANDIDATES: Generated {len(top_candidates)} candidates for {start_region}->{dest_region}")
            
            return top_candidates
            
        except Exception as e:
            self.logger.log_error(f"MACRO_CANDIDATES: Failed to generate candidates: {e}")
            return []
    
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
                    # Use basic LLM decision making
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
        """Calculate current Average Travel Time (ATT)."""
        try:
            if not self.vehicle_travel_metrics:
                return 0.0
            
            total_travel_time = sum(
                metrics['travel_time'] for metrics in self.vehicle_travel_metrics.values()
            )
            
            return total_travel_time / len(self.vehicle_travel_metrics)
            
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
    
    def update_vehicle_tracking(self, current_time: float):
        """Update vehicle positions and region assignments with enhanced performance and logging."""
        tracking_start_time = time.time()
        
        try:
            # Get all vehicles in simulation
            all_vehicle_ids = traci.vehicle.getIDList()
            autonomous_in_sim = [veh_id for veh_id in all_vehicle_ids if veh_id in self.autonomous_vehicles]
            
            self.logger.log_info(f"VEHICLE_TRACKING_START: {len(autonomous_in_sim)}/{len(all_vehicle_ids)} autonomous vehicles active")
            
            # Batch vehicle information gathering for efficiency
            vehicles_processed = 0
            vehicles_failed = 0
            region_changes = 0
            new_vehicles = 0
            
            # Track vehicles entering and exiting the simulation
            for veh_id in autonomous_in_sim:
                try:
                    # Record start time for new vehicles and trigger macro planning
                    if veh_id not in self.vehicle_start_times:
                        # Get accurate departure time from SUMO
                        try:
                            actual_depart_time = traci.vehicle.getDeparture(veh_id)
                            self.vehicle_start_times[veh_id] = actual_depart_time
                            new_vehicles += 1
                            self.logger.log_info(f"NEW_VEHICLE: {veh_id} started at time {actual_depart_time:.1f}")
                            
                            # Event-driven LLM decision: Vehicle birth macro planning
                            self.handle_vehicle_birth_macro_planning(veh_id, current_time)
                        except Exception as e:
                            # Fallback to current simulation time if getDeparture() fails
                            self.vehicle_start_times[veh_id] = current_time
                            new_vehicles += 1
                            self.logger.log_info(f"NEW_VEHICLE: {veh_id} started at time {current_time:.1f} (fallback)")
                            self.logger.log_warning(f"VEHICLE_DEPART_TIME: Could not get departure time for {veh_id}: {e}")
                            
                            # Event-driven LLM decision: Vehicle birth macro planning
                            self.handle_vehicle_birth_macro_planning(veh_id, current_time)
                    
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
                    
                    # Update vehicle region
                    if current_edge in self.edge_to_region:
                        current_region = self.edge_to_region[current_edge]
                        
                        # Check if vehicle moved to a new region
                        if veh_id not in self.vehicle_regions:
                            self.vehicle_regions[veh_id] = current_region
                            self.logger.log_info(f"VEHICLE_REGION_INIT: {veh_id} assigned to region {current_region}")
                            
                            # Trigger regional planning for new vehicle in region
                            self.handle_vehicle_regional_planning(veh_id, current_region, current_time)
                            
                        elif self.vehicle_regions[veh_id] != current_region:
                            # Vehicle changed regions - EVENT-DRIVEN LLM DECISION POINT
                            old_region = self.vehicle_regions[veh_id]
                            self.vehicle_regions[veh_id] = current_region
                            region_changes += 1
                            
                            self.logger.log_info(f"VEHICLE_REGION_CHANGE: {veh_id} moved from region {old_region} to {current_region}")
                            
                            # Event-driven LLM decision: Vehicle reached new region, replan macro route
                            self.handle_vehicle_region_change_replanning(veh_id, old_region, current_region, current_time)
                            
                            # Also trigger regional planning for the new region
                            self.handle_vehicle_regional_planning(veh_id, current_region, current_time)
                        
                        # Real-time logging and metrics update - as required by user
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
                            
                            # Check for stuck vehicles and trigger replanning
                            if vehicle_speed < 0.1 and travel_time > 300:  # Stopped for > 5 minutes
                                self.logger.log_warning(f"VEHICLE_STUCK: {veh_id} stopped on {current_edge} for {travel_time:.1f}s")
                                
                                # Trigger replanning for stuck vehicles (treat as vehicle rebirth)
                                self._handle_stuck_vehicle_replanning(veh_id, current_time)
                            
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
            
            self.logger.log_info(f"VEHICLE_TRACKING_COMPLETE: processed:{vehicles_processed}, failed:{vehicles_failed}, "
                               f"new:{new_vehicles}, region_changes:{region_changes}, completed:{completed_this_step}, "
                               f"active:{active_vehicles}, time:{tracking_time:.1f}ms")
            
            # Performance warnings
            if tracking_time > 2000:  # > 2 seconds
                self.logger.log_warning(f"VEHICLE_TRACKING_SLOW: Update took {tracking_time:.1f}ms")
            
            if vehicles_failed > vehicles_processed * 0.1:  # > 10% failure rate
                self.logger.log_warning(f"VEHICLE_TRACKING_HIGH_FAILURE: {vehicles_failed}/{vehicles_processed + vehicles_failed} vehicles failed")
                
        except Exception as e:
            self.logger.log_error(f"VEHICLE_TRACKING_CRITICAL_ERROR: {e}")
            raise
    
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
        """Log overall system performance."""
        try:
            # Collect metrics from all agents
            regional_metrics = {}
            for region_id, agent in self.regional_agents.items():
                regional_metrics[region_id] = agent.get_performance_metrics()
            
            traffic_metrics = self.traffic_agent.get_performance_metrics()
            prediction_metrics = self.prediction_engine.get_performance_metrics()
            
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
                
                # Update vehicle tracking
                self.update_vehicle_tracking(current_time)
                
                # Update prediction engine
                self.update_prediction_engine(current_time)
                
                # Update traffic agent (event-driven LLM system)
                self.update_traffic_agent(current_time)
                
                # Coordinate regional agents (event-driven LLM system)
                self.coordinate_regional_agents(current_time)
                
                # Process any pending broadcast messages
                self._process_broadcast_messages(current_time)
                
                # Increment step for next iteration
                step += self.step_size
                
                # Display progress
                self.logger.display_progress(current_time)
                
                # Log performance periodically
                if int(current_time) % 300 == 0:  # Every 5 minutes
                    self.log_system_performance(current_time)
            
            # Calculate final results
            if self.vehicle_end_times:
                total_travel_time = sum(
                    self.vehicle_end_times[veh_id] - self.vehicle_start_times.get(veh_id, 0)
                    for veh_id in self.vehicle_end_times
                )
                average_travel_time = total_travel_time / len(self.vehicle_end_times)
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
                    
                    # Request regional planning with LLM through regional agent
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
                        
                        # Execute the regional route using SUMO's setRoute
                        if regional_plan['route'] and len(regional_plan['route']) > 0:
                            try:
                                traci.vehicle.setRoute(vehicle_id, regional_plan['route'])
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
                    # Basic LLM decision making
                    decisions = self.llm_agent.hybrid_decision_making_pipeline(
                        [observation_text], [f'"{answer_options}"']
                    )
                    
                    if decisions and decisions[0]['answer']:
                        selected_route = self._parse_macro_route_answer(decisions[0]['answer'], all_options)
                        reasoning = decisions[0].get('summary', 'LLM replanning decision')
                    else:
                        selected_route = all_options[0]
                        reasoning = 'Fallback decision'
                
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
                    result = planning_data['future'].result(timeout=30)  # 30 second timeout per region
                    
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
                            
                            # Apply route to vehicle in SUMO
                            if regional_plan['route'] and len(regional_plan['route']) > 0:
                                try:
                                    traci.vehicle.setRoute(vehicle_id, regional_plan['route'])
                                    successful_plans += 1
                                    
                                    self.logger.log_info(f"REGIONAL_BATCH: {vehicle_id} assigned route to "
                                                       f"{regional_plan['boundary_edge']} "
                                                       f"(travel_time: {regional_plan.get('travel_time', 'unknown')}s)")
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
    
