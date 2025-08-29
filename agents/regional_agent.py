"""
Regional Agent for Multi-Agent Traffic Control System

Handles intra-regional route planning, lane assignment, and vehicle coordination
within a specific region to optimize traffic flow to boundary edges.
"""

import traci
import networkx as nx
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict, deque
from dataclasses import dataclass
import time

@dataclass
class VehicleDecision:
    """Decision made for a vehicle."""
    vehicle_id: str
    current_edge: str
    target_edge: str
    route: List[str]
    lane_assignment: Optional[int]
    priority: int
    decision_time: float
    reasoning: str


@dataclass
class RegionalRecommendation:
    """Recommendation from Traffic Agent to Regional Agent."""
    target_boundary_edges: List[str]
    congestion_weights: Dict[str, float]
    priority_vehicles: List[str]
    avoid_edges: List[str]


class RegionalAgent:
    """
    Regional Agent for intra-regional traffic management.
    
    Responsible for:
    - Planning fastest routes to boundary edges within the region
    - Assigning lanes to avoid congestion
    - Coordinating vehicles for green wave traffic flow
    - Using LLM for decision making with traffic context
    - Tracking planned routes to prevent overcrowding
    """
    
    def __init__(self, region_id: int, boundary_edges: Dict, edge_to_region: Dict,
                 road_info: Dict, road_network: nx.DiGraph, llm_agent, logger, prediction_engine=None):
        """
        Initialize Regional Agent.
        
        Args:
            region_id: ID of the region this agent manages
            boundary_edges: Dictionary of boundary edge information
            edge_to_region: Mapping of edges to region IDs
            road_info: Road information dictionary
            road_network: NetworkX graph of road network
            llm_agent: Language model agent for decision making
            logger: Agent logger instance
            prediction_engine: Prediction engine for traffic forecasting (optional)
        """
        self.region_id = region_id
        self.boundary_edges = boundary_edges
        self.edge_to_region = edge_to_region
        self.road_info = road_info
        self.road_network = road_network
        self.llm_agent = llm_agent
        self.logger = logger
        self.prediction_engine = prediction_engine  # Store prediction engine for advanced lane optimization
        
        # Identify edges and boundary edges for this region
        self._initialize_region_topology()
        
        # Vehicle tracking
        self.region_vehicles: Set[str] = set()
        self.vehicle_routes: Dict[str, List[str]] = {}
        self.vehicle_targets: Dict[str, str] = {}
        self.vehicle_last_update: Dict[str, float] = {}
        self.planned_routes: Dict[str, int] = defaultdict(int)  # Edge usage count
        
        # Decision tracking
        self.recent_decisions: deque = deque(maxlen=100)
        self.route_effectiveness: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        
        # Performance metrics
        self.total_decisions = 0
        self.successful_decisions = 0
        self.avg_travel_time = 0.0
        self.congestion_reduction = 0.0
        
        # Green wave coordination
        self.traffic_light_phases: Dict[str, Dict] = {}
        self.green_wave_routes: Dict[str, List[str]] = {}
        
        # Initialize advanced lane optimization system
        self._initialize_lane_optimization_system()
        
        self.logger.log_info(f"Regional Agent {region_id} initialized: "
                           f"{len(self.region_edges)} edges, "
                           f"{len(self.boundary_connections)} boundary connections, "
                           f"Advanced Lane Optimization: {'✓' if self.prediction_engine else '✓ (basic)'}")
    
    def _initialize_region_topology(self):
        """Initialize region topology information."""
        # Find all edges in this region
        self.region_edges = [
            edge_id for edge_id, region in self.edge_to_region.items()
            if region == self.region_id
        ]
        
        # Find boundary connections for this region
        self.boundary_connections = {}
        self.outgoing_boundaries = []  # Boundary edges leaving this region
        self.incoming_boundaries = []  # Boundary edges entering this region
        
        for boundary_info in self.boundary_edges:
            edge_id = boundary_info['edge_id']
            from_region = boundary_info['from_region']
            to_region = boundary_info['to_region']
            
            if from_region == self.region_id:
                self.outgoing_boundaries.append(edge_id)
                self.boundary_connections[edge_id] = to_region
            elif to_region == self.region_id:
                self.incoming_boundaries.append(edge_id)
                self.boundary_connections[edge_id] = from_region
        
        # Create regional subgraph
        self.regional_network = self.road_network.subgraph(self.region_edges)
        
        self.logger.log_info(f"Region {self.region_id} topology: "
                           f"{len(self.region_edges)} edges, "
                           f"{len(self.outgoing_boundaries)} outgoing boundaries, "
                           f"{len(self.incoming_boundaries)} incoming boundaries")
    
    def update_vehicle_status(self, current_time: float):
        """Update status of vehicles in this region."""
        try:
            # Get current vehicles in simulation
            current_vehicle_ids = set(traci.vehicle.getIDList())
            
            # Update vehicles in this region
            updated_vehicles = set()
            
            for vehicle_id in current_vehicle_ids:
                try:
                    current_edge = traci.vehicle.getRoadID(vehicle_id)
                    
                    # Check if vehicle is in this region
                    if current_edge in self.edge_to_region:
                        vehicle_region = self.edge_to_region[current_edge]
                        
                        if vehicle_region == self.region_id:
                            updated_vehicles.add(vehicle_id)
                            
                            # Update vehicle tracking
                            if vehicle_id not in self.region_vehicles:
                                # New vehicle entered region
                                self.region_vehicles.add(vehicle_id)
                                self.logger.log_info(f"Vehicle {vehicle_id} entered region {self.region_id}")
                            
                            self.vehicle_last_update[vehicle_id] = current_time
                            
                            # Update route tracking
                            if vehicle_id in self.vehicle_routes:
                                route = traci.vehicle.getRoute(vehicle_id)
                                if route != self.vehicle_routes[vehicle_id]:
                                    # Route changed, update tracking
                                    old_route = self.vehicle_routes[vehicle_id]
                                    for edge in old_route:
                                        if edge in self.planned_routes:
                                            self.planned_routes[edge] = max(0, self.planned_routes[edge] - 1)
                                    
                                    self.vehicle_routes[vehicle_id] = list(route)
                                    for edge in route:
                                        if edge in self.region_edges:
                                            self.planned_routes[edge] += 1
                
                except Exception:
                    continue
            
            # Remove vehicles that left the region
            vehicles_left = self.region_vehicles - updated_vehicles
            for vehicle_id in vehicles_left:
                self._remove_vehicle_tracking(vehicle_id)
            
            self.region_vehicles = updated_vehicles
            
        except Exception as e:
            self.logger.log_error(f"Regional Agent {self.region_id} vehicle status update failed: {e}")
    
    def _remove_vehicle_tracking(self, vehicle_id: str):
        """Remove tracking for a vehicle that left the region."""
        if vehicle_id in self.vehicle_routes:
            # Decrease route usage counts
            for edge in self.vehicle_routes[vehicle_id]:
                if edge in self.planned_routes:
                    self.planned_routes[edge] = max(0, self.planned_routes[edge] - 1)
            
            del self.vehicle_routes[vehicle_id]
        
        if vehicle_id in self.vehicle_targets:
            del self.vehicle_targets[vehicle_id]
        
        if vehicle_id in self.vehicle_last_update:
            del self.vehicle_last_update[vehicle_id]
        
        self.logger.log_info(f"Vehicle {vehicle_id} left region {self.region_id}")
    
    def make_regional_route_planning(self, vehicle_id: str, target_region: int, current_time: float) -> Optional[Dict]:
        """
        LLM-driven regional route planning optimized for minimum travel time.
        
        Uses SUMO's built-in route finding with LLM decision making to select optimal
        boundary edge based on current traffic conditions and travel time minimization.
        
        Args:
            vehicle_id: ID of vehicle needing planning
            target_region: ID of target region to reach
            current_time: Current simulation time
            
        Returns:
            Dictionary with selected boundary edge and route, or None if failed
        """
        try:
            # Input validation
            if not isinstance(vehicle_id, str) or not vehicle_id.strip():
                self.logger.log_error(f"REGIONAL_PLANNING: Invalid vehicle_id: {vehicle_id}")
                return None
                
            if not isinstance(target_region, int) or target_region < 0:
                self.logger.log_error(f"REGIONAL_PLANNING: Invalid target_region: {target_region}")
                return None
            
            self.logger.log_info(f"REGIONAL_PLANNING: Vehicle {vehicle_id} in region {self.region_id} -> target region {target_region}")
            
            # Get current vehicle position using SUMO API
            try:
                current_edge = traci.vehicle.getRoadID(vehicle_id)
                if not current_edge or current_edge.startswith(':'):
                    self.logger.log_warning(f"REGIONAL_PLANNING: Vehicle {vehicle_id} on invalid edge: {current_edge}")
                    return None
                    
                # Verify vehicle is actually in our region
                if current_edge not in self.edge_to_region or self.edge_to_region[current_edge] != self.region_id:
                    self.logger.log_warning(f"REGIONAL_PLANNING: Vehicle {vehicle_id} edge {current_edge} not in region {self.region_id}")
                    return None
                    
            except Exception as traci_error:
                self.logger.log_error(f"REGIONAL_PLANNING: TraCI error for vehicle {vehicle_id}: {traci_error}")
                return None
            
            # Find boundary edges leading to target region using existing topology
            boundary_candidates = self._get_boundary_candidates_to_region(target_region)
            
            if not boundary_candidates:
                self.logger.log_warning(f"REGIONAL_PLANNING: No direct boundary from region {self.region_id} to {target_region}")
                # Fallback: use any outgoing boundary edge (indirect routing)
                boundary_candidates = self.outgoing_boundaries[:3] if self.outgoing_boundaries else []
                if not boundary_candidates:
                    self.logger.log_error(f"REGIONAL_PLANNING: No outgoing boundaries from region {self.region_id}")
                    return None
            
            # Generate route candidates using SUMO's route finding
            route_candidates = self._generate_regional_route_candidates(
                current_edge, boundary_candidates, current_time
            )
            
            if not route_candidates:
                self.logger.log_warning(f"REGIONAL_PLANNING: No valid routes found for {vehicle_id}")
                # Fallback: try direct route to first boundary
                if boundary_candidates:
                    fallback_route = self._create_fallback_route(current_edge, boundary_candidates[0])
                    if fallback_route:
                        return fallback_route
                return None
            
            # Use LLM to select optimal route based on travel time minimization
            selected_plan = self._llm_select_regional_route(
                vehicle_id, current_edge, route_candidates, target_region, current_time
            )
            
            if selected_plan:
                # Validate selected plan structure
                required_keys = ['boundary_edge', 'route']
                for key in required_keys:
                    if key not in selected_plan:
                        self.logger.log_error(f"REGIONAL_PLANNING: Selected plan missing key {key}")
                        return None
                
                # Update tracking for coordination
                self.vehicle_routes[vehicle_id] = selected_plan['route']
                self.vehicle_targets[vehicle_id] = selected_plan['boundary_edge']
                self.vehicle_last_update[vehicle_id] = current_time
                
                # Update planned route counts for load balancing
                for edge in selected_plan['route']:
                    if edge in self.region_edges:
                        self.planned_routes[edge] += 1
                
                self.logger.log_info(f"REGIONAL_PLANNING: {vehicle_id} assigned optimal route to {selected_plan['boundary_edge']} "
                                   f"(travel_time: {selected_plan.get('travel_time', 'unknown')}s)")
                
                return selected_plan
            else:
                self.logger.log_error(f"REGIONAL_PLANNING: LLM failed to select route for {vehicle_id}")
                # Return best heuristic candidate as fallback
                if route_candidates:
                    return route_candidates[0]
                else:
                    return None
                
        except Exception as e:
            self.logger.log_error(f"REGIONAL_PLANNING: Critical error for {vehicle_id}: {e}")
            return None
    
    def _get_boundary_candidates_to_region(self, target_region: int) -> List[str]:
        """Get boundary edges that connect current region to target region."""
        candidates = []
        
        # Find outgoing boundary edges to target region
        for boundary_info in self.boundary_edges:
            if (boundary_info['from_region'] == self.region_id and 
                boundary_info['to_region'] == target_region):
                candidates.append(boundary_info['edge_id'])
        
        return candidates
    
    def _generate_regional_route_candidates(self, current_edge: str, boundary_candidates: List[str], current_time: float) -> List[Dict]:
        """Generate route candidates to boundary edges optimized for minimum travel time."""
        candidates = []
        
        for boundary_edge in boundary_candidates:
            try:
                # Use SUMO's findRoute for accurate travel time calculation
                route_result = traci.simulation.findRoute(current_edge, boundary_edge)
                
                if route_result and route_result.edges and route_result.travelTime > 0:
                    route_edges = list(route_result.edges)
                    
                    # Allow routes with intermediate edges outside region (SUMO optimization)
                    # Only validate that start edge is in region and boundary edge is accessible
                    valid_route = True
                    
                    # Check start edge is in our region
                    if route_edges and route_edges[0] in self.edge_to_region:
                        start_edge_region = self.edge_to_region[route_edges[0]]
                        if start_edge_region != self.region_id:
                            valid_route = False
                            self.logger.log_warning(f"ROUTE_GENERATION: Route starts outside region {self.region_id}")
                    
                    # Allow intermediate edges to be outside region for SUMO path optimization
                    # This is normal behavior when SUMO finds optimal paths across region boundaries
                    if not valid_route:
                        continue
                    
                    # Evaluate route for travel time optimization
                    evaluation = self._evaluate_regional_route_candidate(
                        route_edges, boundary_edge, current_time
                    )
                    
                    candidate = {
                        'boundary_edge': str(boundary_edge),  # Ensure string type
                        'route': list(route_edges),  # Ensure list type
                        'travel_time': float(route_result.travelTime),  # Ensure numeric type
                        'distance': float(route_result.length),  # Ensure numeric type
                        'evaluation': evaluation,
                        'description': self._create_route_description(route_edges, boundary_edge, evaluation)
                    }
                    
                    candidates.append(candidate)
                    
                else:
                    self.logger.log_warning(f"ROUTE_GENERATION: No valid route from {current_edge} to {boundary_edge}")
                    
            except Exception as e:
                self.logger.log_warning(f"ROUTE_GENERATION: Failed for {boundary_edge}: {e}")
        
        if not candidates:
            self.logger.log_error(f"ROUTE_GENERATION: No valid candidates generated from {current_edge}")
            return []
        
        # Sort by travel time first (primary objective), then by evaluation score
        candidates.sort(key=lambda x: (x['travel_time'], -x['evaluation']['total_score']))
        
        # Return top candidates optimized for travel time
        return candidates[:3]  # Reduced to 3 for faster LLM decision
    
    def _create_fallback_route(self, current_edge: str, boundary_edge: str) -> Optional[Dict]:
        """Create fallback route when normal route generation fails."""
        try:
            # Try simple SUMO route finding
            route_result = traci.simulation.findRoute(current_edge, boundary_edge)
            
            if route_result and route_result.edges:
                return {
                    'boundary_edge': str(boundary_edge),  # Ensure string type
                    'route': list(route_result.edges),
                    'travel_time': float(route_result.travelTime),  # Ensure numeric type
                    'distance': float(route_result.length),  # Ensure numeric type
                    'evaluation': {'total_score': 0.0},
                    'description': f"Fallback route to {boundary_edge}",
                    'reasoning': 'Emergency fallback route'
                }
            
            # Last resort: direct edge if reachable
            if boundary_edge in self.regional_network.neighbors(current_edge):
                return {
                    'boundary_edge': str(boundary_edge),  # Ensure string type
                    'route': [str(current_edge), str(boundary_edge)],  # Ensure string types
                    'travel_time': 60.0,  # Estimated 1 minute
                    'distance': 100.0,    # Estimated 100m
                    'evaluation': {'total_score': 0.0},
                    'description': f"Direct fallback to {boundary_edge}",
                    'reasoning': 'Direct connection fallback'
                }
                
            return None
            
        except Exception as e:
            self.logger.log_error(f"FALLBACK_ROUTE: Failed for {current_edge} -> {boundary_edge}: {e}")
            return None
    
    def _evaluate_regional_route_candidate(self, route: List[str], boundary_edge: str, current_time: float) -> Dict:
        """Evaluate regional route candidate based on multiple factors."""
        try:
            evaluation = {
                'congestion_score': 0.0,
                'distance_score': 0.0,
                'utilization_score': 0.0,
                'boundary_score': 0.0,
                'total_score': 0.0
            }
            
            # Factor 1: Route congestion (lower is better)
            total_congestion = 0.0
            valid_edges = 0
            
            for edge in route:
                if edge in self.road_info:
                    congestion = self.road_info[edge].get('congestion_level', 0)
                    total_congestion += congestion
                    valid_edges += 1
            
            if valid_edges > 0:
                avg_congestion = total_congestion / valid_edges
                evaluation['congestion_score'] = max(0, 5 - avg_congestion)  # 5 is max score when no congestion
            
            # Factor 2: Route length (shorter is better)
            route_length = len(route)
            evaluation['distance_score'] = max(0, 10 - route_length * 0.5)  # Penalty for longer routes
            
            # Factor 3: Current route utilization (avoid overused routes)
            utilization_penalty = 0
            for edge in route:
                planned_usage = self.planned_routes.get(edge, 0)
                utilization_penalty += planned_usage * 0.5
            
            evaluation['utilization_score'] = max(0, 5 - utilization_penalty)
            
            # Factor 4: Boundary edge quality
            if boundary_edge in self.road_info:
                boundary_congestion = self.road_info[boundary_edge].get('congestion_level', 0)
                evaluation['boundary_score'] = max(0, 3 - boundary_congestion)
            else:
                evaluation['boundary_score'] = 1.5  # Neutral score
            
            # Calculate total score
            evaluation['total_score'] = (
                evaluation['congestion_score'] * 0.4 +
                evaluation['distance_score'] * 0.2 +
                evaluation['utilization_score'] * 0.3 +
                evaluation['boundary_score'] * 0.1
            )
            
            return evaluation
            
        except Exception as e:
            self.logger.log_error(f"ROUTE_EVALUATION: Failed to evaluate route: {e}")
            return {
                'congestion_score': 0.0, 'distance_score': 0.0,
                'utilization_score': 0.0, 'boundary_score': 0.0, 'total_score': 0.0
            }
    
    def _create_route_description(self, route: List[str], boundary_edge: str, evaluation: Dict) -> str:
        """Create human-readable route description for LLM."""
        try:
            parts = []
            parts.append(f"Route to {boundary_edge}")
            parts.append(f"Length: {len(route)} edges")
            parts.append(f"Score: {evaluation['total_score']:.1f}")
            parts.append(f"Congestion: {evaluation['congestion_score']:.1f}/5")
            parts.append(f"Utilization: {evaluation['utilization_score']:.1f}/5")
            
            return " | ".join(parts)
            
        except Exception as e:
            return f"Route to {boundary_edge} | Error: {e}"
    
    def _llm_select_regional_route(self, vehicle_id: str, current_edge: str, 
                                  route_candidates: List[Dict], target_region: int, current_time: float) -> Optional[Dict]:
        """Use LLM to select the best regional route from candidates."""
        try:
            # Create observation text for LLM
            observation_text = self._create_regional_planning_observation(
                vehicle_id, current_edge, route_candidates, target_region, current_time
            )
            
            # Create answer options as option numbers for LLM with validation
            valid_candidates = [candidate for candidate in route_candidates if candidate.get('boundary_edge') is not None]
            if not valid_candidates:
                self.logger.log_error(f"REGIONAL_ROUTING: No valid candidates for {vehicle_id}")
                # Check if route_candidates has any elements before accessing
                if route_candidates:
                    return route_candidates[0]
                else:
                    return None
            
            # Use option indices (1, 2, 3, etc.) instead of boundary edge IDs
            answer_options = "/".join([str(i+1) for i in range(len(valid_candidates))])
            
            if not answer_options:
                self.logger.log_error(f"REGIONAL_ROUTING: No valid answer options for {vehicle_id}")
                # Check if route_candidates has any elements before accessing
                if route_candidates:
                    return route_candidates[0]
                else:
                    return None
            
            # Use LLM for decision making
            call_id = self.logger.log_llm_call_start(
                "RegionalRouting", f"R{self.region_id}_{vehicle_id}", len(observation_text)
            )
            
            try:
                # Use the same LLM decision method as original single-vehicle system
                # as required by user: "区域内规划需要和原系统的LLM参与的规划方式一样"
                decisions = self.llm_agent.hybrid_decision_making_pipeline(
                    [observation_text], [answer_options]
                )
                
                # Simplified option-index based response processing
                if decisions and len(decisions) > 0 and isinstance(decisions[0], dict) and 'answer' in decisions[0]:
                    llm_answer = decisions[0]['answer']
                    
                    # Parse option index from LLM response
                    selected_boundary = None
                    reasoning = 'LLM regional route decision'
                    
                    if llm_answer is not None:
                        try:
                            # Convert answer to integer option index
                            if isinstance(llm_answer, (int, float)):
                                option_index = int(llm_answer)
                            elif isinstance(llm_answer, str) and llm_answer.strip().isdigit():
                                option_index = int(llm_answer.strip())
                            else:
                                raise ValueError(f"Invalid option format: {llm_answer}")
                            
                            # Validate option index range
                            if 1 <= option_index <= len(valid_candidates):
                                selected_boundary = valid_candidates[option_index - 1]['boundary_edge']
                                reasoning = decisions[0].get('summary', f'Selected option {option_index}')
                                if not isinstance(reasoning, str):
                                    reasoning = str(reasoning) if reasoning is not None else f'Selected option {option_index}'
                            else:
                                self.logger.log_warning(f"REGIONAL_ROUTING: Option index {option_index} out of range [1, {len(valid_candidates)}] for {vehicle_id}")
                                selected_boundary = valid_candidates[0]['boundary_edge']
                                reasoning = f'Option index out of range, using fallback (option 1)'
                                
                        except (ValueError, TypeError) as e:
                            self.logger.log_warning(f"REGIONAL_ROUTING: Invalid LLM answer '{llm_answer}' for {vehicle_id}: {e}")
                            selected_boundary = valid_candidates[0]['boundary_edge']
                            reasoning = f'Invalid LLM response format, using fallback (option 1)'
                    else:
                        self.logger.log_warning(f"REGIONAL_ROUTING: LLM returned None answer for {vehicle_id}")
                        selected_boundary = valid_candidates[0]['boundary_edge']
                        reasoning = 'LLM returned None answer, using fallback (option 1)'
                else:
                    # LLM response validation failed
                    if valid_candidates:
                        selected_boundary = valid_candidates[0]['boundary_edge']
                    elif route_candidates and route_candidates[0].get('boundary_edge'):
                        selected_boundary = route_candidates[0]['boundary_edge']
                    else:
                        selected_boundary = None
                    
                    reasoning = 'Invalid LLM response structure, using fallback (option 1)'
                    if decisions:
                        self.logger.log_warning(f"REGIONAL_ROUTING: Invalid LLM response: {type(decisions[0])}, len={len(decisions)}")
                    else:
                        self.logger.log_warning("REGIONAL_ROUTING: LLM returned empty response")
                
                # Find the selected candidate
                selected_candidate = None
                for candidate in route_candidates:
                    if candidate['boundary_edge'] == selected_boundary:
                        selected_candidate = candidate
                        break
                
                # If LLM selection invalid, use best scored candidate
                if not selected_candidate and route_candidates:
                    selected_candidate = route_candidates[0]
                    reasoning = f'Invalid LLM selection {selected_boundary}, using best candidate'
                elif not selected_candidate:
                    # No valid candidates at all, return None
                    self.logger.log_error(f"REGIONAL_ROUTING: No valid route candidates for {vehicle_id}")
                    return None
                
                # Add reasoning to result
                selected_candidate['reasoning'] = reasoning
                
                self.logger.log_llm_call_end(
                    call_id, True, f"Selected {selected_candidate['boundary_edge']}. Reasoning: {reasoning}",
                    len(observation_text)
                )
                
                return selected_candidate
                
            except Exception as llm_error:
                self.logger.log_llm_call_end(
                    call_id, False, "LLM regional route selection failed",
                    len(observation_text), str(llm_error)
                )
                
                # Fallback: return best scored candidate
                if route_candidates:
                    return route_candidates[0]
                else:
                    return None
                
        except Exception as e:
            self.logger.log_error(f"LLM_REGIONAL_SELECT: Failed for {vehicle_id}: {e}")
            if route_candidates:
                return route_candidates[0]
            else:
                return None
    
    def _create_regional_planning_observation(self, vehicle_id: str, current_edge: str,
                                            route_candidates: List[Dict], target_region: int, current_time: float) -> str:
        """Create observation text for regional planning LLM decision."""
        observation_parts = []
        
        observation_parts.append(f"REGIONAL ROUTE PLANNING FOR VEHICLE {vehicle_id}")
        observation_parts.append(f"Current region: {self.region_id}, Target region: {target_region}")
        observation_parts.append(f"Current edge: {current_edge}")
        observation_parts.append(f"Current time: {current_time:.1f}s")
        observation_parts.append("")
        
        # Show route candidates
        observation_parts.append("ROUTE CANDIDATES:")
        for i, candidate in enumerate(route_candidates):
            observation_parts.append(f"Option {i+1}: {candidate['description']}")
            observation_parts.append(f"  Travel time: {candidate['travel_time']:.1f}s")
            observation_parts.append(f"  Distance: {candidate['distance']:.1f}m")
        observation_parts.append("")
        
        # Regional context
        observation_parts.append(f"REGION {self.region_id} STATUS:")
        observation_parts.append(f"Active vehicles: {len(self.region_vehicles)}")
        observation_parts.append(f"Outgoing boundaries: {len(self.outgoing_boundaries)}")
        
        # Current utilization
        high_util_edges = []
        for edge, count in self.planned_routes.items():
            if count > 5:  # High utilization threshold
                high_util_edges.append(f"{edge}:{count}")
        
        if high_util_edges:
            observation_parts.append(f"High utilization edges: {', '.join(high_util_edges[:5])}")
        observation_parts.append("")
        
        observation_parts.append("OBJECTIVE: Select the best route to the target region boundary")
        observation_parts.append("while minimizing travel time, congestion, and balancing edge utilization.")
        observation_parts.append("")
        observation_parts.append("IMPORTANT: You must respond with ONLY the option number (1, 2, 3, etc.)")
        observation_parts.append("Do NOT return route descriptions or boundary edge IDs.")
        
        return "\n".join(observation_parts)
    
    def make_decisions(self, current_time: float, 
                      recommendations: Optional[RegionalRecommendation] = None) -> List[VehicleDecision]:
        """
        Make routing decisions for vehicles in the region.
        
        Args:
            current_time: Current simulation time
            recommendations: Recommendations from Traffic Agent
            
        Returns:
            List of vehicle decisions
        """
        try:
            # Get vehicles that need decisions
            vehicles_needing_decisions = self._get_vehicles_needing_decisions(current_time)
            
            if not vehicles_needing_decisions:
                return []
            
            # Prepare decision context
            decision_context = self._prepare_decision_context(
                vehicles_needing_decisions, recommendations, current_time
            )
            
            # Use LLM for batch decision making
            call_id = self.logger.log_llm_call_start(
                "RegionalAgent", str(self.region_id), 
                len(str(decision_context))
            )
            
            decisions = []
            try:
                llm_decisions = self._make_llm_decisions(decision_context, current_time)
                
                # Process LLM decisions into VehicleDecision objects
                decisions = self._process_llm_decisions(
                    llm_decisions, vehicles_needing_decisions, current_time
                )
                
                decision_summary = f"Made {len(decisions)} routing decisions"
                self.logger.log_llm_call_end(
                    call_id, True, decision_summary, len(str(decision_context))
                )
                
                self.successful_decisions += len(decisions)
                
            except Exception as e:
                self.logger.log_llm_call_end(
                    call_id, False, "Decision making failed", 
                    len(str(decision_context)), str(e)
                )
                
                # Fallback to heuristic decisions
                decisions = self._make_heuristic_decisions(
                    vehicles_needing_decisions, recommendations, current_time
                )
            
            self.total_decisions += len(vehicles_needing_decisions)
            
            # Track decisions
            for decision in decisions:
                self.recent_decisions.append(decision)
            
            return decisions
            
        except Exception as e:
            self.logger.log_error(f"Regional Agent {self.region_id} decision making failed: {e}")
            return []
    
    def _get_vehicles_needing_decisions(self, current_time: float) -> List[str]:
        """Get list of vehicles that need routing decisions."""
        vehicles_needing_decisions = []
        
        for vehicle_id in self.region_vehicles:
            try:
                # Check if vehicle needs a new decision
                needs_decision = False
                
                # Vehicle just entered region
                if vehicle_id not in self.vehicle_routes:
                    needs_decision = True
                
                # Vehicle hasn't been updated recently
                elif current_time - self.vehicle_last_update.get(vehicle_id, 0) > 300:  # 5 minutes
                    needs_decision = True
                
                # Vehicle doesn't have a clear path to boundary
                elif vehicle_id not in self.vehicle_targets:
                    needs_decision = True
                
                # Check if current route is still valid
                elif vehicle_id in self.vehicle_routes:
                    current_edge = traci.vehicle.getRoadID(vehicle_id)
                    route = self.vehicle_routes[vehicle_id]
                    
                    if current_edge not in route:
                        needs_decision = True
                
                if needs_decision:
                    vehicles_needing_decisions.append(vehicle_id)
            
            except Exception:
                continue
        
        return vehicles_needing_decisions
    
    def _prepare_decision_context(self, vehicle_ids: List[str], 
                                recommendations: Optional[RegionalRecommendation],
                                current_time: float) -> Dict[str, Any]:
        """Prepare context for LLM decision making."""
        context = {
            'region_id': self.region_id,
            'current_time': current_time,
            'vehicles': [],
            'boundary_edges': self.outgoing_boundaries,
            'regional_state': self._get_regional_state(),
            'recommendations': recommendations.__dict__ if recommendations else None
        }
        
        for vehicle_id in vehicle_ids:
            try:
                current_edge = traci.vehicle.getRoadID(vehicle_id)
                route = traci.vehicle.getRoute(vehicle_id)
                destination = route[-1] if route else None
                
                # Get candidate boundary edges
                candidate_boundaries = self._get_candidate_boundaries(current_edge, destination)
                
                # Get road observation data (similar to existing system)
                _, data_text, answer_options = self._get_vehicle_observation(
                    vehicle_id, current_edge, candidate_boundaries
                )
                
                vehicle_context = {
                    'vehicle_id': vehicle_id,
                    'current_edge': current_edge,
                    'destination': destination,
                    'candidate_boundaries': candidate_boundaries,
                    'observation_data': data_text,
                    'answer_options': answer_options,
                    'route_history': self.vehicle_routes.get(vehicle_id, [])
                }
                
                context['vehicles'].append(vehicle_context)
                
            except Exception:
                continue
        
        return context
    
    def _get_regional_state(self) -> Dict[str, Any]:
        """Get current state of the region."""
        # Calculate average congestion
        total_congestion = 0
        edge_count = 0
        
        for edge_id in self.region_edges:
            if edge_id in self.road_info:
                total_congestion += self.road_info[edge_id].get('congestion_level', 0)
                edge_count += 1
        
        avg_congestion = total_congestion / max(1, edge_count)
        
        # Get boundary edge states
        boundary_states = {}
        for boundary_edge in self.outgoing_boundaries:
            if boundary_edge in self.road_info:
                boundary_states[boundary_edge] = {
                    'congestion_level': self.road_info[boundary_edge].get('congestion_level', 0),
                    'vehicle_count': self.road_info[boundary_edge].get('vehicle_num', 0),
                    'planned_usage': self.planned_routes.get(boundary_edge, 0)
                }
        
        return {
            'avg_congestion': avg_congestion,
            'total_vehicles': len(self.region_vehicles),
            'boundary_states': boundary_states,
            'planned_route_usage': dict(self.planned_routes)
        }
    
    def _get_candidate_boundaries(self, current_edge: str, destination: str) -> List[str]:
        """Get candidate boundary edges for a vehicle."""
        candidate_boundaries = []
        
        # If destination is outside region, find appropriate boundary
        if destination and destination in self.edge_to_region:
            dest_region = self.edge_to_region[destination]
            
            if dest_region != self.region_id:
                # Find boundary edges that lead toward destination region
                for boundary_edge in self.outgoing_boundaries:
                    if boundary_edge in self.boundary_connections:
                        target_region = self.boundary_connections[boundary_edge]
                        
                        # Direct connection to destination region
                        if target_region == dest_region:
                            candidate_boundaries.append(boundary_edge)
                        # Add other boundaries as alternatives
                        elif len(candidate_boundaries) < 3:
                            candidate_boundaries.append(boundary_edge)
        
        # If no specific candidates, use all outgoing boundaries
        if not candidate_boundaries:
            candidate_boundaries = self.outgoing_boundaries[:3]  # Limit to top 3
        
        return candidate_boundaries
    
    def _get_vehicle_observation(self, vehicle_id: str, current_edge: str, 
                               candidate_boundaries: List[str]) -> Tuple[Dict, str, str]:
        """Get observation data for a vehicle similar to existing system."""
        try:
            # Build simplified observation for regional context
            road_candidates = self._get_regional_candidates(current_edge, candidate_boundaries)
            obs_text, answer_options = self._build_observation_text(
                vehicle_id, current_edge, candidate_boundaries, road_candidates
            )
            
            # Filter candidates to prioritize boundary edges
            filtered_candidates = {}
            
            # First, add boundary edges if they're reachable
            for boundary in candidate_boundaries:
                if boundary in road_candidates:
                    filtered_candidates[boundary] = road_candidates[boundary]
            
            # Then add other candidates up to limit
            for edge, data in road_candidates.items():
                if len(filtered_candidates) >= 5:  # Limit to 5 total candidates
                    break
                if edge not in filtered_candidates:
                    filtered_candidates[edge] = data
            
            return filtered_candidates, obs_text, answer_options
            
        except Exception as e:
            # Fallback to simple observation
            return {}, f"Vehicle {vehicle_id} at {current_edge}", ""
    
    def _get_regional_candidates(self, current_edge: str, candidate_boundaries: List[str]) -> Dict:
        """Get candidate edges for vehicle routing within the region."""
        candidates = {}
        
        # Add boundary edges as primary candidates
        for boundary in candidate_boundaries:
            if boundary in self.road_info:
                edge_info = self.road_info[boundary]
                candidates[boundary] = {
                    'congestion_level': edge_info.get('congestion_level', 0),
                    'vehicle_count': edge_info.get('vehicle_num', 0),
                    'planned_usage': self.planned_routes.get(boundary, 0),
                    'road_len': edge_info.get('road_len', 100),
                    'is_boundary': True
                }
        
        # Add neighboring edges as alternatives
        if current_edge in self.regional_network:
            neighbors = list(self.regional_network.neighbors(current_edge))[:3]
            for neighbor in neighbors:
                if neighbor in self.road_info and neighbor not in candidates:
                    edge_info = self.road_info[neighbor]
                    candidates[neighbor] = {
                        'congestion_level': edge_info.get('congestion_level', 0),
                        'vehicle_count': edge_info.get('vehicle_num', 0),
                        'planned_usage': self.planned_routes.get(neighbor, 0),
                        'road_len': edge_info.get('road_len', 100),
                        'is_boundary': False
                    }
        
        return candidates
    
    def _build_observation_text(self, vehicle_id: str, current_edge: str, 
                              candidate_boundaries: List[str], candidates: Dict) -> Tuple[str, str]:
        """Build observation text for LLM decision making."""
        observation_parts = []
        
        observation_parts.append(f"Vehicle {vehicle_id} route planning in Region {self.region_id}")
        observation_parts.append(f"Current edge: {current_edge}")
        observation_parts.append("")
        
        # Target boundary edges (compressed format: edge_id:cong|veh|usage)
        if candidate_boundaries:
            boundary_list = []
            for boundary in candidate_boundaries:
                if boundary in candidates:
                    info = candidates[boundary]
                    boundary_compact = f"{boundary}:{info['congestion_level']:.1f}|{info['vehicle_count']}|{info['planned_usage']}"
                    boundary_list.append(boundary_compact)
            observation_parts.append("Target_Boundaries: " + " | ".join(boundary_list))
            observation_parts.append("")
        
        # Alternative edges (compressed format: edge_id:cong|veh)
        non_boundary_candidates = [edge for edge, info in candidates.items() 
                                 if not info.get('is_boundary', False)]
        if non_boundary_candidates:
            alt_list = []
            for edge in non_boundary_candidates[:3]:
                info = candidates[edge]
                alt_compact = f"{edge}:{info['congestion_level']:.1f}|{info['vehicle_count']}"
                alt_list.append(alt_compact)
            observation_parts.append("Alt_Edges: " + " | ".join(alt_list))
            observation_parts.append("")
        
        # Regional context (compressed format: R[id]_status:vehicles|boundaries)
        observation_parts.append(f"R{self.region_id}_Status: {len(self.region_vehicles)}veh|{len(self.outgoing_boundaries)}bdry")
        
        # Create answer options
        answer_options = "/".join(list(candidates.keys())[:5])
        
        return "\n".join(observation_parts), f'"{answer_options}"'
    
    def _make_llm_decisions(self, context: Dict[str, Any], current_time: float) -> List[Dict]:
        """Make coordinated decisions using enhanced LLM."""
        try:
            # Use the new regional coordination decision method
            if hasattr(self.llm_agent, 'regional_coordination_decision'):
                # Prepare regional context for LLM
                regional_context = {
                    'region_id': self.region_id,
                    'total_vehicles': len(self.region_vehicles),
                    'outgoing_boundaries': len(self.outgoing_boundaries),
                    'avg_congestion': context['regional_state']['avg_congestion'],
                    'current_time': current_time
                }
                
                # Prepare vehicles data for LLM
                vehicles_data = []
                for vehicle_data in context['vehicles']:
                    vehicles_data.append({
                        'vehicle_id': vehicle_data['vehicle_id'],
                        'current_edge': vehicle_data['current_edge'], 
                        'destination': vehicle_data['destination'],
                        'candidate_boundaries': vehicle_data['candidate_boundaries'],
                        'observation': vehicle_data['observation_data'],
                        'route_history': vehicle_data.get('route_history', [])
                    })
                
                # Prepare boundary status
                boundary_status = context['regional_state']['boundary_states']
                
                # Prepare coordination messages
                coordination_messages = []
                if context.get('recommendations'):
                    recommendations_data = context['recommendations']
                    print(f"REGIONAL_AGENT_DEBUG: Processing recommendations data type: {type(recommendations_data)}")
                    coordination_messages.append({
                        'from': 'TrafficAgent',
                        'message': f"Target boundaries: {recommendations_data.get('target_boundary_edges', [])}",
                        'priority_vehicles': recommendations_data.get('priority_vehicles', []),
                        'avoid_edges': recommendations_data.get('avoid_edges', [])
                    })
                
                # Get traffic predictions for this region
                traffic_predictions = self._get_traffic_predictions_for_region()
                
                # Prepare route options
                route_options = {}
                for vehicle_data in context['vehicles']:
                    vehicle_id = vehicle_data['vehicle_id']
                    route_options[vehicle_id] = vehicle_data['candidate_boundaries']
                
                # Call the enhanced LLM decision method
                llm_result = self.llm_agent.regional_coordination_decision(
                    regional_context=regional_context,
                    vehicles_data=vehicles_data,
                    boundary_status=boundary_status,
                    coordination_messages=coordination_messages,
                    traffic_predictions=traffic_predictions,
                    route_options=route_options,
                    region_id=self.region_id
                )
                
                # Convert LLM result to expected format
                llm_decisions = []
                vehicle_decisions = llm_result.get('vehicle_decisions', [])
                
                for i, vehicle_data in enumerate(context['vehicles']):
                    vehicle_id = vehicle_data['vehicle_id']
                    
                    # Find corresponding decision
                    decision_found = False
                    for vd in vehicle_decisions:
                        if vd.get('vehicle_id') == vehicle_id:
                            llm_decisions.append({
                                'answer': vd.get('target_boundary'),
                                'summary': vd.get('reasoning', 'Regional coordination decision'),
                                'data_analysis': vd.get('coordination_impact', 'System coordination'),
                                'route': vd.get('route', []),
                                'lane_assignment': vd.get('lane_assignment')
                            })
                            decision_found = True
                            break
                    
                    if not decision_found:
                        # Fallback decision
                        llm_decisions.append({
                            'answer': vehicle_data['candidate_boundaries'][0] if vehicle_data['candidate_boundaries'] else None,
                            'summary': 'Fallback regional decision',
                            'data_analysis': 'No specific LLM decision found'
                        })
                
                # Store coordination messages for inter-agent communication
                self._process_inter_region_communication(llm_result.get('inter_region_communication', ''))
                
                return llm_decisions
                
            else:
                # Fallback to enhanced hybrid decision making
                return self._make_enhanced_hybrid_decisions(context, current_time)
                
        except Exception as e:
            print(f"REGIONAL_AGENT_ERROR_DEBUG: Exception in LLM decisions: {e}")
            print(f"REGIONAL_AGENT_ERROR_DEBUG: Context recommendations type: {type(context.get('recommendations'))}")
            self.logger.log_error(f"Regional LLM decisions failed: {e}")
            return self._make_enhanced_hybrid_decisions(context, current_time)

    def _make_enhanced_hybrid_decisions(self, context: Dict[str, Any], current_time: float) -> List[Dict]:
        """Make enhanced hybrid decisions with multi-agent context."""
        try:
            # Prepare data for enhanced hybrid decision making
            data_texts = []
            answer_options = []
            
            for vehicle_data in context['vehicles']:
                data_texts.append(vehicle_data['observation_data'])
                answer_options.append(vehicle_data['answer_options'])
            
            # Prepare multi-agent context
            system_state = {
                'region_id': self.region_id,
                'regional_state': context['regional_state'],
                'current_time': current_time,
                'total_vehicles': len(self.region_vehicles)
            }
            
            agent_communication = []
            if context.get('recommendations'):
                recommendations_data = context['recommendations']
                print(f"ENHANCED_HYBRID_DEBUG: Processing recommendations data type: {type(recommendations_data)}")
                agent_communication.append({
                    'from': 'TrafficAgent', 
                    'recommendations': recommendations_data
                })
            
            regional_coordination = {
                'boundary_utilization': context['regional_state']['boundary_states'],
                'coordination_opportunities': len(context['vehicles'])
            }
            
            traffic_predictions = self._get_traffic_predictions_for_region()
            
            # Use enhanced hybrid decision making
            if hasattr(self.llm_agent, 'enhanced_hybrid_decision_making_pipeline'):
                llm_decisions = self.llm_agent.enhanced_hybrid_decision_making_pipeline(
                    data_texts=data_texts,
                    answer_option_forms=answer_options,
                    decision_type="regional_coordination",
                    decision_context=f"Region {self.region_id} vehicle coordination",
                    system_state=system_state,
                    agent_communication=agent_communication,
                    regional_coordination=regional_coordination,
                    traffic_predictions=traffic_predictions
                )
            else:
                # Basic hybrid decision making fallback
                llm_decisions = self.llm_agent.hybrid_decision_making_pipeline(
                    data_texts, answer_options
                ) if hasattr(self.llm_agent, 'hybrid_decision_making_pipeline') else []
            
            return llm_decisions if llm_decisions else self._make_basic_fallback_decisions(context)
            
        except Exception as e:
            print(f"ENHANCED_HYBRID_ERROR_DEBUG: Exception in enhanced hybrid decisions: {e}")
            print(f"ENHANCED_HYBRID_ERROR_DEBUG: Context recommendations type: {type(context.get('recommendations'))}")
            self.logger.log_error(f"Enhanced hybrid decisions failed: {e}")
            return self._make_basic_fallback_decisions(context)

    def _make_basic_fallback_decisions(self, context: Dict[str, Any]) -> List[Dict]:
        """Basic fallback decisions when LLM fails."""
        llm_decisions = []
        for vehicle_data in context['vehicles']:
            llm_decisions.append({
                'answer': vehicle_data['candidate_boundaries'][0] if vehicle_data['candidate_boundaries'] else None,
                'summary': 'Basic fallback decision - LLM unavailable',
                'data_analysis': 'No advanced analysis available'
            })
        return llm_decisions

    def _get_traffic_predictions_for_region(self) -> Dict[str, Any]:
        """Get traffic predictions for this region from prediction engine."""
        try:
            # This would interface with the prediction engine
            # For now, return basic predictions
            predictions = {
                'boundary_congestion_forecast': {},
                'regional_traffic_forecast': 'stable',
                'prediction_confidence': 0.7
            }
            
            # Add boundary-specific predictions
            for boundary in self.outgoing_boundaries:
                if boundary in self.road_info:
                    current_congestion = self.road_info[boundary].get('congestion_level', 0)
                    predictions['boundary_congestion_forecast'][boundary] = {
                        'current': current_congestion,
                        'predicted_15min': min(5, current_congestion + 1),
                        'predicted_30min': min(5, current_congestion),
                        'trend': 'stable'
                    }
            
            return predictions
            
        except Exception:
            return {'status': 'predictions_unavailable'}

    def _process_inter_region_communication(self, communication_message: str):
        """Process and store inter-region communication for coordination."""
        if communication_message and communication_message != 'Communication failed':
            # Store for coordination with other regions
            self.inter_region_messages = getattr(self, 'inter_region_messages', [])
            self.inter_region_messages.append({
                'timestamp': time.time(),
                'from_region': self.region_id,
                'message': communication_message,
                'type': 'coordination'
            })
            
            # Limit message history
            self.inter_region_messages = self.inter_region_messages[-10:]
    
    def _process_llm_decisions(self, llm_decisions: List[Dict], 
                             vehicle_ids: List[str], current_time: float) -> List[VehicleDecision]:
        """Process LLM decisions into VehicleDecision objects."""
        decisions = []
        
        for i, vehicle_id in enumerate(vehicle_ids):
            if i >= len(llm_decisions):
                continue
            
            decision_data = llm_decisions[i]
            target_edge = decision_data.get('answer')
            reasoning = decision_data.get('summary', 'LLM decision')
            
            if target_edge and target_edge in self.region_edges:
                try:
                    current_edge = traci.vehicle.getRoadID(vehicle_id)
                    
                    # Plan route to target edge
                    route = self._plan_route(current_edge, target_edge)
                    
                    if route:
                        # Assign lane if needed
                        lane_assignment = self._assign_lane(vehicle_id, target_edge)
                        
                        # Determine priority
                        priority = self._calculate_priority(vehicle_id, target_edge)
                        
                        decision = VehicleDecision(
                            vehicle_id=vehicle_id,
                            current_edge=current_edge,
                            target_edge=target_edge,
                            route=route,
                            lane_assignment=lane_assignment,
                            priority=priority,
                            decision_time=current_time,
                            reasoning=reasoning
                        )
                        
                        decisions.append(decision)
                
                except Exception as e:
                    self.logger.log_error(f"Failed to process decision for vehicle {vehicle_id}: {e}")
                    continue
        
        return decisions
    
    def _make_heuristic_decisions(self, vehicle_ids: List[str], 
                                recommendations: Optional[RegionalRecommendation],
                                current_time: float) -> List[VehicleDecision]:
        """Make heuristic decisions as fallback."""
        decisions = []
        
        for vehicle_id in vehicle_ids:
            try:
                current_edge = traci.vehicle.getRoadID(vehicle_id)
                route = traci.vehicle.getRoute(vehicle_id)
                destination = route[-1] if route else None
                
                # Find best boundary edge using simple heuristics
                target_boundary = self._select_best_boundary_heuristic(
                    current_edge, destination, recommendations
                )
                
                if target_boundary:
                    # Plan route
                    route_to_boundary = self._plan_route(current_edge, target_boundary)
                    
                    if route_to_boundary:
                        decision = VehicleDecision(
                            vehicle_id=vehicle_id,
                            current_edge=current_edge,
                            target_edge=target_boundary,
                            route=route_to_boundary,
                            lane_assignment=None,
                            priority=1,
                            decision_time=current_time,
                            reasoning="Heuristic decision - LLM failed"
                        )
                        
                        decisions.append(decision)
            
            except Exception:
                continue
        
        return decisions
    
    def _select_best_boundary_heuristic(self, current_edge: str, destination: str,
                                      recommendations: Optional[RegionalRecommendation]) -> Optional[str]:
        """Select best boundary edge using heuristics."""
        if not self.outgoing_boundaries:
            return None
        
        boundary_scores = {}
        
        for boundary in self.outgoing_boundaries:
            score = 0
            
            # Factor 1: Distance (prefer closer boundaries)
            try:
                distance = nx.shortest_path_length(
                    self.regional_network, current_edge, boundary, weight='weight'
                )
                score += 1000 / max(1, distance)  # Higher score for shorter distance
            except:
                score += 0  # Not reachable
            
            # Factor 2: Congestion (prefer less congested)
            if boundary in self.road_info:
                congestion = self.road_info[boundary].get('congestion_level', 0)
                score += (5 - congestion) * 100  # Higher score for lower congestion
            
            # Factor 3: Current usage (prefer less used)
            usage = self.planned_routes.get(boundary, 0)
            score += max(0, 50 - usage * 10)  # Penalty for high usage
            
            # Factor 4: Recommendations
            if recommendations and boundary in recommendations.target_boundary_edges:
                score += 200
            
            boundary_scores[boundary] = score
        
        # Return boundary with highest score
        return max(boundary_scores.items(), key=lambda x: x[1])[0]
    
    def _plan_route(self, start_edge: str, target_edge: str) -> Optional[List[str]]:
        """Plan route between two edges within the region."""
        try:
            if start_edge == target_edge:
                return [start_edge]
            
            # Use NetworkX to find shortest path
            path = nx.shortest_path(
                self.regional_network, start_edge, target_edge, weight='weight'
            )
            
            return path
            
        except Exception as e:
            # Fallback: use SUMO's route finding
            try:
                route_result = traci.simulation.findRoute(start_edge, target_edge)
                return list(route_result.edges) if route_result.edges else None
            except:
                return None
    
    def _assign_lane(self, vehicle_id: str, target_edge: str) -> Optional[int]:
        """Advanced lane assignment using multi-criteria optimization.
        
        Considers:
        - Traffic density per lane (30%)
        - Speed efficiency per lane (25%)
        - Route compatibility (20%)
        - Lane change complexity (15%)
        - Predicted conditions (10%)
        """
        try:
            current_edge = traci.vehicle.getRoadID(vehicle_id)
            if not current_edge or current_edge.startswith(':'):
                return self._assign_lane_fallback(vehicle_id)
            
            # Get lane information
            num_lanes = traci.edge.getLaneNumber(current_edge)
            if num_lanes <= 1:
                return 0
                
            # Get vehicle's current lane
            try:
                current_lane_index = traci.vehicle.getLaneIndex(vehicle_id)
            except:
                current_lane_index = 0
                
            # Score each available lane
            lane_scores = []
            for lane_idx in range(num_lanes):
                score = self._calculate_lane_score(
                    current_edge, lane_idx, vehicle_id, target_edge, current_lane_index
                )
                lane_scores.append((lane_idx, score))
                
            # Select best lane
            lane_scores.sort(key=lambda x: x[1], reverse=True)
            best_lane = lane_scores[0][0]
            best_score = lane_scores[0][1]
            
            # Detailed decision logging for performance analysis
            self._log_lane_decision_analysis(vehicle_id, current_edge, lane_scores, current_lane_index, target_edge)
            
            # Only suggest lane change if significant improvement or safety concern
            if best_lane != current_lane_index:
                score_improvement = best_score - lane_scores[current_lane_index][1] if current_lane_index < len(lane_scores) else best_score
                if score_improvement > 0.15:  # Require 15% improvement to change lanes
                    self.logger.log_info(f"LANE_OPTIMIZATION: {vehicle_id} changing from lane {current_lane_index} to {best_lane} "
                                       f"(score improvement: {score_improvement:.3f}, best_score: {best_score:.3f})")
                    self._record_lane_optimization_decision(vehicle_id, current_lane_index, best_lane, score_improvement, "optimization")
                    return best_lane
                else:
                    self.logger.log_info(f"LANE_STAY: {vehicle_id} staying in lane {current_lane_index} "
                                       f"(improvement: {score_improvement:.3f} < threshold: 0.15)")
                    self._record_lane_optimization_decision(vehicle_id, current_lane_index, current_lane_index, score_improvement, "stay")
                    return current_lane_index  # Stay in current lane if improvement is marginal
            
            return best_lane
            
        except Exception as e:
            self.logger.log_error(f"LANE_ASSIGNMENT_ERROR: {vehicle_id} -> {e}")
            return self._assign_lane_fallback(vehicle_id)
    
    def _assign_lane_fallback(self, vehicle_id: str) -> Optional[int]:
        """Fallback lane assignment when advanced algorithm fails."""
        try:
            current_edge = traci.vehicle.getRoadID(vehicle_id)
            num_lanes = traci.edge.getLaneNumber(current_edge)
            
            if num_lanes <= 1:
                return 0
            
            # Conservative fallback: prefer middle lanes or current lane
            try:
                current_lane = traci.vehicle.getLaneIndex(vehicle_id)
                return current_lane  # Stay in current lane as safest option
            except:
                return min(1, num_lanes - 1)  # Default to lane 1 or rightmost
                
        except:
            return 0
    
    def _calculate_lane_score(self, edge_id: str, lane_idx: int, vehicle_id: str, 
                            target_edge: str, current_lane: int) -> float:
        """Calculate comprehensive score for a specific lane.
        
        Returns score between 0.0 and 1.0 where higher is better.
        """
        score = 0.0
        lane_id = f"{edge_id}_{lane_idx}"
        
        try:
            # Dynamic weight calculation based on current traffic conditions
            weights = self._calculate_dynamic_weights(edge_id, current_lane, lane_idx)
            
            # 1. Traffic density score
            density_score = self._score_lane_density(lane_id)
            score += density_score * weights['density']
            
            # 2. Speed efficiency score
            speed_score = self._score_lane_speed(lane_id)
            score += speed_score * weights['speed']
            
            # 3. Route compatibility score
            route_score = self._score_route_compatibility(lane_idx, target_edge, edge_id)
            score += route_score * weights['route']
            
            # 4. Lane change cost score
            change_score = self._score_lane_change_cost(current_lane, lane_idx)
            score += change_score * weights['change_cost']
            
            # 5. Predicted conditions score
            prediction_score = self._score_lane_predictions(edge_id)
            score += prediction_score * weights['prediction']
            
            # 6. Context-aware bonus/penalty
            context_adjustment = self._calculate_context_adjustment(edge_id, lane_idx, vehicle_id)
            score += context_adjustment
            
        except Exception as e:
            self.logger.log_warning(f"LANE_SCORE_ERROR: {lane_id} -> {e}")
            score = 0.5  # Neutral score on error
            
        return max(0.0, min(1.0, score))
    
    def _score_lane_density(self, lane_id: str) -> float:
        """Score based on current lane density (lower density = higher score)."""
        try:
            vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
            lane_length = traci.lane.getLength(lane_id)
            
            if lane_length <= 0:
                return 0.5
                
            # Calculate density (vehicles per 100m)
            density = (vehicle_count / lane_length) * 100
            
            # Convert to score (lower density = higher score)
            # Assume max reasonable density of 10 vehicles per 100m
            max_density = 10.0
            score = max(0.0, 1.0 - (density / max_density))
            
            return score
            
        except Exception as e:
            return 0.5  # Neutral score if data unavailable
    
    def _score_lane_speed(self, lane_id: str) -> float:
        """Score based on lane speed efficiency (higher relative speed = higher score)."""
        try:
            mean_speed = traci.lane.getLastStepMeanSpeed(lane_id)
            max_speed = traci.lane.getMaxSpeed(lane_id)
            
            if max_speed <= 0:
                return 0.5
                
            # Calculate speed efficiency ratio
            speed_ratio = mean_speed / max_speed
            
            # Apply sigmoid-like function to reward good speeds and penalize very slow speeds
            if speed_ratio >= 0.8:
                return 1.0  # Excellent speed
            elif speed_ratio >= 0.6:
                return 0.8  # Good speed
            elif speed_ratio >= 0.4:
                return 0.6  # Moderate speed
            elif speed_ratio >= 0.2:
                return 0.4  # Slow speed
            else:
                return 0.1  # Very slow speed
                
        except Exception as e:
            return 0.5
    
    def _score_route_compatibility(self, lane_idx: int, target_edge: str, current_edge: str) -> float:
        """Score based on how well lane supports route to target edge."""
        try:
            lane_id = f"{current_edge}_{lane_idx}"
            
            # Get lane connections and analyze route compatibility
            try:
                connections = traci.lane.getLinks(lane_id)
                
                # Check if any connections lead toward target direction
                target_compatible = False
                total_connections = len(connections)
                
                if total_connections == 0:
                    return 0.3  # Dead end lane
                    
                # Analyze connection destinations
                for connection in connections:
                    target_lane = connection[0]
                    if target_lane:
                        # Extract edge from lane ID
                        target_lane_edge = target_lane.split('_')[0]
                        if target_edge.startswith(target_lane_edge) or target_lane_edge.startswith(target_edge.split('_')[0]):
                            target_compatible = True
                            break
                            
                if target_compatible:
                    return 1.0  # Perfect route compatibility
                    
            except Exception:
                pass  # Fall through to default scoring
                
            # Default scoring based on lane position
            edge_lanes = traci.edge.getLaneNumber(current_edge)
            
            if edge_lanes <= 1:
                return 1.0  # Only lane available
            elif edge_lanes == 2:
                return 0.8  # Both lanes generally good
            else:
                # For multi-lane roads, middle lanes often provide more flexibility
                if 0 < lane_idx < edge_lanes - 1:
                    return 0.9  # Middle lane - good flexibility
                elif lane_idx == 0:
                    return 0.7  # Leftmost lane - limited for turns
                else:
                    return 0.8  # Rightmost lane - good for exits
                    
        except Exception as e:
            return 0.5
    
    def _score_lane_change_cost(self, current_lane: int, target_lane: int) -> float:
        """Score based on difficulty/cost of lane change (fewer changes = higher score)."""
        lane_diff = abs(target_lane - current_lane)
        
        # Reward staying in same lane or making minimal changes
        if lane_diff == 0:
            return 1.0  # No change needed
        elif lane_diff == 1:
            return 0.8  # Single lane change - reasonable
        elif lane_diff == 2:
            return 0.6  # Two lane change - more complex
        elif lane_diff == 3:
            return 0.4  # Three lane change - difficult
        else:
            return 0.2  # Very complex lane change
    
    def _score_lane_predictions(self, edge_id: str) -> float:
        """Score based on advanced traffic predictions using prediction engine.
        
        Uses both current conditions and future forecasts to make intelligent
        lane selection decisions.
        """
        try:
            score = 0.0
            
            # 1. Current conditions score (40% weight)
            current_score = self._score_current_conditions(edge_id)
            score += current_score * 0.4
            
            # 2. Future congestion forecast score (35% weight)
            if self.prediction_engine:
                forecast_score = self._score_congestion_forecast(edge_id)
                score += forecast_score * 0.35
            else:
                # Fallback to trend analysis if no prediction engine
                trend_score = self._score_traffic_trend(edge_id)
                score += trend_score * 0.35
            
            # 3. Predictive traffic flow score (25% weight)
            flow_score = self._score_predictive_flow(edge_id)
            score += flow_score * 0.25
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.log_warning(f"LANE_PREDICTION_ERROR: {edge_id} -> {e}")
            return 0.5
    
    def _score_current_conditions(self, edge_id: str) -> float:
        """Score based on current real-time traffic conditions."""
        try:
            if edge_id in self.road_info:
                edge_data = self.road_info[edge_id]
                
                # Factor in recent congestion level
                recent_congestion = edge_data.get('congestion_level', 0)
                occupancy_rate = edge_data.get('occupancy_rate', 0)
                vehicle_speed = edge_data.get('vehicle_speed', 0)
                speed_limit = edge_data.get('speed_limit', 13.89)  # Default ~50 km/h
                
                # Convert congestion to score (lower congestion = higher score)
                max_congestion = 5.0
                congestion_score = max(0.0, 1.0 - (recent_congestion / max_congestion))
                
                # Factor in occupancy rate (lower occupancy = higher score)
                occupancy_score = max(0.0, 1.0 - min(1.0, occupancy_rate))
                
                # Factor in speed efficiency (higher relative speed = higher score)
                speed_efficiency = vehicle_speed / speed_limit if speed_limit > 0 else 0.5
                speed_score = max(0.0, min(1.0, speed_efficiency))
                
                # Weighted combination of factors
                current_score = (congestion_score * 0.4) + (occupancy_score * 0.35) + (speed_score * 0.25)
                
                return current_score
                
            return 0.5  # Neutral if no data available
            
        except Exception:
            return 0.5
    
    def _score_congestion_forecast(self, edge_id: str) -> float:
        """Score based on predicted congestion using prediction engine."""
        try:
            if not self.prediction_engine:
                return 0.5
                
            # Get congestion forecast for next 15-30 minutes
            forecast = self.prediction_engine.get_congestion_forecast([edge_id], 1800)  # 30 minutes
            
            if edge_id in forecast and forecast[edge_id]:
                congestion_forecast = forecast[edge_id]
                
                # Calculate weighted average with more weight on near-term predictions
                weighted_congestion = 0.0
                total_weight = 0.0
                
                for i, future_congestion in enumerate(congestion_forecast[:6]):  # Next 6 time steps
                    # Weight decreases with time: 0.4, 0.3, 0.2, 0.1, 0.05, 0.05
                    weight = max(0.05, 0.4 * (0.7 ** i))
                    weighted_congestion += future_congestion * weight
                    total_weight += weight
                    
                if total_weight > 0:
                    avg_forecast = weighted_congestion / total_weight
                    
                    # Convert to score (lower predicted congestion = higher score)
                    max_congestion = 5.0
                    forecast_score = max(0.0, 1.0 - (avg_forecast / max_congestion))
                    
                    # Apply confidence factor if available
                    confidence_factor = 0.8  # Conservative confidence
                    return forecast_score * confidence_factor + 0.5 * (1 - confidence_factor)
                    
            return 0.5  # Neutral if no forecast available
            
        except Exception as e:
            self.logger.log_warning(f"CONGESTION_FORECAST_ERROR: {edge_id} -> {e}")
            return 0.5
    
    def _score_traffic_trend(self, edge_id: str) -> float:
        """Score based on recent traffic trend analysis (fallback method)."""
        try:
            if edge_id in self.road_info:
                edge_data = self.road_info[edge_id]
                
                # Get recent update time to assess data freshness
                last_update = edge_data.get('last_update', 0)
                current_time = time.time()
                data_age = current_time - last_update
                
                # Penalize stale data
                if data_age > 300:  # More than 5 minutes old
                    freshness_factor = max(0.3, 1.0 - (data_age - 300) / 600)  # Decay over 10 minutes
                else:
                    freshness_factor = 1.0
                    
                # Simple trend: compare current to recent average
                current_congestion = edge_data.get('congestion_level', 0)
                
                # Assume improving trend if congestion is low
                if current_congestion <= 1.0:
                    trend_score = 0.8  # Good trend
                elif current_congestion <= 2.5:
                    trend_score = 0.6  # Moderate trend
                elif current_congestion <= 4.0:
                    trend_score = 0.4  # Poor trend
                else:
                    trend_score = 0.2  # Very poor trend
                    
                return trend_score * freshness_factor
                
            return 0.5
            
        except Exception:
            return 0.5
    
    def _score_predictive_flow(self, edge_id: str) -> float:
        """Score based on predictive traffic flow analysis."""
        try:
            # Use prediction engine for multi-metric predictions if available
            if self.prediction_engine:
                predictions = self.prediction_engine.get_predictions(edge_id, 360, 2)  # 6-min window, 2 steps ahead
                
                if predictions:
                    flow_scores = []
                    
                    for prediction in predictions:
                        confidence = getattr(prediction, 'confidence', 0.5)
                        
                        # Vehicle count prediction
                        if hasattr(prediction, 'predicted_vehicle_count'):
                            vehicle_count = prediction.predicted_vehicle_count
                            # Lower predicted count = higher score
                            count_score = max(0.0, 1.0 - min(1.0, vehicle_count / 20.0))  # Normalize by expected max
                            flow_scores.append(count_score * confidence)
                            
                        # Speed prediction
                        if hasattr(prediction, 'predicted_avg_speed'):
                            avg_speed = prediction.predicted_avg_speed
                            # Higher predicted speed = higher score
                            speed_score = min(1.0, avg_speed / 13.89)  # Normalize by typical max speed
                            flow_scores.append(speed_score * confidence)
                            
                    if flow_scores:
                        return sum(flow_scores) / len(flow_scores)
                        
            # Fallback: analyze current flow characteristics
            if edge_id in self.road_info:
                edge_data = self.road_info[edge_id]
                vehicle_num = edge_data.get('vehicle_num', 0)
                road_len = edge_data.get('road_len', 100)
                
                # Calculate flow density
                density = vehicle_num / max(1, road_len / 100)  # Vehicles per 100m
                
                # Convert density to flow score
                optimal_density = 2.0  # Vehicles per 100m
                if density <= optimal_density:
                    flow_score = 1.0 - (density / optimal_density) * 0.3  # Slight penalty for increasing density
                else:
                    flow_score = 0.7 * max(0.0, 1.0 - (density - optimal_density) / 8.0)  # Penalty for overcrowding
                    
                return max(0.0, min(1.0, flow_score))
                
            return 0.5
            
        except Exception as e:
            self.logger.log_warning(f"PREDICTIVE_FLOW_ERROR: {edge_id} -> {e}")
            return 0.5
    
    def _calculate_dynamic_weights(self, edge_id: str, current_lane: int, target_lane: int) -> Dict[str, float]:
        """Calculate dynamic weights for lane scoring based on current conditions.
        
        Adjusts the importance of different factors based on:
        - Current traffic density
        - System congestion level
        - Lane change complexity
        - Time-critical situations
        """
        try:
            # Default weights
            weights = {
                'density': 0.30,
                'speed': 0.25,
                'route': 0.20,
                'change_cost': 0.15,
                'prediction': 0.10
            }
            
            # Adjust weights based on current conditions
            if edge_id in self.road_info:
                edge_data = self.road_info[edge_id]
                current_congestion = edge_data.get('congestion_level', 0)
                occupancy_rate = edge_data.get('occupancy_rate', 0)
                
                # High congestion: prioritize speed and prediction
                if current_congestion >= 3.0 or occupancy_rate >= 0.8:
                    weights['speed'] += 0.10  # Increase speed importance
                    weights['prediction'] += 0.10  # Increase prediction importance
                    weights['density'] -= 0.10  # Decrease density importance
                    weights['route'] -= 0.10  # Decrease route importance
                    
                # Low congestion: prioritize route efficiency
                elif current_congestion <= 1.0 and occupancy_rate <= 0.3:
                    weights['route'] += 0.15  # Increase route importance
                    weights['change_cost'] += 0.05  # Slightly increase change cost importance
                    weights['density'] -= 0.10  # Decrease density importance
                    weights['speed'] -= 0.10  # Decrease speed importance
                    
                # Complex lane change: prioritize safety (change cost)
                lane_diff = abs(target_lane - current_lane)
                if lane_diff >= 2:
                    weights['change_cost'] += 0.15  # Heavily prioritize safety
                    weights['prediction'] += 0.05  # Consider future conditions
                    weights['density'] -= 0.10  # Less focus on current density
                    weights['speed'] -= 0.10  # Less focus on current speed
                    
            # Normalize weights to sum to 1.0
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
                
            return weights
            
        except Exception as e:
            self.logger.log_warning(f"DYNAMIC_WEIGHTS_ERROR: {e}")
            # Return default weights on error
            return {
                'density': 0.30,
                'speed': 0.25, 
                'route': 0.20,
                'change_cost': 0.15,
                'prediction': 0.10
            }
    
    def _calculate_context_adjustment(self, edge_id: str, lane_idx: int, vehicle_id: str) -> float:
        """Calculate context-aware bonus/penalty for lane selection.
        
        Considers:
        - Nearby vehicle behavior
        - Emergency situations
        - System-wide optimization needs
        - Historical performance
        """
        adjustment = 0.0
        
        try:
            # 1. Nearby vehicle analysis
            nearby_adjustment = self._analyze_nearby_vehicles(edge_id, lane_idx)
            adjustment += nearby_adjustment * 0.4
            
            # 2. Emergency/priority situation detection
            priority_adjustment = self._detect_priority_situations(vehicle_id, edge_id)
            adjustment += priority_adjustment * 0.3
            
            # 3. System-wide load balancing
            balance_adjustment = self._calculate_load_balance_adjustment(edge_id, lane_idx)
            adjustment += balance_adjustment * 0.2
            
            # 4. Historical performance bonus
            history_adjustment = self._calculate_historical_performance_bonus(edge_id, lane_idx)
            adjustment += history_adjustment * 0.1
            
            # Cap adjustment to reasonable range
            return max(-0.2, min(0.2, adjustment))
            
        except Exception as e:
            self.logger.log_warning(f"CONTEXT_ADJUSTMENT_ERROR: {vehicle_id} -> {e}")
            return 0.0
    
    def _analyze_nearby_vehicles(self, edge_id: str, lane_idx: int) -> float:
        """Analyze nearby vehicle patterns for cooperative lane selection."""
        try:
            lane_id = f"{edge_id}_{lane_idx}"
            
            # Get vehicles in this lane
            vehicles_in_lane = traci.lane.getLastStepVehicleIDs(lane_id)
            
            if not vehicles_in_lane:
                return 0.1  # Bonus for empty lane
                
            # Analyze vehicle speeds and gaps
            vehicle_speeds = []
            for veh_id in vehicles_in_lane:
                try:
                    speed = traci.vehicle.getSpeed(veh_id)
                    vehicle_speeds.append(speed)
                except:
                    continue
                    
            if vehicle_speeds:
                avg_speed = sum(vehicle_speeds) / len(vehicle_speeds)
                speed_variance = sum((s - avg_speed) ** 2 for s in vehicle_speeds) / len(vehicle_speeds)
                
                # Prefer lanes with consistent, good speeds
                if avg_speed > 8.0 and speed_variance < 4.0:  # Good flow
                    return 0.05
                elif avg_speed < 3.0 or speed_variance > 16.0:  # Poor flow
                    return -0.05
                    
            return 0.0  # Neutral
            
        except Exception:
            return 0.0
    
    def _detect_priority_situations(self, vehicle_id: str, edge_id: str) -> float:
        """Detect emergency or priority situations requiring special lane handling."""
        try:
            # Check if vehicle has been stuck or delayed
            if vehicle_id in self.vehicle_start_times:
                travel_time = time.time() - self.vehicle_start_times.get(vehicle_id, time.time())
                
                # Priority boost for vehicles that have been traveling for a long time
                if travel_time > 600:  # More than 10 minutes
                    return 0.1  # Priority boost
                elif travel_time > 300:  # More than 5 minutes
                    return 0.05  # Moderate boost
                    
            # Check for boundary approach (vehicles nearing region exits get priority)
            if edge_id in [b['edge_id'] for b in self.boundary_edges if b.get('from_region') == self.region_id]:
                return 0.08  # Boost for boundary approach
                
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_load_balance_adjustment(self, edge_id: str, lane_idx: int) -> float:
        """Calculate adjustment for system-wide load balancing."""
        try:
            # Get edge information
            if edge_id not in self.road_info:
                return 0.0
                
            edge_data = self.road_info[edge_id]
            num_lanes = traci.edge.getLaneNumber(edge_id) if edge_id in traci.edge.getIDList() else 1
            
            if num_lanes <= 1:
                return 0.0
                
            # Analyze lane distribution
            lane_loads = []
            for lane_i in range(num_lanes):
                lane_id = f"{edge_id}_{lane_i}"
                try:
                    vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
                    lane_loads.append(vehicle_count)
                except:
                    lane_loads.append(0)
                    
            if not lane_loads:
                return 0.0
                
            avg_load = sum(lane_loads) / len(lane_loads)
            target_lane_load = lane_loads[lane_idx] if lane_idx < len(lane_loads) else avg_load
            
            # Bonus for choosing underutilized lanes
            if target_lane_load < avg_load * 0.8:
                return 0.03  # Bonus for load balancing
            elif target_lane_load > avg_load * 1.2:
                return -0.03  # Penalty for overloading
                
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_historical_performance_bonus(self, edge_id: str, lane_idx: int) -> float:
        """Calculate bonus based on historical lane performance."""
        try:
            # Check if we have lane change statistics
            if not hasattr(self, 'lane_change_stats'):
                return 0.0
                
            recent_changes = self.lane_change_stats.get('recent_changes', [])
            
            # Find successful changes to this lane on this edge
            successful_to_lane = 0
            total_to_lane = 0
            
            for change in recent_changes:
                if change.get('to_lane') == lane_idx:
                    total_to_lane += 1
                    if self._was_lane_change_successful(change):
                        successful_to_lane += 1
                        
            if total_to_lane >= 3:  # Need minimum sample size
                success_rate = successful_to_lane / total_to_lane
                if success_rate > 0.8:
                    return 0.02  # Bonus for historically good lane
                elif success_rate < 0.5:
                    return -0.02  # Penalty for historically poor lane
                    
            return 0.0
            
        except Exception:
            return 0.0
    
    def _log_lane_decision_analysis(self, vehicle_id: str, edge_id: str, lane_scores: List[Tuple[int, float]], 
                                   current_lane: int, target_edge: str):
        """Log detailed lane decision analysis for performance monitoring."""
        try:
            # Log comprehensive lane analysis
            score_summary = ", ".join([f"L{lane}:{score:.3f}" for lane, score in lane_scores])
            self.logger.log_info(f"LANE_ANALYSIS: {vehicle_id} on {edge_id} -> {score_summary} (current: L{current_lane}, target: {target_edge})")
            
            # Log top 2 choices for detailed analysis
            if len(lane_scores) >= 2:
                best_lane, best_score = lane_scores[0]
                second_lane, second_score = lane_scores[1]
                score_gap = best_score - second_score
                
                self.logger.log_info(f"LANE_TOP_CHOICES: {vehicle_id} -> Best: L{best_lane}({best_score:.3f}), "
                                   f"Second: L{second_lane}({second_score:.3f}), Gap: {score_gap:.3f}")
                
        except Exception as e:
            self.logger.log_warning(f"LANE_DECISION_LOG_ERROR: {vehicle_id} -> {e}")
    
    def _record_lane_optimization_decision(self, vehicle_id: str, from_lane: int, to_lane: int, 
                                         score_improvement: float, decision_type: str):
        """Record lane optimization decision for performance tracking."""
        try:
            # Initialize tracking structure if needed
            if not hasattr(self, 'lane_optimization_history'):
                self.lane_optimization_history = deque(maxlen=200)  # Keep last 200 decisions
            
            # Record the decision
            decision_record = {
                'vehicle_id': vehicle_id,
                'timestamp': time.time(),
                'from_lane': from_lane,
                'to_lane': to_lane,
                'score_improvement': score_improvement,
                'decision_type': decision_type,  # 'optimization', 'stay', 'safety'
                'region_id': self.region_id
            }
            
            self.lane_optimization_history.append(decision_record)
            
            # Update aggregated statistics
            self._update_lane_optimization_stats(decision_record)
            
        except Exception as e:
            self.logger.log_warning(f"LANE_DECISION_RECORD_ERROR: {vehicle_id} -> {e}")
    
    def _update_lane_optimization_stats(self, decision_record: Dict):
        """Update aggregated lane optimization statistics."""
        try:
            # Initialize stats if needed
            if not hasattr(self, 'lane_optimization_stats'):
                self.lane_optimization_stats = {
                    'total_decisions': 0,
                    'optimization_decisions': 0,
                    'stay_decisions': 0,
                    'avg_score_improvement': 0.0,
                    'avg_lanes_changed': 0.0,
                    'decision_distribution': {'optimization': 0, 'stay': 0, 'safety': 0}
                }
            
            stats = self.lane_optimization_stats
            
            # Update counters
            stats['total_decisions'] += 1
            decision_type = decision_record['decision_type']
            stats['decision_distribution'][decision_type] = stats['decision_distribution'].get(decision_type, 0) + 1
            
            if decision_type == 'optimization':
                stats['optimization_decisions'] += 1
            elif decision_type == 'stay':
                stats['stay_decisions'] += 1
            
            # Update averages
            old_count = stats['total_decisions'] - 1
            if old_count > 0:
                # Running average calculation
                old_avg_improvement = stats['avg_score_improvement']
                new_improvement = decision_record['score_improvement']
                stats['avg_score_improvement'] = (old_avg_improvement * old_count + new_improvement) / stats['total_decisions']
                
                old_avg_lanes = stats['avg_lanes_changed']
                new_lanes_changed = abs(decision_record['to_lane'] - decision_record['from_lane'])
                stats['avg_lanes_changed'] = (old_avg_lanes * old_count + new_lanes_changed) / stats['total_decisions']
            else:
                stats['avg_score_improvement'] = decision_record['score_improvement']
                stats['avg_lanes_changed'] = abs(decision_record['to_lane'] - decision_record['from_lane'])
                
        except Exception as e:
            self.logger.log_warning(f"LANE_STATS_UPDATE_ERROR: {e}")
    
    def get_lane_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive lane optimization performance report."""
        try:
            report = {
                'region_id': self.region_id,
                'timestamp': time.time(),
                'basic_stats': self._get_lane_optimization_metrics(),
                'decision_quality': {},
                'efficiency_metrics': {},
                'recommendation': 'optimal'
            }
            
            # Decision quality analysis
            if hasattr(self, 'lane_optimization_stats'):
                stats = self.lane_optimization_stats
                total = stats['total_decisions']
                
                if total > 0:
                    optimization_rate = stats['optimization_decisions'] / total
                    stay_rate = stats['stay_decisions'] / total
                    
                    report['decision_quality'] = {
                        'optimization_rate': optimization_rate,
                        'stay_rate': stay_rate,
                        'avg_improvement_when_optimizing': stats['avg_score_improvement'],
                        'avg_lanes_changed': stats['avg_lanes_changed'],
                        'decision_confidence': optimization_rate if optimization_rate < 0.8 else 0.8 + (0.2 * stay_rate)
                    }
            
            # Efficiency metrics
            if hasattr(self, 'lane_change_stats'):
                change_stats = self.lane_change_stats
                recent_changes = change_stats.get('recent_changes', [])
                
                if recent_changes:
                    # Calculate success rate and efficiency
                    successful = sum(1 for change in recent_changes if self._was_lane_change_successful(change))
                    success_rate = successful / len(recent_changes)
                    
                    avg_duration = sum(change['duration'] for change in recent_changes) / len(recent_changes)
                    
                    report['efficiency_metrics'] = {
                        'execution_success_rate': success_rate,
                        'avg_execution_duration': avg_duration,
                        'total_lane_changes': len(recent_changes),
                        'efficiency_score': success_rate * (1.0 - min(0.5, avg_duration / 10000))  # Normalize duration
                    }
            
            # Generate recommendation
            report['recommendation'] = self._generate_optimization_recommendation(report)
            
            return report
            
        except Exception as e:
            self.logger.log_error(f"LANE_OPTIMIZATION_REPORT_ERROR: {e}")
            return {
                'region_id': self.region_id,
                'timestamp': time.time(),
                'error': str(e),
                'recommendation': 'unknown'
            }
    
    def _generate_optimization_recommendation(self, report: Dict) -> str:
        """Generate optimization recommendation based on performance metrics."""
        try:
            decision_quality = report.get('decision_quality', {})
            efficiency_metrics = report.get('efficiency_metrics', {})
            
            optimization_rate = decision_quality.get('optimization_rate', 0.5)
            success_rate = efficiency_metrics.get('execution_success_rate', 0.8)
            efficiency_score = efficiency_metrics.get('efficiency_score', 0.7)
            
            # Generate recommendation based on performance
            if success_rate > 0.85 and efficiency_score > 0.8:
                return 'optimal'
            elif optimization_rate > 0.7 and success_rate > 0.75:
                return 'good'
            elif success_rate < 0.6 or efficiency_score < 0.5:
                return 'needs_tuning'
            elif optimization_rate < 0.3:
                return 'conservative'
            else:
                return 'acceptable'
                
        except Exception:
            return 'unknown'
    
    def _initialize_lane_optimization_system(self):
        """Initialize the advanced lane optimization system components."""
        try:
            # Initialize optimization tracking structures
            self.lane_optimization_history = deque(maxlen=200)
            self.lane_optimization_stats = {
                'total_decisions': 0,
                'optimization_decisions': 0,
                'stay_decisions': 0,
                'avg_score_improvement': 0.0,
                'avg_lanes_changed': 0.0,
                'decision_distribution': {'optimization': 0, 'stay': 0, 'safety': 0}
            }
            
            # Initialize lane change tracking
            self.lane_change_stats = {
                'total_attempts': 0,
                'successful_changes': 0,
                'avg_duration': 0,
                'recent_changes': deque(maxlen=50)
            }
            
            # Log initialization success
            prediction_status = "with prediction engine" if self.prediction_engine else "basic mode"
            self.logger.log_info(f"LANE_OPTIMIZATION_INIT: Region {self.region_id} advanced lane optimization system initialized ({prediction_status})")
            
        except Exception as e:
            self.logger.log_error(f"LANE_OPTIMIZATION_INIT_ERROR: Region {self.region_id} -> {e}")
    
    def validate_lane_optimization_integration(self) -> Dict[str, bool]:
        """Validate that all lane optimization components are properly integrated."""
        validation_results = {
            'prediction_engine_available': self.prediction_engine is not None,
            'optimization_stats_initialized': hasattr(self, 'lane_optimization_stats'),
            'lane_change_tracking_initialized': hasattr(self, 'lane_change_stats'),
            'scoring_methods_available': True,
            'execution_methods_available': True,
            'logging_methods_available': True
        }
        
        # Test core scoring methods
        try:
            test_weights = self._calculate_dynamic_weights('test_edge', 0, 1)
            validation_results['dynamic_weights_functional'] = isinstance(test_weights, dict) and 'density' in test_weights
        except:
            validation_results['dynamic_weights_functional'] = False
            validation_results['scoring_methods_available'] = False
        
        # Test prediction integration
        try:
            test_score = self._score_lane_predictions('test_edge')
            validation_results['prediction_scoring_functional'] = isinstance(test_score, (int, float))
        except:
            validation_results['prediction_scoring_functional'] = False
        
        # Overall integration status
        validation_results['overall_integration_status'] = all([
            validation_results['optimization_stats_initialized'],
            validation_results['lane_change_tracking_initialized'],
            validation_results['scoring_methods_available'],
            validation_results['execution_methods_available'],
            validation_results['dynamic_weights_functional']
        ])
        
        # Log validation results
        status = "✓ PASSED" if validation_results['overall_integration_status'] else "✗ FAILED"
        self.logger.log_info(f"LANE_OPTIMIZATION_VALIDATION: Region {self.region_id} -> {status}")
        
        if not validation_results['overall_integration_status']:
            failed_components = [k for k, v in validation_results.items() if not v and k != 'overall_integration_status']
            self.logger.log_warning(f"LANE_OPTIMIZATION_VALIDATION_ISSUES: {failed_components}")
        
        return validation_results
    
    def _execute_lane_change(self, vehicle_id: str, target_lane: int) -> bool:
        """Execute lane change with intelligent timing and safety checks.
        
        Returns:
            bool: True if lane change was initiated successfully
        """
        try:
            # Verify vehicle is still active
            if vehicle_id not in traci.vehicle.getIDList():
                return False
                
            # Get current vehicle state
            current_edge = traci.vehicle.getRoadID(vehicle_id)
            if current_edge.startswith(':'):
                # Vehicle is in junction, defer lane change
                self.logger.log_info(f"LANE_CHANGE_DEFERRED: {vehicle_id} in junction, deferring lane change")
                return False
                
            current_lane = traci.vehicle.getLaneIndex(vehicle_id)
            vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
            
            # Check if already in target lane
            if current_lane == target_lane:
                return True
                
            # Safety check: don't change lanes if vehicle is very slow or stopped
            if vehicle_speed < 2.0:  # Less than 2 m/s
                self.logger.log_info(f"LANE_CHANGE_DEFERRED: {vehicle_id} too slow for lane change (speed: {vehicle_speed:.1f})")
                return False
                
            # Calculate appropriate change duration based on speed and lane difference
            lane_diff = abs(target_lane - current_lane)
            
            # Base duration: 3-5 seconds depending on complexity
            if lane_diff == 1:
                duration = max(3000, int(5000 / max(vehicle_speed, 1)))  # 3-5s for single change
            elif lane_diff == 2:
                duration = max(5000, int(8000 / max(vehicle_speed, 1)))  # 5-8s for double change
            else:
                duration = max(7000, int(10000 / max(vehicle_speed, 1))) # 7-10s for complex change
                
            # Cap duration at reasonable maximum
            duration = min(duration, 15000)  # Max 15 seconds
            
            # Execute the lane change
            traci.vehicle.changeLane(vehicle_id, target_lane, duration)
            
            # Log successful initiation
            self.logger.log_info(f"LANE_CHANGE_EXECUTED: {vehicle_id} changing from lane {current_lane} to {target_lane} "
                               f"(duration: {duration}ms, speed: {vehicle_speed:.1f} m/s)")
            
            # Track lane change performance
            self._track_lane_change_performance(vehicle_id, current_lane, target_lane, duration)
            
            return True
            
        except Exception as e:
            self.logger.log_error(f"LANE_CHANGE_FAILED: {vehicle_id} -> {e}")
            return False
    
    def _track_lane_change_performance(self, vehicle_id: str, from_lane: int, to_lane: int, duration: int):
        """Track lane change performance for optimization."""
        try:
            # Initialize tracking structure if needed
            if not hasattr(self, 'lane_change_stats'):
                self.lane_change_stats = {
                    'total_attempts': 0,
                    'successful_changes': 0,
                    'avg_duration': 0,
                    'recent_changes': deque(maxlen=50)
                }
            
            # Record this lane change
            change_record = {
                'vehicle_id': vehicle_id,
                'from_lane': from_lane,
                'to_lane': to_lane,
                'duration': duration,
                'timestamp': time.time()
            }
            
            self.lane_change_stats['total_attempts'] += 1
            self.lane_change_stats['recent_changes'].append(change_record)
            
            # Update average duration
            recent_durations = [change['duration'] for change in self.lane_change_stats['recent_changes']]
            if recent_durations:
                self.lane_change_stats['avg_duration'] = sum(recent_durations) / len(recent_durations)
                
        except Exception as e:
            self.logger.log_warning(f"LANE_CHANGE_TRACKING_ERROR: {e}")
    
    def _calculate_priority(self, vehicle_id: str, target_edge: str) -> int:
        """Calculate priority for vehicle (1-5, higher is more priority)."""
        priority = 3  # Default priority
        
        try:
            # Higher priority for vehicles going to less congested boundaries
            if target_edge in self.road_info:
                congestion = self.road_info[target_edge].get('congestion_level', 0)
                priority += (5 - congestion)  # Less congested = higher priority
            
            # Adjust based on route usage
            usage = self.planned_routes.get(target_edge, 0)
            if usage < 3:
                priority += 1  # Bonus for underused routes
            elif usage > 10:
                priority -= 1  # Penalty for overused routes
            
            return max(1, min(5, priority))
            
        except:
            return 3
    
    def execute_decisions(self, decisions: List[VehicleDecision]):
        """Execute routing decisions for vehicles."""
        executed_count = 0
        
        for decision in decisions:
            try:
                # Update route tracking
                self.vehicle_routes[decision.vehicle_id] = decision.route
                self.vehicle_targets[decision.vehicle_id] = decision.target_edge
                
                # Update planned route usage
                for edge in decision.route:
                    if edge in self.region_edges:
                        self.planned_routes[edge] += 1
                
                # Set vehicle route in SUMO
                traci.vehicle.setRoute(decision.vehicle_id, decision.route)
                
                # Execute intelligent lane assignment if specified
                if decision.lane_assignment is not None:
                    self._execute_lane_change(decision.vehicle_id, decision.lane_assignment)
                
                executed_count += 1
                
                self.logger.log_info(f"Executed decision for vehicle {decision.vehicle_id}: "
                                   f"Route to {decision.target_edge}")
                
            except Exception as e:
                self.logger.log_error(f"Failed to execute decision for vehicle "
                                    f"{decision.vehicle_id}: {e}")
        
        if executed_count > 0:
            self.logger.log_info(f"Regional Agent {self.region_id}: "
                               f"Executed {executed_count}/{len(decisions)} decisions")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for this regional agent."""
        success_rate = (self.successful_decisions / max(1, self.total_decisions)) * 100
        
        # Calculate average congestion in region
        total_congestion = 0
        edge_count = 0
        
        for edge_id in self.region_edges:
            if edge_id in self.road_info:
                total_congestion += self.road_info[edge_id].get('congestion_level', 0)
                edge_count += 1
        
        avg_congestion = total_congestion / max(1, edge_count)
        
        # Calculate route efficiency
        total_usage = sum(self.planned_routes.values())
        usage_efficiency = len(self.region_edges) / max(1, total_usage) if total_usage > 0 else 1.0
        
        # Get lane optimization metrics
        lane_metrics = self._get_lane_optimization_metrics()
        
        return {
            'region_id': self.region_id,
            'active_vehicles': len(self.region_vehicles),
            'total_decisions': self.total_decisions,
            'successful_decisions': self.successful_decisions,
            'success_rate': success_rate,
            'avg_congestion': avg_congestion,
            'boundary_utilization': len([e for e in self.outgoing_boundaries 
                                       if self.planned_routes.get(e, 0) > 0]),
            'route_efficiency': usage_efficiency,
            'lane_change_attempts': lane_metrics['total_attempts'],
            'lane_change_success_rate': lane_metrics['success_rate'],
            'avg_lane_change_duration': lane_metrics['avg_duration']
        }
    
    def _get_lane_optimization_metrics(self) -> Dict[str, float]:
        """Get comprehensive lane optimization performance metrics."""
        # Initialize default metrics
        default_metrics = {
            'total_attempts': 0,
            'success_rate': 0.0,
            'avg_duration': 0.0,
            'optimization_efficiency': 1.0,
            'recent_performance': 1.0
        }
        
        try:
            if not hasattr(self, 'lane_change_stats'):
                return default_metrics
                
            stats = self.lane_change_stats
            total_attempts = stats.get('total_attempts', 0)
            
            if total_attempts == 0:
                return default_metrics
                
            # Calculate success rate based on actual completions
            recent_changes = stats.get('recent_changes', [])
            successful_changes = len([c for c in recent_changes 
                                    if self._was_lane_change_successful(c)])
            
            success_rate = (successful_changes / len(recent_changes)) * 100 if recent_changes else 0
            
            # Calculate optimization efficiency
            efficiency = self._calculate_lane_optimization_efficiency(recent_changes)
            
            # Calculate recent performance trend
            recent_performance = self._calculate_recent_performance_trend(recent_changes)
            
            return {
                'total_attempts': total_attempts,
                'success_rate': success_rate,
                'avg_duration': stats.get('avg_duration', 0.0),
                'optimization_efficiency': efficiency,
                'recent_performance': recent_performance
            }
            
        except Exception as e:
            self.logger.log_error(f"LANE_METRICS_ERROR: {e}")
            return default_metrics
    
    def _was_lane_change_successful(self, change_record: Dict) -> bool:
        """Determine if a lane change was actually successful."""
        try:
            vehicle_id = change_record['vehicle_id']
            target_lane = change_record['to_lane']
            timestamp = change_record['timestamp']
            
            # If change was recent and vehicle still exists, check current lane
            current_time = time.time()
            if current_time - timestamp < 30:  # Within last 30 seconds
                if vehicle_id in traci.vehicle.getIDList():
                    try:
                        current_lane = traci.vehicle.getLaneIndex(vehicle_id)
                        return current_lane == target_lane
                    except:
                        pass
                        
            # For older changes, assume successful (vehicle completed journey)
            return True
            
        except Exception:
            return True  # Conservative assumption
    
    def _calculate_lane_optimization_efficiency(self, recent_changes: List[Dict]) -> float:
        """Calculate efficiency of lane optimization decisions."""
        if not recent_changes:
            return 1.0
            
        try:
            efficiency_scores = []
            
            for change in recent_changes[-20:]:  # Last 20 changes
                vehicle_id = change['vehicle_id']
                from_lane = change['from_lane']
                to_lane = change['to_lane']
                
                # Calculate efficiency based on lane change complexity vs benefit
                complexity = abs(to_lane - from_lane)
                
                # Simple efficiency heuristic: fewer lane changes = more efficient
                if complexity == 0:
                    efficiency = 1.0  # Perfect - no change needed
                elif complexity == 1:
                    efficiency = 0.8  # Good - single lane change
                elif complexity == 2:
                    efficiency = 0.6  # Moderate - double lane change
                else:
                    efficiency = 0.4  # Complex - multiple lane changes
                    
                efficiency_scores.append(efficiency)
                
            return sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 1.0
            
        except Exception:
            return 1.0
    
    def _calculate_recent_performance_trend(self, recent_changes: List[Dict]) -> float:
        """Calculate recent performance trend (improving/declining)."""
        if len(recent_changes) < 5:
            return 1.0
            
        try:
            # Compare recent performance to baseline
            recent_10 = recent_changes[-10:] if len(recent_changes) >= 10 else recent_changes
            older_10 = recent_changes[-20:-10] if len(recent_changes) >= 20 else recent_changes[:-10]
            
            if not older_10:
                return 1.0
                
            # Calculate average duration for both periods
            recent_avg_duration = sum(c['duration'] for c in recent_10) / len(recent_10)
            older_avg_duration = sum(c['duration'] for c in older_10) / len(older_10)
            
            # Shorter duration indicates better performance
            if older_avg_duration == 0:
                return 1.0
                
            performance_ratio = older_avg_duration / recent_avg_duration
            
            # Cap the ratio for reasonable bounds
            return max(0.5, min(2.0, performance_ratio))
            
        except Exception:
            return 1.0