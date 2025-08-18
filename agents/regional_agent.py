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
                 road_info: Dict, road_network: nx.DiGraph, llm_agent, logger):
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
        """
        self.region_id = region_id
        self.boundary_edges = boundary_edges
        self.edge_to_region = edge_to_region
        self.road_info = road_info
        self.road_network = road_network
        self.llm_agent = llm_agent
        self.logger = logger
        
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
        
        self.logger.log_info(f"Regional Agent {region_id} initialized: "
                           f"{len(self.region_edges)} edges, "
                           f"{len(self.boundary_connections)} boundary connections")
    
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
                return route_candidates[0] if route_candidates else None
                
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
            
            # Create answer options for LLM with validation
            answer_options = "/".join([
                str(candidate['boundary_edge']) for candidate in route_candidates
                if candidate.get('boundary_edge') is not None
            ])
            
            if not answer_options:
                self.logger.log_error(f"REGIONAL_ROUTING: No valid answer options for {vehicle_id}")
                return route_candidates[0] if route_candidates else None
            
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
                
                # Robust response validation and processing
                if decisions and len(decisions) > 0 and isinstance(decisions[0], dict) and 'answer' in decisions[0]:
                    llm_answer = decisions[0]['answer']
                    
                    # Enhanced LLM response processing: prioritize edge ID matching over index mapping
                    if llm_answer is None:
                        selected_boundary = route_candidates[0]['boundary_edge']
                        reasoning = 'LLM returned None answer, using fallback'
                    else:
                        # Convert to string for processing
                        answer_str = str(llm_answer).strip().strip('"\'') if llm_answer else ""
                        
                        # Strategy 1: Direct boundary edge ID match (primary method)
                        selected_boundary = None
                        for candidate in route_candidates:
                            if str(candidate['boundary_edge']) == answer_str:
                                selected_boundary = candidate['boundary_edge']
                                break
                        
                        # Strategy 2: If no direct match, try option index mapping
                        if selected_boundary is None and isinstance(llm_answer, (int, float)):
                            try:
                                option_index = int(llm_answer)
                                if 1 <= option_index <= len(route_candidates):
                                    selected_boundary = route_candidates[option_index - 1]['boundary_edge']
                            except (ValueError, IndexError, TypeError):
                                pass
                        
                        # Strategy 3: Try option index from string
                        if selected_boundary is None and answer_str.isdigit():
                            try:
                                option_index = int(answer_str)
                                if 1 <= option_index <= len(route_candidates):
                                    selected_boundary = route_candidates[option_index - 1]['boundary_edge']
                            except (ValueError, IndexError):
                                pass
                        
                        # Fallback: use best candidate
                        if selected_boundary is None:
                            selected_boundary = route_candidates[0]['boundary_edge']
                            self.logger.log_warning(f"REGIONAL_ROUTING: No match found for LLM answer '{answer_str}', using fallback")
                    
                    reasoning = decisions[0].get('summary', 'LLM regional route decision')
                    if not isinstance(reasoning, str):
                        reasoning = str(reasoning) if reasoning is not None else 'LLM regional route decision'
                else:
                    # LLM response validation failed
                    selected_boundary = route_candidates[0]['boundary_edge']
                    reasoning = 'Invalid LLM response structure, using fallback'
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
                if not selected_candidate:
                    selected_candidate = route_candidates[0]
                    reasoning = f'Invalid LLM selection {selected_boundary}, using best candidate'
                
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
                return route_candidates[0] if route_candidates else None
                
        except Exception as e:
            self.logger.log_error(f"LLM_REGIONAL_SELECT: Failed for {vehicle_id}: {e}")
            return route_candidates[0] if route_candidates else None
    
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
        
        if candidate_boundaries:
            observation_parts.append("Target boundary edges (to exit region):")
            for boundary in candidate_boundaries:
                if boundary in candidates:
                    info = candidates[boundary]
                    observation_parts.append(f"  {boundary}: congestion={info['congestion_level']}, "
                                           f"vehicles={info['vehicle_count']}, "
                                           f"planned_usage={info['planned_usage']}")
            observation_parts.append("")
        
        # Add alternative edges
        non_boundary_candidates = [edge for edge, info in candidates.items() 
                                 if not info.get('is_boundary', False)]
        if non_boundary_candidates:
            observation_parts.append("Alternative edges within region:")
            for edge in non_boundary_candidates[:3]:
                info = candidates[edge]
                observation_parts.append(f"  {edge}: congestion={info['congestion_level']}, "
                                       f"vehicles={info['vehicle_count']}")
            observation_parts.append("")
        
        # Add regional context
        observation_parts.append(f"Region {self.region_id} status:")
        observation_parts.append(f"  Active vehicles: {len(self.region_vehicles)}")
        observation_parts.append(f"  Total boundary edges: {len(self.outgoing_boundaries)}")
        
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
        """Assign lane for vehicle (simple implementation)."""
        try:
            # Get number of lanes on current edge
            current_edge = traci.vehicle.getRoadID(vehicle_id)
            num_lanes = traci.edge.getLaneNumber(current_edge)
            
            if num_lanes <= 1:
                return 0
            
            # Simple assignment: use rightmost available lane
            return min(1, num_lanes - 1)
            
        except:
            return None
    
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
                
                # Set lane assignment if specified
                if decision.lane_assignment is not None:
                    try:
                        traci.vehicle.changeLane(decision.vehicle_id, decision.lane_assignment, 500)
                    except:
                        pass  # Lane change might fail
                
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
        
        return {
            'region_id': self.region_id,
            'active_vehicles': len(self.region_vehicles),
            'total_decisions': self.total_decisions,
            'successful_decisions': self.successful_decisions,
            'success_rate': success_rate,
            'avg_congestion': avg_congestion,
            'boundary_utilization': len([e for e in self.outgoing_boundaries 
                                       if self.planned_routes.get(e, 0) > 0]),
            'route_efficiency': usage_efficiency
        }