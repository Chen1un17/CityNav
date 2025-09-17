"""
Traffic Agent for Multi-Agent Traffic Control System

Handles macro route planning between regions, inter-regional coordination,
and provides recommendations to Regional Agents based on global traffic state.
"""

import traci
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, deque
from dataclasses import dataclass

from agents.regional_agent import RegionalRecommendation
from agents.prediction_engine import PredictionEngine


@dataclass
class MacroRoute:
    """Macro route between regions."""
    vehicle_id: str
    start_region: int
    end_region: int
    region_sequence: List[int]
    boundary_edges: List[str]
    estimated_travel_time: float
    creation_time: float
    last_update: float


@dataclass
class GlobalTrafficState:
    """Global traffic state information."""
    timestamp: float
    regional_congestion: Dict[int, float]
    boundary_congestion: Dict[str, float]
    total_vehicles: int
    completed_vehicles: int
    avg_travel_time: float


@dataclass
class VehicleMacroRequest:
    """Request for macro route planning."""
    vehicle_id: str
    start_region: int
    end_region: int
    current_time: float


class TrafficAgent:
    """
    Traffic Agent for inter-regional coordination and macro planning.
    
    Responsible for:
    - Planning macro routes between regions
    - Monitoring global traffic state
    - Providing recommendations to Regional Agents
    - Managing boundary edge traffic flow
    - Coordinating vehicles across regions
    """
    
    def __init__(self, boundary_edges: List[Dict], edge_to_region: Dict,
                 road_info: Dict, num_regions: int, llm_agent, logger, 
                 prediction_engine: PredictionEngine, raw_llm_agent=None):
        """
        Initialize Traffic Agent.
        
        Args:
            boundary_edges: List of boundary edge information
            edge_to_region: Mapping of edges to region IDs
            road_info: Road information dictionary
            num_regions: Total number of regions
            llm_agent: Language model agent for decision making
            logger: Agent logger instance
            prediction_engine: Prediction engine for traffic forecasting
            raw_llm_agent: Optional raw LLM instance for macro planning (no concurrency wrapper)
        """
        self.boundary_edges = boundary_edges
        self.edge_to_region = edge_to_region
        self.road_info = road_info
        self.num_regions = num_regions
        self.llm_agent = llm_agent
        self.raw_llm_agent = raw_llm_agent or llm_agent  # Use raw LLM for macro planning if available
        self.logger = logger
        self.prediction_engine = prediction_engine
        
        # Build region connectivity graph
        self._build_region_graph()
        
        # Vehicle tracking
        self.vehicle_macro_routes: Dict[str, MacroRoute] = {}
        self.region_vehicle_counts: Dict[int, int] = defaultdict(int)
        self.boundary_traffic_flow: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        
        # Global state tracking
        self.global_state_history: deque = deque(maxlen=100)
        self.regional_recommendations: Dict[int, RegionalRecommendation] = {}
        
        # Performance metrics
        self.total_macro_routes = 0
        self.successful_macro_routes = 0
        self.avg_macro_travel_time = 0.0
        self.boundary_utilization: Dict[str, float] = {}
        
        # Region traversal statistics for autonomous vehicles (real ATT basis)
        # Structure: {region_id: { 'total_time': float, 'count': int }}
        self.region_travel_stats: Dict[int, Dict[str, float]] = defaultdict(lambda: {'total_time': 0.0, 'count': 0})
        
        # Coordination parameters
        self.congestion_threshold = 3.0  # Congestion level threshold
        self.load_balancing_factor = 0.3  # How aggressively to load balance
        self.prediction_horizon = 1800  # 30 minutes prediction horizon
        
        self.logger.log_info(f"Traffic Agent initialized: "
                           f"{num_regions} regions, "
                           f"{len(boundary_edges)} boundary edges, "
                           f"{self.region_graph.number_of_edges()} region connections")
    
    def _build_region_graph(self):
        """Build graph representing connections between regions."""
        self.region_graph = nx.DiGraph()
        
        # Add all regions as nodes
        for region_id in range(self.num_regions):
            self.region_graph.add_node(region_id)
        
        # Add edges between connected regions
        self.region_connections: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        
        for boundary_info in self.boundary_edges:
            from_region = boundary_info['from_region']
            to_region = boundary_info['to_region']
            edge_id = boundary_info['edge_id']
            
            # Add edge to region graph if not exists
            if not self.region_graph.has_edge(from_region, to_region):
                self.region_graph.add_edge(from_region, to_region, weight=1.0, edges=[])
            
            # Store edge information
            self.region_connections[(from_region, to_region)].append(edge_id)
            self.region_graph[from_region][to_region]['edges'].append(edge_id)
        
        self.logger.log_info(f"Region graph built: {self.region_graph.number_of_nodes()} nodes, "
                           f"{self.region_graph.number_of_edges()} connections")
    
    def update_global_traffic_state(self, current_time: float):
        """Update global traffic state information."""
        try:
            # Calculate regional congestion levels
            regional_congestion = {}
            
            for region_id in range(self.num_regions):
                region_edges = [
                    edge_id for edge_id, region in self.edge_to_region.items()
                    if region == region_id
                ]
                
                total_congestion = 0
                edge_count = 0
                
                for edge_id in region_edges:
                    if edge_id in self.road_info:
                        total_congestion += self.road_info[edge_id].get('congestion_level', 0)
                        edge_count += 1
                
                avg_congestion = total_congestion / max(1, edge_count)
                regional_congestion[region_id] = avg_congestion
            
            # Calculate boundary edge congestion
            boundary_congestion = {}
            for boundary_info in self.boundary_edges:
                edge_id = boundary_info['edge_id']
                if edge_id in self.road_info:
                    boundary_congestion[edge_id] = self.road_info[edge_id].get('congestion_level', 0)
            
            # Get vehicle counts
            total_vehicles = len(traci.vehicle.getIDList())
            completed_vehicles = len(traci.simulation.getArrivedIDList())
            
            # Calculate average travel time (simplified)
            avg_travel_time = self._calculate_avg_travel_time()
            
            # Create global state
            global_state = GlobalTrafficState(
                timestamp=current_time,
                regional_congestion=regional_congestion,
                boundary_congestion=boundary_congestion,
                total_vehicles=total_vehicles,
                completed_vehicles=completed_vehicles,
                avg_travel_time=avg_travel_time
            )
            
            self.global_state_history.append(global_state)
            
            # Update region graph weights based on congestion
            self._update_region_graph_weights(regional_congestion, boundary_congestion)
            
            # Update boundary traffic flow tracking
            self._update_boundary_traffic_flow(current_time)
            
            self._emit_hotspot_throttle_guidance(current_time)
            
        except Exception as e:
            self.logger.log_error(f"Traffic Agent global state update failed: {e}")
    
    def _emit_hotspot_throttle_guidance(self, current_time: float) -> None:
        """Issue temporary global guidance to avoid hotspot edges and throttle inflow."""
        try:
            if not hasattr(self, 'parent_env'):
                return
            env = self.parent_env
            # Require recent hamper stats
            if not hasattr(env, 'exit_hamper_counts'):
                return
            # Select top hotspot edges above minimal threshold
            items = sorted(env.exit_hamper_counts.items(), key=lambda x: x[1], reverse=True)[:8]
            hotspot_edges = [e for e, c in items if c >= 3]
            if not hotspot_edges:
                return
            # Update global guidance with short TTL
            try:
                env.global_macro_guidance['data'] = {
                    'avoid_edges': hotspot_edges,
                    'message': 'Hotspot throttle: avoid high-hamper edges'
                }
                env.global_macro_guidance['expire_at'] = float(traci.simulation.getTime() + 240.0)
            except Exception:
                pass
        except Exception:
            pass

    def collect_regional_congestion_report(self, current_time: float, regional_status_reports: Dict = None) -> Dict:
        """
        Collect real-time congestion report from all regions with coordination data.
        
        This method provides the traffic agent with comprehensive regional status
        for LLM decision making and coordination.
        
        Args:
            current_time: Current simulation time
            regional_status_reports: Optional dict of regional status reports from regional agents
            
        Returns:
            Dictionary containing detailed regional congestion and coordination information
        """
        try:
            regional_report = {
                'timestamp': current_time,
                'regions': {},
                'boundaries': {},
                'system_overview': {},
                'trends': {},
                'coordination_data': {}
            }
            
            # Collect regional congestion data
            for region_id in range(self.num_regions):
                # Get all edges in this region
                region_edges = [
                    edge_id for edge_id, region in self.edge_to_region.items()
                    if region == region_id
                ]
                
                region_metrics = self._calculate_regional_metrics(region_edges, region_id)
                regional_report['regions'][region_id] = region_metrics
            
            # Collect boundary edge status
            for boundary_info in self.boundary_edges:
                edge_id = boundary_info['edge_id']
                boundary_metrics = self._calculate_boundary_metrics(edge_id, boundary_info)
                regional_report['boundaries'][edge_id] = boundary_metrics
            
            # System overview
            current_state = self.global_state_history[-1] if self.global_state_history else None
            if current_state:
                regional_report['system_overview'] = {
                    'total_vehicles': current_state.total_vehicles,
                    'avg_travel_time': current_state.avg_travel_time,
                    'system_congestion': sum(current_state.regional_congestion.values()) / max(1, len(current_state.regional_congestion)),
                    'boundary_congestion': sum(current_state.boundary_congestion.values()) / max(1, len(current_state.boundary_congestion))
                }
            
            # Calculate trends if we have historical data
            if len(self.global_state_history) >= 2:
                regional_report['trends'] = self._calculate_congestion_trends()
            
            # Add coordination data from regional status reports
            if regional_status_reports:
                coordination_summary = self._analyze_coordination_data(regional_status_reports)
                regional_report['coordination_data'] = coordination_summary
            
            self.logger.log_info(f"REGIONAL_REPORT: Collected congestion data for {len(regional_report['regions'])} regions with coordination data")
            
            return regional_report
            
        except Exception as e:
            self.logger.log_error(f"REGIONAL_REPORT: Failed to collect regional congestion report: {e}")
            return {'timestamp': current_time, 'regions': {}, 'boundaries': {}, 'system_overview': {}, 'trends': {}, 'coordination_data': {}}

    def record_region_dwell_time(self, region_id: int, dwell_time: float):
        """Record a single traversal dwell time for a region (autonomous vehicles only).
        Accumulates total time and count for ATT calculation.
        """
        try:
            if region_id is None or region_id < 0:
                return
            if dwell_time is None or dwell_time <= 0:
                return
            stats = self.region_travel_stats[region_id]
            stats['total_time'] += float(dwell_time)
            stats['count'] += 1
        except Exception as e:
            self.logger.log_error(f"REGION_DWELL_RECORD_ERROR: region {region_id}: {e}")

    def get_region_att(self, region_id: int) -> float:
        """Get average traversal time (ATT) for a region based on recorded dwell times.
        Returns a neutral default (300s) if no data is available yet.
        """
        stats = self.region_travel_stats.get(region_id)
        if not stats or stats.get('count', 0) <= 0:
            return 300.0
        return stats['total_time'] / max(1, int(stats['count']))
    
    def _calculate_regional_metrics(self, region_edges: List[str], region_id: int) -> Dict:
        """Calculate comprehensive metrics for a region."""
        try:
            metrics = {
                'region_id': region_id,
                'total_edges': len(region_edges),
                'congestion_level': 0.0,
                'vehicle_count': 0,
                'capacity_utilization': 0.0,
                'avg_speed': 0.0,
                'status': 'normal'
            }
            
            total_congestion = 0.0
            total_vehicles = 0
            total_capacity = 0
            total_speed = 0.0
            valid_edges = 0
            
            for edge_id in region_edges:
                if edge_id in self.road_info:
                    edge_data = self.road_info[edge_id]
                    
                    congestion = edge_data.get('congestion_level', 0)
                    vehicles = edge_data.get('vehicle_num', 0)
                    speed = edge_data.get('avg_speed', 0)
                    
                    total_congestion += congestion
                    total_vehicles += vehicles
                    total_speed += speed
                    valid_edges += 1
                    
                    # Estimate capacity
                    road_len = edge_data.get('road_len', 100)
                    lane_num = edge_data.get('lane_num', 1)
                    estimated_capacity = (road_len * lane_num) / 8.0  # Rough estimate
                    total_capacity += estimated_capacity
            
            if valid_edges > 0:
                metrics['congestion_level'] = total_congestion / valid_edges
                metrics['avg_speed'] = total_speed / valid_edges
                metrics['vehicle_count'] = total_vehicles
                
                if total_capacity > 0:
                    metrics['capacity_utilization'] = min(1.0, total_vehicles / total_capacity)
                
                # Determine status
                if metrics['congestion_level'] > 4.0:
                    metrics['status'] = 'heavily_congested'
                elif metrics['congestion_level'] > 2.0:
                    metrics['status'] = 'congested'
                elif metrics['congestion_level'] > 1.0:
                    metrics['status'] = 'moderate'
                else:
                    metrics['status'] = 'normal'
            
            return metrics
            
        except Exception as e:
            self.logger.log_error(f"REGIONAL_METRICS: Failed for region {region_id}: {e}")
            return {'region_id': region_id, 'total_edges': 0, 'congestion_level': 0.0, 
                   'vehicle_count': 0, 'capacity_utilization': 0.0, 'avg_speed': 0.0, 'status': 'unknown'}
    
    def _calculate_boundary_metrics(self, edge_id: str, boundary_info: Dict) -> Dict:
        """Calculate enhanced metrics for a boundary edge including coordination data."""
        try:
            metrics = {
                'edge_id': edge_id,
                'from_region': boundary_info['from_region'],
                'to_region': boundary_info['to_region'],
                'congestion_level': 0.0,
                'vehicle_count': 0,
                'utilization': 0.0,
                'flow_rate': 0.0,
                'status': 'normal',
                'active_vehicles': 0,
                'capacity_warning': False,
                'output_availability': 1.0,
                'av_targets': 0
            }
            
            if edge_id in self.road_info:
                edge_data = self.road_info[edge_id]
                
                congestion = edge_data.get('congestion_level', 0)
                vehicles = edge_data.get('vehicle_num', 0)
                
                metrics['congestion_level'] = congestion
                metrics['vehicle_count'] = vehicles
                metrics['active_vehicles'] = vehicles
                
                # Calculate utilization
                road_len = edge_data.get('road_len', 100)
                lane_num = edge_data.get('lane_num', 1)
                capacity = (road_len * lane_num) / 8.0
                
                if capacity > 0:
                    utilization = min(1.0, vehicles / capacity)
                    metrics['utilization'] = utilization
                
                # Enhanced capacity warning based on congestion and utilization
                metrics['capacity_warning'] = congestion > 3.0 or metrics['utilization'] > 0.8
                
                # Output availability (higher is better, inverse of congestion)
                metrics['output_availability'] = max(0.1, min(1.0, 1.0 - congestion / 5.0))
                
                # Estimate autonomous vehicles targeting this edge
                # Based on vehicles planned to cross this boundary
                base_targets = max(1, int(vehicles * 0.3))
                
                # Check if vehicles are planned to use this boundary from macro routes
                planned_vehicles = 0
                for vehicle_id, macro_route in self.vehicle_macro_routes.items():
                    if edge_id in macro_route.boundary_edges:
                        planned_vehicles += 1
                
                metrics['av_targets'] = max(base_targets, planned_vehicles)
                
                # Calculate flow rate from historical data
                if edge_id in self.boundary_traffic_flow:
                    flow_data = list(self.boundary_traffic_flow[edge_id])
                    if len(flow_data) >= 2:
                        recent_flow = flow_data[-5:] if len(flow_data) >= 5 else flow_data
                        avg_vehicles = sum(flow[1] for flow in recent_flow) / len(recent_flow)
                        metrics['flow_rate'] = avg_vehicles * 60  # Convert to veh/h
                
                # Enhanced status determination
                if congestion > 4.0 or metrics['utilization'] > 0.9:
                    metrics['status'] = 'blocked'
                elif congestion > 3.0 or metrics['utilization'] > 0.8:
                    metrics['status'] = 'saturated'
                elif congestion > 2.0 or metrics['utilization'] > 0.6:
                    metrics['status'] = 'congested'
                elif congestion > 1.0 or metrics['utilization'] > 0.4:
                    metrics['status'] = 'busy'
                else:
                    metrics['status'] = 'normal'
            
            return metrics
            
        except Exception as e:
            self.logger.log_error(f"BOUNDARY_METRICS: Failed for edge {edge_id}: {e}")
            return {'edge_id': edge_id, 'from_region': boundary_info.get('from_region', -1), 
                   'to_region': boundary_info.get('to_region', -1), 'congestion_level': 0.0, 
                   'vehicle_count': 0, 'utilization': 0.0, 'flow_rate': 0.0, 'status': 'unknown',
                   'active_vehicles': 0, 'capacity_warning': False, 'output_availability': 0.0, 'av_targets': 0}
    
    def _calculate_congestion_trends(self) -> Dict:
        """Calculate congestion trends from historical data."""
        try:
            if len(self.global_state_history) < 2:
                return {}
            
            current_state = self.global_state_history[-1]
            previous_state = self.global_state_history[-2]
            
            trends = {
                'regional_trends': {},
                'boundary_trends': {},
                'system_trend': 'stable'
            }
            
            # Regional trends
            for region_id in current_state.regional_congestion:
                current_congestion = current_state.regional_congestion[region_id]
                previous_congestion = previous_state.regional_congestion.get(region_id, current_congestion)
                
                change = current_congestion - previous_congestion
                
                if change > 0.5:
                    trend = 'increasing'
                elif change < -0.5:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
                
                trends['regional_trends'][region_id] = {
                    'trend': trend,
                    'change': change,
                    'current': current_congestion,
                    'previous': previous_congestion
                }
            
            # Boundary trends
            for edge_id in current_state.boundary_congestion:
                current_congestion = current_state.boundary_congestion[edge_id]
                previous_congestion = previous_state.boundary_congestion.get(edge_id, current_congestion)
                
                change = current_congestion - previous_congestion
                
                if change > 0.3:
                    trend = 'increasing'
                elif change < -0.3:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
                
                trends['boundary_trends'][edge_id] = {
                    'trend': trend,
                    'change': change
                }
            
            # System-wide trend
            current_avg = sum(current_state.regional_congestion.values()) / max(1, len(current_state.regional_congestion))
            previous_avg = sum(previous_state.regional_congestion.values()) / max(1, len(previous_state.regional_congestion))
            
            system_change = current_avg - previous_avg
            
            if system_change > 0.3:
                trends['system_trend'] = 'deteriorating'
            elif system_change < -0.3:
                trends['system_trend'] = 'improving'
            else:
                trends['system_trend'] = 'stable'
            
            return trends
            
        except Exception as e:
            self.logger.log_error(f"CONGESTION_TRENDS: Failed to calculate trends: {e}")
            return {}
    
    def _analyze_coordination_data(self, regional_status_reports: Dict) -> Dict:
        """Analyze coordination data from regional status reports for efficient macro planning.
        
        Args:
            regional_status_reports: Dict mapping region_id to status report
            
        Returns:
            Coordination analysis summary for LLM decision making
        """
        try:
            # Count vehicle flow intentions
            vehicle_flow_matrix = {}
            overloaded_regions = []
            available_regions = []
            total_active_vehicles = 0
            
            for region_id, report in regional_status_reports.items():
                active_vehicles = report.get('active_vehicles', {})
                capacity_status = report.get('capacity_status', {})
                
                # Count active vehicles
                current_count = active_vehicles.get('current_count', 0)
                total_active_vehicles += current_count
                
                # Analyze next targets for flow matrix
                next_targets = active_vehicles.get('next_targets', {})
                for vehicle_id, target_region in next_targets.items():
                    if target_region not in vehicle_flow_matrix:
                        vehicle_flow_matrix[target_region] = 0
                    vehicle_flow_matrix[target_region] += 1
                
                # Identify overloaded/available regions
                if capacity_status.get('congestion_warning', False):
                    overloaded_regions.append(region_id)
                
                # Check boundary availability
                boundary_availability = capacity_status.get('outgoing_boundary_availability', {})
                available_capacity = sum(boundary_availability.values())
                if available_capacity > 2:  # Threshold for availability
                    available_regions.append(region_id)
            
            # Calculate load balance score using existing method
            region_loads = [regional_status_reports[rid].get('capacity_status', {}).get('current_load', 0) 
                           for rid in regional_status_reports.keys()]
            
            load_balance_score = 1.0
            if region_loads and max(region_loads) > 0:
                load_balance_score = 1 - (max(region_loads) - min(region_loads)) / max(region_loads)
            
            return {
                'vehicle_flow_matrix': vehicle_flow_matrix,
                'overloaded_regions': overloaded_regions[:5],  # Limit to top 5 for efficiency
                'available_regions': available_regions[:8],     # Limit to top 8 for efficiency
                'load_balance_score': round(load_balance_score, 3),
                'total_active_vehicles': total_active_vehicles,
                'coordination_opportunities': len(available_regions) > 0 and len(overloaded_regions) > 0
            }
            
        except Exception as e:
            self.logger.log_error(f"COORDINATION_ANALYSIS: Failed to analyze coordination data: {e}")
            return {
                'vehicle_flow_matrix': {},
                'overloaded_regions': [],
                'available_regions': [],
                'load_balance_score': 1.0,
                'total_active_vehicles': 0,
                'coordination_opportunities': False
            }
    
    def _calculate_avg_travel_time(self) -> float:
        """Calculate system average travel time using recorded region traversal ATT.
        Uses aggregated per-region dwell stats across autonomous vehicles.
        """
        try:
            total_time = 0.0
            total_count = 0
            for stats in self.region_travel_stats.values():
                total_time += stats.get('total_time', 0.0)
                total_count += int(stats.get('count', 0))
            if total_count <= 0:
                return 0.0
            return total_time / total_count
        except Exception:
            return 0.0
    
    def _update_region_graph_weights(self, regional_congestion: Dict[int, float],
                                   boundary_congestion: Dict[str, float]):
        """Update region graph edge weights based on congestion."""
        for from_region, to_region in self.region_graph.edges():
            # Calculate weight based on target region congestion and boundary congestion
            target_congestion = regional_congestion.get(to_region, 0)
            
            # Get boundary edges for this connection
            boundary_edges = self.region_connections.get((from_region, to_region), [])
            avg_boundary_congestion = 0
            
            if boundary_edges:
                total_boundary_congestion = sum(
                    boundary_congestion.get(edge, 0) for edge in boundary_edges
                )
                avg_boundary_congestion = total_boundary_congestion / len(boundary_edges)
            
            # Weight combines target region and boundary congestion
            weight = 1.0 + (target_congestion * 0.5) + (avg_boundary_congestion * 0.3)
            
            self.region_graph[from_region][to_region]['weight'] = weight
    
    def _update_boundary_traffic_flow(self, current_time: float):
        """Update boundary traffic flow tracking."""
        for boundary_info in self.boundary_edges:
            edge_id = boundary_info['edge_id']
            
            if edge_id in self.road_info:
                vehicle_count = self.road_info[edge_id].get('vehicle_num', 0)
                self.boundary_traffic_flow[edge_id].append((current_time, vehicle_count))
                
                # Calculate utilization
                capacity = self.road_info[edge_id].get('road_len', 100) / 8.0  # Approximate capacity
                utilization = vehicle_count / max(1, capacity)
                self.boundary_utilization[edge_id] = min(1.0, utilization)
    
    def plan_single_macro_route(self, vehicle_id: str, start_region: int, end_region: int, current_time: float, coordination_data: Dict = None) -> Optional[MacroRoute]:
        """
        Plan macro route for a single vehicle.
        
        Args:
            vehicle_id: ID of vehicle needing planning
            start_region: Current region
            end_region: Target region
            current_time: Current simulation time
            coordination_data: Optional coordination data
            
        Returns:
            MacroRoute object or None if failed
        """
        try:
            request = {
                'vehicle_id': vehicle_id,
                'start_region': start_region,
                'end_region': end_region
            }
            
            # Use existing batch method for single vehicle
            routes = self.batch_macro_planning([request], current_time, coordination_data)
            return routes[0] if routes else None
            
        except Exception as e:
            self.logger.log_error(f"SINGLE_MACRO_PLANNING: Failed for {vehicle_id}: {e}")
            return None

    def batch_macro_planning(self, requests: List[Dict], current_time: float, coordination_data: Dict = None) -> List[MacroRoute]:
        """
        Plan macro routes for multiple vehicles simultaneously with coordination.
        
        Args:
            requests: List of vehicle route requests
            current_time: Current simulation time
            coordination_data: Optional coordination data from regional status reports
            
        Returns:
            List of macro routes
        """
        try:
            if not requests:
                return []
            
            # Prepare context for LLM-based planning with coordination data
            planning_context = self._prepare_macro_planning_context(requests, current_time, coordination_data)
            
            # Generate the actual LLM input texts to log what LLM really sees
            llm_input_texts = self._prepare_llm_input_texts(planning_context)
            combined_llm_input = "\n\n=== BATCH REQUEST ===\n\n".join(llm_input_texts)
            
            # Use LLM for intelligent route planning with coordination
            call_id = self.logger.log_llm_call_start(
                "TrafficAgent", "coordinated_macro_planning", len(combined_llm_input),
                "decision", "", combined_llm_input
            )
            
            macro_routes = []
            try:
                llm_routes = self._make_llm_macro_decisions(planning_context, current_time)
                
                # Process LLM decisions
                macro_routes = self._process_llm_macro_decisions(
                    llm_routes, requests, current_time
                )
                
                decision_summary = f"Planned {len(macro_routes)} coordinated macro routes"
                self.logger.log_llm_call_end(
                    call_id, True, decision_summary, len(str(planning_context))
                )
                
                self.successful_macro_routes += len(macro_routes)
                
            except Exception as e:
                self.logger.log_llm_call_end(
                    call_id, False, "Coordinated macro planning failed", 
                    len(str(planning_context)), str(e)
                )
                
                # Fallback to heuristic planning
                macro_routes = self._make_heuristic_macro_plans(requests, current_time)
            
            # Store macro routes
            for route in macro_routes:
                self.vehicle_macro_routes[route.vehicle_id] = route
            
            self.total_macro_routes += len(requests)
            
            return macro_routes
            
        except Exception as e:
            self.logger.log_error(f"Traffic Agent coordinated batch macro planning failed: {e}")
            return []
    
    def _prepare_macro_planning_context(self, requests: List[Dict], 
                                      current_time: float, coordination_data: Dict = None) -> Dict[str, Any]:
        """Prepare context for LLM-based macro planning with coordination data."""
        # Get current global state
        current_state = self.global_state_history[-1] if self.global_state_history else None
        
        # Extract relevant regions for optimization
        relevant_regions = set()
        if requests:
            for req in requests:
                relevant_regions.add(req['start_region'])
                relevant_regions.add(req['end_region'])
        
        # Add top congested regions
        full_regional_congestion = current_state.regional_congestion if current_state else {}
        full_boundary_congestion = current_state.boundary_congestion if current_state else {}
        
        if full_regional_congestion:
            top_congested_regions = sorted(full_regional_congestion.items(), 
                                         key=lambda x: x[1], reverse=True)[:5]
            for region_id, _ in top_congested_regions:
                relevant_regions.add(region_id)
        
        # Limit to max 10 regions
        relevant_regions = list(relevant_regions)[:10]
        
        # Create compact data structures (will be overridden by real performance if available)
        compact_regional_congestion = {
            region_id: full_regional_congestion.get(region_id, 0) 
            for region_id in relevant_regions
        }

        # Try to use real-time performance metrics from logger to populate regional congestion
        performance_regional_metrics = {}
        system_avg_travel_time = 0.0
        try:
            # Prefer in-memory history for freshest data
            if hasattr(self.logger, 'performance_history') and self.logger.performance_history:
                last_perf = self.logger.performance_history[-1]
                # last_perf is PerformanceMetrics dataclass
                performance_regional_metrics = getattr(last_perf, 'regional_metrics', {}) or {}
                system_avg_travel_time = getattr(last_perf, 'average_travel_time', 0.0) or 0.0
        except Exception:
            pass

        # Override congestion using real avg_congestion if available from performance logs
        if performance_regional_metrics:
            for region_id in list(compact_regional_congestion.keys()):
                metrics = performance_regional_metrics.get(region_id, {})
                if isinstance(metrics, dict) and 'avg_congestion' in metrics:
                    compact_regional_congestion[region_id] = metrics.get('avg_congestion', compact_regional_congestion.get(region_id, 0))

        # If coordination_data is a full regional report, merge its real metrics
        if coordination_data and isinstance(coordination_data, dict) and (
            'regions' in coordination_data or 'boundaries' in coordination_data or 'system_overview' in coordination_data
        ):
            try:
                # Derive regional congestion from report
                report_regions = coordination_data.get('regions', {})
                if isinstance(report_regions, dict) and report_regions:
                    for region_id_str, metrics in report_regions.items():
                        try:
                            region_id_int = int(region_id_str)
                        except Exception:
                            region_id_int = region_id_str
                        if region_id_int in relevant_regions and isinstance(metrics, dict):
                            cong_val = metrics.get('congestion_level', None)
                            if cong_val is not None:
                                compact_regional_congestion[region_id_int] = cong_val
                            # Merge into regional metrics to expose active_vehicles etc.
                            perf_metrics_for_region = performance_regional_metrics.get(region_id_int, {}) if isinstance(performance_regional_metrics, dict) else {}
                            merged_metrics = {
                                **perf_metrics_for_region,
                                'avg_congestion': cong_val if cong_val is not None else perf_metrics_for_region.get('avg_congestion', 0.0),
                                'active_vehicles': metrics.get('vehicle_count', perf_metrics_for_region.get('active_vehicles', 0)),
                                'capacity_utilization': metrics.get('capacity_utilization', perf_metrics_for_region.get('capacity_utilization', 0.0))
                            }
                            if isinstance(performance_regional_metrics, dict):
                                performance_regional_metrics[region_id_int] = merged_metrics
                
                # Derive boundary congestion from report
                report_boundaries = coordination_data.get('boundaries', {})
                compact_boundary_congestion_from_report = {}
                if isinstance(report_boundaries, dict) and report_boundaries:
                    for edge_id, bmetrics in report_boundaries.items():
                        if isinstance(bmetrics, dict):
                            cong_val = bmetrics.get('congestion_level', None)
                            if cong_val is not None:
                                compact_boundary_congestion_from_report[edge_id] = cong_val
                
                # System ATT from report
                system_avg_travel_time = coordination_data.get('system_overview', {}).get('avg_travel_time', system_avg_travel_time)
                
                # Build coordination summary similar to _analyze_coordination_data
                overloaded_regions = []
                available_regions = []
                region_loads = []
                if isinstance(report_regions, dict):
                    for rid_str, metrics in report_regions.items():
                        try:
                            rid = int(rid_str)
                        except Exception:
                            rid = rid_str
                        if not isinstance(metrics, dict):
                            continue
                        cong = metrics.get('congestion_level', 0.0)
                        status = metrics.get('status', 'normal')
                        cap_util = metrics.get('capacity_utilization', 0.0)
                        region_loads.append(cap_util if cap_util is not None else 0.0)
                        if cong is not None and cong > 3.0 or status in ('congested', 'heavily_congested'):
                            overloaded_regions.append(rid)
                        elif cong is not None and cong < 2.0 and cap_util is not None and cap_util < 0.6:
                            available_regions.append(rid)
                load_balance_score = 1.0
                if region_loads and max(region_loads) > 0:
                    load_balance_score = 1 - (max(region_loads) - min(region_loads)) / max(region_loads)
                coordination_summary = {
                    'overloaded_regions': overloaded_regions[:5],
                    'available_regions': available_regions[:8],
                    'load_balance_score': round(load_balance_score, 3)
                }
                
                # Prefer report-based boundary congestion if present
                if compact_boundary_congestion_from_report:
                    compact_boundary_congestion = compact_boundary_congestion_from_report
                
                # Prefer report-derived coordination summary
                coordination_data = coordination_summary
            except Exception:
                # If merging fails, keep existing data
                pass
        
        # Only keep top 10 most congested boundaries
        compact_boundary_congestion = {}
        if full_boundary_congestion:
            top_congested_boundaries = sorted(full_boundary_congestion.items(), 
                                            key=lambda x: x[1], reverse=True)[:10]
            compact_boundary_congestion = dict(top_congested_boundaries)
        
        # Compact performance metrics (only essential metrics)
        # Use logger counters to compute real LLM success rate matching performance logs
        try:
            llm_success_rate = (self.logger.successful_llm_calls / max(1, self.logger.total_llm_calls)) * 100.0
        except Exception:
            llm_success_rate = 0.0

        traffic_perf = self.get_performance_metrics()
        compact_performance = {
            'success_rate': llm_success_rate,
            'regional_balance': traffic_perf.get('regional_balance', 0.0),
            'active_macro_routes': traffic_perf.get('active_macro_routes', 0)
        }
        
        # Gather stuck hotspots from logger/environment if available (exposed via performance metrics placeholder)
        stuck_hotspots = {}
        try:
            # Use vehicle logger performance file if embedded; otherwise rely on regional reports later
            if hasattr(self, 'stuck_edge_blacklist') and isinstance(self.stuck_edge_blacklist, dict):
                stuck_hotspots = self.stuck_edge_blacklist
        except Exception:
            pass

        # Optimize: Only include essential data to reduce context size
        context = {
            'current_time': current_time,
            'num_regions': min(self.num_regions, 20),  # Limit reported regions
            'requests': requests[:10],  # Max 10 requests
            'regional_congestion': compact_regional_congestion,
            'boundary_congestion': compact_boundary_congestion,
            'performance_metrics': compact_performance,
            'coordination_data': coordination_data if coordination_data else {},
            'stuck_hotspots': stuck_hotspots,
            # expose real performance regional metrics and system ATT for downstream observation building
            'regional_metrics': performance_regional_metrics,
            'system_avg_travel_time': system_avg_travel_time
        }

        # Attach current global macro guidance (if environment provided)
        try:
            if hasattr(self, 'parent_env') and hasattr(self.parent_env, '_get_current_global_macro_guidance'):
                gg = self.parent_env._get_current_global_macro_guidance()
                if gg:
                    context['global_guidance'] = gg
        except Exception:
            pass
        
        # Optional: Add very limited congestion forecast only for most relevant boundaries
        if requests and len(requests) <= 5:  # Only for small batches
            relevant_boundaries = []
            for boundary_info in self.boundary_edges[:5]:  # Only check first 5 boundaries
                if (boundary_info['from_region'] in relevant_regions or 
                    boundary_info['to_region'] in relevant_regions):
                    relevant_boundaries.append(boundary_info['edge_id'])
                    if len(relevant_boundaries) >= 3:  # Max 3 boundaries
                        break
            
            if relevant_boundaries:
                try:
                    congestion_forecast = self.prediction_engine.get_congestion_forecast(
                        relevant_boundaries, min(600, self.prediction_horizon)  # Max 10 minutes
                    )
                    # Further limit forecast data
                    limited_forecast = {}
                    for edge_id, forecast_data in congestion_forecast.items():
                        if isinstance(forecast_data, list) and len(forecast_data) > 0:
                            limited_forecast[edge_id] = forecast_data[:2]  # Only first 2 data points
                        else:
                            limited_forecast[edge_id] = forecast_data
                    context['congestion_forecast'] = limited_forecast
                except:
                    pass  # Skip forecast if it fails
        
        return context
    
    def _make_llm_macro_decisions(self, context: Dict[str, Any], 
                                current_time: float) -> List[Dict]:
        """Make macro routing decisions using enhanced LLM coordination."""
        try:
            # Use the new macro route planning method if available
            if hasattr(self.llm_agent, 'macro_route_planning'):
                # Prepare compact global state information
                current_state = self.global_state_history[-1] if self.global_state_history else None
                
                # Extract only relevant regions for requests
                relevant_regions = set()
                for request in context['requests']:
                    relevant_regions.add(request['start_region'])
                    relevant_regions.add(request['end_region'])
                
                # Add top 5 most congested regions
                full_regional_congestion = context.get('regional_congestion', {})
                if full_regional_congestion:
                    top_congested = sorted(full_regional_congestion.items(), 
                                         key=lambda x: x[1], reverse=True)[:5]
                    for region_id, _ in top_congested:
                        relevant_regions.add(region_id)
                
                # Limit to max 10 regions
                relevant_regions = list(relevant_regions)[:10]
                
                # Create compact regional congestion
                compact_regional_congestion = {
                    region_id: full_regional_congestion.get(region_id, 0) 
                    for region_id in relevant_regions
                }
                
                # Create compact boundary congestion (only top 10 most congested)
                full_boundary_congestion = context.get('boundary_congestion', {})
                compact_boundary_congestion = {}
                if full_boundary_congestion:
                    top_boundary_congested = sorted(full_boundary_congestion.items(), 
                                                  key=lambda x: x[1], reverse=True)[:10]
                    compact_boundary_congestion = dict(top_boundary_congested)
                
                global_state = {
                    'current_time': current_time,
                    'total_vehicles': current_state.total_vehicles if current_state else 0,
                    'regional_congestion': compact_regional_congestion,
                    'boundary_congestion': compact_boundary_congestion,
                    'avg_travel_time': current_state.avg_travel_time if current_state else 0
                }
                
                # Prepare route requests for LLM
                route_requests = []
                for request in context['requests']:
                    route_requests.append({
                        'vehicle_id': request['vehicle_id'],
                        'start_region': request['start_region'],
                        'end_region': request['end_region'],
                        'possible_routes': self._get_possible_regional_routes(
                            request['start_region'], request['end_region']
                        ),
                        'route_urgency': 'normal',
                        'special_requirements': None
                    })
                
                # Build raw planned movements (region/boundary incoming vehicle lists)
                planned_region_incoming: Dict[int, List[str]] = defaultdict(list)
                planned_boundary_incoming: Dict[str, List[str]] = defaultdict(list)
                try:
                    for vid, mroute in self.vehicle_macro_routes.items():
                        # Regions the vehicle plans to traverse (excluding potential current region)
                        try:
                            for rid in mroute.region_sequence[1:]:
                                planned_region_incoming[rid].append(vid)
                        except Exception:
                            pass
                        # Boundary edges along its macro route
                        try:
                            for be in mroute.boundary_edges:
                                planned_boundary_incoming[be].append(vid)
                        except Exception:
                            pass
                except Exception:
                    pass

                # Prepare regional conditions (raw view only)
                regional_conditions = {}
                for region_id in relevant_regions:
                    congestion = compact_regional_congestion.get(region_id, 0)
                    # Derive raw vehicle_count and avg_speed from current road_info (no scores/thresholds)
                    region_edges = [e for e, r in self.edge_to_region.items() if r == region_id]
                    vehicle_count = sum(self.road_info.get(edge, {}).get('vehicle_num', 0) for edge in region_edges[:10])
                    speeds = [self.road_info.get(edge, {}).get('avg_speed', 0.0) for edge in region_edges[:10]]
                    avg_speed = (sum(speeds) / len(speeds)) if speeds else 0.0
                    regional_conditions[region_id] = {
                        'region_id': region_id,
                        'congestion_level': congestion,
                        'vehicle_count': int(vehicle_count),
                        'avg_speed_ms': float(avg_speed),
                        'planned_incoming_vehicle_ids': planned_region_incoming.get(region_id, [])
                    }
                
                # Prepare boundary analysis (raw; only top 10 most relevant)
                boundary_analysis = {}
                relevant_boundary_edges = []
                
                # Find boundaries connecting relevant regions
                for boundary_info in self.boundary_edges[:20]:  # Limit search to first 20
                    if (boundary_info['from_region'] in relevant_regions or 
                        boundary_info['to_region'] in relevant_regions):
                        relevant_boundary_edges.append(boundary_info)
                    if len(relevant_boundary_edges) >= 10:  # Max 10 boundaries
                        break
                
                for boundary_info in relevant_boundary_edges:
                    edge_id = boundary_info['edge_id']
                    # Raw metrics from road_info if available
                    edge_data = self.road_info.get(edge_id, {})
                    boundary_analysis[edge_id] = {
                        'edge_id': edge_id,
                        'from_region': boundary_info['from_region'],
                        'to_region': boundary_info['to_region'],
                        'congestion_level': edge_data.get('congestion_level', compact_boundary_congestion.get(edge_id, 0)),
                        'vehicle_count': edge_data.get('vehicle_num', 0),
                        'avg_speed_ms': edge_data.get('avg_speed', 0.0),
                        'flow_rate_vehph': 0.0,
                        'planned_incoming_vehicle_ids': planned_boundary_incoming.get(edge_id, [])
                    }
                
                # Prepare compact flow predictions (limited forecast data)
                congestion_forecast = context.get('congestion_forecast', {})
                # Only keep forecast for relevant boundaries and limit to 2 time points
                compact_forecast = {}
                if congestion_forecast:
                    for edge_id in list(boundary_analysis.keys())[:5]:  # Max 5 edges
                        if edge_id in congestion_forecast:
                            forecast_data = congestion_forecast[edge_id]
                            if isinstance(forecast_data, list) and len(forecast_data) > 0:
                                compact_forecast[edge_id] = forecast_data[:2]  # Only first 2 time points
                
                flow_predictions = {
                    'time_horizon_s': min(self.prediction_horizon, 900),
                    'compact_forecast': compact_forecast
                }
                
                # Inject global macro guidance into coordination_needs to influence macro planning
                coordination_needs = {}
                try:
                    if 'global_guidance' in context:
                        gg = context['global_guidance']
                        coordination_needs = {
                            'priority_goals': gg.get('priority_goals', []),
                            'avoid_regions': gg.get('avoid_regions', []),
                            'avoid_edges': gg.get('avoid_edges', []),
                            'message': gg.get('message', '')
                        }
                except Exception:
                    pass
                
                # Prepare region routes information (raw; max 2 routes per request; no recommendations)
                region_routes = {}
                for request in context['requests'][:5]:  # Max 5 requests
                    start_region = request['start_region']
                    end_region = request['end_region']
                    
                    available_routes = self._get_possible_regional_routes(start_region, end_region, max_routes=2)
                    
                    region_routes[f"{start_region}-{end_region}"] = {
                        'available_routes': available_routes[:2]
                    }
                
                # Call the enhanced LLM macro planning method with raw LLM to avoid double concurrency control
                llm_result = self.raw_llm_agent.macro_route_planning(
                    global_state=global_state,
                    route_requests=route_requests,
                    regional_conditions=regional_conditions,
                    boundary_analysis=boundary_analysis,
                    flow_predictions=flow_predictions,
                    coordination_needs=coordination_needs,
                    region_routes=region_routes
                )
                
                # Convert LLM result to expected format
                llm_decisions = []
                macro_routes = llm_result.get('macro_routes', [])
                
                for i, request in enumerate(context['requests']):
                    vehicle_id = request['vehicle_id']
                    
                    # Find corresponding macro route decision
                    decision_found = False
                    for macro_route in macro_routes:
                        if macro_route.get('vehicle_id') == vehicle_id:
                            planned_route = macro_route.get('planned_route', [])
                            llm_decisions.append({
                                'answer': str(planned_route),
                                'summary': macro_route.get('reasoning', 'Macro route planning decision'),
                                'data_analysis': f"System optimization: {llm_result.get('system_optimization', 'N/A')}",
                                'coordination_strategy': llm_result.get('load_balancing', 'Standard balancing'),
                                'estimated_travel_time': macro_route.get('estimated_travel_time', 0)
                            })
                            decision_found = True
                            break
                    
                    if not decision_found:
                        # Fallback decision
                        fallback_route = self._get_shortest_regional_route(
                            request['start_region'], request['end_region']
                        )
                        llm_decisions.append({
                            'answer': str(fallback_route),
                            'summary': 'Fallback macro route - no LLM decision found',
                            'data_analysis': 'System optimization: N/A'
                        })
                
                # Store regional coordination messages for future use
                coordination_messages = llm_result.get('regional_coordination_messages', {})
                if coordination_messages:
                    self.coordination_messages = getattr(self, 'coordination_messages', {})
                    self.coordination_messages.update(coordination_messages)
                
                return llm_decisions
                
            else:
                # Fallback to enhanced hybrid decision making
                return self._make_enhanced_macro_decisions(context, current_time)
                
        except Exception as e:
            self.logger.log_error(f"Enhanced macro LLM decisions failed: {e}")
            return self._make_enhanced_macro_decisions(context, current_time)

    def _make_enhanced_macro_decisions(self, context: Dict[str, Any], 
                                     current_time: float) -> List[Dict]:
        """Enhanced macro decisions using hybrid approach."""
        # Prepare data for LLM
        data_texts = []
        answer_options = []
        routes_per_request: List[List[List[int]]] = []
        
        for request in context['requests']:
            start_region = request['start_region']
            end_region = request['end_region']
            
            # Create observation text for this route request
            data_text = self._create_macro_route_observation(
                start_region, end_region, context
            )
            
            # Create possible route options
            possible_routes = self._get_possible_regional_routes(start_region, end_region)
            routes_per_request.append(possible_routes)
            # Use numeric option form for strict protocol
            route_options = "/".join([str(i+1) for i in range(len(possible_routes))])
            
            data_texts.append(data_text)
            answer_options.append(f'"{route_options}"')
        
        # Use enhanced hybrid decision making if available
        if hasattr(self.llm_agent, 'enhanced_hybrid_decision_making_pipeline'):
            # Prepare compact system state for enhanced hybrid decisions
            current_state = self.global_state_history[-1] if self.global_state_history else None
            
            system_state = {
                'agent_type': 'TrafficAgent',
                'current_time': current_time,
                'total_requests': len(context['requests']),
                'total_vehicles': current_state.total_vehicles if current_state else 0,
                'avg_travel_time': current_state.avg_travel_time if current_state else 0
            }
            
            agent_communication = []
            coordination_opportunities = {
                'macro_coordination': True,
                'boundary_load_balancing': True,
                'conflict_resolution': len(context['requests']) > 1
            }
            
            # Limit traffic predictions to essential data
            traffic_predictions = {
                'time_horizon': min(self.prediction_horizon, 900),  # Max 15 minutes
                'has_forecast': 'congestion_forecast' in context
            }
            
            llm_decisions = self.llm_agent.enhanced_hybrid_decision_making_pipeline(
                data_texts=data_texts,
                answer_option_forms=answer_options,
                decision_type="macro_planning",
                decision_context="Inter-regional macro route planning",
                system_state=system_state,
                agent_communication=agent_communication,
                regional_coordination=coordination_opportunities,
                traffic_predictions=traffic_predictions
            )
            # Map numeric option answers back to route strings to keep downstream compatibility
            try:
                for i, decision in enumerate(llm_decisions or []):
                    ans = decision.get('answer') if isinstance(decision, dict) else None
                    if ans is None:
                        continue
                    try:
                        option_index = int(str(ans).strip())
                        if 1 <= option_index <= len(routes_per_request[i]):
                            decision['answer'] = str(routes_per_request[i][option_index - 1])
                    except Exception:
                        continue
            except Exception:
                pass
        elif hasattr(self.llm_agent, 'hybrid_decision_making_pipeline'):
            llm_decisions = self.llm_agent.hybrid_decision_making_pipeline(
                data_texts, answer_options
            )
            # Map numeric answers to route strings
            try:
                for i, decision in enumerate(llm_decisions or []):
                    ans = decision.get('answer') if isinstance(decision, dict) else None
                    if ans is None:
                        continue
                    try:
                        option_index = int(str(ans).strip())
                        if 1 <= option_index <= len(routes_per_request[i]):
                            decision['answer'] = str(routes_per_request[i][option_index - 1])
                    except Exception:
                        continue
            except Exception:
                pass
        else:
            # Basic fallback decisions
            llm_decisions = []
            for i, request in enumerate(context['requests']):
                shortest_route = self._get_shortest_regional_route(
                    request['start_region'], request['end_region']
                )
                llm_decisions.append({
                    'answer': str(shortest_route),
                    'summary': 'Fallback shortest route',
                    'data_analysis': 'No enhanced LLM available'
                })
        
        return llm_decisions
    
    def _create_macro_route_observation(self, start_region: int, end_region: int,
                                      context: Dict[str, Any]) -> str:
        """Create enhanced observation text for macro route planning with comprehensive coordination data."""
        observation_parts = []
        
        # Header with vehicle context  
        requests = context.get('requests', [])
        current_request = None
        for req in requests:
            if req.get('start_region') == start_region and req.get('end_region') == end_region:
                current_request = req
                break
        
        observation_parts.append(f"Coordinated Macro Route Planning: Region {start_region} -> Region {end_region}")
        observation_parts.append("")
        
        # Vehicle Information
        if current_request:
            vehicle_id = current_request.get('vehicle_id', 'Unknown')
            observation_parts.append("Vehicle:")
            observation_parts.append("")
            observation_parts.append(f"vehicle_id: {vehicle_id}")
            # Note: start_edge and end_edge would need to be passed in context for full implementation
            observation_parts.append(f"start_region: {start_region}, dest_region: {end_region}")

        # Global Macro Guidance (if available)
        try:
            gg = context.get('global_guidance')
            if gg:
                observation_parts.append("")
                observation_parts.append("GLOBAL_GUIDANCE:")
                observation_parts.append("")
                goals = gg.get('priority_goals', []) or []
                avoid_regions = gg.get('avoid_regions', []) or []
                avoid_edges = gg.get('avoid_edges', []) or []
                msg = gg.get('message', '') or ''
                if goals:
                    observation_parts.append("Goals: " + ", ".join([str(g) for g in goals[:3]]))
                if avoid_regions:
                    observation_parts.append("AvoidRegions: " + ",".join([f"R{int(r)}" for r in avoid_regions[:8]]))
                if avoid_edges:
                    observation_parts.append("AvoidEdges: " + ",".join([str(e) for e in avoid_edges[:10]]))
                if msg:
                    observation_parts.append(f"Note: {msg}")
                observation_parts.append("")
        except Exception:
            pass
        
        # Regional congestion for key regions (raw metrics only)
        regional_congestion = context.get('regional_congestion', {})
        relevant_regions = {start_region, end_region}
        
        # Add top 5 most congested regions  
        if regional_congestion:
            top_congested_regions = sorted(regional_congestion.items(), 
                                         key=lambda x: x[1], reverse=True)[:5]
            for region_id, _ in top_congested_regions:
                relevant_regions.add(region_id)
        
        congestion_list = []
        for region_id in sorted(relevant_regions)[:8]:  # Limit to 8 regions
            congestion = regional_congestion.get(region_id, 0)
            congestion_list.append(f"R{region_id}:{congestion:.2f}")
        
        if congestion_list:
            observation_parts.append("Key_Regional_Congestion: " + "|".join(congestion_list))
            observation_parts.append("")
        
        # Region State Snapshot (raw real-time metrics; no scoring)
        observation_parts.append("Region_State_Snapshot:")
        observation_parts.append("")
        
        # Generate enhanced region state information
        current_state = self.global_state_history[-1] if self.global_state_history else None
        regional_metrics = context.get('regional_metrics', {}) or {}
        for region_id in sorted(relevant_regions)[:6]:  # Limit to 6 regions for prompt efficiency
            # Prefer real metrics from performance logs
            metrics = regional_metrics.get(region_id, {}) if isinstance(regional_metrics, dict) else {}
            congestion = metrics.get('avg_congestion', regional_congestion.get(region_id, 0))
            vehicle_count = metrics.get('active_vehicles', None)
            if vehicle_count is None:
                region_edges = [e for e, r in self.edge_to_region.items() if r == region_id]
                vehicle_count = sum(self.road_info.get(edge, {}).get('vehicle_num', 0) for edge in region_edges[:5])
            avg_speed = metrics.get('avg_speed', None)
            if avg_speed is None:
                # compute average speed from edges if available
                region_edges = [e for e, r in self.edge_to_region.items() if r == region_id]
                speeds = [self.road_info.get(edge, {}).get('avg_speed', 0.0) for edge in region_edges[:5]]
                avg_speed = (sum(speeds) / len(speeds)) if speeds else 0.0
            capacity_util = metrics.get('capacity_utilization', 0.0)
            region_line = (f"R{region_id}: vehicles={int(vehicle_count)}, avg_speed={avg_speed:.1f}m/s, cong={congestion:.2f}, "
                         f"capacity_util={capacity_util:.2f}")
            observation_parts.append(region_line)
        
        observation_parts.append("")
        
        # Routes Analysis（为每条候选附带逐区域的原始序列数据；不做评分）
        possible_routes = self._get_possible_regional_routes(start_region, end_region, max_routes=4)
        route_list = []
        
        for i, route in enumerate(possible_routes):
            if len(route) <= 1:
                continue
                
            route_str = ">".join([f"R{r}" for r in route])
            # 构造逐区域原始序列（拥堵、容量利用率、均速）
            cong_series_parts = []
            cap_series_parts = []
            speed_series_parts = []
            for rid in route:
                rmetrics = regional_metrics.get(rid, {}) if isinstance(regional_metrics, dict) else {}
                r_cong = rmetrics.get('avg_congestion', regional_congestion.get(rid, 0.0))
                r_cap = rmetrics.get('capacity_utilization', 0.0)
                if 'avg_speed' in rmetrics:
                    r_speed = rmetrics.get('avg_speed', 0.0)
                else:
                    # 估算均速
                    region_edges = [e for e, r in self.edge_to_region.items() if r == rid]
                    speeds = [self.road_info.get(edge, {}).get('avg_speed', 0.0) for edge in region_edges[:5]]
                    r_speed = (sum(speeds) / len(speeds)) if speeds else 0.0
                cong_series_parts.append(f"R{rid}:{r_cong:.2f}")
                cap_series_parts.append(f"R{rid}:{r_cap:.2f}")
                speed_series_parts.append(f"R{rid}:{r_speed:.1f}")
            cong_series = "|".join(cong_series_parts)
            cap_series = "|".join(cap_series_parts)
            speed_series = "|".join(speed_series_parts)
            route_line = (f"Rt{i+1}: {route_str} | cong_series: {cong_series} | "
                          f"cap_series: {cap_series} | speed_series: {speed_series}")
            route_list.append(route_line)
        
        if route_list:
            observation_parts.append("Routes:")
            observation_parts.append("")
            for route in route_list:
                observation_parts.append(route)
            observation_parts.append("")
        
        # Coordination Raw（原始列表形式，不做平衡评分/份额推荐）
        try:
            overloaded_items = []
            available_items = []
            # 基于regional_metrics推导原始候选集
            for rid in sorted(relevant_regions)[:8]:
                rmetrics = regional_metrics.get(rid, {}) if isinstance(regional_metrics, dict) else {}
                cong = rmetrics.get('avg_congestion', regional_congestion.get(rid, 0.0))
                capu = rmetrics.get('capacity_utilization', 0.0)
                if cong is not None and (cong > 3.0):
                    overloaded_items.append(f"R{rid}(cong:{cong:.2f},cap:{capu:.2f})")
                if cong is not None and capu is not None and (cong < 2.0 and capu < 0.6):
                    available_items.append(f"R{rid}(cong:{cong:.2f},cap:{capu:.2f})")
            if overloaded_items or available_items:
                observation_parts.append("Coordination_Raw:")
                observation_parts.append("")
                if overloaded_items:
                    observation_parts.append("overloaded: " + ", ".join(overloaded_items[:6]))
                if available_items:
                    observation_parts.append("available: " + ", ".join(available_items[:6]))
                observation_parts.append("")
        except Exception:
            pass
        
        # Cut-Edge Status (raw metrics only)
        observation_parts.append("Cut-Edge_Status:")
        observation_parts.append("")
        
        # Generate boundary edge status for relevant connections
        relevant_boundaries = []
        for boundary_info in self.boundary_edges:
            if (boundary_info['from_region'] in relevant_regions or 
                boundary_info['to_region'] in relevant_regions):
                relevant_boundaries.append(boundary_info)
        
        for boundary_info in relevant_boundaries[:6]:  # Limit to 6 boundaries
            edge_id = boundary_info['edge_id'] 
            from_region = boundary_info['from_region']
            to_region = boundary_info['to_region']
            # Use calculated boundary metrics for raw values
            bmetrics = self._calculate_boundary_metrics(edge_id, boundary_info)
            boundary_line = (f"BE_{from_region}_{to_region}: vehicles={bmetrics.get('vehicle_count', 0)}, "
                           f"cong={bmetrics.get('congestion_level', 0.0):.2f}, util={bmetrics.get('utilization', 0.0):.2f}, "
                           f"flow_rate={bmetrics.get('flow_rate', 0.0):.1f}veh/h")
            observation_parts.append(boundary_line)
        
        # Do not include system performance/score summaries in the prompt
        
        observation_parts.append("Objective: Select route based on current raw regional and boundary metrics; avoid known congested areas and stuck edges.")
        observation_parts.append("")
        
        # Route options for decision (only include once at final combination stage)
        # Keep observation focused on factual state; options will be appended in _prepare_llm_input_texts
        
        observation_text = "\n".join(observation_parts)
        
        # Allow sufficient context but prevent excessive length
        max_context_length = 2500
        if len(observation_text) > max_context_length:
            observation_text = observation_text[:max_context_length-3] + "..."
        
        return observation_text
    
    def _prepare_llm_input_texts(self, context: Dict[str, Any]) -> List[str]:
        """Prepare the exact text inputs that LLM will see for logging purposes."""
        llm_input_texts = []
        
        for request in context['requests']:
            start_region = request['start_region']
            end_region = request['end_region']
            
            # Generate the observation text (what LLM sees)
            observation_text = self._create_macro_route_observation(
                start_region, end_region, context
            )
            
            # Generate the route options (what LLM chooses from)
            possible_routes = self._get_possible_regional_routes(start_region, end_region)
            # Enforce numeric option protocol by logging numbered options for clarity
            option_strs = []
            for idx, route in enumerate(possible_routes, start=1):
                option_strs.append(f"{idx}:{route}")
            route_options = "/".join([str(route) for route in possible_routes])
            
            # Combine observation and options (single source of truth for options)
            full_input = f"{observation_text}\n\nROUTE OPTIONS: {route_options}"
            llm_input_texts.append(full_input)
        
        return llm_input_texts
    
    def _get_possible_regional_routes(self, start_region: int, 
                                    end_region: int, max_routes: int = 3) -> List[List[int]]:
        """Get possible routes between regions."""
        if start_region == end_region:
            return [[start_region]]
        
        try:
            # Find shortest paths allowing for alternatives
            routes = []
            
            # Primary shortest path
            try:
                shortest = nx.shortest_path(self.region_graph, start_region, end_region)
                routes.append(shortest)
            except nx.NetworkXNoPath:
                return [[start_region, end_region]]  # Direct fallback
            
            # Alternative paths (if they exist and are reasonable)
            try:
                for path in nx.all_simple_paths(
                    self.region_graph, start_region, end_region, cutoff=5
                ):
                    if len(routes) >= max_routes:
                        break
                    if path not in routes and len(path) <= len(shortest) + 2:
                        routes.append(path)
            except:
                pass
            
            return routes if routes else [[start_region, end_region]]
            
        except Exception:
            return [[start_region, end_region]]
    
    def _get_shortest_regional_route(self, start_region: int, end_region: int) -> List[int]:
        """Get shortest route between regions."""
        try:
            return nx.shortest_path(self.region_graph, start_region, end_region)
        except:
            return [start_region, end_region]
    
    def _process_llm_macro_decisions(self, llm_decisions: List[Dict], 
                                   requests: List[Dict], current_time: float) -> List[MacroRoute]:
        """Process LLM decisions into MacroRoute objects."""
        macro_routes = []
        
        for i, request in enumerate(requests):
            if i >= len(llm_decisions):
                continue
            
            decision_data = llm_decisions[i]
            route_answer = decision_data.get('answer', '')
            
            try:
                # Parse route from answer
                if '/' in route_answer:
                    # Multiple route options - take first
                    route_str = route_answer.split('/')[0]
                else:
                    route_str = route_answer
                
                # Parse region sequence
                region_sequence = self._parse_region_sequence(route_str)
                
                if not region_sequence:
                    # Fallback to shortest path
                    region_sequence = self._get_shortest_regional_route(
                        request['start_region'], request['end_region']
                    )
                
                # Get boundary edges for the route
                boundary_edges = self._get_boundary_edges_for_route(region_sequence)
                
                # Estimate travel time
                estimated_time = self._estimate_macro_travel_time(region_sequence)
                
                macro_route = MacroRoute(
                    vehicle_id=request['vehicle_id'],
                    start_region=request['start_region'],
                    end_region=request['end_region'],
                    region_sequence=region_sequence,
                    boundary_edges=boundary_edges,
                    estimated_travel_time=estimated_time,
                    creation_time=current_time,
                    last_update=current_time
                )
                
                macro_routes.append(macro_route)
                
            except Exception as e:
                self.logger.log_error(f"Failed to process macro decision for vehicle "
                                    f"{request['vehicle_id']}: {e}")
                continue
        
        return macro_routes
    
    def _parse_region_sequence(self, route_str: str) -> List[int]:
        """Parse region sequence from string."""
        try:
            # Handle different formats
            route_str = route_str.strip().strip('"\'')
            
            # Look for numbers or R-prefixed numbers
            if '→' in route_str:
                parts = route_str.split('→')
            elif '->' in route_str:
                parts = route_str.split('->')
            elif ',' in route_str:
                parts = route_str.split(',')
            elif ' ' in route_str:
                parts = route_str.split()
            else:
                parts = [route_str]
            
            sequence = []
            for part in parts:
                part = part.strip().replace('R', '').replace('r', '')
                try:
                    region_id = int(part)
                    if 0 <= region_id < self.num_regions:
                        sequence.append(region_id)
                except ValueError:
                    continue
            
            return sequence if len(sequence) >= 2 else []
            
        except:
            return []
    
    def _get_boundary_edges_for_route(self, region_sequence: List[int]) -> List[str]:
        """Get boundary edges for a region sequence."""
        boundary_edges = []
        
        for i in range(len(region_sequence) - 1):
            from_region = region_sequence[i]
            to_region = region_sequence[i + 1]
            
            edges = self.region_connections.get((from_region, to_region), [])
            if edges:
                # Select best boundary edge (least congested)
                best_edge = min(edges, key=lambda e: self.road_info.get(e, {}).get('congestion_level', 0))
                boundary_edges.append(best_edge)
        
        return boundary_edges
    
    def _estimate_macro_travel_time(self, region_sequence: List[int]) -> float:
        """Estimate travel time for a macro route by summing recorded regional ATT.
        Falls back to neutral default for regions without data.
        """
        try:
            total_time = 0.0
            for region_id in region_sequence:
                total_time += self.get_region_att(region_id)
            return total_time
        except Exception:
            # Fallback: neutral default per region if any unexpected error occurs
            return 300.0 * len(region_sequence)
    
    def _make_heuristic_macro_plans(self, requests: List[Dict], 
                                  current_time: float) -> List[MacroRoute]:
        """Make heuristic macro plans as fallback."""
        macro_routes = []
        
        for request in requests:
            try:
                # Use shortest path
                region_sequence = self._get_shortest_regional_route(
                    request['start_region'], request['end_region']
                )
                
                boundary_edges = self._get_boundary_edges_for_route(region_sequence)
                estimated_time = self._estimate_macro_travel_time(region_sequence)
                
                macro_route = MacroRoute(
                    vehicle_id=request['vehicle_id'],
                    start_region=request['start_region'],
                    end_region=request['end_region'],
                    region_sequence=region_sequence,
                    boundary_edges=boundary_edges,
                    estimated_travel_time=estimated_time,
                    creation_time=current_time,
                    last_update=current_time
                )
                
                macro_routes.append(macro_route)
                
            except Exception:
                continue
        
        return macro_routes
    
    def update_vehicle_macro_route(self, vehicle_id: str, current_region: int, 
                                 current_time: float):
        """Update vehicle macro route when it changes regions."""
        if vehicle_id in self.vehicle_macro_routes:
            macro_route = self.vehicle_macro_routes[vehicle_id]
            
            # Check if vehicle is following the planned route
            if current_region in macro_route.region_sequence:
                # If vehicle has progressed as expected, update
                macro_route.last_update = current_time
                
                # If vehicle reached destination, mark as completed
                if current_region == macro_route.end_region:
                    self.logger.log_info(f"Vehicle {vehicle_id} completed macro route "
                                       f"from region {macro_route.start_region} "
                                       f"to region {macro_route.end_region}")
                    del self.vehicle_macro_routes[vehicle_id]
            else:
                # Vehicle deviated from route, replan
                self.logger.log_info(f"Vehicle {vehicle_id} deviated from macro route, replanning")
                
                # Use single vehicle macro planning for replanning
                new_route = self.plan_single_macro_route(
                    vehicle_id=vehicle_id,
                    start_region=current_region,
                    end_region=macro_route.end_region,
                    current_time=current_time
                )
                if new_route:
                    self.vehicle_macro_routes[vehicle_id] = new_route
    
    def get_regional_recommendations(self, region_id: int, 
                                   current_time: float) -> RegionalRecommendation:
        """Get recommendations for a specific regional agent."""
        try:
            # Analyze current traffic state
            current_state = self.global_state_history[-1] if self.global_state_history else None
            
            if not current_state:
                return RegionalRecommendation(
                    target_boundary_edges=[],
                    congestion_weights={},
                    priority_vehicles=[],
                    avoid_edges=[]
                )
            
            # Find target boundary edges (less congested outgoing boundaries)
            outgoing_boundaries = []
            for boundary_info in self.boundary_edges:
                if boundary_info['from_region'] == region_id:
                    outgoing_boundaries.append(boundary_info['edge_id'])
            
            # Sort boundaries by congestion (ascending)
            boundary_congestion = current_state.boundary_congestion
            target_boundaries = sorted(
                outgoing_boundaries,
                key=lambda e: boundary_congestion.get(e, 0)
            )
            
            # Calculate congestion weights
            congestion_weights = {}
            for edge in outgoing_boundaries:
                congestion = boundary_congestion.get(edge, 0)
                # Higher weight for less congested edges
                weight = max(0.1, 1.0 - (congestion / 5.0))
                congestion_weights[edge] = weight
            
            # Identify priority vehicles (those with macro routes)
            priority_vehicles = []
            for vehicle_id, macro_route in self.vehicle_macro_routes.items():
                if region_id in macro_route.region_sequence:
                    priority_vehicles.append(vehicle_id)
            
            # Identify edges to avoid (highly congested)
            avoid_edges = []
            for edge, congestion in boundary_congestion.items():
                if congestion >= self.congestion_threshold:
                    avoid_edges.append(edge)
            
            recommendation = RegionalRecommendation(
                target_boundary_edges=target_boundaries,
                congestion_weights=congestion_weights,
                priority_vehicles=priority_vehicles,
                avoid_edges=avoid_edges
            )
            
            self.regional_recommendations[region_id] = recommendation
            return recommendation
            
        except Exception as e:
            self.logger.log_error(f"Failed to generate recommendations for region {region_id}: {e}")
            return RegionalRecommendation(
                target_boundary_edges=[],
                congestion_weights={},
                priority_vehicles=[],
                avoid_edges=[]
            )

    def get_global_guidance_overlay(self, parent_env=None) -> Dict[str, Any]:
        """Expose current global macro guidance for downstream agents. Returns dict or empty."""
        try:
            if parent_env and hasattr(parent_env, '_get_current_global_macro_guidance'):
                guidance = parent_env._get_current_global_macro_guidance()
                return guidance or {}
        except Exception:
            pass
        return {}
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get Traffic Agent performance metrics."""
        success_rate = (self.successful_macro_routes / max(1, self.total_macro_routes)) * 100
        
        # Calculate average boundary utilization
        avg_boundary_utilization = np.mean(list(self.boundary_utilization.values())) if self.boundary_utilization else 0.0
        
        # Calculate regional load balance
        current_state = self.global_state_history[-1] if self.global_state_history else None
        regional_balance = 0.0
        
        if current_state:
            congestion_values = list(current_state.regional_congestion.values())
            if congestion_values:
                regional_balance = 1.0 - (np.std(congestion_values) / max(1.0, np.mean(congestion_values)))
        
        return {
            'total_macro_routes': self.total_macro_routes,
            'successful_macro_routes': self.successful_macro_routes,
            'success_rate': success_rate,
            'avg_travel_time': self.avg_macro_travel_time,
            'boundary_utilization': avg_boundary_utilization,
            'regional_balance': regional_balance,
            'active_macro_routes': len(self.vehicle_macro_routes),
            'tracked_boundaries': len(self.boundary_utilization)
        }