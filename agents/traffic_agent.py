"""
Traffic Agent for Multi-Agent Traffic Control System

Handles macro route planning between regions, inter-regional coordination,
and provides recommendations to Regional Agents based on global traffic state.
"""

import traci
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any
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
                 prediction_engine: PredictionEngine):
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
        """
        self.boundary_edges = boundary_edges
        self.edge_to_region = edge_to_region
        self.road_info = road_info
        self.num_regions = num_regions
        self.llm_agent = llm_agent
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
            
        except Exception as e:
            self.logger.log_error(f"Traffic Agent global state update failed: {e}")
    
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
        """Calculate metrics for a boundary edge."""
        try:
            metrics = {
                'edge_id': edge_id,
                'from_region': boundary_info['from_region'],
                'to_region': boundary_info['to_region'],
                'congestion_level': 0.0,
                'vehicle_count': 0,
                'utilization': 0.0,
                'flow_rate': 0.0,
                'status': 'normal'
            }
            
            if edge_id in self.road_info:
                edge_data = self.road_info[edge_id]
                
                congestion = edge_data.get('congestion_level', 0)
                vehicles = edge_data.get('vehicle_num', 0)
                
                metrics['congestion_level'] = congestion
                metrics['vehicle_count'] = vehicles
                
                # Calculate utilization
                road_len = edge_data.get('road_len', 100)
                lane_num = edge_data.get('lane_num', 1)
                capacity = (road_len * lane_num) / 8.0
                
                if capacity > 0:
                    metrics['utilization'] = min(1.0, vehicles / capacity)
                
                # Calculate flow rate from historical data
                if edge_id in self.boundary_traffic_flow:
                    flow_data = list(self.boundary_traffic_flow[edge_id])
                    if len(flow_data) >= 2:
                        recent_flow = flow_data[-5:] if len(flow_data) >= 5 else flow_data
                        avg_vehicles = sum(flow[1] for flow in recent_flow) / len(recent_flow)
                        metrics['flow_rate'] = avg_vehicles
                
                # Determine status
                if congestion > 4.0:
                    metrics['status'] = 'blocked'
                elif congestion > 2.0:
                    metrics['status'] = 'congested'
                elif congestion > 1.0:
                    metrics['status'] = 'busy'
                else:
                    metrics['status'] = 'normal'
            
            return metrics
            
        except Exception as e:
            self.logger.log_error(f"BOUNDARY_METRICS: Failed for edge {edge_id}: {e}")
            return {'edge_id': edge_id, 'from_region': boundary_info.get('from_region', -1), 
                   'to_region': boundary_info.get('to_region', -1), 'congestion_level': 0.0, 
                   'vehicle_count': 0, 'utilization': 0.0, 'flow_rate': 0.0, 'status': 'unknown'}
    
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
        """Calculate average travel time (simplified implementation)."""
        try:
            # This would ideally track actual travel times
            # For now, use a simple heuristic based on congestion
            total_congestion = 0
            count = 0
            
            for edge_data in self.road_info.values():
                if 'congestion_level' in edge_data:
                    total_congestion += edge_data['congestion_level']
                    count += 1
            
            avg_congestion = total_congestion / max(1, count)
            
            # Convert congestion to estimated travel time factor
            base_time = 300  # Base 5 minutes
            congestion_factor = 1.0 + (avg_congestion * 0.5)  # 50% increase per congestion level
            
            return base_time * congestion_factor
            
        except:
            return 300.0  # Default 5 minutes
    
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
            
            # Use LLM for intelligent route planning with coordination
            call_id = self.logger.log_llm_call_start(
                "TrafficAgent", "coordinated_macro_planning", len(str(planning_context))
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
        
        # Create compact data structures
        compact_regional_congestion = {
            region_id: full_regional_congestion.get(region_id, 0) 
            for region_id in relevant_regions
        }
        
        # Only keep top 10 most congested boundaries
        compact_boundary_congestion = {}
        if full_boundary_congestion:
            top_congested_boundaries = sorted(full_boundary_congestion.items(), 
                                            key=lambda x: x[1], reverse=True)[:10]
            compact_boundary_congestion = dict(top_congested_boundaries)
        
        # Compact performance metrics (only essential metrics)
        performance = self.get_performance_metrics()
        compact_performance = {
            'success_rate': performance.get('success_rate', 0),
            'regional_balance': performance.get('regional_balance', 0),
            'active_macro_routes': performance.get('active_macro_routes', 0)
        }
        
        # Optimize: Only include essential data to reduce context size
        context = {
            'current_time': current_time,
            'num_regions': min(self.num_regions, 20),  # Limit reported regions
            'requests': requests[:10],  # Max 10 requests
            'regional_congestion': compact_regional_congestion,
            'boundary_congestion': compact_boundary_congestion,
            'performance_metrics': compact_performance,
            'coordination_data': coordination_data if coordination_data else {}
        }
        
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
                
                # Prepare compact regional conditions (only relevant regions)
                regional_conditions = {}
                for region_id in relevant_regions:
                    congestion = compact_regional_congestion.get(region_id, 0)
                    regional_conditions[region_id] = {
                        'congestion_level': congestion,
                        'capacity_utilization': min(1.0, congestion / 5.0),
                        'status': 'congested' if congestion > 3.0 else 'normal'
                    }
                
                # Prepare compact boundary analysis (only top 10 most relevant)
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
                    boundary_analysis[edge_id] = {
                        'from_region': boundary_info['from_region'],
                        'to_region': boundary_info['to_region'],
                        'congestion_level': compact_boundary_congestion.get(edge_id, 0),
                        'utilization': min(0.8, compact_boundary_congestion.get(edge_id, 0) / 5.0),
                        'predicted_flow': 'stable'
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
                    'time_horizon': min(self.prediction_horizon, 900),  # Max 15 minutes
                    'compact_forecast': compact_forecast,
                    'regional_trend': 'stable'
                }
                
                # Prepare compact coordination needs
                coordination_needs = {
                    'load_balancing_required': any(
                        util > 0.8 for util in compact_boundary_congestion.values()
                    ),
                    'conflict_resolution_needed': len(context['requests']) > 1,
                    'priority_routing': len(context['requests']) > 3,
                    'system_optimization_level': 'regional'  # Focus on regional instead of global
                }
                
                # Prepare compact region routes information (max 2 routes per request)
                region_routes = {}
                for request in context['requests'][:5]:  # Max 5 requests
                    start_region = request['start_region']
                    end_region = request['end_region']
                    
                    available_routes = self._get_possible_regional_routes(start_region, end_region, max_routes=2)
                    recommended_route = self._get_shortest_regional_route(start_region, end_region)
                    
                    region_routes[f"{start_region}-{end_region}"] = {
                        'available_routes': available_routes[:2],  # Max 2 routes
                        'recommended_route': recommended_route,
                        'route_count': min(len(available_routes), 2)
                    }
                
                # Call the enhanced LLM macro planning method
                llm_result = self.llm_agent.macro_route_planning(
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
        
        for request in context['requests']:
            start_region = request['start_region']
            end_region = request['end_region']
            
            # Create observation text for this route request
            data_text = self._create_macro_route_observation(
                start_region, end_region, context
            )
            
            # Create possible route options
            possible_routes = self._get_possible_regional_routes(start_region, end_region)
            route_options = "/".join([str(route) for route in possible_routes])
            
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
        elif hasattr(self.llm_agent, 'hybrid_decision_making_pipeline'):
            llm_decisions = self.llm_agent.hybrid_decision_making_pipeline(
                data_texts, answer_options
            )
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
        """Create observation text for macro route planning with coordination data."""
        observation_parts = []
        
        observation_parts.append(f"Coordinated Macro Route Planning: Region {start_region} -> Region {end_region}")
        observation_parts.append("")
        
        # Coordination data first for efficiency
        coordination_data = context.get('coordination_data', {})
        if coordination_data and coordination_data.get('coordination_opportunities', False):
            overloaded = coordination_data.get('overloaded_regions', [])[:3]  # Top 3 overloaded
            available = coordination_data.get('available_regions', [])[:3]    # Top 3 available
            load_balance = coordination_data.get('load_balance_score', 1.0)
            
            observation_parts.append(f"Coordination: Avoid_R{overloaded} | Use_R{available} | Balance:{load_balance:.2f}")
            observation_parts.append("")
        
        # Current regional congestion - 只显示相关区域
        regional_congestion = context.get('regional_congestion', {})
        congestion_list = []
        
        # Only show congestion for start/end regions and highly congested regions
        relevant_regions = {start_region, end_region}
        
        # Add top 5 most congested regions
        congested_regions = sorted(regional_congestion.items(), 
                                 key=lambda x: x[1], reverse=True)[:5]
        for region_id, _ in congested_regions:
            relevant_regions.add(region_id)
        
        # Limit to max 10 regions to show
        for region_id in sorted(relevant_regions)[:10]:
            congestion = regional_congestion.get(region_id, 0)
            congestion_list.append(f"R{region_id}:{congestion:.1f}")
        
        if congestion_list:
            observation_parts.append("Key_Regional_Congestion: " + "|".join(congestion_list))
            observation_parts.append("")
        
        # Possible routes - 限制最多3条路径
        possible_routes = self._get_possible_regional_routes(start_region, end_region, max_routes=3)
        route_list = []
        
        for i, route in enumerate(possible_routes):
            if len(route) <= 1:
                continue
                
            route_str = ">".join([f"R{r}" for r in route])
            
            # Calculate route congestion
            route_congestion = np.mean([
                regional_congestion.get(region, 0) for region in route[1:]  # Exclude start
            ])
            
            # Check if route uses overloaded regions
            overloaded_penalty = 0
            if coordination_data:
                overloaded_regions = set(coordination_data.get('overloaded_regions', []))
                overloaded_penalty = len(set(route).intersection(overloaded_regions))
            
            route_compact = f"Rt{i+1}:{route_str}[cong:{route_congestion:.1f},avoid:{overloaded_penalty}]"
            route_list.append(route_compact)
        
        if route_list:
            observation_parts.append("Routes: " + " | ".join(route_list))
            observation_parts.append("")
        
        # Simplified performance - 只显示关键指标
        performance = context.get('performance_metrics', {})
        if performance:
            perf_list = [
                f"success:{performance.get('success_rate', 0):.1f}%",
                f"balance:{performance.get('regional_balance', 0):.1f}"
            ]
            observation_parts.append("Performance: " + "|".join(perf_list))
            observation_parts.append("")
        
        observation_parts.append("Objective: Select route avoiding overloaded regions, optimizing load balance")
        
        observation_text = "\n".join(observation_parts)
        # 严格的上下文长度控制 - 限制在800字符以内
        max_context_length = 800
        if len(observation_text) > max_context_length:
            observation_text = observation_text[:max_context_length-3] + "..."
        
        return observation_text
    
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
        """Estimate travel time for a macro route."""
        base_time_per_region = 300  # 5 minutes base time per region
        
        total_time = 0
        current_state = self.global_state_history[-1] if self.global_state_history else None
        
        for region_id in region_sequence:
            region_time = base_time_per_region
            
            # Adjust for congestion
            if current_state and region_id in current_state.regional_congestion:
                congestion = current_state.regional_congestion[region_id]
                region_time *= (1.0 + congestion * 0.3)  # 30% increase per congestion level
            
            total_time += region_time
        
        return total_time
    
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
                
                new_requests = [{
                    'vehicle_id': vehicle_id,
                    'start_region': current_region,
                    'end_region': macro_route.end_region
                }]
                
                new_routes = self.batch_macro_planning(new_requests, current_time)
                if new_routes:
                    self.vehicle_macro_routes[vehicle_id] = new_routes[0]
    
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