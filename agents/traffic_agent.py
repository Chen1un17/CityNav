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
    
    def collect_regional_congestion_report(self, current_time: float) -> Dict:
        """
        Collect real-time congestion report from all regions as required by user.
        
        This method provides the traffic agent with comprehensive regional status
        for LLM decision making.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Dictionary containing detailed regional congestion and status information
        """
        try:
            regional_report = {
                'timestamp': current_time,
                'regions': {},
                'boundaries': {},
                'system_overview': {},
                'trends': {}
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
            
            self.logger.log_info(f"REGIONAL_REPORT: Collected congestion data for {len(regional_report['regions'])} regions")
            
            return regional_report
            
        except Exception as e:
            self.logger.log_error(f"REGIONAL_REPORT: Failed to collect regional congestion report: {e}")
            return {'timestamp': current_time, 'regions': {}, 'boundaries': {}, 'system_overview': {}, 'trends': {}}
    
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
    
    def batch_macro_planning(self, requests: List[Dict], current_time: float) -> List[MacroRoute]:
        """
        Plan macro routes for multiple vehicles simultaneously.
        
        Args:
            requests: List of vehicle route requests
            current_time: Current simulation time
            
        Returns:
            List of macro routes
        """
        try:
            if not requests:
                return []
            
            # Prepare context for LLM-based planning
            planning_context = self._prepare_macro_planning_context(requests, current_time)
            
            # Use LLM for intelligent route planning
            call_id = self.logger.log_llm_call_start(
                "TrafficAgent", "macro_planning", len(str(planning_context))
            )
            
            macro_routes = []
            try:
                llm_routes = self._make_llm_macro_decisions(planning_context, current_time)
                
                # Process LLM decisions
                macro_routes = self._process_llm_macro_decisions(
                    llm_routes, requests, current_time
                )
                
                decision_summary = f"Planned {len(macro_routes)} macro routes"
                self.logger.log_llm_call_end(
                    call_id, True, decision_summary, len(str(planning_context))
                )
                
                self.successful_macro_routes += len(macro_routes)
                
            except Exception:
                self.logger.log_llm_call_end(
                    call_id, False, "Macro planning failed", 
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
            self.logger.log_error(f"Traffic Agent batch macro planning failed: {e}")
            return []
    
    def _prepare_macro_planning_context(self, requests: List[Dict], 
                                      current_time: float) -> Dict[str, Any]:
        """Prepare context for LLM-based macro planning."""
        # Get global traffic predictions
        boundary_edge_ids = [info['edge_id'] for info in self.boundary_edges]
        congestion_forecast = self.prediction_engine.get_congestion_forecast(
            boundary_edge_ids, self.prediction_horizon
        )
        
        # Get current global state
        current_state = self.global_state_history[-1] if self.global_state_history else None
        
        context = {
            'current_time': current_time,
            'num_regions': self.num_regions,
            'requests': requests,
            'global_state': current_state.__dict__ if current_state else None,
            'regional_congestion': current_state.regional_congestion if current_state else {},
            'boundary_congestion': current_state.boundary_congestion if current_state else {},
            'congestion_forecast': congestion_forecast,
            'boundary_utilization': self.boundary_utilization,
            'region_connections': dict(self.region_connections),
            'performance_metrics': self.get_performance_metrics()
        }
        
        return context
    
    def _make_llm_macro_decisions(self, context: Dict[str, Any], 
                                current_time: float) -> List[Dict]:
        """Make macro routing decisions using enhanced LLM coordination."""
        try:
            # Use the new macro route planning method if available
            if hasattr(self.llm_agent, 'macro_route_planning'):
                # Prepare global state information
                current_state = context.get('global_state', {})
                global_state = {
                    'current_time': current_time,
                    'total_vehicles': current_state.get('total_vehicles', 0),
                    'regional_congestion': context.get('regional_congestion', {}),
                    'boundary_congestion': context.get('boundary_congestion', {}),
                    'avg_travel_time': current_state.get('avg_travel_time', 0)
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
                
                # Prepare regional conditions
                regional_conditions = {}
                for region_id in range(context['num_regions']):
                    congestion = context.get('regional_congestion', {}).get(region_id, 0)
                    regional_conditions[region_id] = {
                        'congestion_level': congestion,
                        'capacity_utilization': min(1.0, congestion / 5.0),
                        'vehicle_count': 0,  # Could be enhanced with actual counts
                        'status': 'congested' if congestion > 3.0 else 'normal'
                    }
                
                # Prepare boundary analysis
                boundary_analysis = {}
                boundary_utilization = context.get('boundary_utilization', {})
                boundary_congestion = context.get('boundary_congestion', {})
                
                for boundary_info in self.boundary_edges:
                    edge_id = boundary_info['edge_id']
                    boundary_analysis[edge_id] = {
                        'from_region': boundary_info['from_region'],
                        'to_region': boundary_info['to_region'],
                        'congestion_level': boundary_congestion.get(edge_id, 0),
                        'utilization': boundary_utilization.get(edge_id, 0),
                        'capacity_remaining': max(0, 1.0 - boundary_utilization.get(edge_id, 0)),
                        'predicted_flow': 'stable'  # Could be enhanced with predictions
                    }
                
                # Prepare flow predictions
                congestion_forecast = context.get('congestion_forecast', {})
                flow_predictions = {
                    'time_horizon': self.prediction_horizon,
                    'boundary_congestion_forecast': congestion_forecast,
                    'regional_trend': 'stable',
                    'system_capacity_forecast': 'normal'
                }
                
                # Prepare coordination needs
                coordination_needs = {
                    'load_balancing_required': any(
                        util > 0.8 for util in boundary_utilization.values()
                    ),
                    'conflict_resolution_needed': len(context['requests']) > 5,
                    'priority_routing': any(
                        req['vehicle_id'] in self.vehicle_macro_routes 
                        for req in context['requests']
                    ),
                    'system_optimization_level': 'global'
                }
                
                # Prepare region routes information
                region_routes = {}
                for request in context['requests']:
                    start_region = request['start_region']
                    end_region = request['end_region']
                    
                    region_routes[f"{start_region}-{end_region}"] = {
                        'available_routes': self._get_possible_regional_routes(start_region, end_region),
                        'recommended_route': self._get_shortest_regional_route(start_region, end_region),
                        'alternative_count': len(self._get_possible_regional_routes(start_region, end_region)),
                        'route_quality': 'optimal'
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
            # Prepare system state for enhanced hybrid decisions
            system_state = {
                'agent_type': 'TrafficAgent',
                'global_state': context.get('global_state', {}),
                'current_time': current_time,
                'total_requests': len(context['requests'])
            }
            
            agent_communication = []
            coordination_opportunities = {
                'macro_coordination': True,
                'boundary_load_balancing': True,
                'conflict_resolution': len(context['requests']) > 1
            }
            
            traffic_predictions = {
                'congestion_forecast': context.get('congestion_forecast', {}),
                'time_horizon': self.prediction_horizon
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
        """Create observation text for macro route planning."""
        observation_parts = []
        
        observation_parts.append(f"Macro Route Planning: Region {start_region} → Region {end_region}")
        observation_parts.append("")
        
        # Current regional congestion
        regional_congestion = context.get('regional_congestion', {})
        observation_parts.append("Regional Congestion Levels:")
        for region_id in range(context['num_regions']):
            congestion = regional_congestion.get(region_id, 0)
            observation_parts.append(f"  Region {region_id}: {congestion:.1f}")
        observation_parts.append("")
        
        # Possible routes
        possible_routes = self._get_possible_regional_routes(start_region, end_region)
        observation_parts.append("Possible Regional Routes:")
        
        for i, route in enumerate(possible_routes):
            if len(route) <= 1:
                continue
                
            route_str = " → ".join([f"R{r}" for r in route])
            
            # Calculate route congestion
            route_congestion = np.mean([
                regional_congestion.get(region, 0) for region in route[1:]  # Exclude start
            ])
            
            # Get boundary information
            boundary_info = []
            for j in range(len(route) - 1):
                from_r, to_r = route[j], route[j + 1]
                boundaries = self.region_connections.get((from_r, to_r), [])
                if boundaries:
                    boundary_congestion = context.get('boundary_congestion', {})
                    avg_boundary_cong = np.mean([
                        boundary_congestion.get(edge, 0) for edge in boundaries
                    ])
                    boundary_info.append(f"boundary_congestion: {avg_boundary_cong:.1f}")
            
            observation_parts.append(f"  Route {i + 1}: {route_str}")
            observation_parts.append(f"    Average region congestion: {route_congestion:.1f}")
            if boundary_info:
                observation_parts.append(f"    {', '.join(boundary_info)}")
            observation_parts.append("")
        
        # Traffic predictions
        congestion_forecast = context.get('congestion_forecast', {})
        if congestion_forecast:
            observation_parts.append("Congestion Forecast (next 30 minutes):")
            for edge_id, forecast in list(congestion_forecast.items())[:5]:  # Show top 5
                avg_forecast = np.mean(forecast) if forecast else 0
                observation_parts.append(f"  Boundary {edge_id}: {avg_forecast:.1f}")
            observation_parts.append("")
        
        # Current performance
        performance = context.get('performance_metrics', {})
        if performance:
            observation_parts.append("System Performance:")
            observation_parts.append(f"  Success rate: {performance.get('success_rate', 0):.1f}%")
            observation_parts.append(f"  Average travel time: {performance.get('avg_travel_time', 0):.1f}s")
            observation_parts.append("")
        
        return "\n".join(observation_parts)
    
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