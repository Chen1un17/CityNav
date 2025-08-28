import argparse
import os
import sys
import random
import time
from tqdm import tqdm
import networkx as nx
import traci
import wandb
from typing import Dict, List, Tuple, Optional
os.environ["WANDB_MODE"] = "offline"
sys.path.append("../")

from env_utils import parse_rou_file
from utils.read_utils import load_json


class DijkstraRoutePlanning(object):
    """
    Traditional Dijkstra-based route planning system.
    
    This class implements shortest path routing using Dijkstra's algorithm
    without any intelligent agents. It provides a baseline comparison for
    evaluating agent-based routing systems.
    """
    
    def __init__(self, location: str, sumo_config: str, route_file: str, 
                 road_info_file: str, adjacency_file: str, step_size: float, 
                 max_steps: int):
        """
        Initialize the Dijkstra routing system.
        
        Args:
            location: Location identifier for the simulation
            sumo_config: Path to SUMO configuration file
            route_file: Path to vehicle route file
            road_info_file: Path to road information JSON file
            adjacency_file: Path to adjacency matrix JSON file
            step_size: Simulation step size in seconds
            max_steps: Maximum number of simulation steps
        """
        self.location = location
        self.sumo_config = sumo_config
        self.route_file = route_file
        self.road_info_file = road_info_file
        self.adjacency_file = adjacency_file
        self.step_size = step_size
        self.max_steps = max_steps
        
        # Vehicle tracking
        self.autonomous_vehicles = set()
        self.vehicle_start_times = {}
        self.vehicle_end_times = {}
        self.vehicle_routes = {}  # Computed routes for autonomous vehicles
        self.processed_vehicles = set()  # Track vehicles already processed for autonomous selection
        self.total_vehicles = 0
        self.completed_vehicles = 0
        self.target_autonomous_count = 0
        self.route_lookup = {}  # Map vehicle_id to (start_edge, end_edge)
        
        # Network data
        self.road_info = None
        self.road_network = None
        self.edges = None
        
        # Performance metrics
        self.average_travel_time = 0.0
        self.route_computation_time = 0.0
        self.total_route_length = 0.0
        self.failed_routes = 0
        
        print(f"Initialized Dijkstra Route Planning for {location}")
    
    def initialize(self):
        """
        Initialize the simulation environment and compute all routes.
        """
        print("Starting SUMO simulation (this may take a while for large networks)...")
        print(f"SUMO config: {self.sumo_config}")
        
        # Add more SUMO options for better performance with large networks
        sumo_cmd = [
            "sumo", "-c", self.sumo_config, 
            "--no-warnings", 
            "--ignore-route-errors",
            "--no-step-log",  # Reduce logging
            "--time-to-teleport", "300",  # Teleport stuck vehicles after 5 minutes
            "--max-depart-delay", "900"  # Allow 15 minute departure delays
        ]
        
        try:
            traci.start(sumo_cmd)
            print("✓ SUMO started successfully")
        except Exception as e:
            print(f"✗ Failed to start SUMO: {e}")
            raise
        
        # Load network data
        print("Loading road network data...")
        self.road_info = load_json(self.road_info_file)
        adjacency_matrix = load_json(self.adjacency_file)
        
        # Build NetworkX graph for Dijkstra algorithm
        print("Building road network graph...")
        self.road_network = nx.DiGraph()
        
        for edge in adjacency_matrix:
            if edge in self.road_info:
                road_len = self.road_info[edge]['road_len']
                for neighbor_edge in adjacency_matrix[edge]:
                    if neighbor_edge in self.road_info:
                        neighbor_len = self.road_info[neighbor_edge]['road_len']
                        # Use road length as edge weight for shortest path calculation
                        self.road_network.add_edge(edge, neighbor_edge, weight=road_len)
        
        print(f"Road network graph built: {self.road_network.number_of_nodes()} nodes, "
              f"{self.road_network.number_of_edges()} edges")
        
        # Parse route file for total vehicle count
        print("Parsing vehicle routes...")
        all_vehicles = parse_rou_file(self.route_file)
        self.total_vehicles = len(all_vehicles)
        
        # Store route info for dynamic route computation
        self.route_lookup = {veh_id: (start_edge, end_edge) 
                           for veh_id, start_edge, end_edge in all_vehicles}
        
        # Dynamic autonomous vehicle selection (will be populated during simulation)
        self.autonomous_vehicles = set()
        self.target_autonomous_count = int(0.02 * self.total_vehicles)
        
        print(f"Total vehicles in route file: {self.total_vehicles}")
        print(f"Target autonomous vehicles (2%): {self.target_autonomous_count}")
        
        # Get edge list from SUMO
        self.edges = traci.edge.getIDList()
        print(f"Retrieved {len(self.edges)} edges from SUMO")
        
        print("Ready for dynamic autonomous vehicle selection and routing")
    
    def _compute_route_for_vehicle(self, veh_id: str, start_edge: str, end_edge: str) -> Optional[Dict]:
        """
        Compute optimal route for a single vehicle using Dijkstra algorithm.
        
        Args:
            veh_id: Vehicle ID
            start_edge: Starting edge ID
            end_edge: Destination edge ID
            
        Returns:
            Dictionary with route info or None if computation failed
        """
        try:
            if (start_edge in self.road_network and 
                end_edge in self.road_network and 
                start_edge != end_edge):
                
                route = nx.dijkstra_path(self.road_network, 
                                       source=start_edge, 
                                       target=end_edge, 
                                       weight='weight')
                
                route_length = nx.dijkstra_path_length(self.road_network,
                                                     source=start_edge,
                                                     target=end_edge,
                                                     weight='weight')
                
                return {
                    'route': route,
                    'length': route_length,
                    'start_edge': start_edge,
                    'end_edge': end_edge
                }
                
            elif start_edge == end_edge:
                # Same start and end edge
                return {
                    'route': [start_edge],
                    'length': 0.0,
                    'start_edge': start_edge,
                    'end_edge': end_edge
                }
                
        except (nx.NetworkXNoPath, nx.NetworkXError, KeyError):
            pass
            
        return None
    
    def run_simulation(self) -> Tuple[float, int]:
        """
        Run the complete simulation with dynamic autonomous vehicle selection 
        and real-time Dijkstra route computation.
        
        Returns:
            Tuple of (average_travel_time, completed_autonomous_vehicles)
        """
        print("Starting Dijkstra-based traffic simulation...")
        print(f"Simulation parameters: step_size={self.step_size}s, max_steps={self.max_steps}")
        print(f"Dynamic selection: targeting {self.target_autonomous_count} autonomous vehicles")
        
        # Set random seed for reproducible autonomous vehicle selection
        random.seed(42)
        
        step = 0.0
        routes_applied = 0
        
        pbar = tqdm(total=self.max_steps, desc="Simulation Progress", unit="steps")
        
        try:
            while step < self.max_steps:
                traci.simulationStep(step)
                current_time = traci.simulation.getTime()
                vehicle_ids = traci.vehicle.getIDList()
                
                # Process newly spawned vehicles for autonomous selection and routing
                for veh_id in vehicle_ids:
                    # Check if this is a new vehicle not yet processed
                    if veh_id not in self.processed_vehicles:
                        # Mark as processed to avoid reprocessing
                        self.processed_vehicles.add(veh_id)
                        
                        # Record start time
                        try:
                            actual_depart_time = traci.vehicle.getDeparture(veh_id)
                            self.vehicle_start_times[veh_id] = actual_depart_time
                        except traci.exceptions.TraCIException:
                            # Some vehicles may not have departure time available immediately
                            self.vehicle_start_times[veh_id] = current_time
                        
                        # Dynamically select as autonomous vehicle using reservoir sampling approach
                        if (veh_id in self.route_lookup and 
                            len(self.autonomous_vehicles) < self.target_autonomous_count):
                            
                            # Calculate dynamic selection probability to reach target exactly
                            processed_eligible = len(self.processed_vehicles)
                            remaining_slots = self.target_autonomous_count - len(self.autonomous_vehicles)
                            remaining_vehicles = max(1, self.total_vehicles - processed_eligible)
                            selection_probability = min(1.0, remaining_slots / remaining_vehicles)
                            
                            if random.random() < selection_probability:
                                self.autonomous_vehicles.add(veh_id)
                                start_edge, end_edge = self.route_lookup[veh_id]
                                
                                # Compute optimal route using Dijkstra
                                route_success = False
                                try:
                                    # Get vehicle's current edge (where it actually is now)
                                    try:
                                        current_edge = traci.vehicle.getRoadID(veh_id)
                                        if current_edge == "":
                                            # Vehicle might be at junction, try route index 0
                                            current_route = traci.vehicle.getRoute(veh_id)
                                            current_edge = current_route[0] if current_route else start_edge
                                    except traci.exceptions.TraCIException:
                                        # Fallback to original start edge
                                        current_edge = start_edge
                                    
                                    if (current_edge in self.road_network and 
                                        end_edge in self.road_network and 
                                        current_edge != end_edge):
                                        
                                        route = nx.dijkstra_path(self.road_network, 
                                                               source=current_edge, 
                                                               target=end_edge, 
                                                               weight='weight')
                                        
                                        route_length = nx.dijkstra_path_length(self.road_network,
                                                                              source=current_edge,
                                                                              target=end_edge,
                                                                              weight='weight')
                                        
                                        # Apply the optimal route using traci
                                        traci.vehicle.setRoute(veh_id, route)
                                        
                                        # Store route info
                                        self.vehicle_routes[veh_id] = {
                                            'route': route,
                                            'length': route_length,
                                            'start_edge': current_edge,
                                            'end_edge': end_edge,
                                            'applied': True
                                        }
                                        
                                        routes_applied += 1
                                        route_success = True
                                        
                                        if routes_applied % 100 == 0:
                                            print(f"Applied {routes_applied} Dijkstra routes")
                                            
                                    elif current_edge == end_edge:
                                        # Already at destination - no routing needed
                                        self.vehicle_routes[veh_id] = {
                                            'route': [current_edge],
                                            'length': 0.0,
                                            'start_edge': current_edge,
                                            'end_edge': end_edge,
                                            'applied': True
                                        }
                                        route_success = True
                                        
                                except (nx.NetworkXNoPath, nx.NetworkXError, KeyError) as e:
                                    # Route computation failed - network path issues
                                    route_success = False
                                    print(f"Route computation failed for vehicle {veh_id}: {current_edge} -> {end_edge}: {e}")
                                except traci.exceptions.TraCIException as e:
                                    # Route application failed - SUMO TraCI issues
                                    route_success = False
                                    print(f"Route application failed for vehicle {veh_id} (current: {current_edge}, target: {end_edge}): {e}")
                                    
                                    # If route application fails, try to keep the vehicle running with original route
                                    # This prevents simulation crashes while maintaining vehicle flow
                                    try:
                                        # Just mark the route as failed but don't interfere with vehicle
                                        self.vehicle_routes[veh_id] = {
                                            'route': [],
                                            'length': 0.0,
                                            'start_edge': current_edge,
                                            'end_edge': end_edge,
                                            'applied': False,
                                            'error': str(e)
                                        }
                                    except:
                                        pass  # Even fallback failed, just continue
                                
                                # If route computation failed, remove from autonomous vehicles
                                if not route_success:
                                    self.autonomous_vehicles.discard(veh_id)
                                    if veh_id in self.vehicle_routes:
                                        del self.vehicle_routes[veh_id]
                
                # Check for completed vehicles and clean up resources
                arrived_vehicles = traci.simulation.getArrivedIDList()
                for veh_id in arrived_vehicles:
                    if (veh_id in self.vehicle_start_times and 
                        veh_id not in self.vehicle_end_times):
                        self.vehicle_end_times[veh_id] = current_time
                        self.completed_vehicles += 1
                        
                        # Clean up completed autonomous vehicles from active tracking
                        # but keep their route info for final metrics calculation
                
                # Update progress
                step += self.step_size
                pbar.update(self.step_size)
                
                # Display progress periodically
                if int(step) % 1800 == 0:  # Every 30 minutes simulation time
                    active_vehicles = len([v for v in vehicle_ids if v in self.autonomous_vehicles])
                    total_vehicles_in_sim = len(vehicle_ids)
                    total_autonomous_selected = len(self.autonomous_vehicles)
                    print(f"Time: {current_time:.0f}s, Total vehicles: {total_vehicles_in_sim}, "
                          f"Active autonomous: {active_vehicles}, Total autonomous: {total_autonomous_selected}, "
                          f"Completed: {self.completed_vehicles}, Routes applied: {routes_applied}")
        
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
            
        finally:
            pbar.close()
            self._calculate_final_metrics()
            
        return self.average_travel_time, self.completed_vehicles
    
    def _calculate_final_metrics(self):
        """Calculate and display final simulation metrics."""
        print("\n" + "="*60)
        print("DIJKSTRA ROUTING SIMULATION RESULTS (Dynamic Selection)")
        print("="*60)
        
        # Calculate average travel time for autonomous vehicles only
        autonomous_travel_times = []
        for veh_id in self.vehicle_end_times:
            if (veh_id in self.autonomous_vehicles and 
                veh_id in self.vehicle_start_times):
                travel_time = self.vehicle_end_times[veh_id] - self.vehicle_start_times[veh_id]
                autonomous_travel_times.append(travel_time)
        
        if autonomous_travel_times:
            self.average_travel_time = sum(autonomous_travel_times) / len(autonomous_travel_times)
        else:
            self.average_travel_time = 0.0
        
        # Calculate completion rate for autonomous vehicles
        autonomous_completed = len(autonomous_travel_times)
        completion_rate = (autonomous_completed / len(self.autonomous_vehicles) * 100 
                         if self.autonomous_vehicles else 0)
        
        # Calculate route metrics
        successfully_routed = len([v for v in self.vehicle_routes 
                                 if self.vehicle_routes[v].get('applied', False)])
        
        total_route_length = sum(
            route_info.get('length', 0) 
            for route_info in self.vehicle_routes.values() 
            if route_info.get('applied', False)
        )
        
        failed_routes = len(self.autonomous_vehicles) - successfully_routed
        
        # Selection efficiency
        total_processed = len(self.processed_vehicles)
        selection_rate = (len(self.autonomous_vehicles) / total_processed * 100 
                        if total_processed > 0 else 0)
        
        print(f"Route Planning Algorithm: Dijkstra Shortest Path (Dynamic)")
        print(f"Total Vehicles in Route File: {self.total_vehicles}")
        print(f"Vehicles Processed: {total_processed}")
        print(f"Target Autonomous Count: {self.target_autonomous_count}")
        print(f"Actual Autonomous Selected: {len(self.autonomous_vehicles)}")
        print(f"Selection Rate: {selection_rate:.2f}% of processed vehicles")
        print(f"Successfully Routed: {successfully_routed}")
        print(f"Failed Routes: {failed_routes}")
        print(f"")
        print(f"PERFORMANCE METRICS:")
        print(f"Average Travel Time (Autonomous): {self.average_travel_time:.2f}s")
        print(f"Completed Autonomous Vehicles: {autonomous_completed}/{len(self.autonomous_vehicles)}")
        print(f"Completion Rate: {completion_rate:.1f}%")
        print(f"Total Route Length: {total_route_length:.0f}m")
        
        if successfully_routed > 0:
            avg_route_length = total_route_length / successfully_routed
            print(f"Average Route Length: {avg_route_length:.2f}m")
        
        print("="*60)
        
        # Cleanup SUMO
        try:
            traci.close()
            print("SUMO simulation closed successfully")
        except:
            pass


def main(location: str, step_size: float, max_steps: int, use_wandb: bool = True):
    """
    Main function to run Dijkstra-based route planning simulation.
    
    Args:
        location: Location identifier for simulation
        step_size: Simulation step size in seconds
        max_steps: Maximum simulation steps
        use_wandb: Whether to log results to Weights & Biases
    """
    # File paths using the new data structure
    sumo_config = f"/data/zhouyuping/LLMNavigation/Data/Region_1/Manhattan_sumo_config.sumocfg"
    route_file = f"/data/zhouyuping/LLMNavigation/Data/Region_1/Manhattan_od_0.01.rou.alt.xml"
    road_info_file = f"/data/zhouyuping/LLMNavigation/Data/Region_1/Manhattan_road_info.json"
    adjacency_file = f"/data/zhouyuping/LLMNavigation/Data/Region_1/Manhattan_adjacency_info.json"
    
    # Verify required files exist
    required_files = [sumo_config, route_file, road_info_file, adjacency_file]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("ERROR: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        sys.exit(1)
    
    # Initialize Weights & Biases logging
    if use_wandb:
        wandb.init(
            project="USTBench-Route-Planning",
            group=f"{location}-Dijkstra-Baseline",
            name="Dijkstra-Shortest-Path",
            config={
                "algorithm": "dijkstra",
                "location": location,
                "step_size": step_size,
                "max_steps": max_steps,
                "autonomous_vehicle_ratio": 0.02
            }
        )
    
    # Create and run simulation
    dijkstra_planner = DijkstraRoutePlanning(
        location=location,
        sumo_config=sumo_config,
        route_file=route_file,
        road_info_file=road_info_file,
        adjacency_file=adjacency_file,
        step_size=step_size,
        max_steps=max_steps
    )
    
    # Initialize simulation
    dijkstra_planner.initialize()
    
    # Run simulation and get results
    average_travel_time, completed_vehicles = dijkstra_planner.run_simulation()
    
    # Log results to W&B
    if use_wandb:
        wandb.log({
            "average_travel_time": average_travel_time,
            "throughput": completed_vehicles,
            "completion_rate": completed_vehicles / len(dijkstra_planner.autonomous_vehicles) * 100,
            "route_computation_time": dijkstra_planner.route_computation_time,
            "failed_routes": dijkstra_planner.failed_routes,
            "total_route_length": dijkstra_planner.total_route_length,
            "total_vehicles": dijkstra_planner.total_vehicles,
            "autonomous_vehicles": len(dijkstra_planner.autonomous_vehicles)
        })
        wandb.finish()
    
    print(f"\nFinal Results:")
    print(f"Average Travel Time: {average_travel_time:.2f}s")
    print(f"Network Throughput: {completed_vehicles} vehicles completed")
    
    return average_travel_time, completed_vehicles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SUMO simulation with Dijkstra shortest-path routing (baseline)."
    )
    parser.add_argument(
        "--location", 
        type=str, 
        default="Manhattan", 
        help="Location of the simulation (default: NewYork)"
    )
    parser.add_argument(
        "--step-size", 
        type=float, 
        default=180.0, 
        help="Simulation step size in seconds (default: 120.0)"
    )
    parser.add_argument(
        "--max-steps", 
        type=int, 
        default=43200, 
        help="Maximum number of simulation steps (default: 86400)"
    )
    parser.add_argument(
        "--no-wandb", 
        action="store_true", 
        help="Disable Weights & Biases logging"
    )
    
    args = parser.parse_args()
    
    print("Dijkstra Shortest Path Route Planning System")
    print("=" * 50)
    print(f"Location: {args.location}")
    print(f"Step Size: {args.step_size}s")
    print(f"Max Steps: {args.max_steps}")
    print(f"W&B Logging: {'Disabled' if args.no_wandb else 'Enabled'}")
    print("=" * 50)
    
    main(
        location=args.location,
        step_size=args.step_size,
        max_steps=args.max_steps,
        use_wandb=not args.no_wandb
    )