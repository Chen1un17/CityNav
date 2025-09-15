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


class GreedyRoutePlanning(object):
    """
    Greedy-based route planning system.
    
    This class implements three different greedy routing strategies:
    1. Minimum free-flow travel latency (minLat)
    2. Minimum travel distance (minDis)
    3. Minimum number of traffic lights (minLig)
    """
    
    def __init__(self, location: str, sumo_config: str, route_file: str, 
                 road_info_file: str, adjacency_file: str, step_size: float, 
                 max_steps: int, algorithm: str):
        """
        Initialize the Greedy routing system.
        
        Args:
            location: Location identifier for the simulation
            sumo_config: Path to SUMO configuration file
            route_file: Path to vehicle route file
            road_info_file: Path to road information JSON file
            adjacency_file: Path to adjacency matrix JSON file
            step_size: Simulation step size in seconds
            max_steps: Maximum number of simulation steps
            algorithm: Greedy algorithm type ('minLat', 'minDis', 'minLig')
        """
        self.location = location
        self.sumo_config = sumo_config
        self.route_file = route_file
        self.road_info_file = road_info_file
        self.adjacency_file = adjacency_file
        self.step_size = step_size
        self.max_steps = max_steps
        self.algorithm = algorithm
        
        # Vehicle tracking
        self.autonomous_vehicles = set()
        self.vehicle_start_times = {}
        self.vehicle_end_times = {}
        self.vehicle_routes = {}
        self.processed_vehicles = set()
        self.total_vehicles = 0
        self.completed_vehicles = 0
        self.target_autonomous_count = 0
        self.route_lookup = {}
        
        # Network data
        self.road_info = None
        self.adjacency_matrix = None
        self.edges = None
        
        # Performance metrics
        self.average_travel_time = 0.0
        self.route_computation_time = 0.0
        self.total_route_length = 0.0
        self.failed_routes = 0
        
        print(f"Initialized Greedy Route Planning for {location} using {algorithm} algorithm")
    
    def initialize(self):
        """
        Initialize the simulation environment and load network data.
        """
        print("Starting SUMO simulation (this may take a while for large networks)...")
        print(f"SUMO config: {self.sumo_config}")
        
        sumo_cmd = [
            "sumo", "-c", self.sumo_config, 
            "--no-warnings", 
            "--ignore-route-errors",
            "--no-step-log",
            "--time-to-teleport", "300",
            "--max-depart-delay", "900"
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
        self.adjacency_matrix = load_json(self.adjacency_file)
        
        print(f"Loaded {len(self.road_info)} road segments and {len(self.adjacency_matrix)} adjacency entries")
        
        # Parse route file for total vehicle count
        print("Parsing vehicle routes...")
        all_vehicles = parse_rou_file(self.route_file)
        self.total_vehicles = len(all_vehicles)
        
        # Store route info for route computation
        self.route_lookup = {veh_id: (start_edge, end_edge) 
                           for veh_id, start_edge, end_edge, _ in all_vehicles}
        
        # Calculate target autonomous vehicles (2% of total vehicles)
        self.target_autonomous_count = int(0.02 * self.total_vehicles)
        self.autonomous_vehicles = set()
        self.vehicles_processed_count = 0  # Counter for sequential selection
        
        print(f"Total vehicles in route file: {self.total_vehicles}")
        print(f"Target autonomous vehicles (2%): {self.target_autonomous_count}")
        
        # Get edge list from SUMO
        self.edges = traci.edge.getIDList()
        print(f"Retrieved {len(self.edges)} edges from SUMO")
        
        print(f"Ready for sequential autonomous vehicle selection with {self.algorithm} algorithm")
    
    def _compute_greedy_route(self, start_edge: str, end_edge: str) -> Optional[Dict]:
        """
        Compute route using greedy algorithm based on the specified strategy.
        
        Args:
            start_edge: Starting edge ID
            end_edge: Destination edge ID
            
        Returns:
            Dictionary with route info or None if computation failed
        """
        if start_edge == end_edge:
            return {
                'route': [start_edge],
                'length': 0.0,
                'start_edge': start_edge,
                'end_edge': end_edge
            }
        
        if start_edge not in self.adjacency_matrix or end_edge not in self.road_info:
            return None
            
        route = [start_edge]
        current_edge = start_edge
        visited = {start_edge}
        total_length = 0.0
        max_iterations = 100  # Prevent infinite loops
        iteration = 0
        
        while current_edge != end_edge and iteration < max_iterations:
            iteration += 1
            
            # Get neighbors of current edge
            if current_edge not in self.adjacency_matrix:
                break
                
            neighbors = self.adjacency_matrix[current_edge]
            
            # Filter out visited neighbors to avoid cycles
            unvisited_neighbors = [n for n in neighbors if n not in visited and n in self.road_info]
            
            if not unvisited_neighbors:
                # No unvisited neighbors, try to find any path to destination
                unvisited_neighbors = [n for n in neighbors if n in self.road_info]
                if not unvisited_neighbors:
                    break
            
            # Select next edge based on greedy strategy
            next_edge = self._select_next_edge_greedy(current_edge, unvisited_neighbors, end_edge)
            
            if next_edge is None:
                break
                
            route.append(next_edge)
            visited.add(next_edge)
            
            # Add to total length
            if current_edge in self.road_info:
                total_length += self.road_info[current_edge]['road_len']
            
            current_edge = next_edge
        
        if current_edge == end_edge:
            return {
                'route': route,
                'length': total_length,
                'start_edge': start_edge,
                'end_edge': end_edge
            }
        else:
            return None
    
    def _select_next_edge_greedy(self, current_edge: str, neighbors: List[str], target_edge: str) -> Optional[str]:
        """
        Select the next edge based on the greedy strategy.
        
        Args:
            current_edge: Current edge ID
            neighbors: List of neighboring edge IDs
            target_edge: Target destination edge ID
            
        Returns:
            Selected next edge ID or None if no valid selection
        """
        if not neighbors:
            return None
            
        if self.algorithm == "minLat":
            # Minimum free-flow travel latency
            return self._select_min_latency(neighbors, target_edge)
        elif self.algorithm == "minDis":
            # Minimum travel distance
            return self._select_min_distance(neighbors, target_edge)
        elif self.algorithm == "minLig":
            # Minimum number of traffic lights
            return self._select_min_traffic_lights(neighbors, target_edge)
        else:
            # Default to distance-based selection
            return self._select_min_distance(neighbors, target_edge)
    
    def _select_min_latency(self, neighbors: List[str], target_edge: str) -> Optional[str]:
        """Select neighbor with minimum free-flow travel latency."""
        min_latency = float('inf')
        best_neighbor = None
        
        for neighbor in neighbors:
            if neighbor in self.road_info:
                road_len = self.road_info[neighbor]['road_len']
                speed_limit = self.road_info[neighbor].get('speed_limit', 13.89)  # Default ~50 km/h
                
                # Calculate free-flow travel time
                latency = road_len / max(speed_limit, 1.0)  # Avoid division by zero
                
                if latency < min_latency:
                    min_latency = latency
                    best_neighbor = neighbor
        
        return best_neighbor
    
    def _select_min_distance(self, neighbors: List[str], target_edge: str) -> Optional[str]:
        """Select neighbor with minimum travel distance."""
        min_distance = float('inf')
        best_neighbor = None
        
        for neighbor in neighbors:
            if neighbor in self.road_info:
                road_len = self.road_info[neighbor]['road_len']
                
                if road_len < min_distance:
                    min_distance = road_len
                    best_neighbor = neighbor
        
        return best_neighbor
    
    def _select_min_traffic_lights(self, neighbors: List[str], target_edge: str) -> Optional[str]:
        """Select neighbor with minimum number of traffic lights."""
        min_lights = float('inf')
        best_neighbor = None
        
        for neighbor in neighbors:
            if neighbor in self.road_info:
                # Use traffic light count if available, otherwise assume based on road type
                traffic_lights = self.road_info[neighbor].get('traffic_lights', 0)
                
                # If traffic light info not available, estimate based on road length
                # Longer roads typically have more traffic lights
                if traffic_lights == 0:
                    road_len = self.road_info[neighbor]['road_len']
                    # Estimate: one traffic light per 500m on average
                    estimated_lights = max(0, int(road_len / 500))
                    traffic_lights = estimated_lights
                
                if traffic_lights < min_lights:
                    min_lights = traffic_lights
                    best_neighbor = neighbor
        
        return best_neighbor
    
    def run_simulation(self) -> Tuple[float, int]:
        """
        Run the complete simulation with dynamic autonomous vehicle selection 
        and real-time greedy route computation.
        
        Returns:
            Tuple of (average_travel_time, completed_autonomous_vehicles)
        """
        print(f"Starting {self.algorithm} greedy-based traffic simulation...")
        print(f"Simulation parameters: step_size={self.step_size}s, max_steps={self.max_steps}")
        print(f"Target autonomous vehicles: {self.target_autonomous_count} vehicles (2%)")
        
        # Random seed already set during initialization for pre-selection
        
        step = 0.0
        routes_applied = 0
        
        pbar = tqdm(total=self.max_steps, desc=f"{self.algorithm} Simulation", unit="steps")
        
        try:
            while step < self.max_steps:
                traci.simulationStep(step)
                current_time = traci.simulation.getTime()
                vehicle_ids = traci.vehicle.getIDList()
                
                # Process newly spawned vehicles for autonomous selection and routing
                for veh_id in vehicle_ids:
                    if veh_id not in self.processed_vehicles:
                        self.processed_vehicles.add(veh_id)
                        
                        # Record start time
                        try:
                            actual_depart_time = traci.vehicle.getDeparture(veh_id)
                            self.vehicle_start_times[veh_id] = actual_depart_time
                        except traci.exceptions.TraCIException:
                            self.vehicle_start_times[veh_id] = current_time
                        
                        # Select first 2% of vehicles as autonomous (sequential selection)
                        self.vehicles_processed_count += 1
                        if len(self.autonomous_vehicles) < self.target_autonomous_count:
                            self.autonomous_vehicles.add(veh_id)
                            
                            # Get start and end edges for route computation from SUMO
                            try:
                                current_route = traci.vehicle.getRoute(veh_id)
                                if len(current_route) >= 2:
                                    start_edge = current_route[0]
                                    end_edge = current_route[-1]
                                else:
                                    # Skip this vehicle if route too short
                                    self.autonomous_vehicles.discard(veh_id)
                                    continue
                            except:
                                # Skip this vehicle if can't get route info
                                self.autonomous_vehicles.discard(veh_id)
                                continue
                            
                            # Compute greedy route
                            route_success = False
                            try:
                                # Get vehicle's current edge
                                try:
                                    current_edge = traci.vehicle.getRoadID(veh_id)
                                    if current_edge == "":
                                        current_route = traci.vehicle.getRoute(veh_id)
                                        current_edge = current_route[0] if current_route else start_edge
                                except traci.exceptions.TraCIException:
                                    current_edge = start_edge
                                
                                # Compute greedy route
                                route_info = self._compute_greedy_route(current_edge, end_edge)
                                
                                if route_info is not None:
                                    # Apply the greedy route using traci
                                    traci.vehicle.setRoute(veh_id, route_info['route'])
                                    
                                    # Store route info
                                    self.vehicle_routes[veh_id] = {
                                        'route': route_info['route'],
                                        'length': route_info['length'],
                                        'start_edge': current_edge,
                                        'end_edge': end_edge,
                                        'applied': True
                                    }
                                    
                                    routes_applied += 1
                                    route_success = True
                                    
                                    if routes_applied % 100 == 0:
                                        print(f"Applied {routes_applied} {self.algorithm} routes")
                            
                            except Exception as e:
                                route_success = False
                                print(f"Route computation/application failed for vehicle {veh_id}: {e}")
                                
                                # Mark route as failed
                                self.vehicle_routes[veh_id] = {
                                    'route': [],
                                    'length': 0.0,
                                    'start_edge': current_edge,
                                    'end_edge': end_edge,
                                    'applied': False,
                                    'error': str(e)
                                }
                            
                            # If route computation failed, remove from autonomous vehicles
                            if not route_success:
                                self.autonomous_vehicles.discard(veh_id)
                                if veh_id in self.vehicle_routes:
                                    del self.vehicle_routes[veh_id]
                
                # Check for completed vehicles
                arrived_vehicles = traci.simulation.getArrivedIDList()
                for veh_id in arrived_vehicles:
                    if (veh_id in self.vehicle_start_times and 
                        veh_id not in self.vehicle_end_times):
                        self.vehicle_end_times[veh_id] = current_time
                        self.completed_vehicles += 1
                
                # Update progress
                step += self.step_size
                pbar.update(self.step_size)
                
                # Display progress periodically
                if int(step) % 360 == 0:  # Every 6 minutes simulation time
                    active_vehicles = len([v for v in vehicle_ids if v in self.autonomous_vehicles])
                    total_vehicles_in_sim = len(vehicle_ids)
                    autonomous_selected = len(self.autonomous_vehicles)
                    total_processed = len(self.processed_vehicles)
                    print(f"Time: {current_time:.0f}s, Active vehicles: {total_vehicles_in_sim}, "
                          f"Processed total: {total_processed}, Autonomous selected: {autonomous_selected}/{self.target_autonomous_count}, "
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
        print(f"GREEDY ROUTING SIMULATION RESULTS ({self.algorithm.upper()})")
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
        
        # Selection statistics
        total_processed = len(self.processed_vehicles)
        autonomous_selected = len(self.autonomous_vehicles)
        selection_rate = (autonomous_selected / self.target_autonomous_count * 100 
                         if self.target_autonomous_count > 0 else 0)
        
        print(f"Route Planning Algorithm: {self.algorithm.upper()} Greedy Strategy")
        print(f"Total Vehicles in Route File: {self.total_vehicles}")
        print(f"Vehicles Processed: {total_processed}")
        print(f"Target Autonomous Count: {self.target_autonomous_count}")
        print(f"Autonomous Vehicles Selected: {autonomous_selected}")
        print(f"Selection Rate: {selection_rate:.1f}% of target")
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


def run_single_experiment(location: str, step_size: float, max_steps: int, 
                         algorithm: str, use_wandb: bool = True) -> Tuple[float, int]:
    """
    Run a single experiment with the specified greedy algorithm.
    
    Args:
        location: Location identifier for simulation
        step_size: Simulation step size in seconds
        max_steps: Maximum simulation steps
        algorithm: Greedy algorithm ('minLat', 'minDis', 'minLig')
        use_wandb: Whether to log results to Weights & Biases
        
    Returns:
        Tuple of (average_travel_time, completed_vehicles)
    """
    # File paths using the specified data structure
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
        return 0.0, 0
    
    # Initialize Weights & Biases logging
    if use_wandb:
        wandb.init(
            project="USTBench-Route-Planning",
            group=f"{location}-Greedy-{algorithm}",
            name=f"Greedy-{algorithm}-Strategy",
            config={
                "algorithm": f"greedy_{algorithm}",
                "location": location,
                "step_size": step_size,
                "max_steps": max_steps,
                "autonomous_vehicle_ratio": 0.02
            }
        )
    
    # Create and run simulation
    greedy_planner = GreedyRoutePlanning(
        location=location,
        sumo_config=sumo_config,
        route_file=route_file,
        road_info_file=road_info_file,
        adjacency_file=adjacency_file,
        step_size=step_size,
        max_steps=max_steps,
        algorithm=algorithm
    )
    
    # Initialize simulation
    greedy_planner.initialize()
    
    # Run simulation and get results
    average_travel_time, completed_vehicles = greedy_planner.run_simulation()
    
    # Log results to W&B
    if use_wandb:
        wandb.log({
            "average_travel_time": average_travel_time,
            "throughput": completed_vehicles,
            "completion_rate": completed_vehicles / len(greedy_planner.autonomous_vehicles) * 100 if greedy_planner.autonomous_vehicles else 0,
            "failed_routes": greedy_planner.failed_routes,
            "total_route_length": greedy_planner.total_route_length,
            "total_vehicles": greedy_planner.total_vehicles,
            "autonomous_vehicles": len(greedy_planner.autonomous_vehicles)
        })
        wandb.finish()
    
    print(f"\nResults for {algorithm}:")
    print(f"Average Travel Time: {average_travel_time:.2f}s")
    print(f"Network Throughput: {completed_vehicles} vehicles completed")
    
    return average_travel_time, completed_vehicles


def main(location: str, step_size: float, max_steps: int, algorithm: str = None, use_wandb: bool = True):
    """
    Main function to run greedy-based route planning experiments.
    
    Args:
        location: Location identifier for simulation
        step_size: Simulation step size in seconds
        max_steps: Maximum simulation steps
        algorithm: Specific algorithm to run, or None to run all
        use_wandb: Whether to log results to Weights & Biases
    """
    print("Greedy Route Planning System")
    print("=" * 50)
    print(f"Location: {location}")
    print(f"Step Size: {step_size}s")
    print(f"Max Steps: {max_steps}")
    print(f"W&B Logging: {'Disabled' if not use_wandb else 'Enabled'}")
    
    algorithms = ['minLat', 'minDis', 'minLig'] if algorithm is None else [algorithm]
    print(f"Algorithms: {', '.join(algorithms)}")
    print("=" * 50)
    
    results = {}
    
    for alg in algorithms:
        print(f"\n{'='*20} Running {alg.upper()} Experiment {'='*20}")
        
        try:
            avg_time, completed = run_single_experiment(
                location=location,
                step_size=step_size,
                max_steps=max_steps,
                algorithm=alg,
                use_wandb=use_wandb
            )
            
            results[alg] = {
                'average_travel_time': avg_time,
                'completed_vehicles': completed
            }
            
            print(f"✓ {alg} experiment completed successfully")
            
        except Exception as e:
            print(f"✗ {alg} experiment failed: {e}")
            results[alg] = {
                'average_travel_time': 0.0,
                'completed_vehicles': 0,
                'error': str(e)
            }
        
        # Add some delay between experiments
        time.sleep(5)
    
    # Print summary results
    print("\n" + "="*60)
    print("SUMMARY OF ALL GREEDY EXPERIMENTS")
    print("="*60)
    
    for alg, result in results.items():
        if 'error' not in result:
            print(f"{alg.upper()}: Avg Time = {result['average_travel_time']:.2f}s, "
                  f"Completed = {result['completed_vehicles']}")
        else:
            print(f"{alg.upper()}: FAILED - {result['error']}")
    
    print("="*60)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SUMO simulation with Greedy routing strategies."
    )
    parser.add_argument(
        "--location", 
        type=str, 
        default="Manhattan", 
        help="Location of the simulation (default: Manhattan)"
    )
    parser.add_argument(
        "--step-size", 
        type=float, 
        default=180.0, 
        help="Simulation step size in seconds (default: 180.0)"
    )
    parser.add_argument(
        "--max-steps", 
        type=int, 
        default=43200, 
        help="Maximum number of simulation steps (default: 43200)"
    )
    parser.add_argument(
        "--algorithm", 
        type=str, 
        choices=['minLat', 'minDis', 'minLig'],
        help="Specific greedy algorithm to run (default: run all three)"
    )
    parser.add_argument(
        "--no-wandb", 
        action="store_true", 
        help="Disable Weights & Biases logging"
    )
    
    args = parser.parse_args()
    
    main(
        location=args.location,
        step_size=args.step_size,
        max_steps=args.max_steps,
        algorithm=args.algorithm,
        use_wandb=not args.no_wandb
    )