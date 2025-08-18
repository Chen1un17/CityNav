import argparse
import os
from env import Simulation
from multi_agent_env import MultiAgentTrafficEnvironment
from utils.read_utils import load_json
from utils.language_model import LLM as LLM
import wandb

class MultiAgent_Route_Planning(object):
    def __init__(self, batch_size, location, sumo_config, route_file, road_info_file, 
                 adjacency_file, task_info_file, llm_path_or_name, step_size, max_steps, 
                 use_reflection, region_data_dir):
        llm_name = llm_path_or_name.split("/")[-1]
        self.llm_name = llm_name
        self.location = location
        
        # Initialize language model
        task_info = load_json(task_info_file)
        self.llm_agent = LLM(llm_path_or_name, batch_size=batch_size, task_info=task_info, 
                           use_reflection=use_reflection)
        
        # Initialize multi-agent environment
        self.sim = MultiAgentTrafficEnvironment(
            location=location,
            sumo_config_file=sumo_config,
            route_file=route_file,
            road_info_file=road_info_file,
            adjacency_file=adjacency_file,
            region_data_dir=region_data_dir,
            llm_agent=self.llm_agent,
            step_size=step_size,
            max_steps=max_steps,
            log_dir=f"logs/{location}_{llm_name}"
        )

        wandb.init(
            project="USTBench-MultiAgent-Route-Planning",
            group=f"{self.location}-{llm_name}{'-w/o reflection' if not use_reflection else ''}",
            name="MultiAgent-Examination"
        )

    def run(self):
        average_travel_time, throughput = self.sim.run_simulation()

        wandb.log({
            "average_travel_time": average_travel_time,
            "throughput": throughput
        })
        wandb.finish()


class Route_Planning(object):
    def __init__(self, batch_size, location, sumo_config, route_file, road_info_file, adjacency_file, task_info_file, llm_path_or_name, step_size, max_steps, use_reflection):
        llm_name = llm_path_or_name.split("/")[-1]
        self.llm_name = llm_name
        self.location = location

        self.sim = Simulation(
            location=location,
            sumo_config_file=sumo_config,
            route_file=route_file,
            road_info_file=road_info_file,
            adjacency_file=adjacency_file,
            step_size=step_size,
            max_steps=max_steps
        )
        self.sim.initialize()

        # initialize language model
        task_info = load_json(task_info_file)
        self.llm_agent = LLM(llm_path_or_name, batch_size=batch_size, task_info=task_info, use_reflection=use_reflection)

        wandb.init(
            project="USTBench-Route-Planning",
            group=f"{self.location}-{llm_name}{'-w/o reflection' if not use_reflection else ''}",
            name="Examination"
        )

    def run(self):
        average_travel_time, throughput = self.sim.run(self.llm_agent)

        wandb.log({
            "average_travel_time": average_travel_time,
            "throughput": throughput
        })
        wandb.finish()

def main(llm_path_or_name, batch_size, location, use_reflection=True, step_size=180.0, 
         max_steps=43200, multi_agent=True):
    """
    Main function to run traffic simulation.
    
    Args:
        llm_path_or_name: Path to or name of the language model
        batch_size: Batch size for LLM processing
        location: Location for simulation (e.g., Manhattan)
        use_reflection: Whether to use reflection in LLM
        step_size: Simulation step size in seconds
        max_steps: Maximum simulation steps
        multi_agent: Whether to use multi-agent architecture (default: True)
    """
    # File paths - updated to use new data structure
    sumo_config = f"./Data/Region_1/{location}_sumo_config.sumocfg"
    route_file = f"./Data/Region_1/{location}_od_0.01.rou.alt.xml"
    road_info_file = f"./Data/Region_1/{location}_road_info.json"
    adjacency_file = f"./Data/Region_1/edge_adjacency_alpha_1.json"
    task_info_file = "./Data/task_info.json"
    
    if multi_agent:
        # Use multi-agent architecture
        region_data_dir = f"./Data/Region_1"  # Region partition data directory
        
        # Check if region data exists
        if not os.path.exists(region_data_dir):
            print(f"Warning: Region data directory {region_data_dir} not found. "
                  f"Falling back to single-agent mode.")
            multi_agent = False
    
    if multi_agent:
        print("Running Multi-Agent Traffic Control System")
        algo = MultiAgent_Route_Planning(
            batch_size, location, sumo_config, route_file, road_info_file, 
            adjacency_file, task_info_file, llm_path_or_name, step_size, 
            max_steps, use_reflection, region_data_dir
        )
    else:
        print("Running Traditional Single-Agent Route Planning")
        algo = Route_Planning(
            batch_size, location, sumo_config, route_file, road_info_file, 
            adjacency_file, task_info_file, llm_path_or_name, step_size, 
            max_steps, use_reflection
        )
    
    algo.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a SUMO simulation with autonomous vehicles.")
    parser.add_argument("--llm-path-or-name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", 
                       help="Path to the language model or its name.")
    parser.add_argument("--batch-size", type=int, default=16, 
                       help="Batch size for the language model.")
    parser.add_argument("--location", type=str, default="Manhattan", 
                       help="Location of the simulation.")
    parser.add_argument("--step-size", type=float, default=180.0, 
                       help="Simulation step size in seconds.")
    parser.add_argument("--max-steps", type=int, default=43200, 
                       help="Maximum number of simulation steps.")
    parser.add_argument("--multi-agent", action="store_true", default=True,
                       help="Use multi-agent architecture (default: True)")
    parser.add_argument("--single-agent", action="store_true", default=False,
                       help="Use traditional single-agent architecture")
    parser.add_argument("--no-reflection", action="store_true", default=False,
                       help="Disable LLM reflection")
    
    args = parser.parse_args()
    
    # Determine which architecture to use
    use_multi_agent = args.multi_agent and not args.single_agent
    use_reflection = not args.no_reflection
    
    print(f"Configuration:")
    print(f"  - Architecture: {'Multi-Agent' if use_multi_agent else 'Single-Agent'}")
    print(f"  - LLM: {args.llm_path_or_name}")
    print(f"  - Reflection: {'Enabled' if use_reflection else 'Disabled'}")
    print(f"  - Location: {args.location}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Step Size: {args.step_size}s")
    print(f"  - Max Steps: {args.max_steps}")
    print()

    main(args.llm_path_or_name, args.batch_size, args.location, use_reflection, 
         args.step_size, args.max_steps, use_multi_agent)
