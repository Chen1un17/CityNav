import argparse
from env import Simulation
import wandb

class Route_Planning(object):
    def __init__(self, location, sumo_config, route_file, road_info_file, adjacency_file, step_size, max_steps):
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

        wandb.init(
            project="USTBench-Route-Planning",
            group=f"{self.location}-Heuristic",
            name="Examination"
        )

    def run(self):
        average_travel_time, throughput = self.sim.run_heuristic()

        wandb.log({
            "average_travel_time": average_travel_time,
            "throughput": throughput
        })
        wandb.finish()

def main(location, step_size, max_steps):
    sumo_config = f"/data/XXXXX/LLMNavigation/Data/NYC/{location}_sumo_config.sumocfg"
    route_file = f"/data/XXXXX/LLMNavigation/Data/NYC/{location}_od_0.1.rou.alt.xml"
    road_info_file = f"/data/XXXXX/LLMNavigation/Data/NYC/{location}_road_info.json"
    adjacency_file = f"/data/XXXXX/LLMNavigation/Data/NYC/Region_1/edge_adjacency_alpha_1.json"

    algo = Route_Planning(location, sumo_config, route_file, road_info_file, adjacency_file, step_size, max_steps)
    algo.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a SUMO simulation with autonomous vehicles.")
    parser.add_argument("--location", type=str, default="/data/XXXXX/LLMNavigation/Data/NYC/ttan", help="Location of the simulation.")
    parser.add_argument("--step-size", type=float, default=120.0, help="Simulation step size in seconds.")
    parser.add_argument("--max-steps", type=int, default=86400, help="Maximum number of simulation steps.")
    args = parser.parse_args()

    main(args.location, args.step_size, args.max_steps)
