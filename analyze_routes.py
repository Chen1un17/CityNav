#!/usr/bin/env python3
"""
Route Analysis Script for SUMO .rou.xml/.rou.alt.xml files

This script analyzes SUMO route files to count vehicles and calculate arrival rate statistics.
Analyzes vehicle departure times to compute arrival rates in 5-minute intervals.
"""

import xml.etree.ElementTree as ET
import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict


def analyze_routes(route_xml_path, interval_minutes=5):
    """
    Analyze a SUMO route file and calculate vehicle statistics and arrival rates.
    
    Args:
        route_xml_path (str): Path to the .rou.xml or .rou.alt.xml file
        interval_minutes (int): Time interval in minutes for arrival rate calculation
        
    Returns:
        dict: Dictionary containing counts and arrival rate statistics
    """
    if not os.path.exists(route_xml_path):
        raise FileNotFoundError(f"Route file not found: {route_xml_path}")
    
    print(f"Analyzing route file: {route_xml_path}")
    print("Loading XML file...")
    
    try:
        tree = ET.parse(route_xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"Error parsing XML file: {e}")
    
    # Extract all vehicles and their departure times
    vehicles = root.findall('.//vehicle')
    total_vehicles = len(vehicles)
    
    if total_vehicles == 0:
        raise ValueError("No vehicles found in the route file")
    
    print(f"Found {total_vehicles} vehicles")
    print("Extracting departure times...")
    
    departure_times = []
    for vehicle in vehicles:
        depart_str = vehicle.get('depart')
        if depart_str:
            try:
                depart_time = float(depart_str)
                departure_times.append(depart_time)
            except ValueError:
                print(f"Warning: Invalid departure time '{depart_str}' for vehicle {vehicle.get('id', 'unknown')}")
    
    if not departure_times:
        raise ValueError("No valid departure times found")
    
    departure_times = np.array(departure_times)
    
    # Convert to minutes and calculate statistics
    departure_times_min = departure_times / 60.0
    simulation_duration_min = departure_times_min.max()
    
    print(f"Simulation duration: {simulation_duration_min:.2f} minutes ({simulation_duration_min/60:.2f} hours)")
    
    # Calculate arrival rates per interval
    interval_seconds = interval_minutes * 60
    max_time_seconds = departure_times.max()
    num_intervals = int(np.ceil(max_time_seconds / interval_seconds))
    
    arrival_counts = np.zeros(num_intervals)
    
    for depart_time in departure_times:
        interval_idx = int(depart_time // interval_seconds)
        if interval_idx < num_intervals:
            arrival_counts[interval_idx] += 1
    
    # Calculate statistics for arrival rates
    mean_arrival_rate = np.mean(arrival_counts)
    std_arrival_rate = np.std(arrival_counts)
    max_arrival_rate = np.max(arrival_counts)
    min_arrival_rate = np.min(arrival_counts)
    
    # Additional statistics
    median_arrival_rate = np.median(arrival_counts)
    percentile_95 = np.percentile(arrival_counts, 95)
    percentile_5 = np.percentile(arrival_counts, 5)
    
    # Time-based statistics
    first_departure = departure_times.min() / 60  # in minutes
    last_departure = departure_times.max() / 60   # in minutes
    
    results = {
        'file_path': route_xml_path,
        'total_vehicles': total_vehicles,
        'interval_minutes': interval_minutes,
        'num_intervals': num_intervals,
        'simulation_duration_min': simulation_duration_min,
        'first_departure_min': first_departure,
        'last_departure_min': last_departure,
        'mean_arrival_rate': mean_arrival_rate,
        'std_arrival_rate': std_arrival_rate,
        'max_arrival_rate': max_arrival_rate,
        'min_arrival_rate': min_arrival_rate,
        'median_arrival_rate': median_arrival_rate,
        'percentile_95_arrival_rate': percentile_95,
        'percentile_5_arrival_rate': percentile_5,
        'arrival_counts': arrival_counts
    }
    
    return results


def save_results(results, output_dir):
    """
    Save analysis results to a file in the outputs directory.
    
    Args:
        results (dict): Analysis results
        output_dir (str): Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract file name from path
    file_path = Path(results['file_path'])
    file_name = file_path.stem  # e.g., "NewYork_od_0.1.rou.alt" from "NewYork_od_0.1.rou.alt.xml"
    
    output_file = os.path.join(output_dir, f"{file_name}_analysis.txt")
    
    with open(output_file, 'w') as f:
        f.write(f"Route Analysis Results for {file_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Source file: {results['file_path']}\n\n")
        
        f.write("Vehicle Statistics:\n")
        f.write(f"  Total vehicles: {results['total_vehicles']:,}\n\n")
        
        f.write("Simulation Time Information:\n")
        f.write(f"  Simulation duration: {results['simulation_duration_min']:.2f} minutes ({results['simulation_duration_min']/60:.2f} hours)\n")
        f.write(f"  First departure: {results['first_departure_min']:.2f} minutes\n")
        f.write(f"  Last departure: {results['last_departure_min']:.2f} minutes\n\n")
        
        f.write(f"Arrival Rate Statistics (vehicles/{results['interval_minutes']}min intervals):\n")
        f.write(f"  Mean: {results['mean_arrival_rate']:.2f}\n")
        f.write(f"  Standard Deviation: {results['std_arrival_rate']:.2f}\n")
        f.write(f"  Maximum: {results['max_arrival_rate']:.0f}\n")
        f.write(f"  Minimum: {results['min_arrival_rate']:.0f}\n")
        f.write(f"  Median: {results['median_arrival_rate']:.2f}\n")
        f.write(f"  95th Percentile: {results['percentile_95_arrival_rate']:.2f}\n")
        f.write(f"  5th Percentile: {results['percentile_5_arrival_rate']:.2f}\n\n")
        
        f.write(f"Total intervals analyzed: {results['num_intervals']}\n")
        
        # Write detailed interval data (first 20 intervals as sample)
        f.write(f"\nSample Arrival Counts (first 20 intervals of {results['interval_minutes']} minutes each):\n")
        sample_intervals = min(20, len(results['arrival_counts']))
        for i in range(sample_intervals):
            start_time = i * results['interval_minutes']
            end_time = (i + 1) * results['interval_minutes']
            f.write(f"  Interval {i+1} ({start_time:3d}-{end_time:3d} min): {results['arrival_counts'][i]:3.0f} vehicles\n")
        
        if len(results['arrival_counts']) > 20:
            f.write(f"  ... (showing first 20 of {len(results['arrival_counts'])} intervals)\n")
    
    print(f"Results saved to: {output_file}")
    return output_file


def main():
    """Main function to analyze route file."""
    # Default file path
    default_path = "/data/zhouyuping/LLMNavigation/Data/Chicago/Chicago_taxi_2015-01-01.rou.xml"
    output_dir = "outputs"
    interval_minutes = 5  # 5-minute intervals as requested
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        route_xml_path = sys.argv[1]
    else:
        route_xml_path = default_path
    
    if len(sys.argv) > 2:
        interval_minutes = int(sys.argv[2])
    
    try:
        # Analyze the routes
        results = analyze_routes(route_xml_path, interval_minutes)
        
        # Display results
        print("\n" + "=" * 60)
        print("Route Analysis Results")
        print("=" * 60)
        print(f"Total Vehicles: {results['total_vehicles']:,}")
        print(f"Simulation Duration: {results['simulation_duration_min']:.2f} minutes ({results['simulation_duration_min']/60:.2f} hours)")
        print(f"\nArrival Rate Statistics (vehicles/{interval_minutes}min):")
        print(f"  Mean:    {results['mean_arrival_rate']:.2f}")
        print(f"  Std:     {results['std_arrival_rate']:.2f}")
        print(f"  Max:     {results['max_arrival_rate']:.0f}")
        print(f"  Min:     {results['min_arrival_rate']:.0f}")
        
        # Save results
        output_file = save_results(results, output_dir)
        print(f"\nAnalysis complete! Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()