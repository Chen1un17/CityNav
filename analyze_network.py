#!/usr/bin/env python3
"""
Network Analysis Script for SUMO .net.xml files

This script analyzes SUMO network files to count intersections and roads.
Intersections are represented as <junction> elements, and roads as <edge> elements.
"""

import xml.etree.ElementTree as ET
import os
import sys
from pathlib import Path


def analyze_network(net_xml_path):
    """
    Analyze a SUMO network file and count intersections and roads.
    
    Args:
        net_xml_path (str): Path to the .net.xml file
        
    Returns:
        dict: Dictionary containing counts and analysis results
    """
    if not os.path.exists(net_xml_path):
        raise FileNotFoundError(f"Network file not found: {net_xml_path}")
    
    print(f"Analyzing network file: {net_xml_path}")
    print("Loading XML file...")
    
    try:
        tree = ET.parse(net_xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"Error parsing XML file: {e}")
    
    # Count junctions (intersections)
    junctions = root.findall('.//junction')
    total_junctions = len(junctions)
    
    # Filter real intersections (exclude internal junctions)
    real_intersections = [j for j in junctions if j.get('type') != 'internal']
    intersection_count = len(real_intersections)
    
    # Count edges (roads)
    edges = root.findall('.//edge')
    total_edges = len(edges)
    
    # Filter real roads (exclude internal edges)
    real_roads = [e for e in edges if not e.get('function') == 'internal']
    road_count = len(real_roads)
    
    # Additional statistics
    internal_junctions = total_junctions - intersection_count
    internal_edges = total_edges - road_count
    
    results = {
        'file_path': net_xml_path,
        'total_intersections': intersection_count,
        'total_roads': road_count,
        'internal_junctions': internal_junctions,
        'internal_edges': internal_edges,
        'total_junctions': total_junctions,
        'total_edges': total_edges
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
    
    # Extract city name from file path
    file_path = Path(results['file_path'])
    city_name = file_path.stem  # e.g., "Chicago" from "Chicago.net.xml"
    
    output_file = os.path.join(output_dir, f"{city_name}_network_analysis.txt")
    
    with open(output_file, 'w') as f:
        f.write(f"Network Analysis Results for {city_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Source file: {results['file_path']}\n\n")
        f.write("Main Statistics:\n")
        f.write(f"  Intersections: {results['total_intersections']:,}\n")
        f.write(f"  Roads: {results['total_roads']:,}\n\n")
        f.write("Detailed Breakdown:\n")
        f.write(f"  Total junctions (including internal): {results['total_junctions']:,}\n")
        f.write(f"  Real intersections: {results['total_intersections']:,}\n")
        f.write(f"  Internal junctions: {results['internal_junctions']:,}\n\n")
        f.write(f"  Total edges (including internal): {results['total_edges']:,}\n")
        f.write(f"  Real roads: {results['total_roads']:,}\n")
        f.write(f"  Internal edges: {results['internal_edges']:,}\n")
    
    print(f"Results saved to: {output_file}")
    return output_file


def main():
    """Main function to analyze network file."""
    # Default file path
    default_path = "/home/zhouyuping/program/baseline/Adaptive-Navigation/environments/sumo/networks/UES_Manhatan/Manhatan-Abstracted-Original.net.xml"
    output_dir = "outputs"
    
    # Allow command line argument for different files
    if len(sys.argv) > 1:
        net_xml_path = sys.argv[1]
    else:
        net_xml_path = default_path
    
    try:
        # Analyze the network
        results = analyze_network(net_xml_path)
        
        # Display results
        print("\n" + "=" * 50)
        print(f"Network Analysis Results")
        print("=" * 50)
        print(f"Intersections: {results['total_intersections']:,}")
        print(f"Roads: {results['total_roads']:,}")
        print("\nDetailed breakdown:")
        print(f"  Total junctions: {results['total_junctions']:,} (including {results['internal_junctions']:,} internal)")
        print(f"  Total edges: {results['total_edges']:,} (including {results['internal_edges']:,} internal)")
        
        # Save results
        output_file = save_results(results, output_dir)
        print(f"\nAnalysis complete! Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()