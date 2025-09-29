#!/usr/bin/env python3
"""
Script to generate SUMO route files from network and trips files.
Converts trips.xml to rou.alt.xml format using SUMO's duarouter.
"""

import os
import sys
import subprocess
import xml.etree.ElementTree as ET
import argparse
import tempfile
import shutil
from pathlib import Path


def validate_files(net_file, trips_file):
    """Validate that input files exist and are readable."""
    if not os.path.exists(net_file):
        raise FileNotFoundError(f"Network file not found: {net_file}")
    if not os.path.exists(trips_file):
        raise FileNotFoundError(f"Trips file not found: {trips_file}")
    
    print(f"Network file: {net_file}")
    print(f"Trips file: {trips_file}")


def create_trips_with_timing(trips_file, output_trips_file, depart_interval=10):
    """
    Convert simple trips to trips with timing information.
    
    Args:
        trips_file: Input trips file path
        output_trips_file: Output trips file with timing
        depart_interval: Time interval between vehicle departures (seconds)
    """
    try:
        tree = ET.parse(trips_file)
        root = tree.getroot()
        
        # Create new trips element
        new_root = ET.Element("trips")
        
        vehicle_id = 0
        depart_time = 0
        
        for trip in root.findall('trip'):
            origin = trip.get('origin')
            destination = trip.get('destination')
            
            if origin and destination:
                # Create trip element with timing
                trip_elem = ET.SubElement(new_root, 'trip')
                trip_elem.set('id', f"vehicle_{vehicle_id}")
                trip_elem.set('depart', f"{depart_time}.00")
                trip_elem.set('from', origin)
                trip_elem.set('to', destination)
                
                vehicle_id += 1
                depart_time += depart_interval
        
        # Write to file
        tree = ET.ElementTree(new_root)
        ET.indent(tree, space="    ")
        tree.write(output_trips_file, encoding='utf-8', xml_declaration=True)
        
        print(f"Created trips file with {vehicle_id} vehicles: {output_trips_file}")
        return vehicle_id
        
    except Exception as e:
        raise Exception(f"Error processing trips file: {e}")


def run_duarouter(net_file, trips_file, output_file, additional_args=None):
    """
    Run SUMO's duarouter to generate routes from trips.
    
    Args:
        net_file: Network file path
        trips_file: Trips file path
        output_file: Output route file path
        additional_args: Additional duarouter arguments
    """
    cmd = [
        'duarouter',
        '--net-file', net_file,
        '--route-files', trips_file,
        '--output-file', output_file,
        '--repair', 'true',
        '--ignore-errors', 'true',
        '--routing-threads', '4',
        '--weights.random-factor', '0.5'
    ]
    
    if additional_args:
        cmd.extend(additional_args)
    
    try:
        print(f"Running duarouter command:")
        print(' '.join(cmd))
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print("Duarouter completed successfully!")
        if result.stdout:
            print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"Duarouter failed with return code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise
    except FileNotFoundError:
        raise Exception("duarouter not found. Please ensure SUMO is installed and in PATH.")


def generate_route_file(net_file, trips_file, output_file, depart_interval=10):
    """
    Main function to generate route file from network and trips.
    
    Args:
        net_file: Path to SUMO network file (.net.xml)
        trips_file: Path to trips file (.xml)
        output_file: Path for output route file (.rou.alt.xml)
        depart_interval: Time interval between departures in seconds
    """
    # Validate input files
    validate_files(net_file, trips_file)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Create temporary file for processed trips
    with tempfile.NamedTemporaryFile(mode='w', suffix='.trips.xml', delete=False) as temp_trips:
        temp_trips_file = temp_trips.name
    
    try:
        # Process trips file to add timing information
        num_vehicles = create_trips_with_timing(trips_file, temp_trips_file, depart_interval)
        
        # Run duarouter to generate routes
        run_duarouter(net_file, temp_trips_file, output_file)
        
        # Verify output file was created
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"Route file generated successfully!")
            print(f"Output file: {output_file}")
            print(f"File size: {file_size:,} bytes")
            print(f"Number of vehicles: {num_vehicles}")
        else:
            raise Exception("Output file was not created")
            
    finally:
        # Clean up temporary file
        if os.path.exists(temp_trips_file):
            os.unlink(temp_trips_file)


def main():
    parser = argparse.ArgumentParser(
        description="Generate SUMO route files from network and trips files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_routes.py --net network.net.xml --trips trips.xml --output routes.rou.alt.xml
    python generate_routes.py --net network.net.xml --trips trips.xml --output routes.rou.alt.xml --interval 5
        """
    )
    
    parser.add_argument('--net', required=True,
                       help='Path to SUMO network file (.net.xml)')
    parser.add_argument('--trips', required=True,
                       help='Path to trips file (.xml)')
    parser.add_argument('--output', required=True,
                       help='Path for output route file (.rou.alt.xml)')
    parser.add_argument('--interval', type=int, default=10,
                       help='Time interval between vehicle departures (default: 10 seconds)')
    
    args = parser.parse_args()
    
    try:
        generate_route_file(args.net, args.trips, args.output, args.interval)
        print("\nRoute generation completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Example usage for your specific files
    if len(sys.argv) == 1:
        print("Example usage for your files:")
        print("python generate_routes.py \\")
        print("  --net '/home/zhouyuping/program/baseline/Adaptive-Navigation/environments/sumo/networks/UES_Manhatan/UES_Manhatan.net.xml' \\")
        print("  --trips '/home/zhouyuping/program/baseline/Adaptive-Navigation/environments/sumo/UES_Manhatan_trips.xml' \\")
        print("  --output './Manhattan_routes.rou.alt.xml' \\")
        print("  --interval 10")
        print("\nOr run with --help for full options")
    else:
        main()