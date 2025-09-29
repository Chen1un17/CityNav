#!/usr/bin/env python3
"""
Simple script to generate Manhattan route files.
Modify the file paths below as needed.
"""

import os
import sys
import subprocess
import xml.etree.ElementTree as ET
import tempfile


# File paths - modify these as needed
NET_FILE = '/home/zhouyuping/program/baseline/Adaptive-Navigation/environments/sumo/networks/UES_Manhatan/UES_Manhatan.net.xml'
TRIPS_FILE = '/home/zhouyuping/program/baseline/Adaptive-Navigation/environments/sumo/UES_Manhatan_trips.xml'
OUTPUT_FILE = './Manhattan_od_routes.rou.alt.xml'
DEPART_INTERVAL = 10  # seconds between vehicle departures


def create_timed_trips_file(input_trips, output_trips, interval=10):
    """Convert simple trips to trips with timing and vehicle IDs."""
    print(f"Processing trips file: {input_trips}")
    
    try:
        tree = ET.parse(input_trips)
        root = tree.getroot()
        
        # Create new trips root
        new_root = ET.Element("trips")
        
        vehicle_id = 0
        depart_time = 0
        
        for trip in root.findall('trip'):
            origin = trip.get('origin')
            destination = trip.get('destination')
            
            if origin and destination:
                # Create new trip with timing
                new_trip = ET.SubElement(new_root, 'trip')
                new_trip.set('id', f"veh_{vehicle_id}")
                new_trip.set('depart', f"{depart_time}.00")
                new_trip.set('from', origin)
                new_trip.set('to', destination)
                
                vehicle_id += 1
                depart_time += interval
        
        # Write to file
        tree = ET.ElementTree(new_root)
        ET.indent(tree, space="  ")
        tree.write(output_trips, encoding='utf-8', xml_declaration=True)
        
        print(f"Created timed trips file with {vehicle_id} vehicles")
        return vehicle_id
        
    except Exception as e:
        print(f"Error processing trips: {e}")
        return 0


def run_duarouter(net_file, trips_file, route_file):
    """Run SUMO duarouter to generate routes."""
    cmd = [
        'duarouter',
        '--net-file', net_file,
        '--route-files', trips_file,
        '--output-file', route_file,
        '--repair', 'true',
        '--ignore-errors', 'true',
        '--routing-threads', '4',
        '--weights.random-factor', '0.5',
        '--verbose'
    ]
    
    print("Running duarouter...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Duarouter completed successfully!")
        else:
            print(f"Duarouter failed with code {result.returncode}")
            if result.stderr:
                print("Error output:", result.stderr)
        
        return result.returncode == 0
        
    except FileNotFoundError:
        print("Error: duarouter not found. Please ensure SUMO is installed.")
        return False


def main():
    """Main function to generate routes."""
    print("Manhattan Route Generator")
    print("=" * 50)
    
    # Check input files
    if not os.path.exists(NET_FILE):
        print(f"Error: Network file not found: {NET_FILE}")
        return False
        
    if not os.path.exists(TRIPS_FILE):
        print(f"Error: Trips file not found: {TRIPS_FILE}")
        return False
    
    print(f"Network file: {NET_FILE}")
    print(f"Trips file: {TRIPS_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Departure interval: {DEPART_INTERVAL} seconds")
    print()
    
    # Create temporary file for timed trips
    with tempfile.NamedTemporaryFile(mode='w', suffix='.trips.xml', delete=False) as temp_file:
        temp_trips_file = temp_file.name
    
    try:
        # Step 1: Create timed trips file
        num_vehicles = create_timed_trips_file(TRIPS_FILE, temp_trips_file, DEPART_INTERVAL)
        
        if num_vehicles == 0:
            print("Error: No valid trips found")
            return False
        
        # Step 2: Generate routes using duarouter
        success = run_duarouter(NET_FILE, temp_trips_file, OUTPUT_FILE)
        
        if success and os.path.exists(OUTPUT_FILE):
            file_size = os.path.getsize(OUTPUT_FILE)
            print()
            print("Route generation completed successfully!")
            print(f"Output file: {OUTPUT_FILE}")
            print(f"File size: {file_size:,} bytes")
            print(f"Number of vehicles: {num_vehicles}")
            
            # Show first few lines of output
            print("\nFirst few lines of generated route file:")
            with open(OUTPUT_FILE, 'r') as f:
                for i, line in enumerate(f):
                    if i < 10:
                        print(f"  {line.rstrip()}")
                    else:
                        break
            
            return True
        else:
            print("Error: Route generation failed")
            return False
            
    finally:
        # Clean up temp file
        if os.path.exists(temp_trips_file):
            os.unlink(temp_trips_file)


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
    
    print("\nDone! You can now use the generated route file in your simulation.")