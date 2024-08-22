#utils/data_mapping.py
import json
import os

def map_data_to_objects(data, output_file):
    """Maps data to objects and saves it to the specified output file."""
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Data mapping saved to {output_file}")

def read_data_from_file(file_path):
    """Reads data from the specified file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data
