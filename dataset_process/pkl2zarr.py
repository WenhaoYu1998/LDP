import os
import pickle
import yaml
import zarr
import numpy as np
from numcodecs import Blosc
import argparse

# Load and process pickle data
def load_pickle_data(pkl_file_path):
    """Load all objects from pickle file into a list"""
    with open(pkl_file_path, 'rb') as file:
        data = []
        while True:
            try:
                data.append(pickle.load(file))
            except EOFError:
                break
    return data

# Process data entries
def process_data_entries(data):
    """Process each data entry and organize them by type"""
    raw_data_dict = {}
    obs_data_list, action_data_list = [], []
    vector_state_list, global_path_list = [], []
    global_map_list, done_list = [], []
    
    for i, entry in enumerate(data):
        for key, value in entry.items():
            if key == "obs_data":
                obs_data_list.append(value.reshape((84, 84, 1)))
            elif key == "action_data":
                action_data_list.append(value)
            elif key in ["global_vector_path", "global_path"]:
                # Handle global path data with shape checking
                if value.shape[1] == 100:
                    global_path_list.append(value.T[[20, 40, 60, 80], :])
                else:
                    global_path_list.append(np.tile(vector_no_path[:2], (4, 1)))
            elif key == "global_map":
                # Reshape global map data
                global_map_list.append(value.reshape((160, 160, 1)))
            elif key == "vector_state":
                vector_no_path = value
                vector_state_list.append(value)
            elif key == "done" and value == 1:
                # Record episode end points
                done_list.append(i+1)

    # Convert lists to numpy arrays
    raw_data_dict.update({
        "obs_data": np.stack(obs_data_list),
        "action_data": np.stack(action_data_list),
        "vector_state": np.stack(vector_state_list),
        "global_path": np.stack(global_path_list),
        "global_map": np.stack(global_map_list),
        "done": np.array(done_list)
    })
    
    return raw_data_dict

def create_zarr_array(data_dict, key, file_name_without_ext):
    """Create and write data to zarr array with specified configuration"""
    # Load config from yaml
    with open(f'config/{key}_config.yaml', 'r') as file:
        configs = yaml.safe_load(file)
    
    # Configure compressor
    compressor = Blosc(
        clevel=configs['compressor']['clevel'],
        blocksize=configs['compressor']['blocksize'],
        cname=configs['compressor']['cname'],
        shuffle=configs['compressor']['shuffle']
    )
    
    # Define zarr path based on data type
    zarr_paths = {
        "done": f'{file_name_without_ext}.zarr/meta/episode_ends',
        "obs_data": f'{file_name_without_ext}.zarr/data/img',
        "action_data": f'{file_name_without_ext}.zarr/data/action',
        "vector_state": f'{file_name_without_ext}.zarr/data/state',
        "global_path": f'{file_name_without_ext}.zarr/data/gpath',
        "global_map": f'{file_name_without_ext}.zarr/data/gmap'
    }
    
    # Create and write to zarr array
    z = zarr.open(
        zarr_paths[key],
        mode='w',
        shape=tuple(configs['shape']),
        chunks=tuple(configs['chunks']),
        dtype=configs['dtype'],
        compressor=compressor,
        fill_value=configs['fill_value'],
        order=configs['order']
    )
    
    # Write data in batches
    batch_size = configs['chunks'][0]
    for start_idx in range(0, configs['shape'][0], batch_size):
        end_idx = min(start_idx + batch_size, configs['shape'][0])
        z[start_idx:end_idx] = data_dict[key][start_idx:end_idx].astype(configs['dtype'])

# Set up command line argument parser
parser = argparse.ArgumentParser(description='PKL to Zarr conversion tool')
parser.add_argument('--pkl_path', type=str, required=True, help='Path to the PKL file')
args = parser.parse_args()

# Use command line arguments
pkl_file_path = args.pkl_path
file_name = os.path.basename(pkl_file_path)  # Get file name
file_name_without_ext = os.path.splitext(file_name)[0]  # Get file name without extension

"""Use: python pkl2zarr.py --pkl_path your_pkl_name.pkl"""

if __name__ == "__main__":
    data = load_pickle_data(pkl_file_path)
    raw_data_dict = process_data_entries(data)
    keys = ["obs_data", "action_data", "vector_state", "global_path", "global_map", "done"]
    for key in keys:
        create_zarr_array(raw_data_dict, key, file_name_without_ext)
