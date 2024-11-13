import pickle
import yaml
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='PKL file visualization tool')
parser.add_argument('--pkl_path', type=str, required=True, help='Path to PKL file')
args = parser.parse_args()

pkl_file_path = args.pkl_path

# Load pickle data
with open(pkl_file_path, 'rb') as file:
    data = []
    count = 0
    while True:
        try:
            obj = pickle.load(file)
            data.append(obj)
            count += 1
        except EOFError:
            break

# Print basic information about the loaded data
print(f"The file contains {count} pickle object(s).")
print("data's keys:{}".format(data[0].keys()))

# Count number of episodes by checking 'done' flags
num_episode = 0
for i in range(count):
    for key, value in data[i].items():
        if key == "done" and value == 1:
            num_episode += 1
print("num_episode:{}".format(num_episode))

# Define zarr configuration templates for different data types
config_data = [
    # Configuration for observation data (84x84x1)
    {
        "chunks": [161, 84, 84, 1],
        "compressor": {"blocksize": 0, "clevel": 5, "cname": "zstd", "id": "blosc", "shuffle": 2},
        "dtype": "<f4", "fill_value": 0.0, "filters": None, "order": "C",
        "shape": [count, 84, 84, 1], "zarr_format": 2
    },
    # Configuration for vector state (3 dimensions)
    {
        "chunks": [161, 3],
        "compressor": {"blocksize": 0, "clevel": 5, "cname": "zstd", "id": "blosc", "shuffle": 2},
        "dtype": "<f4", "fill_value": 0.0, "filters": None, "order": "C",
        "shape": [count, 3], "zarr_format": 2
    },
    # Configuration for action data (2 dimensions)
    {
        "chunks": [161, 2],
        "compressor": {"blocksize": 0, "clevel": 5, "cname": "zstd", "id": "blosc", "shuffle": 2},
        "dtype": "<f4", "fill_value": 0.0, "filters": None, "order": "C",
        "shape": [count, 2], "zarr_format": 2
    },
    # Configuration for global path (4x2)
    {
        "chunks": [161, 4, 2],
        "compressor": {"blocksize": 0, "clevel": 5, "cname": "zstd", "id": "blosc", "shuffle": 2},
        "dtype": "<f4", "fill_value": 0.0, "filters": None, "order": "C",
        "shape": [count, 4, 2], "zarr_format": 2
    },
    # Configuration for global map (160x160x1)
    {
        "chunks": [161, 160, 160, 1],
        "compressor": {"blocksize": 0, "clevel": 5, "cname": "zstd", "id": "blosc", "shuffle": 2},
        "dtype": "<f4", "fill_value": 0.0, "filters": None, "order": "C",
        "shape": [count, 160, 160, 1], "zarr_format": 2
    },
    # Configuration for done flags
    {
        "chunks": [208],
        "compressor": {"blocksize": 0, "clevel": 5, "cname": "zstd", "id": "blosc", "shuffle": 2},
        "dtype": "<i8", "fill_value": 0, "filters": None, "order": "C",
        "shape": [num_episode], "zarr_format": 2
    }
]

# Map configurations to their respective keys
configs = {
    "obs_data": config_data[0],
    "vector_state": config_data[1],
    "action_data": config_data[2],
    "global_path": config_data[3],
    "global_map": config_data[4],
    "done": config_data[5]
}

# Save each configuration to a separate YAML file
keys = ["obs_data", "vector_state", "action_data", "global_path", "global_map", "done"]
for key in keys:
    yaml_file_path = 'config/' + key + '_config.yaml'
    with open(yaml_file_path, 'w') as file:
        yaml.dump(configs[key], file, default_flow_style=False)
