# File: data_generation/record.py
import os
import yaml
import numpy as np
# Removed unused matplotlib import
from grid.env_graph_gridv1 import GraphEnv # Use relative import from data_generation/
from tqdm import tqdm # Import tqdm for progress bar

# Note: Removed sys.path.append - rely on project structure or PYTHONPATH

def make_env(case_path, config):
    """
    Creates a GraphEnv environment instance based on the input.yaml
    found in the specified case directory.
    """
    input_yaml_path = os.path.join(case_path, "input.yaml")
    if not os.path.exists(input_yaml_path):
        # print(f"Warning: input.yaml not found in {case_path}. Skipping env creation.")
        return None

    try:
        with open(input_yaml_path, 'r') as input_params:
            params = yaml.safe_load(input_params) # Use safe_load
    except yaml.YAMLError as e:
        print(f"Error loading YAML from {input_yaml_path}: {e}")
        return None
    except Exception as e:
        print(f"Error reading {input_yaml_path}: {e}")
        return None

    if not params or "agents" not in params or "map" not in params:
        print(f"Warning: Invalid or incomplete input.yaml structure in {case_path}.")
        return None

    nb_agents = len(params["agents"])
    if nb_agents == 0:
        # print(f"Warning: No agents defined in {input_yaml_path}.") # Less verbose
        return None # Or handle appropriately if an env with 0 agents is valid

    dimensions = params["map"]["dimensions"]
    obstacles = params["map"].get("obstacles", []) # Handle missing obstacles key

    # Initialize numpy arrays
    starting_pos = np.zeros((nb_agents, 2), dtype=np.int32)
    goals = np.zeros((nb_agents, 2), dtype=np.int32)
    obstacles_list = np.array(obstacles, dtype=np.int32).reshape(-1, 2) # More robust conversion

    for i, agent_data in enumerate(params["agents"]):
        try:
            starting_pos[i, :] = np.array(agent_data["start"], dtype=np.int32)
            goals[i, :] = np.array(agent_data["goal"], dtype=np.int32)
        except (KeyError, IndexError, ValueError) as e:
             print(f"Error processing agent {i} data in {input_yaml_path}: {e}")
             return None # Invalid agent data

    # Ensure required config keys exist
    required_keys = ["sensor_range"] # Add other keys GraphEnv needs from config
    if not all(key in config for key in required_keys):
         print(f"Error: Missing required key(s) in config for GraphEnv: {[k for k in required_keys if k not in config]}")
         return None
    # Add 'board_size' to config if not present, derived from dimensions
    if 'board_size' not in config:
         config['board_size'] = dimensions # Assuming board_size is [width, height]

    try:
        env = GraphEnv(
            config=config,
            goal=goals,
            # board_size=int(dimensions[0]), # Using config['board_size'] now
            starting_positions=starting_pos,
            obstacles=obstacles_list,
            sensing_range=config["sensor_range"],
            # Pass other necessary parameters from config if GraphEnv expects them
        )
        return env
    except Exception as e:
        print(f"Error initializing GraphEnv for {case_path}: {e}")
        return None


def record_env(path, config):
    """
    Processes trajectories in a given dataset path, simulates them in the environment,
    and records observations (FOV, GSO).
    """
    try:
        # List only directories starting with 'case_'
        cases = sorted([
            d for d in os.listdir(path)
            if d.startswith("case_") and os.path.isdir(os.path.join(path, d))
        ], key=lambda x: int(x.split('_')[-1])) # Sort numerically
    except FileNotFoundError:
        print(f"Error: Directory not found: {path}")
        return
    except Exception as e:
        print(f"Error listing cases in {path}: {e}")
        return

    if not cases:
        print(f"No 'case_*' directories found in {path}. Nothing to record.")
        return

    print(f"Found {len(cases)} potential cases in {path}.")

    # --- Calculate Trajectory Statistics ---
    trajectory_lengths = []
    valid_cases_for_stats = []
    print("Analyzing trajectory lengths...")
    for case_dir in tqdm(cases, desc="Checking Trajectories", unit="case"):
        case_path = os.path.join(path, case_dir)
        trajectory_path = os.path.join(case_path, "trajectory.npy")

        if os.path.exists(trajectory_path):
            try:
                trajectory = np.load(trajectory_path, allow_pickle=True)
                # Ensure trajectory has at least 2 dimensions (agents, steps)
                if trajectory.ndim >= 2 and trajectory.shape[1] > 0:
                    trajectory_lengths.append(trajectory.shape[1])
                    valid_cases_for_stats.append(case_dir) # Keep track of cases used for stats
                # else: # Optionally warn about empty trajectories
                #    print(f"Warning: Empty or invalid trajectory shape {trajectory.shape} found in {case_dir}. Skipping for stats.")
            except Exception as e:
                print(f"Warning: Could not load or process {trajectory_path}: {e}. Skipping for stats.")
        # else: # Optionally warn about missing trajectories
            # print(f"Warning: trajectory.npy not found in {case_dir}. Skipping for stats.")


    if not trajectory_lengths:
        print("No valid trajectories found to calculate statistics or record.")
        return

    t_lengths = np.array(trajectory_lengths)
    max_steps = np.max(t_lengths)
    min_steps = np.min(t_lengths)
    mean_steps = np.mean(t_lengths)

    print(f"\nTrajectory Statistics (based on {len(valid_cases_for_stats)} valid cases):")
    print(f"  Max steps: {max_steps}")
    print(f"  Min steps: {min_steps}")
    print(f"  Mean steps: {mean_steps:.2f}")

    # Write stats to file
    try:
        stats_path = os.path.join(path, "recording_stats.txt") # Use a different name maybe
        with open(stats_path, "w") as f:
            f.write(f"Stats based on {len(valid_cases_for_stats)} valid cases.\n")
            f.write(f"max steps: {max_steps}\n")
            f.write(f"min steps: {min_steps}\n")
            f.write(f"mean steps: {mean_steps:.2f}\n")
    except IOError as e:
        print(f"Warning: Could not write stats file {stats_path}: {e}")

    # --- Record Environment States ---
    print(f"\nRecording states for {len(valid_cases_for_stats)} cases with valid trajectories...")
    recorded_count = 0
    skipped_count = 0

    # Iterate only through the cases confirmed to have valid trajectories
    for case_dir in tqdm(valid_cases_for_stats, desc="Recording Env States", unit="case"):
        case_path = os.path.join(path, case_dir)
        trajectory_path = os.path.join(case_path, "trajectory.npy") # We know this exists

        try:
            # Create environment for this case
            env = make_env(case_path, config)
            if env is None:
                # print(f"Skipping {case_dir}: Failed to create environment.") # Already printed in make_env
                skipped_count += 1
                continue

            # Load trajectory (we know it exists and is loadable from stat calculation)
            trajectory = np.load(trajectory_path, allow_pickle=True)

            # Assuming trajectory.npy contains actions from t=0..T-1
            num_timesteps = trajectory.shape[1]
            agent_nb = trajectory.shape[0]

            if num_timesteps == 0:
                 # print(f"Skipping {case_dir}: Trajectory has zero timesteps after processing.") # Less verbose
                 skipped_count += 1
                 continue

            if agent_nb != env.nb_agents:
                print(f"Skipping {case_dir}: Agent count mismatch. Trajectory={agent_nb}, Env={env.nb_agents}")
                skipped_count += 1
                continue

            # Initialize recording arrays
            initial_obs = env.reset() # Get initial state
            fov_shape = initial_obs['fov'].shape # Should be (num_agents, channels, height, width)
            adj_shape = initial_obs['adj_matrix'].shape # Should be (num_agents, ...) depends on GraphEnv

            # === THE FIX IS HERE ===
            # Initialize recordings array with the *full* observation shape including the agent dimension.
            # Shape: (Num_States, Num_Agents, Channels, FOV_H, FOV_W) - T+1 states for T actions
            recordings = np.zeros((num_timesteps + 1,) + fov_shape, dtype=initial_obs['fov'].dtype)
            # Shape: (Num_States, Num_Agents, ...) depends on GraphEnv output
            adj_record = np.zeros((num_timesteps + 1,) + adj_shape, dtype=initial_obs['adj_matrix'].dtype)
            # === END FIX ===

            # Store initial observation (state at t=0)
            recordings[0] = initial_obs['fov']
            adj_record[0] = initial_obs['adj_matrix']

            # Placeholder for embedding - Check if GraphEnv uses/updates this
            emb = np.ones(env.nb_agents) # Static embedding?

            # Simulate trajectory
            obs = initial_obs
            for i in range(num_timesteps):
                actions = trajectory[:, i] # Actions for timestep i
                obs, _, _, _ = env.step(actions, emb) # Env state is now after action i (state at t=i+1)

                # Store the observation *after* action i (state at t=i+1)
                recordings[i + 1] = obs['fov']
                adj_record[i + 1] = obs['adj_matrix']

            # Save recorded data
            states_save_path = os.path.join(case_path, "states.npy")
            gso_save_path = os.path.join(case_path, "gso.npy")
            np.save(states_save_path, recordings)
            np.save(gso_save_path, adj_record)
            # Removed saving trajectory_record.npy as it's redundant

            recorded_count += 1

        except Exception as e:
            print(f"\nError processing {case_dir}: {e}")
            # import traceback # Optional: for detailed debugging
            # traceback.print_exc()
            skipped_count += 1

    print(f"\nRecording finished for path: {path}")
    print(f"Successfully recorded: {recorded_count} cases.")
    print(f"Skipped: {skipped_count} cases (due to errors or mismatches).")


# No __main__ block needed, this module is called by main_data.py