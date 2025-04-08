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
    Requires 'config' to contain necessary parameters for GraphEnv,
    like 'sensing_range', 'pad', 'board_size', 'num_agents'.
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

    nb_agents_from_yaml = len(params["agents"])
    if nb_agents_from_yaml == 0:
        # print(f"Warning: No agents defined in {input_yaml_path}.") # Less verbose
        return None # Or handle appropriately if an env with 0 agents is valid

    # Check consistency with main config
    if nb_agents_from_yaml != config.get("num_agents"):
         print(f"Warning: Agent count mismatch in {case_path}. YAML={nb_agents_from_yaml}, Config={config.get('num_agents')}. Using value from config.")
         # Decide which one takes precedence. Let's assume config is primary.

    dimensions = params["map"]["dimensions"] # Should be [width, height]
    # Make sure dimensions match config['board_size'] = [rows, cols]
    if dimensions[1] != config['board_size'][0] or dimensions[0] != config['board_size'][1]:
        print(f"Warning: Map dimensions mismatch in {case_path}. YAML=[w={dimensions[0]}, h={dimensions[1]}], Config=[rows={config['board_size'][0]}, cols={config['board_size'][1]}]. Using value from config.")
        # Ensure config board_size is used by GraphEnv

    obstacles_yaml = params["map"].get("obstacles", []) # List of [x, y]
    # Convert CBS obstacles [x,y] to GraphEnv obstacles [row, col]
    obstacles_list = np.array([[item[1], item[0]] for item in obstacles_yaml], dtype=np.int32).reshape(-1, 2) if obstacles_yaml else np.empty((0,2), dtype=int)

    # Initialize numpy arrays for start/goal [row, col]
    starting_pos = np.zeros((config["num_agents"], 2), dtype=np.int32)
    goals_env = np.zeros((config["num_agents"], 2), dtype=np.int32)

    for i, agent_data in enumerate(params["agents"]):
         if i >= config["num_agents"]: break # Only process up to num_agents specified in config
         try:
            # CBS start/goal are [x, y] -> convert to [row, col]
            starting_pos[i, :] = np.array([agent_data["start"][1], agent_data["start"][0]], dtype=np.int32) # [row, col]
            goals_env[i, :] = np.array([agent_data["goal"][1], agent_data["goal"][0]], dtype=np.int32)       # [row, col]
         except (KeyError, IndexError, ValueError) as e:
             print(f"Error processing agent {i} data in {input_yaml_path}: {e}")
             return None # Invalid agent data

    # Ensure required config keys exist for GraphEnv
    required_keys = ["sensing_range", "pad", "board_size", "num_agents", "max_time", "min_time"] # Add others if needed
    if not all(key in config for key in required_keys):
         missing_keys = [k for k in required_keys if k not in config]
         print(f"Error: Config missing required key(s) for GraphEnv: {missing_keys}")
         return None

    try:
        # Pass necessary parameters from config
        env = GraphEnv(
            config=config, # Pass the whole config dict
            goal=goals_env, # Use the converted [row, col] goals
            sensing_range=config["sensing_range"],
            pad=config["pad"],
            starting_positions=starting_pos, # Use the converted [row, col] starts
            obstacles=obstacles_list, # Use the converted [row, col] obstacles
        )
        return env
    except Exception as e:
        print(f"Error initializing GraphEnv for {case_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def record_env(path, config):
    """
    Processes expert trajectories in a given dataset path, simulates them in the
    modified GraphEnv, and records 3-channel FOVs ('states.npy') and GSOs ('gso.npy').
    """
    try:
        cases = sorted([
            d for d in os.listdir(path)
            if d.startswith("case_") and os.path.isdir(os.path.join(path, d))
        ], key=lambda x: int(x.split('_')[-1]))
    except FileNotFoundError:
        print(f"Error: Directory not found: {path}")
        return
    except Exception as e:
        print(f"Error listing cases in {path}: {e}")
        return

    if not cases:
        print(f"No 'case_*' directories found in {path}. Nothing to record.")
        return

    print(f"Found {len(cases)} potential cases in {path} for recording.")

    # --- Calculate Trajectory Statistics (Optional but helpful) ---
    trajectory_lengths = []
    valid_cases_for_stats = []
    print("Analyzing trajectory lengths...")
    for case_dir in tqdm(cases, desc="Checking Trajectories", unit="case", leave=False):
        case_path = os.path.join(path, case_dir)
        trajectory_path = os.path.join(case_path, "trajectory.npy")
        if os.path.exists(trajectory_path):
            try:
                trajectory = np.load(trajectory_path) # Removed allow_pickle
                if trajectory.ndim == 2 and trajectory.shape[1] > 0 and trajectory.shape[0] == config['num_agents']:
                    trajectory_lengths.append(trajectory.shape[1])
                    valid_cases_for_stats.append(case_dir)
            except Exception as e:
                # print(f"Warning: Could not load/process {trajectory_path}: {e}") # Less verbose
                pass

    if not trajectory_lengths:
        print("No valid trajectories found to calculate statistics or record.")
        return

    t_lengths = np.array(trajectory_lengths)
    max_steps = np.max(t_lengths) if trajectory_lengths else 0
    min_steps = np.min(t_lengths) if trajectory_lengths else 0
    mean_steps = np.mean(t_lengths) if trajectory_lengths else 0

    print(f"\nTrajectory Statistics (based on {len(valid_cases_for_stats)} valid cases):")
    print(f"  Max steps: {max_steps}")
    print(f"  Min steps: {min_steps}")
    print(f"  Mean steps: {mean_steps:.2f}")
    # --- End Statistics ---

    # --- Record Environment States ---
    print(f"\nRecording states for {len(valid_cases_for_stats)} cases with valid trajectories...")
    recorded_count = 0
    skipped_count = 0

    for case_dir in tqdm(valid_cases_for_stats, desc="Recording Env States", unit="case"):
        case_path = os.path.join(path, case_dir)
        trajectory_path = os.path.join(case_path, "trajectory.npy")

        try:
            # Create environment for this case
            env = make_env(case_path, config)
            if env is None:
                skipped_count += 1
                continue

            # Load trajectory actions [N, T]
            trajectory_actions = np.load(trajectory_path)
            num_timesteps = trajectory_actions.shape[1] # T
            agent_nb = trajectory_actions.shape[0]

            if num_timesteps == 0 or agent_nb != env.nb_agents:
                 skipped_count += 1
                 continue

            # --- Initialize recording arrays ---
            # Get initial observation to determine shapes
            # Need to reset env to the actual starting positions from input.yaml
            _ , initial_info = env.reset() # Reset ensures correct start state
            initial_obs = env.getObservations() # Now get obs from start state

            # FOV shape: (num_agents, num_fov_channels, fov_size, fov_size)
            fov_shape = initial_obs['fov'].shape
            # GSO shape: (num_agents, num_agents)
            adj_shape = initial_obs['adj_matrix'].shape

            # State recordings: (T+1, N, C, H, W) - Includes initial state
            recordings_fov = np.zeros((num_timesteps + 1,) + fov_shape, dtype=initial_obs['fov'].dtype)
            # GSO recordings: (T+1, N, N) - Includes initial state GSO
            recordings_gso = np.zeros((num_timesteps + 1,) + adj_shape, dtype=initial_obs['adj_matrix'].dtype)
            # --- --------------------------- ---

            # Store initial state (t=0)
            recordings_fov[0] = initial_obs['fov']
            recordings_gso[0] = initial_obs['adj_matrix']

            # Simulate trajectory steps
            current_obs = initial_obs
            for i in range(num_timesteps): # Simulate T actions
                actions_at_step_i = trajectory_actions[:, i]

                # Step the environment using the expert action
                # Pass dummy embedding if env.step requires it, otherwise None
                # current_emb = current_obs['embeddings'] # Or env.getEmbedding()?
                current_obs, _, terminated, truncated, _ = env.step(actions_at_step_i, emb=None)

                # Store observation *after* action i (state at t=i+1)
                recordings_fov[i + 1] = current_obs['fov']
                recordings_gso[i + 1] = current_obs['adj_matrix']

                # Optional: Check for early termination if needed
                # if terminated or truncated: break

            # Save recorded data (FOV as states.npy, GSO as gso.npy)
            states_save_path = os.path.join(case_path, "states.npy")
            gso_save_path = os.path.join(case_path, "gso.npy")
            np.save(states_save_path, recordings_fov)
            np.save(gso_save_path, recordings_gso)

            recorded_count += 1

        except Exception as e:
            print(f"\nError processing {case_dir}: {e}")
            import traceback
            traceback.print_exc()
            skipped_count += 1

    print(f"\nRecording finished for path: {path}")
    print(f"Successfully recorded: {recorded_count} cases.")
    print(f"Skipped: {skipped_count} cases (due to errors or mismatches).")


# No __main__ block needed, this module is called by main_data.py