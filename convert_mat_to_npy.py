# File: convert_mat_to_npy.py

import numpy as np
import scipy.io
from pathlib import Path
import yaml
from tqdm import tqdm
import logging
import argparse
import shutil

# --- Setup Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Action Mapping ---
# Mapping from (delta_row, delta_col) to your required action index (0-4)
# IMPORTANT: Verify this matches the action definition in your GraphEnv!
ACTION_MAP_DELTA_RC_TO_ACTION = {
    (0, 0): 0,  # Idle
    (0, 1): 1,  # Right (dx=1)
    (-1, 0): 2, # Up    (dy=-1)
    (0, -1): 3, # Left  (dx=-1)
    (1, 0): 4,  # Down  (dy=1)
}

def get_action_from_delta(prev_pos_rc, next_pos_rc):
    """Calculates action index based on position change."""
    delta_row = int(round(next_pos_rc[0] - prev_pos_rc[0]))
    delta_col = int(round(next_pos_rc[1] - prev_pos_rc[1]))
    action_tuple = (delta_row, delta_col)
    return ACTION_MAP_DELTA_RC_TO_ACTION.get(action_tuple, 0) # Default to Idle (0)

def convert_mat_file(mat_file_path: Path, output_case_dir: Path, target_num_agents: int, target_pad: int):
    """Converts a single .mat file to the npy/yaml format in a case directory."""
    try:
        output_case_dir.mkdir(parents=True, exist_ok=True)
        mat_data = scipy.io.loadmat(mat_file_path)

        # --- Extract Data ---
        map_data = mat_data.get('map')              # (H, W) -> (Rows, Cols)
        goal_data = mat_data.get('goal')            # (N, 2) -> Agent goals [row, col]? Verify order!
        fov_data = mat_data.get('inputTensor')      # (T, N, C, H_fov, W_fov)
        target_onehot = mat_data.get('target')      # (T, N, 5) - Assumed one-hot
        gso_data = mat_data.get('GSO')              # (T, N, N)
        pos_data = mat_data.get('inputState')       # (T, N, 2) -> Agent positions [row, col]? Verify order!
        # makespan = mat_data.get('makespan')         # Optional

        # --- Basic Validation ---
        if any(d is None for d in [map_data, goal_data, fov_data, target_onehot, gso_data, pos_data]):
            logger.warning(f"Skipping {mat_file_path.name}: Missing required keys.")
            shutil.rmtree(output_case_dir) # Clean up incomplete case
            return False, "missing_keys"

        # --- Dimension Checks ---
        map_rows, map_cols = map_data.shape
        T_fov, N_fov, C_fov, H_fov, W_fov = fov_data.shape
        T_tgt, N_tgt, A_tgt = target_onehot.shape
        T_gso, N_gso1, N_gso2 = gso_data.shape
        T_pos, N_pos, D_pos = pos_data.shape

        # Check time consistency (allow position data to be longer if needed)
        if not (T_fov == T_tgt == T_gso) or T_pos < T_fov:
             logger.warning(f"Skipping {mat_file_path.name}: Inconsistent time dimensions (Fov/Tgt/GSO: {T_fov}, Pos: {T_pos}).")
             shutil.rmtree(output_case_dir); return False, "time_dim_mismatch"
        # Check agent consistency
        if not (N_fov == N_tgt == N_gso1 == N_gso2 == N_pos):
             logger.warning(f"Skipping {mat_file_path.name}: Inconsistent agent dimensions.")
             shutil.rmtree(output_case_dir); return False, "agent_dim_mismatch"
        # Check against target config
        if N_fov != target_num_agents:
            logger.warning(f"Skipping {mat_file_path.name}: Agent count ({N_fov}) != target ({target_num_agents}).")
            shutil.rmtree(output_case_dir); return False, "config_agent_mismatch"
        # Check FOV dimensions
        expected_fov_size = (target_pad * 2) - 1
        if H_fov != expected_fov_size or W_fov != expected_fov_size:
             logger.warning(f"Skipping {mat_file_path.name}: FOV size ({H_fov}x{W_fov}) != expected ({expected_fov_size}x{expected_fov_size} from pad={target_pad}).")
             shutil.rmtree(output_case_dir); return False, "config_fov_mismatch"

        num_agents = N_fov
        num_timesteps = T_fov # Use length of FOV/Target/GSO data

        # --- 1. Generate input.yaml ---
        # Find obstacle coordinates [col, row] format for yaml
        obstacle_coords_cr = np.argwhere(map_data == 1.0) # Assuming 1.0 marks obstacles
        obstacles_xy_list = [[int(c), int(r)] for r, c in obstacle_coords_cr]

        # Get start positions [col, row] from pos_data at t=0
        # IMPORTANT: Double-check if pos_data is [row, col] or [col, row]
        # Assuming pos_data is [row, col] based on map shape convention
        start_pos_rc = pos_data[0, :, :] # Shape (N, 2)
        start_pos_xy_list = [[int(pos[1]), int(pos[0])] for pos in start_pos_rc]

        # Get goal positions [col, row]
        # IMPORTANT: Double-check if goal_data is [row, col] or [col, row]
        # Assuming goal_data is [row, col]
        goal_pos_rc = goal_data
        goal_pos_xy_list = [[int(pos[1]), int(pos[0])] for pos in goal_pos_rc]

        agents_list_yaml = []
        for i in range(num_agents):
            agents_list_yaml.append({
                "start": start_pos_xy_list[i],
                "goal": goal_pos_xy_list[i],
                "name": f"agent{i}"
            })

        input_yaml_content = {
            "agents": agents_list_yaml,
            "map": {
                "dimensions": [int(map_cols), int(map_rows)], # [width, height]
                "obstacles": obstacles_xy_list
            }
        }
        with open(output_case_dir / "input.yaml", 'w') as f:
            yaml.dump(input_yaml_content, f, default_flow_style=None, sort_keys=False)

        # --- 2. Generate trajectory.npy ---
        # Convert one-hot target to action indices
        # Actions correspond to transitions from t to t+1
        num_actions = num_timesteps - 1
        if num_actions < 0: # Only one state, no actions
            logger.warning(f"Skipping {mat_file_path.name}: Only one timestep found, no actions.")
            shutil.rmtree(output_case_dir); return False, "no_actions"

        trajectory_npy = np.zeros((num_agents, num_actions), dtype=np.int64)
        target_indices = np.argmax(target_onehot, axis=2) # Shape (T, N)

        # Note: target[t] is the action taken *at* time t, leading to state t+1
        # So, trajectory_npy[:, t] should be derived from target_indices[t, :]
        for t in range(num_actions):
            trajectory_npy[:, t] = target_indices[t, :]

        np.save(output_case_dir / "trajectory.npy", trajectory_npy)


        # --- 3. Generate solution.yaml ---
        # Reconstruct schedule from pos_data [row, col] -> [x=col, y=row]
        schedule_yaml = {}
        for i in range(num_agents):
            agent_path = []
            # Use pos_data up to num_timesteps (length of FOV/Target/GSO data)
            for t in range(num_timesteps):
                 pos_rc = pos_data[t, i, :]
                 agent_path.append({'t': t, 'x': int(pos_rc[1]), 'y': int(pos_rc[0])})
            schedule_yaml[f"agent{i}"] = agent_path

        # Calculate cost (Sum of Costs = Sum of path lengths - N)
        cost = sum(len(p) - 1 for p in schedule_yaml.values() if p)

        solution_yaml_content = {
            "cost": cost,
            "schedule": schedule_yaml
        }
        with open(output_case_dir / "solution.yaml", 'w') as f:
            yaml.dump(solution_yaml_content, f, default_flow_style=None, sort_keys=False)

        # --- 4. Save states.npy ---
        # Input shape is (T, N, C, H_fov, W_fov)
        # Required shape is (T, N, C, H_fov, W_fov) - seems correct already
        if fov_data.dtype != np.float32: fov_data = fov_data.astype(np.float32)
        np.save(output_case_dir / "states.npy", fov_data)

        # --- 5. Save gso.npy ---
        # Input shape is (T, N, N)
        # Required shape is (T, N, N) - seems correct already
        if gso_data.dtype != np.float32: gso_data = gso_data.astype(np.float32)
        np.save(output_case_dir / "gso.npy", gso_data)

        return True, "success"

    except Exception as e:
        logger.error(f"Failed processing {mat_file_path.name}: {e}", exc_info=True)
        if output_case_dir.exists(): # Clean up on any processing error
            try: shutil.rmtree(output_case_dir)
            except OSError: pass
        return False, f"processing_error_{type(e).__name__}"


# --- Main Script Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .mat dataset to npy/yaml format.")
    parser.add_argument("mat_dir", type=Path, help="Directory containing the input .mat files (e.g., .../train).")
    parser.add_argument("output_dir", type=Path, help="Directory to save the output 'case_XXXXX' folders.")
    parser.add_argument("--num_agents", type=int, required=True, help="Expected number of agents (N) in the .mat files.")
    parser.add_argument("--pad", type=int, required=True, help="Expected 'pad' value corresponding to the FOV size in .mat files (e.g., pad=6 for 11x11).")
    parser.add_argument("--start_case_idx", type=int, default=0, help="Starting index for output case directories (case_XXXXX).")

    args = parser.parse_args()

    if not args.mat_dir.is_dir():
        logger.error(f"Input directory not found: {args.mat_dir}")
        exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    mat_files = sorted(list(args.mat_dir.glob("*.mat")))
    if not mat_files:
        logger.warning(f"No .mat files found in {args.mat_dir}")
        exit(0)

    logger.info(f"Found {len(mat_files)} .mat files in {args.mat_dir}.")
    logger.info(f"Output will be saved to: {args.output_dir}")
    logger.info(f"Expecting {args.num_agents} agents and pad={args.pad}.")

    success_count = 0
    fail_count = 0
    failure_details = {}

    pbar = tqdm(mat_files, desc="Converting", unit="file")
    for i, mat_file in enumerate(pbar):
        case_index = args.start_case_idx + i
        output_case_path = args.output_dir / f"case_{case_index:05d}" # Pad index with zeros

        # Optional: Skip if output already exists fully
        # if (output_case_path / "states.npy").exists():
        #     logger.debug(f"Skipping case {case_index}, already exists.")
        #     success_count +=1 # Assume existing is success for count
        #     continue

        status, reason = convert_mat_file(mat_file, output_case_path, args.num_agents, args.pad)

        if status:
            success_count += 1
        else:
            fail_count += 1
            failure_details[reason] = failure_details.get(reason, 0) + 1

        pbar.set_postfix({"Success": success_count, "Failed": fail_count})

    logger.info("\n--- Conversion Summary ---")
    logger.info(f"Successfully converted: {success_count} files.")
    logger.info(f"Failed to convert: {fail_count} files.")
    if failure_details:
        logger.info("Failure reasons:")
        for reason, count in failure_details.items():
            logger.info(f"  - {reason}: {count}")