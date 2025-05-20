# File: data_generation/main_data.py
# (MODIFIED FOR RE-RECORDING - _record_one_case_worker defined locally)

import os
import yaml
import numpy as np
from pathlib import Path
import traceback
import logging
import time
import shutil
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
# signal import not needed if not running full data_gen
# torch import not needed for this specific task if not using GPU for record workers

# --- Setup Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s'
)

# --- Imports ---
try:
    # parse_traject is run sequentially
    from .trayectory_parser import parse_traject
    # make_env is needed
    from .record import make_env # Only make_env is needed from record.py
    logger.info("Successfully imported data generation submodules for re-recording.")
except ImportError as e:
    logger.error(f"FATAL ERROR: Failed to import submodules: {e}", exc_info=True)
    exit(1)


# --- WORKER FUNCTION DEFINED LOCALLY FOR RE-RECORDING ---
def _record_one_case_worker(case_path: Path, config: dict) -> tuple[str, bool]:
    """Worker function to record states/GSO for ONE case."""
    env = None
    try:
        # --- Determine expected shapes ---
        N = int(config["num_agents"])
        C = 3  # Fixed channels: Obstacles/Agents, Goal, Self
        pad_val = int(config["pad"])
        H = W = (pad_val * 2) - 1
        expected_fov_shape = (N, C, H, W)
        expected_gso_shape = (N, N)

        # --- Load Trajectory ---
        trajectory_path = case_path / "trajectory.npy"
        if not trajectory_path.exists():
            # logger.debug(f"Recording skipped for {case_path.name}: trajectory.npy missing.")
            return case_path.name, False # Cannot record without trajectory

        trajectory_actions = np.load(trajectory_path) # Shape (N, T_actions)
        if trajectory_actions.ndim != 2:
            logger.warning(f"Recording failed for {case_path.name}: Trajectory ndim != 2.")
            return case_path.name, False
        num_agents_traj = trajectory_actions.shape[0]
        num_timesteps = trajectory_actions.shape[1] # T_actions

        if num_timesteps == 0:
            # logger.debug(f"Recording skipped for {case_path.name}: Trajectory has 0 timesteps.")
            return case_path.name, False
        max_traj_len = config.get("max_trajectory_length_recording", 500)
        if num_timesteps > max_traj_len:
            logger.warning(f"Recording skipped for {case_path.name}: Trajectory too long ({num_timesteps} > {max_traj_len}).")
            return case_path.name, False

        # --- Create Env ---
        env = make_env(case_path, config) # make_env uses num_agents from config
        if env is None:
            logger.warning(f"Recording failed for {case_path.name}: Env creation failed.")
            return case_path.name, False
        if num_agents_traj != env.nb_agents: # Ensure trajectory matches env agent count
             logger.warning(f"Recording failed for {case_path.name}: Agent mismatch traj ({num_agents_traj}) vs env ({env.nb_agents}).")
             return case_path.name, False


        # --- Simulate and Record ---
        initial_obs, _ = env.reset(seed=np.random.randint(1e6)) # Random seed per case
        initial_fov = initial_obs.get('fov')
        initial_gso = initial_obs.get('adj_matrix')

        # Check initial observation shapes
        if initial_fov is None or initial_fov.shape != expected_fov_shape:
             logger.warning(f"Recording failed for {case_path.name}: Bad initial FOV shape {initial_fov.shape if initial_fov is not None else 'None'} != {expected_fov_shape}.")
             return case_path.name, False
        if initial_gso is None or initial_gso.shape != expected_gso_shape:
             logger.warning(f"Recording failed for {case_path.name}: Bad initial GSO shape {initial_gso.shape if initial_gso is not None else 'None'} != {expected_gso_shape}.")
             return case_path.name, False

        recorded_fov_list = [initial_fov]
        recorded_gso_list = [initial_gso]
        sim_successful = True

        for i in range(num_timesteps): # Iterate through T_actions
            actions_at_step_i = trajectory_actions[:, i]
            current_obs, _, terminated, truncated, info = env.step(actions_at_step_i)
            obs_fov = current_obs.get('fov')
            obs_gso = current_obs.get('adj_matrix')

            # Check shapes during sim
            if obs_fov is None or obs_fov.shape != expected_fov_shape or \
               obs_gso is None or obs_gso.shape != expected_gso_shape:
                logger.warning(f"Recording failed for {case_path.name}: Bad shape at step {i+1}.")
                sim_successful = False; break

            recorded_fov_list.append(obs_fov)
            recorded_gso_list.append(obs_gso)

            if terminated or truncated: # Env finished early or hit internal max_time
                break

        # --- Save ---
        if sim_successful:
            recordings_fov = np.array(recorded_fov_list, dtype=np.float32)
            recordings_gso = np.array(recorded_gso_list, dtype=np.float32)

            if recordings_fov.shape[0] != recordings_gso.shape[0]:
                 logger.error(f"Recording internal error for {case_path.name}: FOV/GSO list length mismatch.")
                 return case_path.name, False

            states_save_path = case_path / "states.npy"
            gso_save_path = case_path / "gso.npy"
            np.save(states_save_path, recordings_fov)
            np.save(gso_save_path, recordings_gso)
            return case_path.name, True
        else:
            return case_path.name, False # Simulation failed

    except Exception as e:
        logger.error(f"Unhandled exception in record worker for case {case_path.name}: {e}", exc_info=True)
        return case_path.name, False
    finally:
        if env is not None:
            env.close()
# --- END OF _record_one_case_worker ---


# --- Parallel Recording Function (uses the locally defined worker) ---
def record_env_parallel(path: Path, config: dict, num_workers: int):
    """Records states/GSO in parallel using imap_unordered."""
    if not path.is_dir():
        logger.error(f"Dataset directory not found: {path}. Cannot record.")
        return

    cases_to_process = []
    try:
        all_trajectories = list(path.glob("case_*/trajectory.npy"))
        for traj_file in tqdm(all_trajectories, desc=f"Checking cases in {path.name} for recording", leave=False, unit="case"):
            case_path = traj_file.parent
            state_file = case_path / "states.npy"
            if not state_file.exists():
                cases_to_process.append(case_path)
    except Exception as e:
        logger.error(f"Error listing cases for recording in {path}: {e}", exc_info=True)
        return

    if not cases_to_process:
        num_with_states = len(list(path.glob("case_*/states.npy")))
        logger.info(f"No cases found in {path.name} needing new recording (based on missing states.npy). {num_with_states} might already exist.")
        logger.info("If you intend to re-record all, ensure states.npy are removed or modify script logic.")
        return

    logger.info(f"\n--- Recording States/GSOs in Parallel for Dataset: {path.name} ---")
    logger.info(f"Found {len(cases_to_process)} cases to process with {num_workers} workers.")

    recorded_count = 0
    failed_count = 0

    worker_func = partial(_record_one_case_worker, config=config) # Uses the local worker

    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(cases_to_process), desc=f"Recording ({path.name})", unit="case") as pbar:
            for case_name, success in pool.imap_unordered(worker_func, cases_to_process):
                if success:
                    recorded_count += 1
                else:
                    failed_count += 1
                pbar.update(1)
                pbar.set_postfix({"RecOK": recorded_count, "RecFail": failed_count})

    logger.info(f"\n--- Recording Finished for: {path.name} ---")
    logger.info(f"Successfully recorded: {recorded_count} cases.")
    if failed_count > 0:
        logger.warning(f"Failed to record for: {failed_count} cases.")


# --- Main Execution Logic for Re-Recording ---
if __name__ == "__main__":

    logger.info("Entered __main__ block - RE-RECORDING SCRIPT.")

    # --- Configuration ---
    try:
        logger.info("Defining configuration for re-recording...")

        # !!! USER: Define parameters for RE-RECORDING !!!
        num_agents_original = 5
        board_rows_original = 10
        board_cols_original = 10

        new_pad_for_fov = 3          # e.g., for 5x5 FOV
        new_sensing_range_for_gso = 6 # e.g., smaller communication range
        max_time_env_recording = 120

        dataset_paths_to_reprocess = [
            Path("dataset/map10x10_r5_o10_p5"),
            Path("dataset/map10x10_r5_o20_p5"),
            Path("dataset/map10x10_r5_o30_p5"),
        ]
        subdirs_to_process = ["train", "val", "test"]

        max_cpu_workers = cpu_count()
        num_parallel_workers = max(1, min(max_cpu_workers - 2, 32))
        logger.info(f"Using {num_parallel_workers} parallel workers for re-recording.")

        re_record_config = {
            "num_agents": num_agents_original,
            "board_size": [board_rows_original, board_cols_original],
            "sensing_range": new_sensing_range_for_gso,
            "pad": new_pad_for_fov,
            "max_time": max_time_env_recording,
            "map_shape": [(new_pad_for_fov * 2) - 1, (new_pad_for_fov * 2) - 1],
            "nb_agents": num_agents_original, # Used by make_env logic to read input.yaml
            "max_trajectory_length_recording": 500,
        }
        logger.info(f"Re-recording config to be used: {re_record_config}")

    except Exception as e:
        logger.error(f"FATAL ERROR during configuration: {e}", exc_info=True)
        exit(1)

    # --- Re-Recording Loop ---
    logger.info("\n--- Starting Re-Recording of States/GSO ---")
    overall_start_time = time.time()

    for base_dataset_path in dataset_paths_to_reprocess:
        for subdir_name in subdirs_to_process:
            current_path_to_process = base_dataset_path / subdir_name
            if not current_path_to_process.is_dir():
                if subdir_name == "" and base_dataset_path == current_path_to_process : # Processing root path
                     logger.warning(f"Base dataset path {base_dataset_path} not found. Skipping.")
                continue

            logger.info(f"\n\n{'='*15} Processing Directory: {current_path_to_process} {'='*15}")

            logger.info(f"Deleting existing states.npy and gso.npy in {current_path_to_process}...")
            deleted_states_count = 0; deleted_gso_count = 0
            for case_dir in current_path_to_process.glob("case_*"):
                if (case_dir / "states.npy").exists():
                    try: (case_dir / "states.npy").unlink(missing_ok=True); deleted_states_count += 1
                    except OSError as e: logger.error(f"Could not delete states.npy in {case_dir}: {e}")
                if (case_dir / "gso.npy").exists():
                    try: (case_dir / "gso.npy").unlink(missing_ok=True); deleted_gso_count += 1
                    except OSError as e: logger.error(f"Could not delete gso.npy in {case_dir}: {e}")
            logger.info(f"Deleted {deleted_states_count} states.npy and {deleted_gso_count} gso.npy files.")

            logger.info("  STEP 1: (Re-)Parsing Trajectories...")
            try:
                parse_traject(current_path_to_process)
            except Exception as e:
                 logger.error(f"  Error during parse_traject for {current_path_to_process}: {e}", exc_info=True)
                 continue

            logger.info("  STEP 2: Re-Recording Env States/GSO with new parameters...")
            try:
                record_env_parallel(current_path_to_process, re_record_config, num_parallel_workers)
            except Exception as e:
                logger.error(f"  Error during record_env_parallel for {current_path_to_process}: {e}", exc_info=True)
                continue

    overall_duration = time.time() - overall_start_time
    logger.info(f"\n\n--- Total Re-Recording Time: {overall_duration:.2f}s ---")