# File: grid/env_graph_gridv1.py
# (COMPLETE REVISED VERSION - Gymnasium Compliance, Collision Fixes, Render Fixes, Logging)

from copy import copy
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors # Import directly
import logging # Use logging for warnings

# --- Added for rgb_array rendering ---
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import gymnasium as gym # Use Gymnasium
from gymnasium import spaces # Use Gymnasium spaces

# Setup logger
logger = logging.getLogger(__name__)
# Basic config, assuming level might be set by main script later
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GraphEnv(gym.Env):
    """
    Environment for Grid-based Multi-Agent Path Finding with Graph Neural Networks.
    Uses partial observations (FOV) and allows for communication graph structure.
    Internally uses (row_y, col_x) coordinate system.

    Observation Space Dict:
        - fov: (N, C, H, W) float32 Box [0, 3]
        - adj_matrix: (N, N) float32 Box [0, 1]
        - embeddings: (N, 1) float32 Box [-inf, inf]

    Action Space: MultiDiscrete([5]*N) - Each agent chooses 0-4.
        0: Idle, 1: Right(x+1), 2: Up(y-1), 3: Left(x-1), 4: Down(y+1)
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10} # Supported modes

    def __init__(
        self,
        config: dict,
        goal: np.ndarray, # Expects shape (num_agents, 2) -> [[row_y, col_x], ...]
        starting_positions: np.ndarray | None = None, # Optional: np.array(num_agents, 2) -> [[row_y, col_x], ...]
        obstacles: np.ndarray | None = None, # Optional: np.array(num_obstacles, 2) -> [[row_y, col_x], ...]
    ):
        super().__init__() # Initialize Gym Env
        self.config = config
        # Ensure necessary config keys exist
        required_config_keys = ["max_time", "board_size", "num_agents", "sensing_range", "pad"]
        for key in required_config_keys:
             if key not in self.config:
                 raise ValueError(f"Config missing required key: {key}")

        self.max_time = int(self.config["max_time"])
        if not isinstance(self.config["board_size"], (list, tuple)) or len(self.config["board_size"]) != 2:
            raise ValueError("config['board_size'] must be a list/tuple of [rows, cols]")
        self.board_rows, self.board_cols = map(int, self.config["board_size"]) # Ensure integers

        self.obstacles = np.empty((0,2), dtype=int)
        if obstacles is not None and obstacles.size > 0:
             if obstacles.ndim != 2 or obstacles.shape[1] != 2:
                  raise ValueError("Obstacles must be a Nx2 array of [row, col]")
             # Filter obstacles outside bounds and convert to int
             valid_mask = (obstacles[:, 0] >= 0) & (obstacles[:, 0] < self.board_rows) & \
                          (obstacles[:, 1] >= 0) & (obstacles[:, 1] < self.board_cols)
             self.obstacles = obstacles[valid_mask].astype(int)
             if len(self.obstacles) < len(obstacles):
                  logger.warning(f"Removed {len(obstacles) - len(self.obstacles)} obstacles outside board bounds.")

        if goal.ndim != 2 or goal.shape[1] != 2:
            raise ValueError("Goal must be a Nx2 array of [row, col]")
        self.nb_agents = int(self.config["num_agents"])
        if goal.shape[0] != self.nb_agents:
             raise ValueError(f"Goal array shape mismatch. Expected ({self.nb_agents}, 2), got {goal.shape}")
        goal_int = goal.astype(int)
        # Ensure goals are within bounds
        if np.any((goal_int[:, 0] < 0) | (goal_int[:, 0] >= self.board_rows) | (goal_int[:, 1] < 0) | (goal_int[:, 1] >= self.board_cols)):
             raise ValueError("One or more goals are outside board boundaries.")
        # Ensure goals don't overlap with obstacles
        if self.obstacles.size > 0 and np.any(np.all(goal_int[:, np.newaxis, :] == self.obstacles[np.newaxis, :, :], axis=2)):
             raise ValueError("One or more goals overlap with obstacles.")
        self.goal = goal_int # Shape (num_agents, 2) -> [row_y, col_x]

        self.board = np.zeros((self.board_rows, self.board_cols), dtype=np.int8) # Use int8 for board codes: 0=empty, 1=agent, 2=obstacle

        # FOV calculation parameters
        self.sensing_range = float(self.config["sensing_range"]) # Used for FOV extent and communication graph
        self.pad = int(self.config["pad"])
        if self.pad <= 0: raise ValueError("pad must be >= 1")
        self.fov_size = (self.pad * 2) - 1 # e.g., pad=3 -> 5x5 FOV
        logger.info(f"GraphEnv Initialized: Board={self.board_rows}x{self.board_cols}, Agents={self.nb_agents}, Pad={self.pad}, FOV={self.fov_size}x{self.fov_size}")

        self.starting_positions = starting_positions # Note: reset uses random if None
        if self.starting_positions is not None:
             self.starting_positions = self.starting_positions.astype(int)

        # Agent state
        self.positionX = np.zeros(self.nb_agents, dtype=np.int32) # Current X (column)
        self.positionY = np.zeros(self.nb_agents, dtype=np.int32) # Current Y (row)
        self.positionX_temp = np.zeros_like(self.positionX) # For collision checking
        self.positionY_temp = np.zeros_like(self.positionY)
        self.embedding = np.ones((self.nb_agents, 1), dtype=np.float32) # Agent embeddings (if used)
        self.reached_goal = np.zeros(self.nb_agents, dtype=bool) # Track who reached goal
        self.time = 0

        # Action space (5 discrete actions)
        self.action_map_dy_dx = {
             0: (0, 0),  # Idle   (delta_row, delta_col)
             1: (0, 1),  # Right
             2: (-1, 0), # Up
             3: (0, -1), # Left
             4: (1, 0),  # Down
        }
        self.action_space = spaces.MultiDiscrete([5] * self.nb_agents) # Each agent chooses 0-4

        # Observation space
        self.num_fov_channels = 3
        self.observation_space = spaces.Dict({
             "fov": spaces.Box(low=0, high=3, shape=(self.nb_agents, self.num_fov_channels, self.fov_size, self.fov_size), dtype=np.float32),
             "adj_matrix": spaces.Box(low=0, high=1, shape=(self.nb_agents, self.nb_agents), dtype=np.float32),
             "embeddings": spaces.Box(low=-np.inf, high=np.inf, shape=(self.nb_agents, 1), dtype=np.float32)
        })

        # Rendering state
        norm = colors.Normalize(vmin=0.0, vmax=1.4, clip=True)
        self.mapper = cm.ScalarMappable(norm=norm, cmap=cm.inferno)
        # Store render mode from config for optional use, but render() method prioritizes passed mode
        self.render_mode = self.config.get('render_mode', None)
        self._plot_fig = None
        self._plot_ax = None
        self._color_obstacle = '#333333'
        self._color_goal = '#2ecc71'
        self._color_agent_base = '#3498db'
        self._color_agent_reached = '#95a5a6'
        self._color_neighbor_line = '#DDDDDD'
        self._color_boundary = 'black'


    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed) # Important for seeding self.np_random

        self.time = 0
        self.board.fill(0) # Reset board to empty (0)

        # Place obstacles on board
        if self.obstacles.size > 0:
             self.board[self.obstacles[:, 0], self.obstacles[:, 1]] = 2 # Use 2 for obstacles

        # Set starting positions
        occupied_mask = self.board != 0
        if self.starting_positions is not None:
            if self.starting_positions.ndim != 2 or self.starting_positions.shape != (self.nb_agents, 2):
                raise ValueError(f"starting_positions shape mismatch, expected ({self.nb_agents}, 2), got {self.starting_positions.shape}")
            self.positionY = self.starting_positions[:, 0].copy() # row
            self.positionX = self.starting_positions[:, 1].copy() # col

            # Validation
            if np.any((self.positionY < 0) | (self.positionY >= self.board_rows) | (self.positionX < 0) | (self.positionX >= self.board_cols)):
                raise ValueError("Provided starting positions are outside board boundaries.")
            if np.any(occupied_mask[self.positionY, self.positionX]):
                 colliding_agents = np.where(occupied_mask[self.positionY, self.positionX])[0]
                 raise ValueError(f"Provided starting positions collide with obstacles for agents: {colliding_agents} at {self.starting_positions[colliding_agents]}.")
            unique_starts, counts = np.unique(self.starting_positions, axis=0, return_counts=True)
            if np.any(counts > 1):
                 raise ValueError(f"Provided starting positions contain duplicates: {unique_starts[counts > 1]}.")
        else:
            # Generate random starting positions
            possible_coords = np.argwhere(~occupied_mask)
            if len(possible_coords) < self.nb_agents:
                 raise RuntimeError(f"Not enough free space ({len(possible_coords)}) for {self.nb_agents} agents.")
            chosen_indices = self.np_random.choice(len(possible_coords), size=self.nb_agents, replace=False)
            start_coords = possible_coords[chosen_indices]
            self.positionY = start_coords[:, 0]
            self.positionX = start_coords[:, 1]

        # Reset agent state variables
        self.embedding = np.ones((self.nb_agents, 1), dtype=np.float32)
        self.reached_goal.fill(False)

        # Check initial goal achievement
        self.reached_goal = np.all(np.stack([self.positionY, self.positionX], axis=1) == self.goal, axis=1)

        # Update board
        self.board[self.positionY, self.positionX] = 1

        # Compute initial graph
        self._compute_comm_graph()
        observation = self.getObservations()
        info = self._get_info()

        return observation, info

    def getObservations(self):
        """Returns the current observation dictionary."""
        self.updateBoard() # Ensure board is up-to-date
        obs = {
            "fov": self.generate_fov(),
            "adj_matrix": self.adj_matrix.copy(),
            "embeddings": self.embedding.copy(),
        }
        return obs

    def _get_info(self):
        """Provides auxiliary information."""
        current_pos = np.stack([self.positionY, self.positionX], axis=1)
        dist_to_goal = np.linalg.norm(current_pos - self.goal, axis=1)
        return {
            "time": self.time,
            "positions": current_pos.copy(),
            "distance_to_goal": dist_to_goal,
            "agents_at_goal": self.reached_goal.copy()
            }

    def get_current_positions(self):
        """Returns current agent positions as (N, 2) array [row, col]."""
        return np.stack([self.positionY, self.positionX], axis=1)

    def _compute_comm_graph(self):
        """Calculates adjacency matrix based on sensing_range."""
        current_pos = np.stack([self.positionY, self.positionX], axis=1)
        delta = current_pos[:, np.newaxis, :] - current_pos[np.newaxis, :, :]
        dist_sq = np.sum(delta**2, axis=2)
        adj = (dist_sq > 1e-9) & (dist_sq < self.sensing_range**2)
        np.fill_diagonal(adj, False)
        self.adj_matrix = adj.astype(np.float32)

    def step(self, actions: np.ndarray):
        """Transitions the environment based on agent actions."""
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions, dtype=int)
        if actions.shape != (self.nb_agents,):
             raise ValueError(f"Actions shape incorrect. Expected ({self.nb_agents},), got {actions.shape}")
        if np.any((actions < 0) | (actions >= len(self.action_map_dy_dx))):
            logger.warning(f"Received invalid action index in {actions}. Clamping to Idle.")
            actions = np.clip(actions, 0, len(self.action_map_dy_dx) - 1)

        self.positionX_temp = self.positionX.copy()
        self.positionY_temp = self.positionY.copy()

        proposedY = self.positionY.copy()
        proposedX = self.positionX.copy()
        did_collide = np.zeros(self.nb_agents, dtype=bool)
        active_agent_mask = ~self.reached_goal

        # Calculate proposed moves
        for agent_id in np.where(active_agent_mask)[0]:
             act = actions[agent_id]
             dy, dx = self.action_map_dy_dx.get(act, (0,0))
             proposedY[agent_id] += dy
             proposedX[agent_id] += dx

        # Clamp to boundaries
        proposedY[active_agent_mask] = np.clip(proposedY[active_agent_mask], 0, self.board_rows - 1)
        proposedX[active_agent_mask] = np.clip(proposedX[active_agent_mask], 0, self.board_cols - 1)

        # Check obstacle collisions
        if self.obstacles.size > 0:
            active_indices_obs = np.where(active_agent_mask)[0]
            if len(active_indices_obs) > 0:
                 proposed_coords_active_obs = np.stack([proposedY[active_indices_obs], proposedX[active_indices_obs]], axis=1)
                 obstacle_collision_mask_obs = np.any(np.all(proposed_coords_active_obs[:, np.newaxis, :] == self.obstacles[np.newaxis, :, :], axis=2), axis=1)
                 colliding_agent_indices_obs = active_indices_obs[obstacle_collision_mask_obs]
                 if colliding_agent_indices_obs.size > 0:
                      proposedY[colliding_agent_indices_obs] = self.positionY_temp[colliding_agent_indices_obs]
                      proposedX[colliding_agent_indices_obs] = self.positionX_temp[colliding_agent_indices_obs]
                      did_collide[colliding_agent_indices_obs] = True
                      active_agent_mask[colliding_agent_indices_obs] = False # Mark as inactive

        # Check agent-agent collisions
        active_indices_agent = np.where(active_agent_mask)[0]
        if len(active_indices_agent) > 1:
            current_coords_check = np.stack([self.positionY_temp[active_indices_agent], self.positionX_temp[active_indices_agent]], axis=1)
            proposed_coords_check = np.stack([proposedY[active_indices_agent], proposedX[active_indices_agent]], axis=1)

            # Vertex collisions
            unique_coords, unique_map, counts = np.unique(proposed_coords_check, axis=0, return_inverse=True, return_counts=True)
            vertex_collision_agents = active_indices_agent[np.isin(unique_map, np.where(counts > 1)[0])]

            # Edge collisions
            swapping_agents_list = []
            rel_idx = np.arange(len(active_indices_agent))
            for i in rel_idx:
                 for j in range(i + 1, len(active_indices_agent)):
                     if np.array_equal(proposed_coords_check[i], current_coords_check[j]) and np.array_equal(proposed_coords_check[j], current_coords_check[i]):
                         swapping_agents_list.extend([active_indices_agent[i], active_indices_agent[j]])
            swapping_collision_agents = np.unique(swapping_agents_list)

            # Combine and revert
            agents_to_shield_idx = np.unique(np.concatenate([vertex_collision_agents, swapping_collision_agents])).astype(int)
            if agents_to_shield_idx.size > 0:
                 proposedY[agents_to_shield_idx] = self.positionY_temp[agents_to_shield_idx]
                 proposedX[agents_to_shield_idx] = self.positionX_temp[agents_to_shield_idx]
                 did_collide[agents_to_shield_idx] = True

        # Final Position Update & Goal Check
        self.positionY = proposedY
        self.positionX = proposedX
        current_pos = np.stack([self.positionY, self.positionX], axis=1)
        self.reached_goal = np.all(current_pos == self.goal, axis=1)

        # Update Env State
        self.time += 1
        self._compute_comm_graph()
        self.updateBoard()

        # Determine termination and truncation
        terminated = np.all(self.reached_goal)
        truncated = self.time >= self.max_time

        # Get observation, reward, info
        observation = self.getObservations()
        step_penalty = -0.01
        collision_penalty = -0.1
        goal_reward = 1.0
        reward = step_penalty * self.nb_agents
        reward += collision_penalty * np.sum(did_collide)
        if terminated:
            reward += goal_reward

        info = self._get_info()
        info["collisions_this_step"] = did_collide

        # Rendering is handled by the caller script (e.g., create_gif)

        return observation, reward, terminated, truncated, info


    def _updateEmbedding(self, H):
        """Allow updating embeddings if needed."""
        if H is not None and isinstance(H, np.ndarray) and H.shape == self.embedding.shape:
             self.embedding = H.astype(np.float32).copy()

    def map_goal_to_fov(self, agent_id: int) -> tuple[int, int]:
        """Maps the agent's absolute goal coordinate to its FOV coordinates."""
        center_offset = self.pad - 1
        relative_y = self.goal[agent_id, 0] - self.positionY[agent_id]
        relative_x = self.goal[agent_id, 1] - self.positionX[agent_id]
        fov_y = center_offset - relative_y
        fov_x = center_offset + relative_x

        if 0 <= fov_y < self.fov_size and 0 <= fov_x < self.fov_size:
             return int(fov_y), int(fov_x)
        else:
             proj_y = np.clip(fov_y, 0, self.fov_size - 1)
             proj_x = np.clip(fov_x, 0, self.fov_size - 1)
             return int(proj_y), int(proj_x)

    def generate_fov(self) -> np.ndarray:
        """Generates the 3-Channel Field of View (FOV) for each agent."""
        map_padded = np.pad(self.board, ((self.pad, self.pad), (self.pad, self.pad)), mode='constant', constant_values=2)
        current_posY_padded = self.positionY + self.pad
        current_posX_padded = self.positionX + self.pad
        FOV = np.zeros((self.nb_agents, self.num_fov_channels, self.fov_size, self.fov_size), dtype=np.float32)
        center_idx = self.pad - 1

        for agent_id in range(self.nb_agents):
            row_start = current_posY_padded[agent_id] - center_idx
            row_end = row_start + self.fov_size
            col_start = current_posX_padded[agent_id] - center_idx
            col_end = col_start + self.fov_size

            # Channel 0: Obstacles and Other Agents
            local_view = map_padded[row_start:row_end, col_start:col_end]
            FOV[agent_id, 0, :, :] = local_view
            FOV[agent_id, 0, center_idx, center_idx] = 0 # Mask self

            # Channel 1: Goal Location
            gy, gx = self.map_goal_to_fov(agent_id)
            FOV[agent_id, 1, gy, gx] = 3

            # Channel 2: Self Position
            FOV[agent_id, 2, center_idx, center_idx] = 1

        return FOV

    def updateBoard(self):
        """Updates self.board representation based on current agent positions."""
        self.board.fill(0)
        if self.obstacles.size > 0:
            valid_obs_mask = (self.obstacles[:, 0] >= 0) & (self.obstacles[:, 0] < self.board_rows) & \
                             (self.obstacles[:, 1] >= 0) & (self.obstacles[:, 1] < self.board_cols)
            obs_rows = self.obstacles[valid_obs_mask, 0]
            obs_cols = self.obstacles[valid_obs_mask, 1]
            self.board[obs_rows, obs_cols] = 2

        rows, cols = self.positionY, self.positionX
        valid_agent_mask = (rows >= 0) & (rows < self.board_rows) & (cols >= 0) & (cols < self.board_cols)
        valid_rows = rows[valid_agent_mask]
        valid_cols = cols[valid_agent_mask]

        if not np.all(valid_agent_mask):
             logger.warning(f"Agent positions out of bounds before board update! Y:{self.positionY}, X:{self.positionX}")

        target_cells = self.board[valid_rows, valid_cols]
        agent_placement_mask = (target_cells != 2)

        if np.any(~agent_placement_mask):
             logger.warning(f"Attempted to place agent(s) on obstacle(s) in updateBoard!")

        self.board[valid_rows[agent_placement_mask], valid_cols[agent_placement_mask]] = 1

    def render(self, mode: str | None = None): # Updated Signature
        """ Renders the environment. Uses self.render_mode if mode is None."""
        # print(f"--- DEBUG: Inside GraphEnv.render --- mode={mode}, self.render_mode={self.render_mode}") # Debug print
        render_mode_to_use = mode if mode is not None else self.render_mode

        if render_mode_to_use is None:
            gym.logger.warn("Cannot render setup with render_mode=None.")
            return
        if render_mode_to_use not in self.metadata["render_modes"]:
             raise ValueError(f"Unsupported render mode: {render_mode_to_use}.")

        # --- Setup Figure ---
        if render_mode_to_use == "human":
            if self._plot_fig is None:
                plt.ion()
                self._plot_fig, self._plot_ax = plt.subplots(figsize=(6, 6))
            ax = self._plot_ax
        elif render_mode_to_use == "rgb_array":
            fig = Figure(figsize=(5, 5), dpi=100)
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
        else: return # Should not happen

        # --- Drawing Logic ---
        ax.clear()
        ax.set_facecolor('#F0F0F0')
        ax.set_xticks(np.arange(-.5, self.board_cols, 1), minor=True)
        ax.set_yticks(np.arange(-.5, self.board_rows, 1), minor=True)
        ax.grid(which='minor', color='#CCCCCC', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', size=0)
        ax.tick_params(which='major', bottom=False, left=False, labelbottom=False, labelleft=False)

        # Draw obstacles
        if self.obstacles.size > 0:
             obs_y, obs_x = self.obstacles[:, 0], self.obstacles[:, 1]
             for r, c in zip(obs_y, obs_x):
                  rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=self._color_obstacle, edgecolor='#222222', linewidth=0.5)
                  ax.add_patch(rect)
        # Draw goals
        if self.goal.size > 0:
             ax.scatter(self.goal[:, 1], self.goal[:, 0], color=self._color_goal, marker="*", s=200, zorder=3, alpha=0.7, edgecolors='#111111', linewidth=0.5)
        # Draw agents
        current_pos_x = self.positionX; current_pos_y = self.positionY
        agent_colors = [self._color_agent_reached if self.reached_goal[i] else self._color_agent_base for i in range(self.nb_agents)]
        ax.scatter(current_pos_x, current_pos_y, s=150, c=agent_colors, zorder=4, edgecolors='black', linewidth=0.5)
        for i in range(self.nb_agents):
             ax.text(current_pos_x[i], current_pos_y[i], str(i), color='white', ha='center', va='center', fontsize=8, fontweight='bold', zorder=5)

        # Set Limits and Title
        ax.set_xlim(-0.5, self.board_cols - 0.5)
        ax.set_ylim(-0.5, self.board_rows - 0.5)
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"Step: {self.time} | Reached: {np.sum(self.reached_goal)}/{self.nb_agents}")
        # --- End Drawing Logic ---

        # --- Finalize Rendering ---
        if render_mode_to_use == "human":
            self._plot_fig.canvas.draw()
            self._plot_fig.canvas.flush_events()
            plt.pause(0.01)
            return None
        elif render_mode_to_use == "rgb_array":
            canvas.draw()
            image = np.asarray(canvas.buffer_rgba())
            # plt.close(fig) # Closing local fig might not be necessary
            return image

    def close(self):
        """Closes any rendering resources."""
        if self._plot_fig is not None:
            plt.close(self._plot_fig)
            self._plot_fig = None
            self._plot_ax = None
            if plt.isinteractive():
                 plt.ioff()

# --- Utility functions outside class ---
# These are kept separate as they don't depend on Env state (self)
def create_goals(board_size: tuple[int, int], num_agents: int, obstacles: np.ndarray | None = None, current_starts: np.ndarray | None = None) -> np.ndarray:
    """Creates random goal/start locations avoiding obstacles and optionally current_starts."""
    rows, cols = board_size
    occupied_mask = np.zeros((rows, cols), dtype=bool)

    if obstacles is not None and obstacles.size > 0:
        valid_obstacles = obstacles[(obstacles[:, 0] >= 0) & (obstacles[:, 0] < rows) & (obstacles[:, 1] >= 0) & (obstacles[:, 1] < cols)]
        if len(valid_obstacles) > 0:
             occupied_mask[valid_obstacles[:, 0], valid_obstacles[:, 1]] = True
    if current_starts is not None and current_starts.size > 0:
         valid_starts = current_starts[(current_starts[:, 0] >= 0) & (current_starts[:, 0] < rows) & (current_starts[:, 1] >= 0) & (current_starts[:, 1] < cols)]
         if len(valid_starts) > 0:
              occupied_mask[valid_starts[:, 0], valid_starts[:, 1]] = True

    available_coords = np.argwhere(~occupied_mask)
    if len(available_coords) < num_agents:
        raise ValueError(f"Not enough free spaces ({len(available_coords)}) to place {num_agents} goals/starts.")

    chosen_indices = np.random.choice(len(available_coords), size=num_agents, replace=False)
    goals = available_coords[chosen_indices]
    return goals.astype(int)

def create_obstacles(board_size: tuple[int, int], nb_obstacles: int) -> np.ndarray:
    """Creates random obstacle locations."""
    rows, cols = board_size
    total_cells = rows * cols
    if nb_obstacles < 0: nb_obstacles = 0
    if nb_obstacles >= total_cells:
         logger.warning(f"Requested obstacles ({nb_obstacles}) >= total cells ({total_cells}). Placing {total_cells - 1} instead.")
         nb_obstacles = total_cells - 1

    all_coords = np.array([(r, c) for r in range(rows) for c in range(cols)])
    chosen_indices = np.random.choice(total_cells, size=nb_obstacles, replace=False)
    obstacles = all_coords[chosen_indices]
    return obstacles.astype(int)


# --- Main block for testing ---
if __name__ == "__main__":
    logger.info("--- Running GraphEnv Example ---")
    # Define parameters for test
    agents_test = 5
    bs_test = 12
    board_dims_test = [bs_test, bs_test]
    num_obstacles_test = 10
    sensing_range_test = 4
    pad_test = 3
    max_time_test = 60

    config_test = {
        "num_agents": agents_test, "board_size": board_dims_test, "max_time": max_time_test,
        "sensing_range": sensing_range_test, "pad": pad_test, "render_mode": "human"
    }
    env = None # Initialize env to None for finally block
    try:
        obstacles_arr = create_obstacles(board_dims_test, num_obstacles_test)
        start_pos_arr = create_goals(board_dims_test, agents_test, obstacles=obstacles_arr)
        goals_arr = create_goals(board_dims_test, agents_test, obstacles=obstacles_arr, current_starts=start_pos_arr)

        env = GraphEnv(config=config_test, goal=goals_arr, starting_positions=start_pos_arr, obstacles=obstacles_arr)
        obs, info = env.reset(seed=42)
        logger.info(f"Initial State: Positions={info['positions'].tolist()}, Reached={info['agents_at_goal']}")
        if env.render_mode == 'human': env.render()
        plt.pause(1)

        total_reward = 0
        for step in range(max_time_test):
            actions = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(actions)
            total_reward += reward
            print(f"\rStep: {info['time']}, Term: {terminated}, Trunc: {truncated}, Reward: {reward:.3f}, Reached: {np.sum(info['agents_at_goal'])}", end="")
            if env.render_mode == 'human': env.render()
            if terminated or truncated:
                print(f"\nEpisode finished at step {info['time']}. Success: {terminated}. Total Reward: {total_reward:.3f}")
                break

        if not (terminated or truncated): print(f"\nEpisode reached step limit ({max_time_test}). Total Reward: {total_reward:.3f}")
        logger.info(f"Final State: Positions={info['positions'].tolist()}, Reached={info['agents_at_goal']}")
        if env.render_mode == 'human': env.render(); plt.pause(2)

    except Exception as e:
        logger.error(f"Error during environment test: {e}", exc_info=True)
    finally:
        if env is not None: env.close()

    logger.info("--- GraphEnv Example Finished ---")