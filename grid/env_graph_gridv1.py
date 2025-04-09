# File: grid/env_graph_gridv1.py
# (Modified Version with Gymnasium Compliance, Collision Fixes, RGB Rendering)

from copy import copy
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors # Import directly

# --- Added for rgb_array rendering ---
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Removed unused sqrtm as it wasn't used
# from scipy.linalg import sqrtm
# from scipy.special import softmax # Removed unused softmax
import gymnasium as gym # Use Gymnasium
from gymnasium import spaces # Use Gymnasium spaces


class GraphEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array', 'plot', 'photo'], 'render_fps': 10} # Added metadata

    def __init__(
        self,
        config,
        goal, # Expects shape (num_agents, 2) -> [[row_y, col_x], ...]
        sensing_range=6, # This acts as r_fov and r_comm
        pad=3, # Determines FOV size = (pad * 2) - 1
        starting_positions=None, # Optional: np.array(num_agents, 2) -> [[row_y, col_x], ...]
        obstacles=None, # Optional: np.array(num_obstacles, 2) -> [[row_y, col_x], ...]
    ):
        """
        Environment for Grid-based Multi-Agent Path Finding with Graph Neural Networks.
        Uses partial observations (FOV) and allows for communication graph structure.
        Coordinates are generally (row_y, col_x).
        """
        super().__init__() # Initialize Gym Env
        self.config = config
        # Ensure necessary config keys exist
        required_config_keys = ["max_time", "board_size", "num_agents", "sensing_range", "pad"]
        for key in required_config_keys:
             if key not in self.config:
                 raise ValueError(f"Config missing required key: {key}")

        self.max_time = self.config["max_time"]
        if not isinstance(self.config["board_size"], (list, tuple)) or len(self.config["board_size"]) != 2:
            raise ValueError("config['board_size'] must be a list/tuple of [rows, cols]")
        self.board_rows, self.board_cols = self.config["board_size"] # Assuming [rows, cols]

        self.obstacles = np.empty((0,2), dtype=int)
        if obstacles is not None and obstacles.size > 0:
             if obstacles.ndim != 2 or obstacles.shape[1] != 2:
                  raise ValueError("Obstacles must be a Nx2 array of [row, col]")
             # Filter obstacles outside bounds
             valid_mask = (obstacles[:, 0] >= 0) & (obstacles[:, 0] < self.board_rows) & \
                          (obstacles[:, 1] >= 0) & (obstacles[:, 1] < self.board_cols)
             self.obstacles = obstacles[valid_mask]
             if len(self.obstacles) < len(obstacles):
                  print(f"Warning: Removed {len(obstacles) - len(self.obstacles)} obstacles outside board bounds.")

        if goal.ndim != 2 or goal.shape[1] != 2:
            raise ValueError("Goal must be a Nx2 array of [row, col]")
        if goal.shape[0] != self.config["num_agents"]:
             raise ValueError(f"Goal array shape mismatch. Expected ({self.config['num_agents']}, 2), got {goal.shape}")
        # Ensure goals are within bounds
        if np.any((goal[:, 0] < 0) | (goal[:, 0] >= self.board_rows) | (goal[:, 1] < 0) | (goal[:, 1] >= self.board_cols)):
             raise ValueError("One or more goals are outside board boundaries.")
        # Ensure goals don't overlap with obstacles
        if self.obstacles.size > 0 and np.any(np.all(goal[:, np.newaxis, :] == self.obstacles[np.newaxis, :, :], axis=2)):
             raise ValueError("One or more goals overlap with obstacles.")
        self.goal = goal # Shape (num_agents, 2) -> [row_y, col_x]

        self.board = np.zeros((self.board_rows, self.board_cols), dtype=np.int8) # Use int8 for board codes: 0=empty, 1=agent, 2=obstacle

        # FOV calculation parameters
        self.sensing_range = self.config["sensing_range"] # Used for FOV extent and communication graph
        self.pad = self.config["pad"]
        self.fov_size = (self.pad * 2) - 1 # e.g., pad=3 -> 5x5 FOV
        print(f"GraphEnv Initialized: Board={self.board_rows}x{self.board_cols}, Agents={self.config['num_agents']}, Pad={self.pad}, FOV={self.fov_size}x{self.fov_size}")

        self.starting_positions = starting_positions # Note: reset uses random if None
        self.nb_agents = self.config["num_agents"]

        # Agent state
        self.positionX = np.zeros((self.nb_agents,), dtype=np.int32) # Current X (column)
        self.positionY = np.zeros((self.nb_agents,), dtype=np.int32) # Current Y (row)
        self.positionX_temp = np.zeros_like(self.positionX) # For collision checking
        self.positionY_temp = np.zeros_like(self.positionY)
        self.embedding = np.ones((self.nb_agents, 1), dtype=np.float32) # Agent embeddings (if used)
        self.reached_goal = np.zeros(self.nb_agents, dtype=bool) # Track who reached goal
        self.time = 0

        # Action space (5 discrete actions)
        # 0: Idle   (0, 0)
        # 1: Right  (0, 1)  -> col+1
        # 2: Up     (-1, 0) -> row-1
        # 3: Left   (0, -1) -> col-1
        # 4: Down   (1, 0)  -> row+1
        self.action_map_dy_dx = {
             0: (0, 0),  # Idle   (delta_row, delta_col)
             1: (0, 1),  # Right
             2: (-1, 0), # Up
             3: (0, -1), # Left
             4: (1, 0),  # Down
        }
        # Action space requires a single Discrete space for all agents in this setup
        # If used with MARL libraries, might need MultiDiscrete or similar
        # For this project's structure (centralized policy giving N actions):
        self.action_space = spaces.MultiDiscrete([5] * self.nb_agents) # Each agent chooses 0-4

        # Observation space (based on getObservations output)
        # Channel definitions for FOV:
        # 0: Obstacles (value 2), Other Agents (value 1)
        # 1: Goal location (value 3)
        # 2: Self position (value 1 at center)
        self.num_fov_channels = 3
        self.observation_space = spaces.Dict({
             # Batch dim is handled by DataLoader/training loop, space defines single obs
             "fov": spaces.Box(low=0, high=3, shape=(self.nb_agents, self.num_fov_channels, self.fov_size, self.fov_size), dtype=np.float32), # FOV per agent
             "adj_matrix": spaces.Box(low=0, high=1, shape=(self.nb_agents, self.nb_agents), dtype=np.float32), # Adjacency
             "embeddings": spaces.Box(low=-np.inf, high=np.inf, shape=(self.nb_agents, 1), dtype=np.float32) # Embeddings per agent
        })

        # Rendering state
        norm = colors.Normalize(vmin=0.0, vmax=1.4, clip=True) # Agent color normalization
        self.mapper = cm.ScalarMappable(norm=norm, cmap=cm.inferno)
        self.render_mode = config.get('render_mode', None) # Store render mode if needed
        self._plot_fig = None
        self._plot_ax = None
        self._color_obstacle = '#333333' # Darker Gray
        self._color_goal = '#2ecc71'   # Green
        self._color_agent_base = '#3498db' # Blue
        self._color_agent_reached = '#95a5a6' # Gray for reached
        self._color_neighbor_line = '#DDDDDD' # Lighter grey
        self._color_boundary = 'black'

        self.window = None # For pygame rendering if needed later
        self.clock = None  # For pygame rendering if needed later

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Important for seeding self.np_random

        self.time = 0
        self.board.fill(0) # Reset board to empty (0)

        # Place obstacles on board
        if self.obstacles.size > 0:
             # Ensure obstacles are within bounds before placing
             valid_mask = (self.obstacles[:, 0] >= 0) & (self.obstacles[:, 0] < self.board_rows) & \
                          (self.obstacles[:, 1] >= 0) & (self.obstacles[:, 1] < self.board_cols)
             valid_obstacles = self.obstacles[valid_mask]
             if len(valid_obstacles) > 0:
                  self.board[valid_obstacles[:, 0], valid_obstacles[:, 1]] = 2 # Use 2 for obstacles

        # Set starting positions
        occupied_mask = self.board != 0 # Mask of occupied cells (currently only obstacles)
        if self.starting_positions is not None:
            if self.starting_positions.ndim != 2 or self.starting_positions.shape != (self.nb_agents, 2):
                raise ValueError(f"starting_positions shape mismatch, expected ({self.nb_agents}, 2), got {self.starting_positions.shape}")
            self.positionY = self.starting_positions[:, 0].copy() # row
            self.positionX = self.starting_positions[:, 1].copy() # col

            # Check if provided starts are valid (within bounds and not on obstacle)
            if np.any((self.positionY < 0) | (self.positionY >= self.board_rows) | (self.positionX < 0) | (self.positionX >= self.board_cols)):
                raise ValueError("Provided starting positions are outside board boundaries.")
            if np.any(occupied_mask[self.positionY, self.positionX]):
                 colliding_agents = np.where(occupied_mask[self.positionY, self.positionX])[0]
                 raise ValueError(f"Provided starting positions collide with obstacles for agents: {colliding_agents}.")
            # Check for duplicates in starting positions
            unique_starts = np.unique(self.starting_positions, axis=0)
            if len(unique_starts) < self.nb_agents:
                 raise ValueError("Provided starting positions contain duplicates.")
        else:
            # Generate random starting positions avoiding obstacles
            possible_coords = np.argwhere(~occupied_mask) # N x 2 array of [row, col] for empty cells
            if len(possible_coords) < self.nb_agents:
                 raise RuntimeError(f"Not enough free space ({len(possible_coords)}) on board to place {self.nb_agents} agents randomly. Board size {self.board_rows}x{self.board_cols}, Obstacles: {len(self.obstacles)}")
            # Use self.np_random for seeded randomness
            chosen_indices = self.np_random.choice(len(possible_coords), size=self.nb_agents, replace=False)
            start_coords = possible_coords[chosen_indices]
            self.positionY = start_coords[:, 0] # Row is Y
            self.positionX = start_coords[:, 1] # Col is X

        # Reset agent state variables
        self.embedding = np.ones((self.nb_agents, 1), dtype=np.float32) # Reset embedding
        self.reached_goal.fill(False) # Reset goal tracker

        # Check if initial positions are already at goals
        self.reached_goal = np.all(np.stack([self.positionY, self.positionX], axis=1) == self.goal, axis=1)

        # Update agent positions on the board (use 1 for agent)
        # Ensure not to overwrite obstacles if start was somehow invalid (though checked above)
        self.board[self.positionY, self.positionX] = 1

        # Compute initial distances/adjacency
        self._compute_comm_graph()
        observation = self.getObservations()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def getObservations(self):
        # Make sure the board reflects current agent positions before calculating FOV
        self.updateBoard() # Updates self.board based on current agent X,Y

        obs = {
            "fov": self.generate_fov(),    # Agent's local view (3 channels)
            "adj_matrix": self.adj_matrix.copy(), # Adjacency matrix
            "embeddings": self.embedding.copy(),     # Agent embeddings
        }
        return obs

    def _get_info(self):
        # Provides auxiliary information, not used for learning directly
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
        """Calculates adjacency matrix based on sensing_range (acting as r_comm)."""
        current_pos = np.stack([self.positionY, self.positionX], axis=1) # Shape (n_agents, 2) [row, col]
        # Efficient pairwise distance calculation
        delta = current_pos[:, np.newaxis, :] - current_pos[np.newaxis, :, :] # Shape (n, n, 2)
        dist_sq = np.sum(delta**2, axis=2) # Shape (n, n)
        # Agents are adjacent if distance > epsilon (not same agent) and < sensing_range
        adj = (dist_sq > 1e-9) & (dist_sq < self.sensing_range**2)
        np.fill_diagonal(adj, False) # Ensure no self-loops in adjacency
        self.adj_matrix = adj.astype(np.float32)

    def step(self, actions): # Removed emb=None, not used in this version's step logic
        """
        Transitions the environment based on agent actions.
        Includes collision checking and goal locking.
        Args: actions (np.ndarray): Array of shape (num_agents,) with action indices (0-4).
        """
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if actions.shape != (self.nb_agents,):
             raise ValueError(f"Actions shape incorrect. Expected ({self.nb_agents},), got {actions.shape}")
        if np.any((actions < 0) | (actions >= len(self.action_map_dy_dx))):
            print(f"Warning: Received invalid action index in {actions}. Clamping/treating as Idle.")
            # Optionally clamp or handle invalid actions, here we proceed and let the loop handle it

        # Store previous positions for collision checks
        self.positionX_temp = self.positionX.copy()
        self.positionY_temp = self.positionY.copy()

        # --- Action Application and Collision Checking ---
        proposedY = self.positionY.copy()
        proposedX = self.positionX.copy()
        did_collide = np.zeros(self.nb_agents, dtype=bool) # Tracks if an agent's move was reverted due to *any* collision

        # 1. Agents already at goal do not move
        active_agent_mask = ~self.reached_goal

        # 2. Calculate proposed moves only for active agents
        for agent_id in np.where(active_agent_mask)[0]:
             act = actions[agent_id]
             # Handle invalid action index
             if act not in self.action_map_dy_dx:
                 # print(f"Warning: Agent {agent_id} received invalid action {act}. Treating as Idle.")
                 act = 0 # Treat invalid action as Idle (action 0)
             dy, dx = self.action_map_dy_dx[act] # Get change in row, col
             proposedY[agent_id] += dy
             proposedX[agent_id] += dx

        # 3. Clamp proposed positions to board boundaries (for active agents)
        # Important: Clamp *before* collision checks involving board elements
        proposedY[active_agent_mask] = np.clip(proposedY[active_agent_mask], 0, self.board_rows - 1)
        proposedX[active_agent_mask] = np.clip(proposedX[active_agent_mask], 0, self.board_cols - 1)

        # 4. Check for collisions with obstacles (for active agents)
        if self.obstacles.size > 0:
            # Get proposed coords only for agents that are still active
            active_indices = np.where(active_agent_mask)[0]
            proposed_coords_active = np.stack([proposedY[active_indices], proposedX[active_indices]], axis=1)

            # Check if any proposed coordinate matches any obstacle coordinate
            # Use broadcasting for efficiency
            obstacle_collision_active_mask = np.any(np.all(proposed_coords_active[:, np.newaxis, :] == self.obstacles[np.newaxis, :, :], axis=2), axis=1)

            # Map back to original agent indices and revert colliding agents
            colliding_agent_indices = active_indices[obstacle_collision_active_mask]
            if colliding_agent_indices.size > 0:
                proposedY[colliding_agent_indices] = self.positionY_temp[colliding_agent_indices]
                proposedX[colliding_agent_indices] = self.positionX_temp[colliding_agent_indices]
                did_collide[colliding_agent_indices] = True
                active_agent_mask[colliding_agent_indices] = False # These agents are no longer active for agent-agent collision checks

        # 5. Check for agent-agent collisions (vertex and swapping)
        # Only consider agents that are still active (not at goal, didn't hit obstacle)
        active_indices = np.where(active_agent_mask)[0]
        if len(active_indices) > 1: # Need at least two agents to collide
            # Get coordinates relevant to this check
            current_coords_check = np.stack([self.positionY_temp[active_indices], self.positionX_temp[active_indices]], axis=1)
            proposed_coords_check = np.stack([proposedY[active_indices], proposedX[active_indices]], axis=1)

            # Vertex collisions: Find proposed locations occupied by more than one agent
            unique_coords, unique_map_indices, counts = np.unique(proposed_coords_check, axis=0, return_inverse=True, return_counts=True)
            colliding_cell_indices = np.where(counts > 1)[0] # Indices into unique_coords
            vertex_collision_mask_rel = np.isin(unique_map_indices, colliding_cell_indices) # Mask relative to 'active_indices'
            vertex_collision_agents_orig_idx = active_indices[vertex_collision_mask_rel] # Original agent indices involved

            # Edge collisions (swapping): Find pairs (i, j) where proposed_i == current_j AND proposed_j == current_i
            swapping_collision_agents_list = []
            # Iterate using relative indices for efficiency
            relative_indices = np.arange(len(active_indices))
            for i_rel in relative_indices:
                 for j_rel in range(i_rel + 1, len(active_indices)):
                     # Check swap condition using relative indices into coord arrays
                     if np.array_equal(proposed_coords_check[i_rel], current_coords_check[j_rel]) and \
                        np.array_equal(proposed_coords_check[j_rel], current_coords_check[i_rel]):
                         # Map back to original agent indices
                         swapping_collision_agents_list.extend([active_indices[i_rel], active_indices[j_rel]])

            swapping_collision_agents_orig_idx = np.unique(swapping_collision_agents_list)

            # Combine indices of agents involved in either type of agent-agent collision
            agent_collision_indices = np.unique(np.concatenate([vertex_collision_agents_orig_idx, swapping_collision_agents_orig_idx]))

            # Revert agents involved in agent-agent collisions
            if agent_collision_indices.size > 0:
                proposedY[agent_collision_indices] = self.positionY_temp[agent_collision_indices]
                proposedX[agent_collision_indices] = self.positionX_temp[agent_collision_indices]
                did_collide[agent_collision_indices] = True
                # No need to update active_agent_mask further here, final positions are set below

        # --- Final Position Update & Goal Check ---
        self.positionY = proposedY
        self.positionX = proposedX

        # Check which agents are now at their goal (includes newly arrived and previously locked)
        current_pos = np.stack([self.positionY, self.positionX], axis=1)
        self.reached_goal = np.all(current_pos == self.goal, axis=1)

        # --- Update Environment State ---
        self.time += 1
        self._compute_comm_graph() # Recompute graph based on final positions
        self.updateBoard() # Update board representation

        # Determine termination and truncation
        terminated = np.all(self.reached_goal) # Episode ends if all agents are at their goals
        truncated = self.time >= self.max_time # Episode ends if max time is reached

        # Get observation, reward, info
        observation = self.getObservations()
        # Reward: Give small penalty for each step, penalty for collision, large reward for success
        # Simple reward structure:
        step_penalty = -0.01
        collision_penalty = -0.1 # Penalty per agent involved in a collision this step
        goal_reward = 1.0 # Reward if *all* agents reach goal
        # Advanced reward idea: -0.001 * sum(dist_to_goal) ?

        reward = step_penalty * self.nb_agents # Penalty for time passing
        reward += collision_penalty * np.sum(did_collide) # Penalty for collisions

        if terminated:
            reward += goal_reward # Bonus for completing the task

        info = self._get_info()
        # Add collision info for debugging/analysis
        info["collisions_this_step"] = did_collide

        if self.render_mode == "human":
             self.render()

        return observation, reward, terminated, truncated, info

    # Removed _calculate_reward method, integrated into step

    def _updateEmbedding(self, H):
        # Allow updating embeddings if needed by the model/training
        if H is not None and isinstance(H, np.ndarray) and H.shape == self.embedding.shape:
             self.embedding = H.copy()

    def map_goal_to_fov(self, agent_id):
        """Maps the agent's absolute goal coordinate to its FOV coordinates."""
        # FOV center corresponds to agent's position
        center_offset = self.pad - 1 # e.g., pad=3 -> fov_size=5 -> center_idx=2

        # Calculate goal relative to agent (delta_row, delta_col)
        relative_y = self.goal[agent_id, 0] - self.positionY[agent_id]
        relative_x = self.goal[agent_id, 1] - self.positionX[agent_id]

        # Convert relative coordinates to FOV coordinates
        # Y-axis in FOV increases downwards, opposite to relative_y increase upwards
        fov_y = center_offset - relative_y
        fov_x = center_offset + relative_x

        # Check if goal is within the FOV boundaries
        if 0 <= fov_y < self.fov_size and 0 <= fov_x < self.fov_size:
             return int(fov_y), int(fov_x)
        else:
             # Goal is outside FOV, project onto boundary (simple clamp)
             # More sophisticated projection could find the intersection point
             proj_y = np.clip(fov_y, 0, self.fov_size - 1)
             proj_x = np.clip(fov_x, 0, self.fov_size - 1)
             # Ensure it's truly on border if goal was outside
             # If clamped point is NOT on border, means goal ray hits corner region.
             # A simple fix is to snap to the *nearest* border cell, but clamping is easier.
             # If 0 < proj_y < self.fov_size - 1 and 0 < proj_x < self.fov_size - 1:
                   # This indicates clamping resulted in an interior point, which shouldn't happen
                   # if the original goal was truly outside. This edge case needs careful thought
                   # if precise border projection is needed. For now, clamped value is used.
                   # print(f"Debug: Goal projection issue for agent {agent_id}?") # Should not happen often with simple clamp
             return int(proj_y), int(proj_x)

    def generate_fov(self):
        """
        Generates the 3-Channel Field of View (FOV) for each agent based on paper Fig 1.
        Channel 0: Obstacles (2) and Other Agents (1)
        Channel 1: Goal location (projected if outside FOV) (value 3)
        Channel 2: Self position (value 1 at center)

        Returns: np.ndarray shape (num_agents, 3, fov_size, fov_size)
        """
        # Pad the main board: obstacles=2, agents=1, empty=0
        # Pad width should be self.pad for correct slicing relative to agent center
        # Example: pad=3 -> fov=5x5. Need 2 cells padding around border.
        # Correction: Need pad amount of padding for full FOV even at edge.
        map_padded = np.pad(self.board, ((self.pad, self.pad), (self.pad, self.pad)), mode='constant', constant_values=2) # Pad with obstacle value

        # Agent positions in the padded map coordinates
        current_posY_padded = self.positionY + self.pad
        current_posX_padded = self.positionX + self.pad

        FOV = np.zeros((self.nb_agents, self.num_fov_channels, self.fov_size, self.fov_size), dtype=np.float32)
        center_idx = self.pad - 1 # Index of the center cell in the FOV grid

        for agent_id in range(self.nb_agents):
            # Calculate slice boundaries in the padded map
            # Start index = center_padded - center_fov_idx
            # End index = center_padded + (fov_size - 1 - center_fov_idx) + 1
            # Simplifies to: center_padded +/- (pad - 1) for start/end range of size fov_size
            row_start = current_posY_padded[agent_id] - center_idx
            row_end = current_posY_padded[agent_id] + center_idx + 1 # Slice end is exclusive
            col_start = current_posX_padded[agent_id] - center_idx
            col_end = current_posX_padded[agent_id] + center_idx + 1

            # --- Channel 0: Obstacles and Other Agents ---
            # Extract local view from padded map
            local_view = map_padded[row_start:row_end, col_start:col_end]

            # Assign to FOV channel 0
            FOV[agent_id, 0, :, :] = local_view
            # Agent doesn't see itself in this channel, set center to 0
            FOV[agent_id, 0, center_idx, center_idx] = 0

            # --- Channel 1: Goal Location ---
            # Map goal to FOV coordinates (handles projection)
            gy, gx = self.map_goal_to_fov(agent_id)
            FOV[agent_id, 1, gy, gx] = 3 # Mark goal position

            # --- Channel 2: Self Position ---
            FOV[agent_id, 2, center_idx, center_idx] = 1 # Mark self at center

        return FOV

    def updateBoard(self):
        """Updates self.board representation based on current agent positions."""
        self.board.fill(0) # Clear previous agent positions
        # Place obstacles
        if self.obstacles.size > 0:
            # Ensure obstacle indices are integers
            obs_rows = self.obstacles[:, 0].astype(int)
            obs_cols = self.obstacles[:, 1].astype(int)
            self.board[obs_rows, obs_cols] = 2

        # Place agents
        # !!! FIX: Ensure indices are integers before using them to index self.board !!!
        rows = self.positionY.astype(int)
        cols = self.positionX.astype(int)

        # Additional safety check: Ensure indices are within bounds AFTER casting
        valid_row_mask = (rows >= 0) & (rows < self.board_rows)
        valid_col_mask = (cols >= 0) & (cols < self.board_cols)
        valid_mask = valid_row_mask & valid_col_mask

        if not np.all(valid_mask):
             print(f"Warning: Agent positions out of bounds before board update! Y:{self.positionY}, X:{self.positionX}")
             # Only update valid positions
             rows = rows[valid_mask]
             cols = cols[valid_mask]
             if rows.size == 0: # No valid agents to place
                  return

        # Place agents (value 1) at their integer coordinates
        # Avoid placing on obstacles (value 2) - though step logic should prevent this
        target_cells = self.board[rows, cols]
        agent_placement_mask = (target_cells != 2) # Only place if not obstacle

        if np.any(~agent_placement_mask):
             print(f"Warning: Attempted to place agent(s) on obstacle(s) in updateBoard!")

        # Apply placement only where mask allows
        self.board[rows[agent_placement_mask], cols[agent_placement_mask]] = 1

    def render(self, mode="human"): # Removed printNeigh argument, controlled internally if needed
        """ Renders the environment.
            - human: Displays window using Matplotlib.
            - rgb_array: Returns an numpy.ndarray suitable for video or image saving.
        """
        current_pos = self.get_current_positions() # Get current [row, col]

        # --- Setup Figure ---
        if mode == "human":
            if self._plot_fig is None: # First render
                plt.ion() # Turn on interactive mode
                self._plot_fig, self._plot_ax = plt.subplots(figsize=(6, 6))
            ax = self._plot_ax
        elif mode == "rgb_array":
            # Create a new figure for each rgb_array call
            # TODO: Could potentially reuse figure object for performance if needed
            fig = Figure(figsize=(5, 5), dpi=100) # Adjust dpi/figsize as needed
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
        else:
            super(GraphEnv, self).render(mode=mode) # Let base class handle unsupported modes
            return None

        ax.clear()
        ax.set_facecolor('#F0F0F0') # Light grey background
        ax.set_xticks(np.arange(-.5, self.board_cols, 1), minor=True)
        ax.set_yticks(np.arange(-.5, self.board_rows, 1), minor=True)
        ax.grid(which='minor', color='#CCCCCC', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', size=0)
        ax.tick_params(which='major', bottom=False, left=False, labelbottom=False, labelleft=False) # Hide major ticks and labels


        # --- Draw Elements ---
        # Draw neighbor lines (optional, can be slow)
        # show_neighbor_lines = False
        # if show_neighbor_lines:
        #     for i in range(self.nb_agents):
        #         for j in range(i + 1, self.nb_agents):
        #             if self.adj_matrix[i, j] > 0:
        #                 ax.plot([current_pos[i, 1], current_pos[j, 1]], [current_pos[i, 0], current_pos[j, 0]], color=self._color_neighbor_line, lw=0.5, zorder=1)

        # Draw obstacles (use squares centered on grid cells)
        if self.obstacles.size > 0:
             obs_y, obs_x = self.obstacles[:, 0], self.obstacles[:, 1]
             for r, c in zip(obs_y, obs_x):
                  rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=self._color_obstacle, edgecolor='#222222', linewidth=0.5)
                  ax.add_patch(rect)

        # Draw goals (stars centered on grid cells)
        if self.goal.size > 0:
             ax.scatter(self.goal[:, 1], self.goal[:, 0], color=self._color_goal, marker="*", s=200, zorder=3, alpha=0.7, edgecolors='#111111', linewidth=0.5)

        # Draw agents (circles centered on grid cells)
        agent_colors = [self._color_agent_reached if self.reached_goal[i] else self._color_agent_base for i in range(self.nb_agents)]
        # If using embeddings for color:
        # agent_colors = self.mapper.to_rgba(self.embedding.flatten())
        ax.scatter(self.positionX, self.positionY, s=150, c=agent_colors, zorder=4, edgecolors='black', linewidth=0.5)
        # Add agent numbers
        for i in range(self.nb_agents):
             ax.text(self.positionX[i], self.positionY[i], str(i), color='white', ha='center', va='center', fontsize=8, fontweight='bold', zorder=5)

        # --- Set Limits and Aspect Ratio ---
        ax.set_xlim(-0.5, self.board_cols - 0.5)
        ax.set_ylim(-0.5, self.board_rows - 0.5) # Matplotlib Y increases upwards
        ax.invert_yaxis() # Invert Y axis to match matrix indexing (row 0 at top)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"Step: {self.time} | Reached: {np.sum(self.reached_goal)}/{self.nb_agents}")

        # --- Finalize Rendering ---
        if mode == "human":
            self._plot_fig.canvas.flush_events()
            plt.pause(0.01) # Short pause to allow plot to update
            return None # human mode doesn't return array
        elif mode == "rgb_array":
            canvas.draw()
            image = np.asarray(canvas.buffer_rgba())
            # Clean up the temporary figure if desired
            plt.close(fig)
            return image

    def close(self):
        if self._plot_fig is not None:
            plt.close(self._plot_fig)
            self._plot_fig = None
            self._plot_ax = None
            plt.ioff() # Turn off interactive mode

# --- Utility functions outside class ---
def create_goals(board_size, num_agents, obstacles=None, current_starts=None):
    """
    Creates random goal/start locations avoiding obstacles and optionally current_starts.
    Args:
        board_size (tuple): (rows, cols)
        num_agents (int): Number of goals/starts to generate.
        obstacles (np.ndarray, optional): Nx2 array of obstacle [row, col].
        current_starts (np.ndarray, optional): Nx2 array of start positions to avoid.

    Returns:
        np.ndarray: Nx2 array of generated [row, col] positions.
    """
    rows, cols = board_size
    occupied_mask = np.zeros((rows, cols), dtype=bool) # Use boolean mask

    # Mark obstacles as occupied
    if obstacles is not None and obstacles.size > 0:
        valid_obstacles = obstacles[(obstacles[:, 0] >= 0) & (obstacles[:, 0] < rows) & (obstacles[:, 1] >= 0) & (obstacles[:, 1] < cols)]
        if len(valid_obstacles) > 0:
             occupied_mask[valid_obstacles[:, 0], valid_obstacles[:, 1]] = True

    # Mark current_starts as occupied
    if current_starts is not None and current_starts.size > 0:
         valid_starts = current_starts[(current_starts[:, 0] >= 0) & (current_starts[:, 0] < rows) & (current_starts[:, 1] >= 0) & (current_starts[:, 1] < cols)]
         if len(valid_starts) > 0:
              occupied_mask[valid_starts[:, 0], valid_starts[:, 1]] = True

    # Find indices where board is False (unoccupied)
    available_coords = np.argwhere(~occupied_mask) # N x 2 array [row, col]

    if len(available_coords) < num_agents:
        raise ValueError(f"Not enough free spaces ({len(available_coords)}) to place {num_agents} goals/starts. Board: {rows}x{cols}, Obstacles: {len(obstacles) if obstacles is not None else 0}, Starts: {len(current_starts) if current_starts is not None else 0}")

    # Use default RNG for utilities, environment uses seeded RNG
    chosen_indices = np.random.choice(len(available_coords), size=num_agents, replace=False)
    goals = available_coords[chosen_indices]
    return goals

def create_obstacles(board_size, nb_obstacles):
    """Creates random obstacle locations."""
    rows, cols = board_size
    total_cells = rows * cols
    if nb_obstacles < 0: nb_obstacles = 0
    if nb_obstacles >= total_cells:
         print(f"Warning: Requested obstacles ({nb_obstacles}) >= total cells ({total_cells}). Placing obstacles everywhere might make task impossible.")
         nb_obstacles = total_cells - 1 # Leave at least one cell free

    # Generate all possible coordinates
    all_coords = np.array([(r, c) for r in range(rows) for c in range(cols)])

    # Use default RNG
    chosen_indices = np.random.choice(total_cells, size=nb_obstacles, replace=False)
    obstacles = all_coords[chosen_indices]
    return obstacles

# --- Main block for testing ---
if __name__ == "__main__":
    print("--- Running GraphEnv Example ---")
    agents = 5
    bs = 12 # Board size
    board_dims = [bs, bs]
    num_obstacles_to_gen = 10
    sensing_range_val = 4
    pad_val = 3 # For 5x5 FOV

    # Example config (mirroring structure potentially used in training)
    config = {
        "num_agents": agents,
        "board_size": board_dims,
        "max_time": 60,
        "sensing_range": sensing_range_val,
        "pad": pad_val,
        "render_mode": "human", # Set render mode for testing
        # Add dummy model config keys if needed by other parts of code using config
        "min_time": 1, "obstacles": num_obstacles_to_gen, # These might be duplicated but help consistency
        "encoder_layers": 1, "encoder_dims": [64], "last_convs": [0],
        "graph_filters": [3], "node_dims": [128], "action_layers": 1, "channels": [16, 16, 16],
    }
    try:
        # Generate obstacles and goals/starts
        obstacles_arr = create_obstacles(board_dims, num_obstacles_to_gen)
        start_pos_arr = create_goals(board_dims, agents, obstacles=obstacles_arr)
        # Ensure goals are different from starts and obstacles
        temp_obstacles_for_goals = np.vstack([obstacles_arr, start_pos_arr]) if obstacles_arr.size > 0 else start_pos_arr
        goals_arr = create_goals(board_dims, agents, obstacles=obstacles_arr, current_starts=start_pos_arr) # Avoid starts too

        env = GraphEnv(
            config=config,
            goal=goals_arr,
            # sensing_range, pad etc. are now taken from config inside __init__
            starting_positions=start_pos_arr,
            obstacles=obstacles_arr,
        )

        obs, info = env.reset(seed=42) # Use seed for reproducibility
        print("Initial State:")
        print("  Positions:", info["positions"].tolist())
        print("  Goals:", env.goal.tolist())
        print("  Reached:", info["agents_at_goal"])
        env.render() # Initial render
        plt.pause(1) # Pause for viewing

        total_reward = 0
        for step in range(config["max_time"]):
            # Sample random actions for testing
            # Need to provide actions for ALL agents, e.g., shape (num_agents,)
            actions = env.action_space.sample() # Get a valid sample from MultiDiscrete
            # print(f"Step {env.time+1} Actions: {actions}") # Debug print

            obs, reward, terminated, truncated, info = env.step(actions)
            total_reward += reward

            print(f"\rStep: {info['time']}, Term: {terminated}, Trunc: {truncated}, Reward: {reward:.3f}, Reached: {np.sum(info['agents_at_goal'])}", end="")
            # env.render() is called inside step if mode is human

            if terminated or truncated:
                print(f"\nEpisode finished at step {info['time']}. Success: {terminated}. Total Reward: {total_reward:.3f}")
                break

        if not terminated and not truncated:
             print(f"\nEpisode reached step limit ({config['max_time']}). Total Reward: {total_reward:.3f}")

        print("Final State:")
        print("  Positions:", info["positions"].tolist())
        print("  Reached:", info["agents_at_goal"])
        env.render() # Final render
        plt.pause(2)

    except Exception as e:
        print(f"\nError during environment test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals() and env is not None:
            env.close()

    print("\n--- GraphEnv Example Finished ---")