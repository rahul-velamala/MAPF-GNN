# File: grid/env_graph_gridv1.py
# (Modified Version with IndexError Fix and Refinements)

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
import gym
from gym import spaces


class GraphEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array', 'plot', 'photo'], 'render_fps': 10} # Added metadata

    def __init__(
        self,
        config,
        goal, # Expects shape (num_agents, 2) -> [[row_y, col_x], ...]
        # max_time=23, # Use config['max_time']
        # board_size=10, # Use config['board_size']
        sensing_range=6, # This acts as r_fov and r_comm
        pad=3, # Determines FOV size = (pad * 2) - 1
        starting_positions=None, # Optional: np.array(num_agents, 2) -> [[row_y, col_x], ...]
        obstacles=None, # Optional: np.array(num_obstacles, 2) -> [[row_y, col_x], ...]
    ):
        super(GraphEnv, self).__init__()
        """
        Environment for Grid-based Multi-Agent Path Finding with Graph Neural Networks.
        Uses partial observations (FOV) and allows for communication graph structure.
        Coordinates are generally (row_y, col_x).
        """
        self.config = config
        self.max_time = self.config["max_time"]
        self.board_rows, self.board_cols = self.config["board_size"] # Assuming [rows, cols]
        if self.board_rows != self.board_cols:
            print("Warning: Non-square boards might have unexpected behavior with padding/FOV.")
        self.board_size = self.board_rows # Convenience for square checks if needed

        self.obstacles = np.empty((0,2), dtype=int)
        if obstacles is not None and obstacles.size > 0:
             # Filter obstacles outside bounds
             valid_mask = (obstacles[:, 0] >= 0) & (obstacles[:, 0] < self.board_rows) & \
                          (obstacles[:, 1] >= 0) & (obstacles[:, 1] < self.board_cols)
             self.obstacles = obstacles[valid_mask]

        self.goal = goal # Shape (num_agents, 2) -> [row_y, col_x]
        if self.goal.shape[0] != self.config["num_agents"]:
             raise ValueError("Goal array shape mismatch with num_agents")

        self.board = np.zeros((self.board_rows, self.board_cols), dtype=np.int8) # Use int8 for board codes

        # FOV calculation parameters
        self.sensing_range = sensing_range # Used for FOV extent and communication graph
        # Use pad from config directly, don't recalculate here
        self.pad = config.get("pad", 3) # Default to 3 if not in config
        # Paper uses r_fov=4. If pad=3, FOV=5x5. If pad=4, FOV=7x7. If pad=5, FOV=9x9.
        self.fov_size = (self.pad * 2) - 1 # e.g., pad=3 -> 5x5 FOV
        # print(f"GraphEnv: using pad={self.pad}, fov_size={self.fov_size}x{self.fov_size}")


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
             0: (0, 0),  # Idle
             1: (0, 1),  # Right
             2: (-1, 0), # Up
             3: (0, -1), # Left
             4: (1, 0),  # Down
        }
        self.action_space = spaces.Discrete(5)

        # Observation space (based on getObservations output)
        # Channel definitions for FOV:
        # 0: Obstacles (value 2), Other Agents (value 1)
        # 1: Goal location (value 3)
        # 2: Self position (value 1 at center)
        self.num_fov_channels = 3
        self.observation_space = spaces.Dict({
             "fov": spaces.Box(low=0, high=3, shape=(self.nb_agents, self.num_fov_channels, self.fov_size, self.fov_size), dtype=np.float32), # FOV
             "adj_matrix": spaces.Box(low=0, high=1, shape=(self.nb_agents, self.nb_agents), dtype=np.float32), # Adjacency
             "embeddings": spaces.Box(low=-np.inf, high=np.inf, shape=(self.nb_agents, 1), dtype=np.float32) # Embeddings
        })

        # Rendering state
        norm = colors.Normalize(vmin=0.0, vmax=1.4, clip=True) # Agent color normalization
        self.mapper = cm.ScalarMappable(norm=norm, cmap=cm.inferno)
        self._plot_fig = None
        self._plot_ax = None
        self._color_obstacle = 'black'
        self._color_goal = 'blue'
        self._color_agent = 'orange' # Base color, will be overridden by embedding map
        self._color_neighbor_line = '#AAAAAA' # Lighter grey
        self._color_boundary = 'black'

        # Initial state setup called within __init__
        # No need to call self.reset() here, it's called externally or after init typically

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Gym API compliance for seeding RNG

        self.time = 0
        self.board.fill(0) # Reset board to empty

        # Place obstacles
        if self.obstacles.size > 0:
             self.board[self.obstacles[:, 0], self.obstacles[:, 1]] = 2 # Use 2 for obstacles

        # Set starting positions
        occupied_mask = self.board != 0 # Mask of occupied cells (obstacles)
        if self.starting_positions is not None:
            if self.starting_positions.shape != (self.nb_agents, 2):
                raise ValueError(f"starting_positions shape mismatch")
            self.positionY = self.starting_positions[:, 0].copy() # row
            self.positionX = self.starting_positions[:, 1].copy() # col
            # Check if provided starts are valid
            if np.any(occupied_mask[self.positionY, self.positionX]):
                 colliding_agents = np.where(occupied_mask[self.positionY, self.positionX])[0]
                 print(f"WARNING: Provided starting positions collide with obstacles for agents: {colliding_agents}. Agent behavior undefined.")
                 # Or raise ValueError("Provided starting positions collide with obstacles.")
        else:
            # Generate random starting positions avoiding obstacles
            possible_coords = list(zip(*np.where(~occupied_mask))) # Find empty cells
            if len(possible_coords) < self.nb_agents:
                 raise ValueError("Not enough free space to place all agents randomly.")
            # Use self.np_random for seeded randomness
            chosen_indices = self.np_random.choice(len(possible_coords), size=self.nb_agents, replace=False)
            start_coords = np.array(possible_coords)[chosen_indices]
            self.positionY = start_coords[:, 0] # Row is Y
            self.positionX = start_coords[:, 1] # Col is X


        # Reset agent state variables
        self.embedding = np.ones((self.nb_agents, 1), dtype=np.float32) # Reset embedding
        self.reached_goal.fill(False) # Reset goal tracker

        # Update agent positions on the board (use 1 for agent)
        self.board[self.positionY, self.positionX] = 1

        # Compute initial distances/adjacency
        self._compute_comm_graph()
        observation = self.getObservations()
        info = self._get_info()

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
        delta = current_pos[:, np.newaxis, :] - current_pos[np.newaxis, :, :] # Shape (n, n, 2)
        dist_sq = np.sum(delta**2, axis=2) # Shape (n, n)
        adj = (dist_sq > 1e-9) & (dist_sq < self.sensing_range**2)
        np.fill_diagonal(adj, False)
        self.adj_matrix = adj.astype(np.float32)

    def step(self, actions, emb=None): # Make emb optional
        """
        Transitions the environment based on agent actions.
        Includes collision checking and goal locking.
        """
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if actions.shape != (self.nb_agents,):
             raise ValueError(f"Actions shape incorrect. Expected ({self.nb_agents},), got {actions.shape}")

        # Update embeddings if provided
        if emb is not None: self._updateEmbedding(emb)

        # Store previous positions for collision checks
        self.positionX_temp = self.positionX.copy()
        self.positionY_temp = self.positionY.copy()

        # --- Action Application and Collision Checking ---
        proposedY = self.positionY.copy()
        proposedX = self.positionX.copy()
        obstacle_collisions = np.zeros(self.nb_agents, dtype=bool)
        agent_collisions = np.zeros(self.nb_agents, dtype=bool) # Tracks agents involved

        # Agents already at goal do not move (goal locking is applied first)
        active_agent_mask = ~self.reached_goal

        # Calculate proposed moves only for active agents
        for agent_id in np.where(active_agent_mask)[0]:
             act = actions[agent_id]
             # Ensure action is valid (e.g., 0-4)
             if act not in self.action_map_dy_dx:
                 # print(f"Warning: Agent {agent_id} received invalid action {act}. Treating as Idle.")
                 act = 0 # Treat invalid action as Idle (action 0)
             dy, dx = self.action_map_dy_dx.get(act, (0,0)) # Get change in row, col
             proposedY[agent_id] += dy
             proposedX[agent_id] += dx

        # Clamp proposed positions to board boundaries (for active agents)
        proposedY[active_agent_mask] = np.clip(proposedY[active_agent_mask], 0, self.board_rows - 1)
        proposedX[active_agent_mask] = np.clip(proposedX[active_agent_mask], 0, self.board_cols - 1)

        # Check for collisions with obstacles (for active agents)
        if self.obstacles.size > 0:
            proposed_coords_active = np.stack([proposedY[active_agent_mask], proposedX[active_agent_mask]], axis=1)
            obstacle_collision_active_mask = np.any(np.all(proposed_coords_active[:, np.newaxis, :] == self.obstacles[np.newaxis, :, :], axis=2), axis=1)
            colliding_agent_indices = np.where(active_agent_mask)[0][obstacle_collision_active_mask]
            proposedY[colliding_agent_indices] = self.positionY_temp[colliding_agent_indices]
            proposedX[colliding_agent_indices] = self.positionX_temp[colliding_agent_indices]
            obstacle_collisions[colliding_agent_indices] = True

        # Check for agent-agent collisions (vertex and swapping) among *potentially* moved agents
        # Use the PROPOSED positions after obstacle checks
        active_indices = np.where(active_agent_mask)[0]
        if len(active_indices) > 1:
            # Indices relative to the 'active_indices' array
            relative_indices = np.arange(len(active_indices))

            proposed_coords_active = np.stack([proposedY[active_indices], proposedX[active_indices]], axis=1)
            original_coords_active = np.stack([self.positionY_temp[active_indices], self.positionX_temp[active_indices]], axis=1)

            # Vertex collisions: Check if any two active agents propose the same cell
            unique_coords, unique_map_indices, counts = np.unique(proposed_coords_active, axis=0, return_inverse=True, return_counts=True)
            colliding_cell_indices = np.where(counts > 1)[0] # Indices into unique_coords array
            vertex_collision_mask_rel = np.isin(unique_map_indices, colliding_cell_indices) # Mask relative to active agents
            vertex_collision_agents = active_indices[vertex_collision_mask_rel] # Map back to original agent indices

            # Edge collisions (swapping): Check A->B and B->A simultaneously
            swapping_collision_agents_list = []
            for i in relative_indices: # Iterate using relative indices
                 for j in range(i + 1, len(active_indices)): # Iterate using relative indices
                     agent_i_idx = active_indices[i] # Get original index
                     agent_j_idx = active_indices[j] # Get original index
                     # Check swap using relative indices into proposed/original _active arrays
                     if np.array_equal(proposed_coords_active[i], original_coords_active[j]) and \
                        np.array_equal(proposed_coords_active[j], original_coords_active[i]):
                         swapping_collision_agents_list.extend([agent_i_idx, agent_j_idx])

            swapping_collision_agents = np.unique(swapping_collision_agents_list)

            # Combine collision masks (relative to ALL agents)
            # Ensure arrays contain integers before concatenating if empty
            if vertex_collision_agents.size == 0 and swapping_collision_agents.size == 0:
                agents_colliding_idx = np.array([], dtype=int)
            elif vertex_collision_agents.size == 0:
                agents_colliding_idx = swapping_collision_agents
            elif swapping_collision_agents.size == 0:
                agents_colliding_idx = vertex_collision_agents
            else:
                agents_colliding_idx = np.unique(np.concatenate([vertex_collision_agents, swapping_collision_agents]))

            # --- START FIX ---
            # Only attempt assignment if there are colliding agents
            if agents_colliding_idx.size > 0:
                agent_collisions[agents_colliding_idx] = True
            # --- END FIX ---

            # Revert agents involved in agent-agent collisions
            # --- START FIX ---
            # Also check if the array is not empty before trying to index
            if agents_colliding_idx.size > 0:
                proposedY[agents_colliding_idx] = self.positionY_temp[agents_colliding_idx]
                proposedX[agents_colliding_idx] = self.positionX_temp[agents_colliding_idx]
            # --- END FIX ---


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
        reward = self._calculate_reward(terminated, truncated, agent_collisions, obstacle_collisions)
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _calculate_reward(self, terminated, truncated, agent_collisions, obstacle_collisions):
        # Simple reward: -0.01 per step, +10 if all reach goal, -0.1 per agent collision, -0.2 per obstacle collision
        reward = -0.01
        reward -= np.sum(agent_collisions) * 0.1
        reward -= np.sum(obstacle_collisions) * 0.2
        if terminated:
            reward += 10.0
        return reward

    def _updateEmbedding(self, H):
        if H is not None and H.shape == self.embedding.shape:
             self.embedding = H.copy()

    def map_goal_to_fov(self, agent_id):
        """Maps the agent's absolute goal coordinate to its FOV coordinates."""
        center_offset = self.pad - 1
        relative_y = self.goal[agent_id, 0] - self.positionY[agent_id]
        relative_x = self.goal[agent_id, 1] - self.positionX[agent_id]
        fov_y = center_offset - relative_y
        fov_x = center_offset + relative_x

        if 0 <= fov_y < self.fov_size and 0 <= fov_x < self.fov_size:
             return int(fov_y), int(fov_x)
        else:
             # Project onto boundary using simple clamping
             proj_y = np.clip(fov_y, 0, self.fov_size - 1)
             proj_x = np.clip(fov_x, 0, self.fov_size - 1)
             # Ensure it's truly on border if goal was outside
             if (0 < proj_y < self.fov_size - 1) and (0 < proj_x < self.fov_size - 1):
                  # If clamping results inside, find nearest border point (more complex)
                  # For simplicity, just return clamped - may not be *exactly* border
                  pass
             return int(proj_y), int(proj_x)

    def generate_fov(self):
        """
        Generates the 3-Channel Field of View (FOV) for each agent based on paper Fig 1.
        Channel 0: Obstacles (2) and Other Agents (1)
        Channel 1: Goal location (projected if outside FOV) (value 3)
        Channel 2: Self position (value 1 at center)
        """
        map_padded = np.pad(self.board, ((self.pad, self.pad), (self.pad, self.pad)), mode='constant', constant_values=2)
        current_posY_padded = self.positionY + self.pad
        current_posX_padded = self.positionX + self.pad
        FOV = np.zeros((self.nb_agents, self.num_fov_channels, self.fov_size, self.fov_size), dtype=np.float32)
        center_idx = self.pad - 1

        for agent_id in range(self.nb_agents):
            row_start = current_posY_padded[agent_id] - center_idx
            row_end = current_posY_padded[agent_id] + self.pad
            col_start = current_posX_padded[agent_id] - center_idx
            col_end = current_posX_padded[agent_id] + self.pad

            # Channel 0: Obstacles and Other Agents
            local_view = map_padded[row_start:row_end, col_start:col_end]
            FOV[agent_id, 0, :, :] = local_view
            FOV[agent_id, 0, center_idx, center_idx] = 0 # Agent doesn't see itself

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
            self.board[self.obstacles[:, 0], self.obstacles[:, 1]] = 2
        valid_agent_mask = self.board[self.positionY, self.positionX] != 2
        self.board[self.positionY[valid_agent_mask], self.positionX[valid_agent_mask]] = 1

    def render(self, mode="human", printNeigh=False):
        # --- RENDER TO RGB ARRAY ---
        if mode == "rgb_array":
            fig = Figure(figsize=(5, 5))
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            ax.clear(); ax.set_facecolor('white'); ax.axis('off')

            current_pos = self.get_current_positions() # Get [row, col]
            # Draw neighbor lines
            if printNeigh:
                for i in range(self.nb_agents):
                    for j in range(i + 1, self.nb_agents):
                        if self.adj_matrix[i, j] > 0:
                            ax.plot([current_pos[i, 1], current_pos[j, 1]], [current_pos[i, 0], current_pos[j, 0]], color=self._color_neighbor_line, linewidth=0.5, zorder=1)
            # Draw obstacles
            if self.obstacles.size > 0: ax.scatter(self.obstacles[:, 1], self.obstacles[:, 0], color=self._color_obstacle, marker="s", s=100, zorder=2)
            # Draw goals
            if self.goal.size > 0: ax.scatter(self.goal[:, 1], self.goal[:, 0], color=self._color_goal, marker="*", s=150, zorder=3, alpha=0.6)
            # Draw agents
            agent_colors = self.mapper.to_rgba(self.embedding.flatten())
            ax.scatter(self.positionX, self.positionY, s=120, c=agent_colors, zorder=4, edgecolors='black', linewidth=0.5)
            for i in range(self.nb_agents): ax.text(self.positionX[i], self.positionY[i], str(i), color='white', ha='center', va='center', fontsize=8, zorder=5)
            # Draw boundaries
            ax.plot([-0.5, self.board_cols - 0.5], [-0.5, -0.5], color=self._color_boundary, lw=1.5)
            ax.plot([-0.5, -0.5], [-0.5, self.board_rows - 0.5], color=self._color_boundary, lw=1.5)
            ax.plot([-0.5, self.board_cols - 0.5], [self.board_rows - 0.5, self.board_rows - 0.5], color=self._color_boundary, lw=1.5)
            ax.plot([self.board_cols - 0.5, self.board_cols - 0.5], [-0.5, self.board_rows - 0.5], color=self._color_boundary, lw=1.5)
            # Set limits and aspect
            ax.set_xlim(-1, self.board_cols); ax.set_ylim(-1, self.board_rows); ax.invert_yaxis(); ax.set_aspect('equal', adjustable='box')
            # Render canvas
            canvas.draw(); image = np.asarray(canvas.buffer_rgba()); plt.close(fig)
            return image

        # --- RENDER TO SCREEN ---
        elif mode == "human" or mode == "plot":
            if self._plot_fig is None: plt.ion(); self._plot_fig, self._plot_ax = plt.subplots(figsize=(6, 6))
            ax = self._plot_ax; ax.clear(); ax.set_facecolor('white'); ax.axis('off')
            current_pos = self.get_current_positions() # Get [row, col]
            if printNeigh:
                for i in range(self.nb_agents):
                    for j in range(i + 1, self.nb_agents):
                        if self.adj_matrix[i, j] > 0: ax.plot([current_pos[i, 1], current_pos[j, 1]], [current_pos[i, 0], current_pos[j, 0]], color=self._color_neighbor_line, lw=0.5, zorder=1)
            if self.obstacles.size > 0: ax.scatter(self.obstacles[:, 1], self.obstacles[:, 0], color=self._color_obstacle, marker="s", s=100, zorder=2)
            if self.goal.size > 0: ax.scatter(self.goal[:, 1], self.goal[:, 0], color=self._color_goal, marker="*", s=150, zorder=3, alpha=0.6)
            agent_colors = self.mapper.to_rgba(self.embedding.flatten())
            ax.scatter(self.positionX, self.positionY, s=120, c=agent_colors, zorder=4, edgecolors='black', lw=0.5)
            for i in range(self.nb_agents): ax.text(self.positionX[i], self.positionY[i], str(i), color='white', ha='center', va='center', fontsize=8, zorder=5)
            ax.plot([-0.5, self.board_cols - 0.5], [-0.5, -0.5], color=self._color_boundary, lw=1.5)
            ax.plot([-0.5, -0.5], [-0.5, self.board_rows - 0.5], color=self._color_boundary, lw=1.5)
            ax.plot([-0.5, self.board_cols - 0.5], [self.board_rows - 0.5, self.board_rows - 0.5], color=self._color_boundary, lw=1.5)
            ax.plot([self.board_cols - 0.5, self.board_cols - 0.5], [-0.5, self.board_rows - 0.5], color=self._color_boundary, lw=1.5)
            ax.set_xlim(-1, self.board_cols); ax.set_ylim(-1, self.board_rows); ax.invert_yaxis(); ax.set_aspect('equal', adjustable='box'); ax.set_title(f"Step: {self.time}")
            plt.draw(); plt.pause(0.01)
            return None
        else:
            print(f"Warning: Unsupported render mode '{mode}'.")
            return None

    def close(self):
        if self._plot_fig is not None:
            plt.close(self._plot_fig)
            self._plot_fig = None; self._plot_ax = None; plt.ioff()

# --- Utility functions outside class ---
def create_goals(board_size, num_agents, obstacles=None):
    """Creates random goal locations avoiding obstacles."""
    rows, cols = board_size
    temp_board = np.zeros((rows, cols), dtype=bool) # Use boolean mask
    if obstacles is not None and obstacles.size > 0:
        valid_obstacles = obstacles[(obstacles[:, 0] >= 0) & (obstacles[:, 0] < rows) & (obstacles[:, 1] >= 0) & (obstacles[:, 1] < cols)]
        if len(valid_obstacles) > 0:
             temp_board[valid_obstacles[:, 0], valid_obstacles[:, 1]] = True # Mark obstacles as occupied

    available_coords = list(zip(*np.where(~temp_board))) # Find indices where board is False (unoccupied)
    if len(available_coords) < num_agents:
        raise ValueError(f"Not enough free spaces ({len(available_coords)}) to place {num_agents} goals/starts.")

    chosen_indices = np.random.choice(len(available_coords), size=num_agents, replace=False)
    goals = np.array(available_coords)[chosen_indices]
    return goals

def create_obstacles(board_size, nb_obstacles):
    """Creates random obstacle locations."""
    rows, cols = board_size
    total_cells = rows * cols
    if nb_obstacles > total_cells: nb_obstacles = total_cells # Cap obstacles

    all_coords = np.array([(r, c) for r in range(rows) for c in range(cols)])
    chosen_indices = np.random.choice(total_cells, size=nb_obstacles, replace=False)
    obstacles = all_coords[chosen_indices]
    return obstacles

# --- Main block for testing ---
if __name__ == "__main__":
    # (Keep the testing __main__ block from the previous version if desired)
    print("--- Running GraphEnv Example ---")
    agents = 4
    bs = 10 # Board size
    board_dims = [bs, bs]
    num_obstacles_to_gen = 8
    sensing_range_val = 4
    pad_val = 3 # For 5x5 FOV

    config = {
        "num_agents": agents,
        "board_size": board_dims,
        "max_time": 50,
        "sensing_range": sensing_range_val,
        "pad": pad_val,
        # Add dummy model config keys needed by __init__ if any (placeholder)
        "min_time": 1, "obstacles": num_obstacles_to_gen,
        "encoder_layers": 1, "encoder_dims": [64], "last_convs": [0],
        "graph_filters": [3], "node_dims": [128], "action_layers": 1, "channels": [16, 16, 16],
    }
    # Generate obstacles and goals/starts
    obstacles_arr = create_obstacles(board_dims, num_obstacles_to_gen)
    start_pos_arr = create_goals(board_dims, agents, obstacles_arr)
    temp_obstacles_for_goals = np.vstack([obstacles_arr, start_pos_arr]) if obstacles_arr.size > 0 else start_pos_arr
    goals_arr = create_goals(board_dims, agents, temp_obstacles_for_goals)
    # ... (rest of the test code from previous version) ...

    try:
        env = GraphEnv(
            config,
            goal=goals_arr,
            sensing_range=sensing_range_val,
            starting_positions=start_pos_arr,
            obstacles=obstacles_arr,
            pad=pad_val
        )
        obs, info = env.reset()
        env.render(mode="human", printNeigh=True)
        plt.pause(1)
        total_reward = 0
        for step in range(config["max_time"] + 5):
            actions = np.random.randint(0, 5, size=agents)
            obs, reward, terminated, truncated, info = env.step(actions)
            total_reward += reward
            print(f"\rStep: {env.time}, Term: {terminated}, Trunc: {truncated}, Reward: {reward:.2f}", end="")
            env.render(mode="human", printNeigh=False) # Render less clutter in loop
            if terminated or truncated: break
        env.close()
        print("\n--- GraphEnv Example Finished ---")
    except Exception as e:
        print(f"\nError during environment test: {e}")
        import traceback
        traceback.print_exc()