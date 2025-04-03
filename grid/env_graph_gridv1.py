# File: grid/env_graph_gridv1.py
# (Modified to support render mode 'rgb_array')

from copy import copy
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors # Import directly

# --- Added for rgb_array rendering ---
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from scipy.linalg import sqrtm
from scipy.special import softmax
import gym
from gym import spaces


class GoalWrapper:
    def __init__(self, env, trayectories):
        self.trayectories = trayectories
        self.env = env

    def step(self, actions, emb):
        obs, _, terminated, info = self.env.step(actions, emb)


class GraphEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array', 'plot', 'photo'], 'render_fps': 10} # Added metadata

    def __init__(
        self,
        config,
        goal,
        max_time=23,
        board_size=10,
        sensing_range=6,
        pad=3,
        starting_positions=None,
        obstacles=None,
    ):
        super(GraphEnv, self).__init__()
        """
        :starting_positions: np.array-> [nb_agents, positions]; positions == [X,Y]
                            [[0,0],
                             [1,1]]
        """
        self.config = config
        self.max_time = self.config["max_time"]
        self.min_time = self.config["min_time"]
        self.board_size = self.config["board_size"][0]
        self.obstacles = obstacles if obstacles is not None else np.empty((0,2), dtype=int) # Ensure obstacles is always an array
        self.goal = goal # Expects shape (num_agents, 2) -> [[gx1, gy1], [gx2, gy2], ...]
        self.board = np.zeros((self.board_size, self.board_size))
        self.pad = pad
        self.starting_positions = starting_positions # Note: reset uses random if None
        self.action_list = {
            1: (1, 0),  # Right
            2: (0, 1),  # Up
            3: (-1, 0), # Left
            4: (0, -1), # Down
            0: (0, 0),  # Idle
        }
        nb_agents = self.config["num_agents"]
        self.positionX = np.zeros((nb_agents,), dtype=np.int32) # Use 1D arrays for positions
        self.positionY = np.zeros((nb_agents,), dtype=np.int32)
        self.nb_agents = nb_agents
        self.sensing_range = sensing_range
        self.obs_shape = self.nb_agents * 4 # This seems unused? FOV shape is different
        self.action_space = spaces.Discrete(5)
        # Observation space might need refinement based on what's actually returned in getObservations
        self.observation_space = spaces.Dict({
             "board": spaces.Box(low=0, high=5, shape=(self.board_size, self.board_size), dtype=np.int32),
             "fov": spaces.Box(low=0, high=3, shape=(self.nb_agents, 2, (self.pad * 2) - 1, (self.pad * 2) - 1), dtype=np.float32), # Assuming FOV values fit
             "adj_matrix": spaces.Box(low=0, high=1, shape=(self.nb_agents, self.nb_agents), dtype=np.float32),
             "distances": spaces.Box(low=0, high=self.sensing_range*2, shape=(self.nb_agents, self.nb_agents), dtype=np.float32), # Approx range
             "embeddings": spaces.Box(low=0, high=np.inf, shape=(self.nb_agents, 1), dtype=np.float32) # Assuming embedding is positive
        })

        self.headings = None
        self.embedding = np.ones((self.nb_agents, 1)) # Ensure embedding is 2D
        norm = colors.Normalize(vmin=0.0, vmax=1.4, clip=True) # Agent color normalization
        self.mapper = cm.ScalarMappable(norm=norm, cmap=cm.inferno)
        self.time = 0

        # --- Render state (for 'human'/'plot' modes using plt) ---
        self._plot_fig = None
        self._plot_ax = None
        # --- Colors for rendering ---
        self._color_obstacle = 'black'
        self._color_goal = 'blue'
        self._color_neighbor_line = 'grey' # Changed from black for less clutter
        self._color_boundary = 'black'

        # Initialize state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Gym API compliance for seeding RNG

        self.time = 0
        self.board = np.zeros((self.board_size, self.board_size))
        if self.obstacles.size > 0:
            # Ensure obstacles are within bounds - clip or error as needed
            valid_obstacles = self.obstacles[ (self.obstacles[:, 0] >= 0) & (self.obstacles[:, 0] < self.board_size) & \
                                              (self.obstacles[:, 1] >= 0) & (self.obstacles[:, 1] < self.board_size)]
            if len(valid_obstacles) != len(self.obstacles):
                print("Warning: Some obstacles were outside board boundaries.")
                self.obstacles = valid_obstacles

            if self.obstacles.size > 0:
                 # Obstacles marked with '2' (consistent with FOV?)
                 # NOTE: Original code used [:,1], [:,0] which swaps x/y if obstacles are (x,y)
                 # Assuming obstacles are (row, col) format matching numpy indexing
                 self.board[self.obstacles[:, 0], self.obstacles[:, 1]] = 2

        if self.starting_positions is not None:
            assert self.starting_positions.shape == (self.nb_agents, 2), f"starting_positions shape mismatch"
            # Assuming starting_positions are (row, col)
            self.positionX = self.starting_positions[:, 1].copy() # Col is X
            self.positionY = self.starting_positions[:, 0].copy() # Row is Y
        else:
            # Generate random starting positions avoiding obstacles and other agents
            possible_coords = list(zip(*np.where(self.board == 0))) # Find empty cells
            if len(possible_coords) < self.nb_agents:
                 raise ValueError("Not enough free space to place all agents randomly.")
            chosen_indices = self.np_random.choice(len(possible_coords), size=self.nb_agents, replace=False)
            start_coords = np.array(possible_coords)[chosen_indices]
            self.positionY = start_coords[:, 0] # Row is Y
            self.positionX = start_coords[:, 1] # Col is X


        # Ensure goals are valid - should be done in create_goals ideally
        # Assuming self.goal is (num_agents, 2) with [row, col] format
        assert self.goal.shape == (self.nb_agents, 2), "Goal shape mismatch"

        self.headings = self.np_random.uniform(-np.pi, np.pi, size=(self.nb_agents)) # Use np.pi
        self.embedding = np.ones((self.nb_agents, 1)) # Reset embedding
        self.reached_goal = np.zeros(self.nb_agents, dtype=bool) # Track who reached goal

        # Update agent positions on the board (use '1' for agent)
        # Clear old agent positions if any were set before obstacle placement
        self.board[self.board == 1] = 0
        self.board[self.positionY, self.positionX] = 1

        self._computeDistance() # Compute initial distances/adjacency
        observation = self.getObservations()
        info = self._get_info() # Add any auxiliary info needed

        return observation, info

    def getObservations(self):
        # Make sure the board reflects current agent positions before calculating FOV
        self.updateBoard() # Updates self.board based on current X,Y

        obs = {
            "board": self.updateBoardGoal(), # Board with goals marked
            "fov": self.preprocessObs(),    # Agent's local view
            "adj_matrix": self.adj_matrix.copy(), # Adjacency matrix
            "distances": self.distance_matrix.copy(), # Distance matrix
            "embeddings": self.embedding.copy(),     # Agent embeddings
        }
        return obs

    def _get_info(self):
        # Example info - distances to goal could be useful
        current_pos = np.stack([self.positionY, self.positionX], axis=1)
        # Using Euclidean distance, shape (num_agents,)
        dist_to_goal = np.linalg.norm(current_pos - self.goal, axis=1)
        return {
            "distance_to_goal": dist_to_goal,
            "agents_at_goal": self.reached_goal.copy()
            }

    def getGraph(self):
        # Deprecated? getObservations returns adj_matrix
        return self.adj_matrix.copy()

    def getEmbedding(self):
        # Deprecated? getObservations returns embeddings
        return copy(self.embedding)

    def getPositions(self):
        # Returns (num_agents, 2) array with [Y, X] (row, col)
        return np.array([self.positionY, self.positionX]).T

    def _computeDistance(self):
        # Calculate pairwise distances between agents
        current_pos = np.stack([self.positionY, self.positionX], axis=1) # Shape (n_agents, 2)
        # Use scipy's distance matrix for efficiency if available, or broadcasting:
        delta = current_pos[:, np.newaxis, :] - current_pos[np.newaxis, :, :] # Shape (n, n, 2)
        D_ij = np.linalg.norm(delta, axis=2) # Shape (n, n)

        # Apply sensing range limit
        D_ij[D_ij >= self.sensing_range] = 0
        self.distance_matrix = D_ij

        # Compute adjacency matrix (binary, based on sensing range)
        # Note: Original code computed "closest 4", which is graph specific.
        # Standard adjacency is usually just based on range. Let's stick to range.
        adj = (D_ij > 0) & (D_ij < self.sensing_range) # Agents within range (excluding self)
        np.fill_diagonal(adj, False) # Agent is not adjacent to itself
        self.adj_matrix = adj.astype(np.float32)


    def computeMetrics(self):
        # Calculate metrics at the end of an episode
        # Success rate: Fraction of agents at their goal location
        final_pos = np.stack([self.positionY, self.positionX], axis=1)
        # Check element-wise equality and then if both coords match for each agent
        at_goal = np.all(final_pos == self.goal, axis=1)
        success_rate = np.mean(at_goal) # Fraction of agents at goal

        # Flow time: Sum of steps taken by agents who reached goal, or max_time*N if not all reach
        # This definition might need refinement depending on exact requirements
        if np.all(at_goal):
            flow_time = self.time # If all reached, use current time (steps)
        else:
            # A common definition if not all finish: sum of finish times for those who did,
            # plus max_time for those who didn't. Here we use the simpler version from original code.
            flow_time = self.nb_agents * self.max_time

        return success_rate, flow_time

    def checkAllInGoal(self):
        # Check if ALL agents are simultaneously at their respective goal locations
        current_pos = np.stack([self.positionY, self.positionX], axis=1)
        return np.array_equal(current_pos, self.goal)

    def check_goals(self):
        # This method seems intended to "lock" agents once they reach their goal.
        current_pos = np.stack([self.positionY, self.positionX], axis=1)
        at_goal = np.all(current_pos == self.goal, axis=1)

        # Update the reached_goal tracker
        self.reached_goal = at_goal

        # Revert position for agents who *were* at goal but moved away? No, original keeps them at goal.
        # Lock agents at goal: If an agent is at its goal, keep its position fixed.
        # We apply this *after* collision checks in _updatePositions.

    def computeFlowTime(self):
       # Deprecated? computeMetrics returns flow time. Keeping for compatibility if called elsewhere.
        if self.checkAllInGoal():
            return self.time
        else:
            # Consider returning sum of times for agents at goal if not all finished?
            # Current implementation matches original code.
            return self.nb_agents * self.max_time

    # Original _computeClosest seems unused if adj_matrix is based purely on range. Removed.
    # @staticmethod
    # def _computeClosest(A): ...

    def step(self, actions, emb=None): # Make emb optional if not always provided
        """
        Actions: list or np.array of integers (0-4) for each agent.
        Emb: Optional embeddings (if provided by external model).
        """
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)

        if actions.shape != (self.nb_agents,):
             raise ValueError(f"Actions shape incorrect. Expected ({self.nb_agents},), got {actions.shape}")

        if emb is not None:
             self._updateEmbedding(emb)

        # Store previous positions for collision resolution
        self.positionX_temp = self.positionX.copy()
        self.positionY_temp = self.positionY.copy()

        # 1. Apply actions to get proposed new positions
        action_dx = np.array([self.action_list[act][0] for act in actions])
        action_dy = np.array([self.action_list[act][1] for act in actions]) # Y is vertical -> row change

        proposedX = self.positionX + action_dx
        proposedY = self.positionY + action_dy

        # 2. Check boundaries
        proposedX = np.clip(proposedX, 0, self.board_size - 1)
        proposedY = np.clip(proposedY, 0, self.board_size - 1)

        # 3. Check collisions with obstacles
        if self.obstacles.size > 0:
            proposed_coords_flat = proposedY * self.board_size + proposedX # Flatten coords for checking
            obstacle_coords_flat = self.obstacles[:, 0] * self.board_size + self.obstacles[:, 1]
            obstacle_collision_mask = np.isin(proposed_coords_flat, obstacle_coords_flat)
            # Revert position if colliding with an obstacle
            proposedX[obstacle_collision_mask] = self.positionX_temp[obstacle_collision_mask]
            proposedY[obstacle_collision_mask] = self.positionY_temp[obstacle_collision_mask]

        # 4. Check collisions between agents (Swapping / Staying)
        # This implements a simple "stay if collision" logic. More complex rules exist (e.g., swapping).
        proposed_coords = np.stack([proposedY, proposedX], axis=1)
        unique_coords, indices, counts = np.unique(proposed_coords, axis=0, return_inverse=True, return_counts=True)

        colliding_indices = np.where(counts > 1)[0]
        agent_collision_mask = np.isin(indices, colliding_indices)

        # Revert position for agents involved in a collision
        proposedX[agent_collision_mask] = self.positionX_temp[agent_collision_mask]
        proposedY[agent_collision_mask] = self.positionY_temp[agent_collision_mask]

        # 5. Check if agents reached goal and lock them there
        current_pos = np.stack([proposedY, proposedX], axis=1)
        newly_at_goal = np.all(current_pos == self.goal, axis=1)
        self.reached_goal = newly_at_goal # Update tracker

        # Apply lock: If an agent was already at goal OR just reached it,
        # ensure its position remains the goal position.
        # This overrides collisions IF the collision happens AT the goal.
        # (Consider if this is desired behavior).
        finalX = np.where(self.reached_goal, self.goal[:, 1], proposedX)
        finalY = np.where(self.reached_goal, self.goal[:, 0], proposedY)

        # 6. Update final positions
        self.positionX = finalX
        self.positionY = finalY

        # 7. Update environment state
        self.time += 1
        self._computeDistance() # Recompute distances/adjacency based on new positions
        self.updateBoard() # Update the board representation AFTER positions are finalized

        # 8. Determine termination and truncation
        terminated = self.checkAllInGoal() # Episode ends if all agents are at their goals
        truncated = self.time >= self.max_time # Episode ends if max time is reached

        # 9. Get observation, reward, info
        observation = self.getObservations()
        reward = self._calculate_reward(terminated, truncated, agent_collision_mask, obstacle_collision_mask) # Implement reward function
        info = self._get_info()


        # Gym API expects obs, reward, terminated, truncated, info
        return observation, reward, terminated, truncated, info

    def _calculate_reward(self, terminated, truncated, agent_collisions, obstacle_collisions):
        # Example reward function (needs tuning)
        reward = 0.0

        # Penalty for time step
        reward -= 0.01

        # Penalty for collisions
        reward -= np.sum(agent_collisions) * 0.1 # Penalty per agent involved in collision
        reward -= np.sum(obstacle_collisions) * 0.2 # Harsher penalty for obstacle collision

        # Reward for reaching goal (individual or collective)
        # Individual reward for reaching goal (first time)
        # For simplicity, let's do a large reward if terminated (all reached)
        if terminated:
            reward += 10.0

        # Potentially add reward based on distance decrease to goal?

        return reward


    def _updatePositions(self, actions):
       # This logic is now integrated into the step() method following Gym API best practices.
       # Kept here temporarily for reference during transition if needed, then remove.
        pass


    def _updateEmbedding(self, H):
        # Update agent embeddings if provided externally
        if H.shape == self.embedding.shape:
             self.embedding = H.copy()
        else:
             print(f"Warning: Embedding shape mismatch. Expected {self.embedding.shape}, got {H.shape}")


    def map_goal(self, agent):
        # Maps the absolute goal coordinate to the agent's FOV coordinates.
        # Assumes FOV is centered on agent, size (pad*2 - 1) x (pad*2 - 1)
        fov_size = (self.pad * 2) - 1
        center_offset = self.pad - 1 # Index of the center cell in FOV

        # Goal coords relative to agent's absolute position
        # Remember: Y is row, X is column
        relative_y = self.goal[agent, 0] - self.positionY[agent]
        relative_x = self.goal[agent, 1] - self.positionX[agent]

        # Coords within the FOV grid (origin at top-left of FOV)
        # Flip Y axis because FOV origin is top-left, but relative_y increases downwards
        fov_y = center_offset - relative_y
        fov_x = center_offset + relative_x

        # Check if the goal coordinate is within the FOV bounds
        if 0 <= fov_y < fov_size and 0 <= fov_x < fov_size:
             # Return integer indices if within FOV
             return int(fov_y), int(fov_x)
        else:
             # Return None or special value if goal is outside FOV
             return None, None


    def preprocessObs(self):
        # Creates the Field of View (FOV) for each agent
        # FOV shape: (num_agents, 2, fov_size, fov_size)
        # Channel 0: Obstacles (2) and other agents (1)
        # Channel 1: Goal position (3) if visible

        fov_size = (self.pad * 2) - 1
        # Pad the board: Values: 0=empty, 1=agent, 2=obstacle. Padding adds 2s.
        map_padded = np.pad(self.board, ((self.pad, self.pad), (self.pad, self.pad)), mode='constant', constant_values=2)

        # Pre-calculate padded agent positions
        current_posY_padded = self.positionY + self.pad
        current_posX_padded = self.positionX + self.pad

        # Initialize FOV array
        FOV = np.zeros((self.nb_agents, 2, fov_size, fov_size), dtype=np.float32)

        # Calculate center index of the FOV array
        center_idx = self.pad - 1 # e.g., if pad=3, fov_size=5, center_idx=2

        for agent in range(self.nb_agents):
            # Calculate slicing indices for the padded map
            row_start = current_posY_padded[agent] - center_idx
            row_end = current_posY_padded[agent] + self.pad # Exclusive end index
            col_start = current_posX_padded[agent] - center_idx
            col_end = current_posX_padded[agent] + self.pad # Exclusive end index

            # Extract the agent's local view from the padded map
            local_view = map_padded[row_start:row_end, col_start:col_end]

            # --- Populate FOV Channel 0: Obstacles and Agents ---
            # Copy the local view. Obstacles (2) and other Agents (1) are present.
            FOV[agent, 0, :, :] = local_view

            # Set the agent's own position in its FOV to 0 (doesn't see itself)
            # The center of the FOV corresponds to the agent's location.
            FOV[agent, 0, center_idx, center_idx] = 0

            # --- Populate FOV Channel 1: Goal ---
            # map_goal returns FOV coordinates (y, x) or (None, None)
            gy, gx = self.map_goal(agent=agent)

            if gy is not None: # Check if goal is within FOV
                # Ensure coordinates are valid integers within FOV bounds
                if 0 <= gy < fov_size and 0 <= gx < fov_size:
                    # Mark goal position with value 3 in the goal channel
                    FOV[agent, 1, int(gy), int(gx)] = 3
                # else: # Optional: Handle case where map_goal returned valid coords somehow outside bounds
                #    print(f"Warning: Agent {agent} map_goal returned ({gy},{gx}) outside FOV {fov_size}x{fov_size}")


            # <<< --- START DEBUG PRINT STATEMENTS --- >>>
            print(f"\n--- FOV Debug: Agent {agent} at Step {self.time} ---")
            print(f"  Agent Position (Y,X): ({self.positionY[agent]}, {self.positionX[agent]})")
            print(f"  Agent Goal     (Y,X): ({self.goal[agent, 0]}, {self.goal[agent, 1]})")
            print(f"  Calculated Goal in FOV (gy, gx): ({gy}, {gx})") # gy=row, gx=col in FOV array
            print(f"  FOV Center Index: {center_idx}")

            # Optional: Print the goal channel of the FOV to see where the '3' is placed
            if gy is not None and 0 <= gy < fov_size and 0 <= gx < fov_size:
                 print(f"  FOV Goal Channel Value at ({int(gy)},{int(gx)}): {FOV[agent, 1, int(gy), int(gx)]}")
                 # Uncomment below for verbose output of the entire goal channel
                 # print(f"  FOV Goal Channel (Channel 1 for Agent {agent}):\n", FOV[agent, 1, :, :])
            else:
                 print("  Goal is calculated to be outside FOV.")
            # <<< --- END DEBUG PRINT STATEMENTS --- >>>

        # Return the completed FOV array for all agents
        return FOV

    def check_boundary(self):
         # Boundary checking is now done within the step() method using np.clip.
         pass


    def updateBoard(self):
        # Updates self.board based on current agent positions
        # Clear previous agent positions (cells marked with 1)
        self.board[self.board == 1] = 0
        # Place agents at their current positions (mark with 1)
        # Ensure agents are not placed on obstacles (collision logic should prevent this)
        valid_agent_mask = self.board[self.positionY, self.positionX] != 2
        self.board[self.positionY[valid_agent_mask], self.positionX[valid_agent_mask]] = 1
        # Note: Obstacles (2) remain unchanged. Goals are not marked on self.board.


    def updateBoardGoal(self):
        # Returns a temporary copy of the board with goals marked for observation
        board_with_goals = self.board.copy()
        # Mark goal locations (use value 4, consistent with original?)
        # Ensure goal locations are valid
        valid_goals = self.goal[ (self.goal[:, 0] >= 0) & (self.goal[:, 0] < self.board_size) & \
                                 (self.goal[:, 1] >= 0) & (self.goal[:, 1] < self.board_size)]
        if len(valid_goals) != len(self.goal):
            print("Warning: Some goals are outside board boundaries.")

        if valid_goals.size > 0:
            # Add 4 to goal cell value (can be 0, 1, or 2 initially)
            # Assuming goal coords are (row, col)
             board_with_goals[valid_goals[:, 0], valid_goals[:, 1]] += 4

        return board_with_goals


    def check_collisions(self):
        # Agent-Agent collision checking is now done within the step() method.
        pass

    def check_collision_obstacle(self):
        # Agent-Obstacle collision checking is now done within the step() method.
        pass

    def printBoard(self):
        # For simple text representation
        self.updateBoard() # Ensure board is current
        # Maybe create a string representation with different chars for agents, obstacles, goals?
        return f"Game Board at step {self.time}:\n{self.board}"

    def render(self, mode="human", agentId=None, printNeigh=False, printFOV=False):
        # --- RENDER TO RGB ARRAY ---
        if mode == "rgb_array":
            fig = Figure(figsize=(5, 5)) # Create a figure
            canvas = FigureCanvasAgg(fig) # Attach Agg canvas
            ax = fig.add_subplot(111)      # Add axes to the figure

            # --- Replicate plotting logic using 'ax' instead of 'plt' ---
            ax.axis('off') # Turn off axis labels/ticks

            # Draw neighbor lines (optional, can be cluttered)
            if printNeigh: # Use the flag passed to render
                for agent in range(self.nb_agents):
                    neighbors = np.where(self.adj_matrix[agent])[0]
                    for neighbor in neighbors:
                        if agent < neighbor: # Avoid drawing lines twice
                            ax.plot(
                                [self.positionX[agent], self.positionX[neighbor]],
                                [self.positionY[agent], self.positionY[neighbor]],
                                color=self._color_neighbor_line, # Use defined color
                                linewidth=0.5, # Make lines thinner
                                zorder=1 # Draw lines behind agents
                            )

            # Draw obstacles
            if self.obstacles.size > 0:
                 ax.scatter(
                     self.obstacles[:, 1], # X coordinates (cols)
                     self.obstacles[:, 0], # Y coordinates (rows)
                     color=self._color_obstacle,
                     marker="s", # Square marker
                     s=150,     # Size
                     zorder=2   # Draw obstacles above lines
                 )

            # Draw goals
            if self.goal.size > 0:
                 ax.scatter(
                     self.goal[:, 1], # X coordinates (cols)
                     self.goal[:, 0], # Y coordinates (rows)
                     color=self._color_goal,
                     marker="*", # Star marker
                     s=150,     # Size
                     zorder=3   # Draw goals above obstacles
                 )

            # Draw agents
            ax.scatter(
                self.positionX,
                self.positionY,
                s=150, # Size
                c=self.mapper.to_rgba(self.embedding.flatten()), # Color based on embedding
                zorder=4 # Draw agents on top
            )

            # Draw boundaries
            ax.plot([-0.5, self.board_size - 0.5], [-0.5, -0.5], color=self._color_boundary) # Bottom
            ax.plot([-0.5, -0.5], [-0.5, self.board_size - 0.5], color=self._color_boundary) # Left
            ax.plot([-0.5, self.board_size - 0.5], [self.board_size - 0.5, self.board_size - 0.5], color=self._color_boundary) # Top
            ax.plot([self.board_size - 0.5, self.board_size - 0.5], [-0.5, self.board_size - 0.5], color=self._color_boundary) # Right

            # Set axis limits slightly expanded
            ax.axis([-1, self.board_size, -1, self.board_size])
            ax.set_aspect('equal', adjustable='box') # Ensure square cells

            # --- Render the canvas to buffer and convert to numpy array ---
            canvas.draw()
            image = np.asarray(canvas.buffer_rgba())
            plt.close(fig) # Close the temporary figure to free memory
            return image

        # --- RENDER TO SCREEN (using global plt, stateful) ---
        elif mode == "human" or mode == "plot":
            if self._plot_fig is None: # Initialize figure only once
                plt.ion() # Turn on interactive mode
                self._plot_fig, self._plot_ax = plt.subplots(figsize=(6, 6))

            ax = self._plot_ax
            ax.clear() # Clear previous frame
            ax.set_facecolor('white') # Set background
            ax.axis('off')

            # Draw neighbor lines (optional)
            if printNeigh:
                 for agent in range(self.nb_agents):
                    neighbors = np.where(self.adj_matrix[agent])[0]
                    for neighbor in neighbors:
                        if agent < neighbor:
                            ax.plot(
                                [self.positionX[agent], self.positionX[neighbor]],
                                [self.positionY[agent], self.positionY[neighbor]],
                                color=self._color_neighbor_line, linewidth=0.5, zorder=1
                            )

            # Draw obstacles
            if self.obstacles.size > 0:
                 ax.scatter(self.obstacles[:, 1], self.obstacles[:, 0], color=self._color_obstacle, marker="s", s=150, zorder=2)

            # Draw goals
            if self.goal.size > 0:
                 ax.scatter(self.goal[:, 1], self.goal[:, 0], color=self._color_goal, marker="*", s=150, zorder=3)

            # Draw agents
            ax.scatter(self.positionX, self.positionY, s=150, c=self.mapper.to_rgba(self.embedding.flatten()), zorder=4)

             # Draw boundaries
            ax.plot([-0.5, self.board_size - 0.5], [-0.5, -0.5], color=self._color_boundary) # Bottom
            ax.plot([-0.5, -0.5], [-0.5, self.board_size - 0.5], color=self._color_boundary) # Left
            ax.plot([-0.5, self.board_size - 0.5], [self.board_size - 0.5, self.board_size - 0.5], color=self._color_boundary) # Top
            ax.plot([self.board_size - 0.5, self.board_size - 0.5], [-0.5, self.board_size - 0.5], color=self._color_boundary) # Right

            # Set axis limits and aspect ratio
            ax.axis([-1, self.board_size, -1, self.board_size])
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(f"Step: {self.time}") # Show current step

            plt.draw()
            plt.pause(0.1) # Pause for visualization
            return None # Human mode doesn't return array

        elif mode == "photo": # Original mode, less common for Gym
             # This mode seems to just show the raw board matrix
             if self._plot_fig is None:
                 plt.ion()
                 self._plot_fig, self._plot_ax = plt.subplots(figsize=(5,5))
             ax = self._plot_ax
             ax.clear()
             ax.imshow(self.updateBoardGoal(), cmap="Greys", origin='lower') # Show board with goals
             ax.set_title(f"Board State - Step: {self.time}")
             plt.draw()
             plt.pause(0.1)
             return None

        else:
            # Handle other modes or raise error
            # return super().render(mode=mode) # Call parent if inheriting from Gym Env directly
            print(f"Warning: Unsupported render mode '{mode}'. Available modes: {self.metadata['render_modes']}")
            return None

    def close(self):
        # Clean up resources, especially the matplotlib figure if used
        if self._plot_fig is not None:
            plt.close(self._plot_fig)
            self._plot_fig = None
            self._plot_ax = None
            plt.ioff() # Turn off interactive mode


########## utils ##########
def create_goals(board_size, num_agents, obstacles=None):
    # Ensure board_size is (rows, cols) tuple/list
    rows, cols = board_size[0], board_size[1]

    # Create a temporary board to mark obstacle locations
    temp_board = np.zeros((rows, cols))
    if obstacles is not None and obstacles.size > 0:
        # Assuming obstacles are (row, col)
        valid_obstacles = obstacles[(obstacles[:, 0] >= 0) & (obstacles[:, 0] < rows) & (obstacles[:, 1] >= 0) & (obstacles[:, 1] < cols)]
        if len(valid_obstacles) > 0:
             temp_board[valid_obstacles[:, 0], valid_obstacles[:, 1]] = 1 # Mark obstacles

    # Find all available coordinates (where temp_board == 0)
    available_coords = list(zip(*np.where(temp_board == 0)))

    if len(available_coords) < num_agents:
        raise ValueError("Not enough free spaces to place all goals.")

    # Randomly choose unique coordinates for goals
    chosen_indices = np.random.choice(len(available_coords), size=num_agents, replace=False)
    goals = np.array(available_coords)[chosen_indices] # Shape (num_agents, 2) -> [[row1, col1], [row2, col2], ...]

    return goals


def create_obstacles(board_size, nb_obstacles):
    # Ensure board_size is (rows, cols) tuple/list
    rows, cols = board_size[0], board_size[1]
    total_cells = rows * cols

    if nb_obstacles > total_cells:
        raise ValueError("Number of obstacles cannot exceed the total number of cells.")

    # Generate all possible coordinates
    all_coords = np.array([(r, c) for r in range(rows) for c in range(cols)])

    # Randomly choose unique coordinates for obstacles
    chosen_indices = np.random.choice(total_cells, size=nb_obstacles, replace=False)
    obstacles = all_coords[chosen_indices] # Shape (nb_obstacles, 2) -> [[row1, col1], ...]

    return obstacles


if __name__ == "__main__":
    # Example Usage (adapted from original)
    agents = 2
    bs = 16 # Board size (square)
    board_dims = [bs, bs]
    config = {
        "num_agents": agents,
        "board_size": board_dims, # Use list/tuple
        "max_time": 60, # Increased max time
        "min_time": 1, # Should be <= max_time
        "obstacles": 10, # Number of obstacles to generate using create_obstacles
        "sensing_range": 4,
        "pad": 3, # Related to FOV size
    }
    sensing = config["sensing_range"]

    # Generate obstacles and goals
    obstacles = create_obstacles(board_dims, config["obstacles"])
    # Generate start positions (must avoid obstacles)
    start_pos = create_goals(board_dims, agents, obstacles) # Use create_goals logic for valid positions
    # Generate goal positions (must avoid obstacles and preferably start positions)
    temp_obstacles_for_goals = np.vstack([obstacles, start_pos]) if obstacles.size > 0 else start_pos
    goals = create_goals(board_dims, agents, temp_obstacles_for_goals)

    print("Obstacles:\n", obstacles)
    print("Start Pos:\n", start_pos)
    print("Goals:\n", goals)


    env = GraphEnv(
        config,
        goal=goals,
        # board_size=board_dims, # board_size is now directly from config
        sensing_range=sensing,
        starting_positions=start_pos,
        obstacles=obstacles,
        pad = config['pad']
    )

    # Simple test loop (random actions)
    obs, info = env.reset()
    print("Initial Observation Keys:", obs.keys())
    print("Initial Info:", info)

    plt.ion() # Ensure interactive mode is on for the test loop

    for step in range(config["max_time"] + 5): # Run a few steps
        # Choose random actions for testing
        actions = env.action_space.sample() # If action space is MultiDiscrete
        if isinstance(env.action_space, spaces.Discrete): # If single Discrete (needs adjustment)
            actions = np.random.randint(0, env.action_space.n, size=agents)


        # Use dummy embeddings if needed by step signature, otherwise omit
        # emb = np.random.rand(agents, 1)
        # obs, reward, terminated, truncated, info = env.step(actions, emb)
        obs, reward, terminated, truncated, info = env.step(actions)

        print(f"\n--- Step {env.time} ---")
        print(f"Actions: {actions}")
        print(f"Reward: {reward:.3f}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")
        print(f"Info: {info}")
        # print("Agent Positions (Y,X):", env.getPositions())

        env.render(mode="human", printNeigh=True) # Render in human mode

        if terminated or truncated:
            print("\nEpisode Finished!")
            print(f"Final Status: {'Success!' if terminated else 'Time Limit Reached.'}")
            final_metrics = env.computeMetrics()
            print(f"Success Rate (Agents at goal): {final_metrics[0]:.2f}")
            print(f"Flow Time: {final_metrics[1]}")
            break

    env.close() # Close the rendering window
    print("\nTest Finished.")