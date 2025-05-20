from manim import *
import numpy as np

class MultiAgentTrainingVisualization(Scene):
    def construct(self):
        # Set up constants and colors
        GRID_SIZE = 10
        SCALE_FACTOR = 0.6
        
        # Colors
        GRID_COLOR = BLUE_B
        OBSTACLE_COLOR = GRAY
        AGENT_COLORS = [RED, GREEN, BLUE]
        GOAL_COLORS = [RED_A, GREEN_A, BLUE_A]
        ACTION_COLORS = [RED_B, GREEN_B, BLUE_B]
        
        # Define the grid
        grid = self.create_grid(GRID_SIZE, SCALE_FACTOR, GRID_COLOR)
        self.play(Create(grid))
        
        # Label axes
        x_labels, y_labels = self.create_grid_labels(GRID_SIZE, SCALE_FACTOR)
        self.play(
            *[Write(label) for label in x_labels + y_labels],
            run_time=1
        )
        
        # Title and description
        title = Text("Multi-Agent Pathfinding: Training Process", font_size=36)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))
        
        # Add obstacles
        obstacle_positions = [(2, 2), (5, 5), (7, 3)]
        obstacles = self.create_obstacles(obstacle_positions, SCALE_FACTOR, OBSTACLE_COLOR)
        obstacle_label = Text("Obstacles", font_size=24, color=OBSTACLE_COLOR)
        obstacle_label.next_to(obstacles[0], UP, buff=0.5)
        
        self.play(
            FadeIn(obstacle_label),
            *[FadeIn(obs) for obs in obstacles],
            run_time=1
        )
        self.wait(1)
        self.play(FadeOut(obstacle_label))
        
        # Initialize agents and goals
        agent_positions = [(1, 1), (1, 3), (3, 1)]
        goal_positions = [(8, 8), (8, 1), (1, 8)]
        
        agents, agent_labels = self.create_agents(agent_positions, SCALE_FACTOR, AGENT_COLORS)
        goals, goal_labels = self.create_goals(goal_positions, SCALE_FACTOR, GOAL_COLORS)
        
        # Show the initial state
        initial_state_title = Text("Initial State (t=0)", font_size=32)
        initial_state_title.to_edge(UP, buff=0.5)
        
        self.play(
            ReplacementTransform(title, initial_state_title),
            *[FadeIn(agent) for agent in agents],
            *[FadeIn(label) for label in agent_labels],
            run_time=1
        )
        
        self.play(
            *[FadeIn(goal) for goal in goals],
            *[FadeIn(label) for label in goal_labels],
            run_time=1
        )
        
        # Show communication ranges
        comm_range = 3
        comm_circles = self.create_communication_circles(
            agent_positions, SCALE_FACTOR, comm_range, AGENT_COLORS
        )
        
        comm_title = Text("Communication Range", font_size=32)
        comm_title.to_edge(UP, buff=0.5)
        
        self.play(
            ReplacementTransform(initial_state_title, comm_title),
            *[Create(circle) for circle in comm_circles],
            run_time=1.5
        )
        self.wait(1)
        
        # Show the Graph Signal Operator (GSO) matrix
        gso_matrix = self.create_gso_matrix(agent_positions, comm_range)
        self.play(
            *[FadeOut(circle) for circle in comm_circles],
        )
        
        # Display FOVs for each agent
        fov_size = 5  # Field of view size (odd number)
        fov_title = Text("Field of View (FOV)", font_size=32)
        fov_title.to_edge(UP, buff=0.5)
        
        self.play(ReplacementTransform(comm_title, fov_title))
        
        # Create FOV visualizations for each agent
        fovs = []
        for i, pos in enumerate(agent_positions):
            fov = self.create_agent_fov(
                i, pos, GRID_SIZE, SCALE_FACTOR, fov_size, 
                agent_positions, goal_positions, obstacle_positions,
                AGENT_COLORS, GOAL_COLORS, OBSTACLE_COLOR
            )
            fovs.append(fov)
        
        # Show FOVs one by one
        for i, fov in enumerate(fovs):
            fov_agent_title = Text(f"Agent {i} FOV", font_size=32, color=AGENT_COLORS[i])
            fov_agent_title.to_edge(UP, buff=0.5)
            
            self.play(ReplacementTransform(fov_title if i == 0 else fovs[i-1], fov))
            if i == 0:
                self.play(ReplacementTransform(fov_title, fov_agent_title))
            else:
                prev_title = Text(f"Agent {i-1} FOV", font_size=32, color=AGENT_COLORS[i-1])
                prev_title.to_edge(UP, buff=0.5)
                self.play(ReplacementTransform(prev_title, fov_agent_title))
            
            self.wait(1)
        
        # Show the CNN, GNN, and MLP process
        model_title = Text("Neural Network Processing", font_size=32)
        model_title.to_edge(UP, buff=0.5)
        self.play(
            ReplacementTransform(Text(f"Agent {len(fovs)-1} FOV", font_size=32, color=AGENT_COLORS[-1]).to_edge(UP, buff=0.5), 
            model_title),
            FadeOut(fovs[-1])
        )
        
        # Create the neural network diagram
        nn_diagram = self.create_nn_diagram()
        self.play(Create(nn_diagram), run_time=2)
        self.wait(1)
        
        # Show the output actions
        actions = ["Idle (0)", "Right (1)", "Up (2)", "Left (3)", "Down (4)"]
        expert_actions = [actions[1], actions[4], actions[0]]  # Right, Down, Idle
        
        action_title = Text("Expert Actions (t=0)", font_size=32)
        action_title.to_edge(UP, buff=0.5)
        
        action_texts = []
        for i, action in enumerate(expert_actions):
            action_text = Text(
                f"Agent {i}: {action}", 
                font_size=28, 
                color=AGENT_COLORS[i]
            )
            action_text.next_to(nn_diagram, DOWN, buff=0.5 + i*0.4)
            action_texts.append(action_text)
        
        self.play(
            ReplacementTransform(model_title, action_title),
            FadeOut(nn_diagram),
            *[Write(text) for text in action_texts],
            run_time=1.5
        )
        self.wait(1)
        
        # Show the next state (t=1)
        next_positions = [(1, 2), (2, 3), (3, 1)]  # After applying expert actions
        next_agents = self.create_agents(
            next_positions, SCALE_FACTOR, AGENT_COLORS, is_filled=False
        )[0]
        
        next_state_title = Text("Next State (t=1)", font_size=32)
        next_state_title.to_edge(UP, buff=0.5)
        
        # Show agent movements with arrows
        movement_arrows = []
        for i, (start, end) in enumerate(zip(agent_positions, next_positions)):
            if start != end:  # Only create arrows for moving agents
                arrow = self.create_movement_arrow(
                    start, end, SCALE_FACTOR, AGENT_COLORS[i]
                )
                movement_arrows.append(arrow)
        
        self.play(
            ReplacementTransform(action_title, next_state_title),
            *[FadeOut(text) for text in action_texts],
            *[Create(arrow) for arrow in movement_arrows],
            run_time=1
        )
        
        self.play(
            *[ReplacementTransform(agents[i], next_agents[i]) for i in range(len(agents))],
            *[FadeOut(label) for label in agent_labels],
            run_time=1.5
        )
        
        # Update agent labels for t=1
        next_agent_labels = []
        for i, pos in enumerate(next_positions):
            label = Text(f"Agent {i} (t=1)", font_size=20, color=AGENT_COLORS[i])
            grid_pos = self.grid_to_screen(pos, SCALE_FACTOR)
            label.next_to(grid_pos, DOWN, buff=0.2)
            next_agent_labels.append(label)
        
        self.play(
            *[FadeIn(label) for label in next_agent_labels],
            *[FadeOut(arrow) for arrow in movement_arrows],
            run_time=1
        )
        self.wait(1)
        
        # Show learning process
        learn_title = Text("Learning Process", font_size=32)
        learn_title.to_edge(UP, buff=0.5)
        
        loss_formula = MathTex(
            r"\text{Loss} = -\frac{1}{N}\sum_{i=0}^{N-1} \log(P_i(a_i^*))",
            font_size=36
        )
        loss_formula.next_to(learn_title, DOWN, buff=0.5)
        
        example_loss = MathTex(
            r"\text{Loss} \approx -\frac{\log(0.5) + \log(0.6) + \log(0.7)}{3}",
            font_size=32
        )
        example_loss.next_to(loss_formula, DOWN, buff=0.5)
        
        self.play(
            ReplacementTransform(next_state_title, learn_title),
            *[FadeOut(agent) for agent in next_agents],
            *[FadeOut(label) for label in next_agent_labels],
            *[FadeOut(goal) for goal in goals],
            *[FadeOut(label) for label in goal_labels],
            Write(loss_formula),
            run_time=1.5
        )
        
        self.play(Write(example_loss), run_time=1)
        self.wait(1)
        
        # Final summary
        summary_title = Text("Training Summary", font_size=32)
        summary_title.to_edge(UP, buff=0.5)
        
        summary_points = [
            "• Model learns state-to-action mapping",
            "• Training uses many expert trajectories",
            "• Loss function minimizes difference between predicted and expert actions",
            "• After training, model should generalize to new scenarios"
        ]
        
        summary_text = VGroup(*[
            Text(point, font_size=28, color=WHITE).align_to(Point(), LEFT)
            for point in summary_points
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        summary_text.next_to(summary_title, DOWN, buff=0.5)
        
        self.play(
            ReplacementTransform(learn_title, summary_title),
            FadeOut(loss_formula),
            FadeOut(example_loss),
            run_time=1
        )
        
        self.play(Write(summary_text), run_time=2)
        self.wait(2)
        
        # End screen
        end_title = Text("Multi-Agent Pathfinding with GNNs", font_size=40)
        end_title.to_edge(UP, buff=1)
        
        self.play(
            FadeOut(summary_title),
            FadeOut(summary_text),
            FadeOut(grid),
            *[FadeOut(obs) for obs in obstacles],
            *[FadeOut(label) for label in x_labels + y_labels],
            run_time=1
        )
        
        self.play(Write(end_title), run_time=1)
        self.wait(2)
        self.play(FadeOut(end_title), run_time=1)

    def create_grid(self, size, scale_factor, color=BLUE_B):
        """Creates a grid of specified size and color."""
        grid = VGroup()
        
        # Create horizontal and vertical lines
        for i in range(size + 1):
            # Horizontal line
            h_line = Line(
                start=scale_factor * np.array([-size/2, i - size/2, 0]),
                end=scale_factor * np.array([size/2, i - size/2, 0]),
                color=color
            )
            grid.add(h_line)
            
            # Vertical line
            v_line = Line(
                start=scale_factor * np.array([i - size/2, -size/2, 0]),
                end=scale_factor * np.array([i - size/2, size/2, 0]),
                color=color
            )
            grid.add(v_line)
            
        return grid
    
    def create_grid_labels(self, size, scale_factor):
        """Creates labels for the grid axes."""
        x_labels = []
        y_labels = []
        
        for i in range(size):
            # X-axis labels (bottom edge)
            x_label = Text(str(i), font_size=16)
            x_pos = scale_factor * np.array([i - size/2 + 0.5, -size/2 - 0.3, 0])
            x_label.move_to(x_pos)
            x_labels.append(x_label)
            
            # Y-axis labels (left edge)
            y_label = Text(str(i), font_size=16)
            y_pos = scale_factor * np.array([-size/2 - 0.3, i - size/2 + 0.5, 0])
            y_label.move_to(y_pos)
            y_labels.append(y_label)
            
        return x_labels, y_labels
    
    def grid_to_screen(self, grid_pos, scale_factor):
        """Converts grid coordinates to screen coordinates."""
        x, y = grid_pos
        grid_size = 10  # Assuming 10x10 grid
        screen_x = scale_factor * (x - grid_size/2 + 0.5)
        screen_y = scale_factor * (y - grid_size/2 + 0.5)
        return np.array([screen_x, screen_y, 0])
    
    def create_obstacles(self, positions, scale_factor, color=GRAY):
        """Creates obstacle squares at specified positions."""
        obstacles = []
        for pos in positions:
            square = Square(
                side_length=scale_factor,
                color=color,
                fill_color=color,
                fill_opacity=0.7
            )
            square.move_to(self.grid_to_screen(pos, scale_factor))
            obstacles.append(square)
            
        return obstacles
    
    def create_agents(self, positions, scale_factor, colors, is_filled=True):
        """Creates agent circles at specified positions."""
        agents = []
        labels = []
        
        for i, (pos, color) in enumerate(zip(positions, colors)):
            circle = Circle(
                radius=scale_factor/3,
                color=color,
                fill_color=color if is_filled else BLACK,
                fill_opacity=0.8 if is_filled else 0
            )
            circle.move_to(self.grid_to_screen(pos, scale_factor))
            agents.append(circle)
            
            # Add label
            time_suffix = " (t=0)" if is_filled else ""
            label = Text(f"Agent {i}{time_suffix}", font_size=20, color=color)
            label.next_to(circle, DOWN, buff=0.2)
            labels.append(label)
            
        return agents, labels
    
    def create_goals(self, positions, scale_factor, colors):
        """Creates goal stars at specified positions."""
        goals = []
        labels = []
        
        for i, (pos, color) in enumerate(zip(positions, colors)):
            star = Star(
                n=5,
                outer_radius=scale_factor/3,
                color=color,
                fill_color=color,
                fill_opacity=0.4
            )
            star.move_to(self.grid_to_screen(pos, scale_factor))
            goals.append(star)
            
            # Add label
            label = Text(f"Goal {i}", font_size=20, color=color)
            label.next_to(star, UP, buff=0.2)
            labels.append(label)
            
        return goals, labels
    
    def create_communication_circles(self, positions, scale_factor, comm_range, colors):
        """Creates circles showing communication range for each agent."""
        circles = []
        
        for pos, color in zip(positions, colors):
            # Convert grid range to screen distance
            radius = comm_range * scale_factor
            
            circle = Circle(
                radius=radius,
                color=color,
                stroke_opacity=0.5,
                fill_opacity=0.1,
                fill_color=color
            )
            circle.move_to(self.grid_to_screen(pos, scale_factor))
            circles.append(circle)
            
        return circles
    
    def create_gso_matrix(self, agent_positions, comm_range):
        """Creates a representation of the GSO adjacency matrix."""
        n_agents = len(agent_positions)
        gso = np.zeros((n_agents, n_agents))
        
        # Fill the GSO matrix based on communication range
        for i, pos_i in enumerate(agent_positions):
            for j, pos_j in enumerate(agent_positions):
                # Calculate Manhattan distance
                dist = abs(pos_i[0] - pos_j[0]) + abs(pos_i[1] - pos_j[1])
                if dist <= comm_range:
                    gso[i, j] = 1
                    
        return gso
    
    def create_agent_fov(self, agent_idx, position, grid_size, scale_factor, fov_size, 
                        agent_positions, goal_positions, obstacle_positions,
                        agent_colors, goal_colors, obstacle_color):
        """Creates a visualization of an agent's field of view."""
        fov_group = VGroup()
        
        # Create the FOV background
        fov_background = Square(
            side_length=fov_size * scale_factor,
            color=agent_colors[agent_idx],
            fill_color=BLACK,
            fill_opacity=0.1,
            stroke_width=2
        )
        fov_background.move_to(self.grid_to_screen(position, scale_factor))
        fov_group.add(fov_background)
        
        # Label for FOV
        fov_label = Text(
            f"Agent {agent_idx} FOV", 
            font_size=20, 
            color=agent_colors[agent_idx]
        )
        fov_label.next_to(fov_background, LEFT, buff=0.2)
        fov_group.add(fov_label)
        
        # Half of the FOV size
        half_fov = fov_size // 2
        
        # Calculate FOV grid boundaries
        min_x = max(0, position[0] - half_fov)
        max_x = min(grid_size - 1, position[0] + half_fov)
        min_y = max(0, position[1] - half_fov)
        max_y = min(grid_size - 1, position[1] + half_fov)
        
        # Create FOV grid lines
        for i in range(min_x, max_x + 2):
            rel_x = i - position[0] + half_fov
            if 0 <= rel_x <= fov_size:
                line = Line(
                    start=scale_factor * np.array([
                        position[0] - half_fov + rel_x - grid_size/2, 
                        position[1] - half_fov - grid_size/2, 
                        0
                    ]),
                    end=scale_factor * np.array([
                        position[0] - half_fov + rel_x - grid_size/2, 
                        position[1] + half_fov - grid_size/2, 
                        0
                    ]),
                    color=BLUE_D,
                    stroke_width=1
                )
                fov_group.add(line)
        
        for j in range(min_y, max_y + 2):
            rel_y = j - position[1] + half_fov
            if 0 <= rel_y <= fov_size:
                line = Line(
                    start=scale_factor * np.array([
                        position[0] - half_fov - grid_size/2, 
                        position[1] - half_fov + rel_y - grid_size/2, 
                        0
                    ]),
                    end=scale_factor * np.array([
                        position[0] + half_fov - grid_size/2, 
                        position[1] - half_fov + rel_y - grid_size/2, 
                        0
                    ]),
                    color=BLUE_D,
                    stroke_width=1
                )
                fov_group.add(line)
        
        # Add obstacles in FOV
        for obs_pos in obstacle_positions:
            if min_x <= obs_pos[0] <= max_x and min_y <= obs_pos[1] <= max_y:
                obs_square = Square(
                    side_length=scale_factor,
                    color=obstacle_color,
                    fill_color=obstacle_color,
                    fill_opacity=0.7
                )
                obs_square.move_to(self.grid_to_screen(obs_pos, scale_factor))
                fov_group.add(obs_square)
        
        # Add other agents in FOV
        for i, (agent_pos, color) in enumerate(zip(agent_positions, agent_colors)):
            if i != agent_idx and min_x <= agent_pos[0] <= max_x and min_y <= agent_pos[1] <= max_y:
                agent_circle = Circle(
                    radius=scale_factor/3,
                    color=color,
                    fill_color=color,
                    fill_opacity=0.8
                )
                agent_circle.move_to(self.grid_to_screen(agent_pos, scale_factor))
                fov_group.add(agent_circle)
        
        # Add agent's goal if in FOV
        goal_pos = goal_positions[agent_idx]
        if min_x <= goal_pos[0] <= max_x and min_y <= goal_pos[1] <= max_y:
            goal_star = Star(
                n=5,
                outer_radius=scale_factor/3,
                color=goal_colors[agent_idx],
                fill_color=goal_colors[agent_idx],
                fill_opacity=0.4
            )
            goal_star.move_to(self.grid_to_screen(goal_pos, scale_factor))
            fov_group.add(goal_star)
        elif goal_pos[0] < min_x or goal_pos[0] > max_x or goal_pos[1] < min_y or goal_pos[1] > max_y:
            # Add arrow pointing to the goal if it's outside FOV
            center_pos = self.grid_to_screen(position, scale_factor)
            
            # Calculate direction to goal
            dir_x = goal_pos[0] - position[0]
            dir_y = goal_pos[1] - position[1]
            
            # Normalize and scale
            magnitude = np.sqrt(dir_x**2 + dir_y**2)
            if magnitude > 0:
                dir_x = dir_x / magnitude
                dir_y = dir_y / magnitude
            
            # Create the arrow
            arrow_start = center_pos
            arrow_end = center_pos + scale_factor * np.array([dir_x, dir_y, 0])
            
            goal_arrow = Arrow(
                start=arrow_start,
                end=arrow_end,
                color=goal_colors[agent_idx],
                buff=0,
                stroke_width=3,
                max_tip_length_to_length_ratio=0.3
            )
            fov_group.add(goal_arrow)
        
        # Add self (agent) at the center
        self_circle = Circle(
            radius=scale_factor/3,
            color=agent_colors[agent_idx],
            fill_color=agent_colors[agent_idx],
            fill_opacity=1.0
        )
        self_circle.move_to(self.grid_to_screen(position, scale_factor))
        fov_group.add(self_circle)
        
        return fov_group
    
    def create_nn_diagram(self):
        """Creates a diagram of the neural network architecture."""
        diagram = VGroup()
        
        # Create boxes for the neural network components
        cnn_box = Rectangle(height=1.5, width=2, color=YELLOW)
        cnn_text = Text("CNN", font_size=30)
        cnn_text.move_to(cnn_box.get_center())
        cnn = VGroup(cnn_box, cnn_text)
        
        gnn_box = Rectangle(height=1.5, width=2, color=GREEN)
        gnn_text = Text("GNN", font_size=30)
        gnn_text.move_to(gnn_box.get_center())
        gnn = VGroup(gnn_box, gnn_text)
        
        mlp_box = Rectangle(height=1.5, width=2, color=BLUE)
        mlp_text = Text("MLP", font_size=30)
        mlp_text.move_to(mlp_box.get_center())
        mlp = VGroup(mlp_box, mlp_text)
        
        # Position the components
        cnn.shift(LEFT * 4)
        mlp.shift(RIGHT * 4)
        
        # Create arrows between components
        arrow1 = Arrow(cnn.get_right(), gnn.get_left(), buff=0.1)
        arrow2 = Arrow(gnn.get_right(), mlp.get_left(), buff=0.1)
        
        # Labels for inputs and outputs
        input_label = Text("Agent FOVs", font_size=24)
        input_label.next_to(cnn, UP, buff=0.3)
        
        gso_label = Text("GSO Matrix", font_size=24)
        gso_label.next_to(gnn, UP, buff=0.3)
        
        output_label = Text("Action Logits", font_size=24)
        output_label.next_to(mlp, UP, buff=0.3)
        
        # Add all components to the diagram
        diagram.add(
            cnn, gnn, mlp,
            arrow1, arrow2,
            input_label, gso_label, output_label
        )
        
        return diagram
    
    def create_movement_arrow(self, start_pos, end_pos, scale_factor, color):
        """Creates an arrow showing agent movement between states."""
        start_screen = self.grid_to_screen(start_pos, scale_factor)
        end_screen = self.grid_to_screen(end_pos, scale_factor)
        
        arrow = Arrow(
            start=start_screen,
            end=end_screen,
            color=color,
            buff=scale_factor/3,  # Leave space for the agent circle
            stroke_width=4
        )
        
        return arrow


class MultiAgentLossExplanation(Scene):
    def construct(self):
        # Title
        title = Text("Learning Process: Loss Calculation", font_size=36)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.wait(1)
        
        # Agent outputs
        agent_outputs = [
            "Agent 0: [0.1, 1.2, -0.3, 0.5, 0.0] → [0.2, 0.5, 0.1, 0.1, 0.1]",
            "Agent 1: [-0.5, 0.2, 0.1, 0.8, 1.5] → [0.1, 0.2, 0.1, 0.2, 0.4]",
            "Agent 2: [1.8, 0.1, -0.1, 0.3, 0.2] → [0.7, 0.1, 0.1, 0.1, 0.1]"
        ]
        
        output_title = Text("Model Outputs (Logits → Probabilities)", font_size=30)
        output_title.next_to(title, DOWN, buff=0.5)
        
        output_texts = []
        for i, output in enumerate(agent_outputs):
            text = Text(output, font_size=24, color=BLUE if i == 0 else GREEN if i == 1 else RED)
            text.next_to(output_title, DOWN, buff=0.3 + i*0.4)
            output_texts.append(text)
        
        self.play(Write(output_title))
        self.play(*[Write(text) for text in output_texts], run_time=2)
        self.wait(1)
        
        # Expert actions
        expert_title = Text("Expert Actions (Ground Truth)", font_size=30)
        expert_title.next_to(output_texts[-1], DOWN, buff=0.7)
        
        expert_actions = [
            "Agent 0: Action 1 (Right)",
            "Agent 1: Action 4 (Down)",
            "Agent 2: Action 0 (Idle)"
        ]
        
        expert_texts = []
        for i, action in enumerate(expert_actions):
            text = Text(action, font_size=24, color=BLUE if i == 0 else GREEN if i == 1 else RED)
            text.next_to(expert_title, DOWN, buff=0.3 + i*0.4)
            expert_texts.append(text)
        
        self.play(Write(expert_title))
        self.play(*[Write(text) for text in expert_texts], run_time=2)