# File: cbs/a_star.py
# (Cleaned Version)

import logging
# Assume State and Location are defined in cbs.py or imported
# If running standalone, you might need: from cbs import State, Location

logger = logging.getLogger(__name__)

class AStar:
    def __init__(self, env):
        """ Initializes A* search.
        Args:
            env: The CBS Environment object which provides agent info, heuristics,
                 goal checks, and neighbor generation considering constraints.
        """
        self.env = env
        # Directly use methods from the passed environment instance
        self.agent_dict = env.agent_dict
        self.admissible_heuristic = env.admissible_heuristic
        self.is_at_goal = env.is_at_goal
        self.get_neighbors = env.get_neighbors

    def reconstruct_path(self, came_from: dict, current) -> list:
        """ Reconstructs the path from the came_from dictionary. """
        total_path = [current]
        while current in came_from: # Use direct check for key existence
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1] # Reverse to get start -> goal order

    def search(self, agent_name: str) -> list | None:
        """
        Performs low-level A* search for a single agent respecting constraints.

        Args:
            agent_name (str): The name of the agent to plan for.

        Returns:
            list: A list of State objects representing the path, or None if no path found.
        """
        try:
            initial_state = self.agent_dict[agent_name]["start"]
        except KeyError:
            logger.error(f"A* Search: Agent '{agent_name}' not found in environment agent_dict.")
            return None

        step_cost = 1 # Assuming uniform cost grid

        closed_set = set()
        # Use a set for efficient membership checking
        open_set = {initial_state}
        # Dictionary to store the predecessor state for path reconstruction
        came_from = {}
        # Dictionary storing the cost from start to a state g(n)
        g_score = {initial_state: 0}
        # Dictionary storing the estimated total cost f(n) = g(n) + h(n)
        f_score = {initial_state: self.admissible_heuristic(initial_state, agent_name)}

        while open_set:
            # Find the state in open_set with the lowest f_score
            # Using a simple min for clarity; for performance, a priority queue (heapq) is better
            current = min(open_set, key=lambda state: f_score.get(state, float("inf")))

            # Check if goal is reached
            if self.is_at_goal(current, agent_name):
                # Found the goal, reconstruct and return the path
                return self.reconstruct_path(came_from, current)

            # Move current state from open to closed set
            open_set.remove(current)
            closed_set.add(current)

            # Explore neighbors
            # get_neighbors should handle constraint checks internally via env.state_valid / env.transition_valid
            neighbor_list = self.get_neighbors(current)

            for neighbor in neighbor_list:
                # Skip neighbors already evaluated
                if neighbor in closed_set:
                    continue

                # Calculate tentative g_score for the neighbor
                tentative_g_score = g_score.get(current, float("inf")) + step_cost

                # If neighbor is not in open_set, add it
                if neighbor not in open_set:
                    open_set.add(neighbor)
                # If this path to neighbor is worse than previously found, skip
                elif tentative_g_score >= g_score.get(neighbor, float("inf")):
                    continue

                # This path is the best so far. Record it.
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + self.admissible_heuristic(neighbor, agent_name)

        # If open_set is empty and goal was not reached, no path exists
        logger.debug(f"A* Search: No path found for agent '{agent_name}'.")
        return None