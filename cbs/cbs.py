# File: cbs/cbs.py
# (Cleaned version - Hashing fixes kept, Immutability enforcement removed)

import sys
import argparse
import yaml
from math import fabs
from itertools import combinations
from copy import deepcopy
import heapq # Import heapq for priority queue
import logging

# Use relative import if running cbs code directly within the package
try:
    from .a_star import AStar
except ImportError:
    # Fallback for direct execution or different structure
    from a_star import AStar

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # Adjust level as needed

class Location:
    """ Represents a 2D location (x, y) on the grid. """
    # __slots__ removed for mutability

    def __init__(self, x=-1, y=-1):
        try:
             self.x = int(x)
             self.y = int(y)
        except (ValueError, TypeError):
             logger.warning(f"Non-integer coordinates ({x}, {y}) passed to Location. Defaulting to (-1, -1).")
             self.x = -1
             self.y = -1

    def __eq__(self, other):
        if isinstance(other, Location):
            return self.x == other.x and self.y == other.y
        return NotImplemented

    def __hash__(self):
        return hash((self.x, self.y)) # Hash tuple of coordinates

    def __str__(self):
        return f"({self.x},{self.y})"

    def __repr__(self):
        return f"Location(x={self.x}, y={self.y})"

class State:
    """ Represents a state (time, location). """
    # __slots__ removed for mutability

    def __init__(self, time: int, location: Location | list | tuple):
        try:
            self.time = int(time)
        except (ValueError, TypeError):
            raise TypeError(f"State time must be an integer, got {time}")

        if isinstance(location, Location):
            self.location = location
        elif isinstance(location, (list, tuple)) and len(location) == 2:
             try:
                 self.location = Location(location[0], location[1]) # Assume [x,y]
             except (TypeError, ValueError) as e:
                 raise TypeError(f"State location list/tuple must contain convertible values, got {location}: {e}")
        else:
            raise TypeError(f"State location must be a Location object or [x,y] list/tuple, got {type(location)}")

    def __eq__(self, other):
        if isinstance(other, State):
            return self.time == other.time and self.location == other.location
        return NotImplemented

    def __hash__(self):
        # Hash tuple of time and the location object (which is hashable)
        return hash((self.time, self.location))

    def is_equal_except_time(self, state) -> bool:
        if isinstance(state, State):
            return self.location == state.location
        return False

    def __str__(self):
        return f"({self.time}, {self.location})"

    def __repr__(self):
        return f"State(time={self.time}, location={repr(self.location)})"

class Conflict:
    """ Represents a conflict between two agents. """
    VERTEX = 1
    EDGE = 2

    def __init__(self):
        self.time: int = -1
        self.type: int = -1 # VERTEX or EDGE
        self.agent_1: str = ""
        self.agent_2: str = ""
        self.location_1: Location = Location() # Vertex location or first edge location
        self.location_2: Location = Location() # Second edge location (edge conflict only)

    def __str__(self):
        loc_str = str(self.location_1)
        if self.type == Conflict.EDGE:
            loc_str += f" <-> {self.location_2}"
        return (
            f"(Time={self.time}, Type={'V' if self.type == Conflict.VERTEX else 'E'}, "
            f"Agents=({self.agent_1}, {self.agent_2}), Loc={loc_str})"
        )
    def __repr__(self):
        return f"Conflict(t={self.time}, type={self.type}, a1='{self.agent_1}', a2='{self.agent_2}', loc1={self.location_1}, loc2={self.location_2})"

class VertexConstraint:
    """ Vertex constraint (location, time). """
    # __slots__ removed

    def __init__(self, time: int, location: Location):
        if not isinstance(location, Location):
             raise TypeError("VertexConstraint location must be a Location object.")
        try:
            self.time = int(time)
        except (ValueError, TypeError):
            raise TypeError(f"VertexConstraint time must be an integer, got {time}")
        self.location = location

    def __eq__(self, other):
        if isinstance(other, VertexConstraint):
            return self.time == other.time and self.location == other.location
        return NotImplemented

    def __hash__(self):
        return hash((self.time, self.location))

    def __str__(self):
        return f"(VC: t={self.time}, loc={self.location})"

    def __repr__(self):
        return f"VertexConstraint(time={self.time}, location={repr(self.location)})"

class EdgeConstraint:
    """ Edge constraint (loc1 -> loc2 at time t). """
    # __slots__ removed

    def __init__(self, time: int, location_1: Location, location_2: Location):
        if not isinstance(location_1, Location): raise TypeError("EdgeConstraint location_1 must be a Location object.")
        if not isinstance(location_2, Location): raise TypeError("EdgeConstraint location_2 must be a Location object.")
        try:
            self.time = int(time) # Time of *leaving* location_1
        except (ValueError, TypeError):
            raise TypeError(f"EdgeConstraint time must be an integer, got {time}")
        self.location_1 = location_1
        self.location_2 = location_2

    def __eq__(self, other):
        if isinstance(other, EdgeConstraint):
            # Order matters for directed edge constraint
            return (self.time == other.time and
                    self.location_1 == other.location_1 and
                    self.location_2 == other.location_2)
        return NotImplemented

    def __hash__(self):
        return hash((self.time, self.location_1, self.location_2))

    def __str__(self):
        return f"(EC: t={self.time}, {self.location_1} -> {self.location_2})"

    def __repr__(self):
        return f"EdgeConstraint(time={self.time}, loc1={repr(self.location_1)}, loc2={repr(self.location_2)})"

class Constraints:
    """ Holds a set of vertex and edge constraints. """
    def __init__(self):
        self.vertex_constraints: set[VertexConstraint] = set()
        self.edge_constraints: set[EdgeConstraint] = set()

    def add_constraint(self, other):
        """ Adds constraints from another Constraints object. Returns self for chaining. """
        if isinstance(other, Constraints):
            self.vertex_constraints.update(other.vertex_constraints)
            self.edge_constraints.update(other.edge_constraints)
        return self

    def __str__(self):
        # Sort constraints for deterministic output
        vc_list = sorted(list(self.vertex_constraints), key=lambda x: (x.time, x.location.x, x.location.y))
        ec_list = sorted(list(self.edge_constraints), key=lambda x: (x.time, x.location_1.x, x.location_1.y, x.location_2.x, x.location_2.y))
        vc_str = ", ".join(map(str, vc_list))
        ec_str = ", ".join(map(str, ec_list))
        return f"VC: {{{vc_str}}}; EC: {{{ec_str}}}"

    def __eq__(self, other):
        if isinstance(other, Constraints):
            return self.vertex_constraints == other.vertex_constraints and \
                   self.edge_constraints == other.edge_constraints
        return NotImplemented

    def __hash__(self):
        # Hash based on frozensets of the constraints for use in sets/dicts
        return hash((frozenset(self.vertex_constraints), frozenset(self.edge_constraints)))

class Environment:
    """ Represents the grid environment configuration for the CBS solver. """
    def __init__(self, dimension: list[int], agents: list[dict], obstacles: list[list[int]]):
        if not isinstance(dimension, (list, tuple)) or len(dimension) != 2:
            raise ValueError("Dimension must be [width, height]")
        self.dimension = dimension
        self.width, self.height = map(int, dimension)

        self.obstacles: set[Location] = set()
        if obstacles:
             for obs in obstacles:
                  try:
                       if isinstance(obs, (list, tuple)) and len(obs) == 2:
                            self.obstacles.add(Location(obs[0], obs[1]))
                       else:
                            logger.warning(f"Invalid obstacle format skipped: {obs}")
                  except (ValueError, TypeError) as e:
                       logger.warning(f"Invalid obstacle value skipped: {obs} ({e})")
        logger.info(f"CBS Environment: Dimensions={self.width}x{self.height}, Obstacles={len(self.obstacles)}")

        self.agents = agents # Store original list if needed
        self.agent_dict: dict[str, dict] = {}
        self.make_agent_dict() # Validates agents and populates agent_dict

        # These will be updated by the CBS algorithm during search
        self.constraints: Constraints = Constraints()
        self.constraint_dict: dict[str, Constraints] = {}

        self.a_star = AStar(self) # Pass self (the environment) to A*
        logger.info(f"CBS Environment: Agents processed={len(self.agent_dict)}")

    def make_agent_dict(self):
        """ Parses agent list, creates State objects, validates positions. """
        self.agent_dict = {}
        if not self.agents: return

        for agent in self.agents:
             name = agent.get("name", "unknown_agent")
             try:
                  if not isinstance(agent, dict) or not all(k in agent for k in ["name", "start", "goal"]):
                      raise ValueError(f"Invalid agent format: {agent}")

                  start_loc = Location(agent["start"][0], agent["start"][1])
                  goal_loc = Location(agent["goal"][0], agent["goal"][1])

                  if not (0 <= start_loc.x < self.width and 0 <= start_loc.y < self.height):
                      raise ValueError(f"Start {start_loc} out of bounds ({self.width}x{self.height})")
                  if start_loc in self.obstacles:
                       raise ValueError(f"Start {start_loc} is on an obstacle")
                  if not (0 <= goal_loc.x < self.width and 0 <= goal_loc.y < self.height):
                      raise ValueError(f"Goal {goal_loc} out of bounds ({self.width}x{self.height})")
                  if goal_loc in self.obstacles:
                       raise ValueError(f"Goal {goal_loc} is on an obstacle")

                  # Store goal as Location for heuristic/goal check, start as State(0, Loc)
                  self.agent_dict[name] = {"start": State(0, start_loc), "goal": goal_loc} # Store goal Location

             except (TypeError, IndexError, ValueError, KeyError) as e:
                  logger.error(f"Error processing agent '{name}': {e}. Skipping agent.")
                  if name in self.agent_dict: del self.agent_dict[name] # Clean up partial entry

    def get_neighbors(self, state: State) -> list[State]:
        """ Returns valid neighbor states (including wait) considering constraints. """
        if not isinstance(state, State): return []
        neighbors = []
        # Moves: Wait, Right, Left, Down, Up (relative to CBS x,y coords)
        for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
            try:
                 next_loc = Location(state.location.x + dx, state.location.y + dy)
                 next_state = State(state.time + 1, next_loc)
                 # Check validity (bounds, obstacles, vertex constraints)
                 # and transition validity (edge constraints)
                 if self.state_valid(next_state) and self.transition_valid(state, next_state):
                      neighbors.append(next_state)
            except TypeError: continue # If Location creation fails
        return neighbors

    def state_valid(self, state: State) -> bool:
        """ Checks if a state is valid (bounds, obstacles, vertex constraints). """
        if not isinstance(state, State): return False
        loc = state.location
        # Check bounds
        if not (0 <= loc.x < self.width and 0 <= loc.y < self.height): return False
        # Check obstacles
        if loc in self.obstacles: return False
        # Check vertex constraints (using the constraints currently active for A*)
        if VertexConstraint(state.time, loc) in self.constraints.vertex_constraints: return False
        return True

    def transition_valid(self, state_1: State, state_2: State) -> bool:
        """ Checks if moving from state_1 to state_2 violates an edge constraint. """
        if not isinstance(state_1, State) or not isinstance(state_2, State): return False
        # Check edge constraints (using the constraints currently active for A*)
        constraint = EdgeConstraint(state_1.time, state_1.location, state_2.location)
        if constraint in self.constraints.edge_constraints: return False
        return True

    def admissible_heuristic(self, state: State, agent_name: str) -> float:
        """ Manhattan distance heuristic. """
        if not isinstance(state, State): return float('inf')
        try:
            goal_loc = self.agent_dict[agent_name]["goal"] # Goal is stored as Location
            return abs(state.location.x - goal_loc.x) + abs(state.location.y - goal_loc.y)
        except KeyError:
            logger.error(f"Agent '{agent_name}' not found in agent_dict during heuristic calculation.")
            return float('inf')

    def is_at_goal(self, state: State, agent_name: str) -> bool:
        """ Checks if a state's location matches the agent's goal location. """
        if not isinstance(state, State): return False
        try:
            # Compare current state's location with the stored goal Location
            return state.location == self.agent_dict[agent_name]["goal"]
        except KeyError:
             logger.error(f"Agent '{agent_name}' not found in agent_dict during goal check.")
             return False

    def get_first_conflict(self, solution: dict[str, list[State]]) -> Conflict | None:
        """ Finds the first vertex or edge conflict in a solution dictionary. """
        if not solution: return None
        max_t = 0
        for path in solution.values():
            if path: max_t = max(max_t, path[-1].time) # Time of the last state

        for t in range(max_t + 1):
            # Check for vertex conflicts at time t
            loc_at_t: dict[Location, list[str]] = {}
            for agent_name, path in solution.items():
                state_t = self.get_state(agent_name, path, t)
                if state_t is None: continue # Should not happen with valid paths
                loc = state_t.location
                if loc not in loc_at_t: loc_at_t[loc] = []
                loc_at_t[loc].append(agent_name)

            for loc, agents_at_loc in loc_at_t.items():
                if len(agents_at_loc) > 1:
                    conflict = Conflict()
                    conflict.time = t
                    conflict.type = Conflict.VERTEX
                    conflict.location_1 = loc
                    conflict.agent_1 = agents_at_loc[0]
                    conflict.agent_2 = agents_at_loc[1]
                    return conflict

            # Check for edge conflicts (swapping) between time t and t+1
            if t < max_t:
                for agent_1, agent_2 in combinations(solution.keys(), 2):
                    path1 = solution.get(agent_1)
                    path2 = solution.get(agent_2)
                    if not path1 or not path2: continue # Skip if agent has no path

                    state1_t = self.get_state(agent_1, path1, t)
                    state1_t_plus_1 = self.get_state(agent_1, path1, t + 1)
                    state2_t = self.get_state(agent_2, path2, t)
                    state2_t_plus_1 = self.get_state(agent_2, path2, t + 1)

                    # Check for swapping condition
                    if state1_t and state1_t_plus_1 and state2_t and state2_t_plus_1 and \
                       state1_t.location == state2_t_plus_1.location and \
                       state1_t_plus_1.location == state2_t.location:
                        conflict = Conflict()
                        # Edge conflict occurs *at* the destination time step t+1
                        conflict.time = t + 1
                        conflict.type = Conflict.EDGE
                        conflict.agent_1 = agent_1
                        conflict.agent_2 = agent_2
                        conflict.location_1 = state1_t.location # Loc at time t
                        conflict.location_2 = state1_t_plus_1.location # Loc at time t+1
                        return conflict
        return None # No conflicts found

    def create_constraints_from_conflict(self, conflict: Conflict) -> dict[str, Constraints]:
        """ Creates constraint dictionaries {agent_name: Constraints} from a conflict. """
        constraints_dict: dict[str, Constraints] = {}
        if conflict is None: return constraints_dict

        agent1 = conflict.agent_1
        agent2 = conflict.agent_2

        if conflict.type == Conflict.VERTEX:
            # Add vertex constraint for both agents at the conflict time and location
            vc = VertexConstraint(conflict.time, conflict.location_1)
            c1 = Constraints(); c1.vertex_constraints.add(vc)
            c2 = Constraints(); c2.vertex_constraints.add(vc)
            constraints_dict[agent1] = c1
            constraints_dict[agent2] = c2

        elif conflict.type == Conflict.EDGE:
            # Add edge constraints for both agents for the conflicting transition
            # Edge constraint time is the time *leaving* the first location (t for conflict at t+1)
            constraint_time = conflict.time - 1
            # Constraint for agent1: cannot move from loc1 -> loc2 at time 'constraint_time'
            ec1 = EdgeConstraint(constraint_time, conflict.location_1, conflict.location_2)
            # Constraint for agent2: cannot move from loc2 -> loc1 at time 'constraint_time'
            ec2 = EdgeConstraint(constraint_time, conflict.location_2, conflict.location_1)
            c1 = Constraints(); c1.edge_constraints.add(ec1)
            c2 = Constraints(); c2.edge_constraints.add(ec2)
            constraints_dict[agent1] = c1
            constraints_dict[agent2] = c2
        return constraints_dict

    def get_state(self, agent_name: str, path: list[State], t: int) -> State | None:
        """ Safely gets an agent's state at time t from its path.
            If t exceeds path length, assumes agent stays at its final location.
        """
        if not path:
             # Attempt to return start state if path is empty
             start_state = self.agent_dict.get(agent_name, {}).get("start")
             if start_state:
                  # Return a new State object at the requested time but start location
                  return State(t, start_state.location)
             else:
                  logger.error(f"Agent '{agent_name}' has no path and no registered start state in get_state.")
                  return None

        if t < len(path):
            return path[t]
        else:
            # Agent waits at its final position
            last_state = path[-1]
            return State(t, last_state.location)

    def compute_solution(self) -> dict[str, list[State]] | None:
        """ Computes paths for all agents using A* based on constraints in self.constraint_dict. """
        solution = {}
        for agent_name in self.agent_dict.keys():
            # Set self.constraints for the A* call for this *specific* agent
            # A* will use self.constraints when calling self.state_valid and self.transition_valid
            self.constraints = self.constraint_dict.get(agent_name, Constraints())
            local_solution = self.a_star.search(agent_name)
            if local_solution is None: # A* failed to find a path
                logger.debug(f"A* failed for agent {agent_name} with constraints: {self.constraints}")
                # Reset constraints for safety, although it will be overwritten next loop
                self.constraints = Constraints()
                return None # Infeasible under current constraints
            solution[agent_name] = local_solution

        # Reset constraints after planning for all agents in this node
        self.constraints = Constraints()
        return solution

    def compute_solution_cost(self, solution: dict[str, list[State]] | None) -> float:
        """ Computes the Sum of Costs (SoC) for a given solution dictionary. Cost is path length - 1. """
        if solution is None: return float('inf')
        total_cost = 0
        for path in solution.values():
            if path and len(path) > 1: # Path must have at least start and goal state (len >= 1)
                total_cost += len(path) - 1 # Cost is number of steps/transitions
        return total_cost

class HighLevelNode:
    """ Represents a node in the CBS Constraint Tree (CT). """
    def __init__(self):
        self.solution: dict[str, list[State]] = {}
        self.constraint_dict: dict[str, Constraints] = {}
        self.cost: float = float('inf')

    # Comparison for priority queue (min-heap)
    def __lt__(self, other):
        if not isinstance(other, HighLevelNode): return NotImplemented
        # Primary sort by cost, secondary by number of constraints (prefer fewer constraints)
        if self.cost != other.cost:
            return self.cost < other.cost
        else:
            num_constraints_self = sum(len(c.vertex_constraints) + len(c.edge_constraints) for c in self.constraint_dict.values())
            num_constraints_other = sum(len(c.vertex_constraints) + len(c.edge_constraints) for c in other.constraint_dict.values())
            return num_constraints_self < num_constraints_other # Tie-break: fewer constraints first

    def __eq__(self, other):
         # Equality check might be needed for closed list if constraints are identical
         if not isinstance(other, HighLevelNode): return NotImplemented
         return self.cost == other.cost and self.constraint_dict == other.constraint_dict

    def __hash__(self):
         # Hash based on constraints for closed list checking
         # Convert dict items to tuple of tuples for hashability
         constraint_items = tuple(sorted( (k, v) for k, v in self.constraint_dict.items() ))
         return hash((self.cost, constraint_items))

    def __repr__(self):
        return f"HLNode(cost={self.cost}, constraints={len(self.constraint_dict)} agents)"

class CBS:
    """ Conflict-Based Search High-Level Solver using A* for low-level search. """
    def __init__(self, environment: Environment, verbose: bool = True):
        if not isinstance(environment, Environment):
             raise TypeError("CBS requires an Environment object.")
        self.env = environment
        self.verbose = verbose
        self.open_list: list[HighLevelNode] = [] # Use list as min-heap managed by heapq
        # Optional: Store hashes/keys of visited constraint sets to avoid re-expanding identical nodes
        self.closed_set: set[int] = set()

    def search(self) -> dict[str, list[dict]] | dict:
        """ Performs the high-level CBS search.
            Returns the plan in the standard YAML output format if successful,
            or an empty dictionary if no solution is found.
        """
        start_node = HighLevelNode()
        # Initialize constraints for all agents as empty
        start_node.constraint_dict = {agent: Constraints() for agent in self.env.agent_dict.keys()}
        # Set the environment's constraint dictionary for the initial planning
        self.env.constraint_dict = start_node.constraint_dict
        start_node.solution = self.env.compute_solution() # Compute initial paths

        if start_node.solution is None:
            logger.error("CBS Error: Initial solution could not be computed (problem might be infeasible from start).")
            return {} # Return empty dict for failure

        start_node.cost = self.env.compute_solution_cost(start_node.solution)
        heapq.heappush(self.open_list, start_node)
        self.closed_set.add(hash(start_node)) # Add start node hash to closed set

        if self.verbose: logger.info(f"Initializing CBS search. Start node cost: {start_node.cost}")
        nodes_generated = 0
        nodes_expanded = 0

        while self.open_list:
            nodes_expanded += 1
            current_node = heapq.heappop(self.open_list)

            if self.verbose and nodes_expanded % 100 == 0:
                 logger.info(f"CBS search... Open list size: {len(self.open_list)}, Nodes expanded: {nodes_expanded}, Current Cost: {current_node.cost}")

            # Find the first conflict in the current node's solution
            first_conflict = self.env.get_first_conflict(current_node.solution)

            # If no conflict, we found a valid solution
            if first_conflict is None:
                if self.verbose: logger.info(f"CBS Success: Solution found after expanding {nodes_expanded} nodes. Final Cost: {current_node.cost}")
                # Convert the internal solution (list of States) to the output format
                return self.generate_plan_from_solution(current_node.solution)

            if self.verbose > 1: # Higher verbosity level
                 logger.debug(f"  Node {nodes_expanded}: Cost={current_node.cost}, Conflict={first_conflict}")

            # Create new constraints based on the conflict
            new_constraint_pairs = self.env.create_constraints_from_conflict(first_conflict)

            # Create successor nodes by adding one constraint at a time
            for agent_involved, added_constraint in new_constraint_pairs.items():
                nodes_generated += 1
                if self.verbose > 1: logger.debug(f"    Branching {nodes_generated} for {agent_involved} with constraint: {added_constraint}")

                # --- Create Successor Node ---
                successor_node = HighLevelNode()
                # Inherit constraints and add the new one
                successor_node.constraint_dict = deepcopy(current_node.constraint_dict)
                successor_node.constraint_dict[agent_involved].add_constraint(added_constraint)

                # Check if this constraint set has been seen before
                successor_hash = hash(successor_node) # Hash based on constraints and cost placeholder
                if successor_hash in self.closed_set:
                     if self.verbose > 1: logger.debug("      Skipping node: Constraint set already expanded.")
                     continue

                # --- Re-plan for the affected agent ---
                self.env.constraint_dict = successor_node.constraint_dict # Set constraints for A*
                # We only need to re-plan for the agent whose constraint changed
                new_path_agent = self.env.a_star.search(agent_involved)

                # If A* fails for this agent, this branch is infeasible
                if new_path_agent is None:
                    if self.verbose > 1: logger.debug(f"      Pruning branch: No path found for {agent_involved}.")
                    continue # Prune this branch

                # --- Update Successor Node Solution and Cost ---
                successor_node.solution = deepcopy(current_node.solution) # Start with parent solution
                successor_node.solution[agent_involved] = new_path_agent # Update the re-planned path
                successor_node.cost = self.env.compute_solution_cost(successor_node.solution) # Recalculate cost

                # Add the valid successor node to the open list and closed set
                heapq.heappush(self.open_list, successor_node)
                self.closed_set.add(hash(successor_node)) # Use hash including constraints and cost
                if self.verbose > 1: logger.debug(f"      Added successor node {nodes_generated}. New cost: {successor_node.cost}")

        # If open list becomes empty, no solution was found
        if self.verbose: logger.warning(f"CBS Failure: Open list empty after expanding {nodes_expanded} nodes. No solution found.")
        return {} # Return empty dict for failure


    def generate_plan_from_solution(self, solution_internal: dict[str, list[State]]) -> dict[str, list[dict]]:
        """ Converts the internal solution (list of State objects) to the output YAML format. """
        plan = {}
        if not solution_internal: return plan
        for agent, path in solution_internal.items():
            if not path:
                 plan[agent] = []
                 continue
            # Convert each State object to a dictionary {t, x, y}
            path_dict_list = [{"t": state.time, "x": state.location.x, "y": state.location.y} for state in path]
            plan[agent] = path_dict_list
        return plan


def main():
    """ Main function to run CBS from command line arguments. """
    parser = argparse.ArgumentParser(description="Run Conflict-Based Search MAPF solver.")
    parser.add_argument("param", help="Input YAML file containing map and agents definition.")
    parser.add_argument("output", help="Output YAML file to write the schedule.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    if args.verbose:
         logger.setLevel(logging.DEBUG)
         logging.basicConfig(level=logging.DEBUG)
    else:
         logger.setLevel(logging.INFO)
         logging.basicConfig(level=logging.INFO)


    try:
        # --- Load Input ---
        logger.info(f"Loading input from: {args.param}")
        with open(args.param, "r") as param_file:
            param = yaml.safe_load(param_file)
            if param is None: raise ValueError("Input file is empty or invalid YAML.")
        # Basic validation
        if "map" not in param or not isinstance(param["map"], dict) or "dimensions" not in param["map"]:
             raise KeyError("Input YAML missing 'map' section or 'map/dimensions'.")
        if not isinstance(param["map"]["dimensions"], list) or len(param["map"]["dimensions"]) != 2:
             raise ValueError("'map/dimensions' must be a list of [width, height].")
        if "agents" not in param or not isinstance(param["agents"], list):
             raise KeyError("Input YAML missing 'agents' list.")

        dimension = param["map"]["dimensions"] # [width, height]
        obstacles_yaml = param["map"].get("obstacles", []) # List of [x,y] or tuples
        agents_yaml = param["agents"]

        # Convert obstacles to list of lists if not already
        obstacles_list = [list(obs) for obs in obstacles_yaml if isinstance(obs, (list, tuple)) and len(obs)==2]

        # --- Setup and Run CBS ---
        logger.info("Setting up CBS Environment...")
        env = Environment(dimension, agents_yaml, obstacles_list)
        logger.info("Initializing CBS Solver...")
        cbs_solver = CBS(env, verbose=args.verbose)
        logger.info("Starting CBS search...")
        solution_output_format = cbs_solver.search() # Search returns dict in output format

        # --- Process and Save Results ---
        if not solution_output_format:
            logger.warning("CBS search completed: Solution not found.")
            output_data = {"schedule": {}, "cost": -1, "status": "No Solution"}
        else:
            # Re-calculate cost from the final plan for verification (optional)
            # Cost calculation based on output format needs careful handling of path length
            calculated_cost = sum(len(path) - 1 for path in solution_output_format.values() if path and len(path) > 0)
            logger.info(f"Solution found! Calculated Cost (SoC): {calculated_cost}")
            output_data = dict()
            output_data["schedule"] = solution_output_format
            output_data["cost"] = calculated_cost # Use the calculated cost
            output_data["status"] = "Success"

        logger.info(f"Writing output to: {args.output}")
        with open(args.output, "w") as output_yaml:
            yaml.safe_dump(output_data, output_yaml, default_flow_style=None, sort_keys=False)
        logger.info("Output written successfully.")

    except FileNotFoundError:
        logger.error(f"Input file not found at {args.param}")
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"Error processing input file or setting up environment: {e}", exc_info=True)
    except yaml.YAMLError as exc:
        logger.error(f"Error parsing YAML input file {args.param}: {exc}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()