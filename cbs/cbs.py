# File: cbs/cbs.py
# (Complete Code - Hashing fixes kept, Immutability enforcement removed)

import sys
# sys.path.insert(0, "../") # Avoid modifying sys.path if possible
import argparse
import yaml
from math import fabs
from itertools import combinations
from copy import deepcopy
import heapq # Import heapq for priority queue

# Use relative import if running cbs code directly within the package
try:
    from .a_star import AStar
except ImportError:
    # Fallback for direct execution or different structure
    from a_star import AStar


class Location(object):
    """ Represents a 2D location (x, y) on the grid. """
    # __slots__ = ['x', 'y'] # REMOVED immutability enforcement

    def __init__(self, x=-1, y=-1):
        try:
             # Ensure coordinates are integers upon initialization
             self.x = int(x)
             self.y = int(y)
        except (ValueError, TypeError):
             print(f"Warning: Non-integer coordinates ({x}, {y}) passed to Location. Defaulting to -1.")
             self.x = -1
             self.y = -1

    def __eq__(self, other):
        """ Checks if two Location objects represent the same coordinates. """
        if isinstance(other, Location):
            return self.x == other.x and self.y == other.y
        return NotImplemented # Indicate comparison is not supported

    def __hash__(self):
        """ Computes the hash based on the coordinates. """
        # Hash a tuple of the coordinates
        return hash((self.x, self.y))

    def __str__(self):
        """ String representation of the location. """
        return str((self.x, self.y))

    def __repr__(self):
        """ Detailed representation for debugging. """
        return f"Location(x={self.x}, y={self.y})"

    # REMOVED custom __setattr__


class State(object):
    """ Represents a state (time, location). """
    # __slots__ = ['time', 'location'] # REMOVED immutability enforcement

    def __init__(self, time, location):
        """
        Initializes a State object.
        Args:
            time (int): The time step.
            location (Location or convertible): The location object or [x,y] list/tuple.
        """
        try:
            self.time = int(time) # Ensure time is an integer
        except (ValueError, TypeError):
            raise TypeError(f"State time must be an integer, got {time}")

        if not isinstance(location, Location):
             # Attempt conversion if a list/tuple is provided
             try:
                 loc_obj = Location(location[0], location[1]) # Assume input might be [x,y]
                 # print(f"Warning: State received non-Location object for location ({location}), converted to {loc_obj}.") # Optional warning
                 self.location = loc_obj
             except (TypeError, IndexError, ValueError):
                 raise TypeError(f"State location must be a Location object or convertible like [x,y], got {location}")
        else:
            self.location = location


    def __eq__(self, other):
        """ Checks if two State objects are identical (same time and location). """
        if isinstance(other, State):
            # Relies on Location.__eq__ for location comparison
            return self.time == other.time and self.location == other.location
        return NotImplemented

    def __hash__(self):
        """ Computes the hash based on time and location. """
        # Hash a tuple of the time and the hashable location object.
        return hash((self.time, self.location))

    def is_equal_except_time(self, state):
        """ Checks if the location of this state matches another state's location. """
        if isinstance(state, State):
            return self.location == state.location
        return False

    def __str__(self):
        """ String representation of the state. """
        return str((self.time, str(self.location)))

    def __repr__(self):
        """ Detailed representation for debugging. """
        return f"State(time={self.time}, location={repr(self.location)})"

    # REMOVED custom __setattr__


class Conflict(object):
    """ Represents a conflict between two agents. """
    VERTEX = 1
    EDGE = 2

    def __init__(self):
        self.time = -1
        self.type = -1 # VERTEX or EDGE
        self.agent_1 = "" # Name of the first agent
        self.agent_2 = "" # Name of the second agent
        self.location_1 = Location() # Location involved (vertex) or first location (edge)
        self.location_2 = Location() # Second location involved (edge conflict only)

    def __str__(self):
        """ String representation of the conflict. """
        loc_str = str(self.location_1)
        if self.type == Conflict.EDGE:
            loc_str += f" <-> {str(self.location_2)}" # Indicate swap for edge conflict
        return (
            f"(Time: {self.time}, Type: {'Vertex' if self.type == Conflict.VERTEX else 'Edge'}, "
            f"Agents: ({self.agent_1}, {self.agent_2}), Loc: {loc_str})"
        )


class VertexConstraint(object):
    """ Vertex constraint (location, time). """
    # __slots__ = ['time', 'location'] # REMOVED immutability enforcement

    def __init__(self, time, location):
        if not isinstance(location, Location):
             raise TypeError("VertexConstraint location must be a Location object.")
        try:
            self.time = int(time)
        except (ValueError, TypeError):
            raise TypeError(f"VertexConstraint time must be an integer, got {time}")
        self.location = location


    def __eq__(self, other):
        """ Checks equality based on time and location. """
        if isinstance(other, VertexConstraint):
            return self.time == other.time and self.location == other.location
        return NotImplemented

    def __hash__(self):
        """ Computes hash based on time and location. """
        return hash((self.time, self.location))

    def __str__(self):
        """ String representation of the vertex constraint. """
        return f"(VC: t={self.time}, loc={self.location})"

    # REMOVED custom __setattr__


class EdgeConstraint(object):
    """ Edge constraint (loc1 -> loc2 at time t). """
    # __slots__ = ['time', 'location_1', 'location_2'] # REMOVED immutability enforcement

    def __init__(self, time, location_1, location_2):
        """
        Args:
            time (int): The time step *when leaving* location_1.
            location_1 (Location): The starting location of the edge.
            location_2 (Location): The ending location of the edge.
        """
        if not isinstance(location_1, Location):
             raise TypeError("EdgeConstraint location_1 must be a Location object.")
        if not isinstance(location_2, Location):
             raise TypeError("EdgeConstraint location_2 must be a Location object.")
        try:
            self.time = int(time) # Time of *leaving* location_1
        except (ValueError, TypeError):
            raise TypeError(f"EdgeConstraint time must be an integer, got {time}")
        self.location_1 = location_1
        self.location_2 = location_2

    def __eq__(self, other):
        """ Checks equality based on time and the directed edge. """
        if isinstance(other, EdgeConstraint):
            # Order matters
            return (self.time == other.time and
                    self.location_1 == other.location_1 and
                    self.location_2 == other.location_2)
        return NotImplemented

    def __hash__(self):
        """ Computes hash based on time and the directed edge. """
        # Hash includes order
        return hash((self.time, self.location_1, self.location_2))

    def __str__(self):
        """ String representation of the edge constraint. """
        return f"(EC: t={self.time}, {self.location_1} -> {self.location_2})"

    # REMOVED custom __setattr__


class Constraints(object):
    """ Holds a set of vertex and edge constraints for an agent or CBS node. """
    def __init__(self):
        self.vertex_constraints = set() # Set of VertexConstraint objects
        self.edge_constraints = set()   # Set of EdgeConstraint objects

    def add_constraint(self, other):
        """ Adds constraints from another Constraints object. """
        if not isinstance(other, Constraints): return
        # Use set union to add constraints
        self.vertex_constraints.update(other.vertex_constraints)
        self.edge_constraints.update(other.edge_constraints)

    def __str__(self):
        """ String representation of all constraints, sorted for consistency. """
        # Sort constraints before joining for deterministic string representation
        vc_list = sorted(list(self.vertex_constraints), key=lambda x: (x.time, x.location.x, x.location.y))
        ec_list = sorted(list(self.edge_constraints), key=lambda x: (x.time, x.location_1.x, x.location_1.y, x.location_2.x, x.location_2.y))
        vc_str = ", ".join(map(str, vc_list))
        ec_str = ", ".join(map(str, ec_list))
        return f"VC: {{{vc_str}}}; EC: {{{ec_str}}}"

    def __eq__(self, other):
        """ Checks if two Constraints objects hold the exact same constraints. """
        if isinstance(other, Constraints):
            return self.vertex_constraints == other.vertex_constraints and \
                   self.edge_constraints == other.edge_constraints
        return NotImplemented

    def __hash__(self):
        """ Computes hash based on frozensets of the constraint sets. """
        return hash((frozenset(self.vertex_constraints), frozenset(self.edge_constraints)))


class Environment(object):
    """ Represents the grid environment configuration for the CBS solver. """
    def __init__(self, dimension, agents, obstacles):
        """
        Initializes the CBS environment.
        Args:
            dimension (list/tuple): Map dimensions [width, height].
            agents (list): List of agent dictionaries [{'name': str, 'start': [x,y], 'goal': [x,y]}].
            obstacles (list): List of obstacle coordinates [[x,y], ...].
        """
        if not isinstance(dimension, (list, tuple)) or len(dimension) != 2:
            raise ValueError("Dimension must be [width, height]")
        self.dimension = dimension
        self.width, self.height = dimension

        self.obstacles = set()
        if obstacles:
             for obs in obstacles:
                  if isinstance(obs, (list, tuple)) and len(obs) == 2:
                       try: self.obstacles.add(Location(obs[0], obs[1]))
                       except (ValueError, TypeError): print(f"Warning: Invalid obstacle format skipped: {obs}")
                  else:
                       print(f"Warning: Invalid obstacle format skipped: {obs}")
        print(f"CBS Environment: Dimensions={self.width}x{self.height}, Obstacles={len(self.obstacles)}")

        self.agents = agents
        self.agent_dict = {}

        self.make_agent_dict()

        self.constraints = Constraints()
        self.constraint_dict = {}

        self.a_star = AStar(self)
        print(f"CBS Environment: Agents processed={len(self.agent_dict)}")


    def make_agent_dict(self):
        """ Parses agent list, creates State objects, validates positions. """
        self.agent_dict = {}
        if not self.agents: return

        processed_agents = 0
        for agent in self.agents:
             if not isinstance(agent, dict) or not all(k in agent for k in ["name", "start", "goal"]):
                  print(f"Warning: Invalid agent format skipped: {agent}")
                  continue
             name = agent["name"]
             try:
                  start_loc = Location(agent["start"][0], agent["start"][1])
                  goal_loc = Location(agent["goal"][0], agent["goal"][1])

                  if not (0 <= start_loc.x < self.width and 0 <= start_loc.y < self.height):
                      raise ValueError(f"Start {start_loc} out of bounds ({self.width}x{self.height}).")
                  if start_loc in self.obstacles:
                       raise ValueError(f"Start {start_loc} is on an obstacle.")
                  if not (0 <= goal_loc.x < self.width and 0 <= goal_loc.y < self.height):
                      raise ValueError(f"Goal {goal_loc} out of bounds ({self.width}x{self.height}).")
                  if goal_loc in self.obstacles:
                       raise ValueError(f"Goal {goal_loc} is on an obstacle.")

                  start_state = State(0, start_loc)
                  goal_state_for_check = State(0, goal_loc)

                  self.agent_dict[name] = {"start": start_state, "goal": goal_state_for_check}
                  processed_agents += 1
             except (TypeError, IndexError, ValueError) as e:
                  print(f"Error processing agent '{agent.get('name', 'N/A')}': {e}. Skipping agent.")
                  if name in self.agent_dict: del self.agent_dict[name]


    def get_neighbors(self, state):
        """ Returns valid neighbor states (including wait) considering constraints. """
        if not isinstance(state, State): return []
        neighbors = []
        for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
            try:
                 next_loc = Location(state.location.x + dx, state.location.y + dy)
                 next_state = State(state.time + 1, next_loc)
            except TypeError: continue

            # self.constraints should hold the current agent's constraints for A*
            if self.state_valid(next_state) and self.transition_valid(state, next_state):
                neighbors.append(next_state)
        return neighbors

    def state_valid(self, state):
        """ Checks if a state is valid (bounds, obstacles, vertex constraints). """
        if not isinstance(state, State): return False
        loc = state.location
        if not (0 <= loc.x < self.width and 0 <= loc.y < self.height): return False
        if loc in self.obstacles: return False
        # Check against self.constraints (set by compute_solution for current A* agent)
        if VertexConstraint(state.time, loc) in self.constraints.vertex_constraints: return False
        return True

    def transition_valid(self, state_1, state_2):
        """ Checks if moving from state_1 to state_2 violates an edge constraint. """
        if not isinstance(state_1, State) or not isinstance(state_2, State): return False
        constraint = EdgeConstraint(state_1.time, state_1.location, state_2.location)
        # Check against self.constraints (set by compute_solution for current A* agent)
        if constraint in self.constraints.edge_constraints: return False
        return True

    def admissible_heuristic(self, state, agent_name):
        """ Manhattan distance heuristic. """
        # ... (remains the same) ...
        if not isinstance(state, State): return float('inf')
        try:
            goal_loc = self.agent_dict[agent_name]["goal"].location
            return abs(state.location.x - goal_loc.x) + abs(state.location.y - goal_loc.y)
        except KeyError:
            print(f"Warning: Agent '{agent_name}' not found in agent_dict during heuristic calculation.")
            return float('inf')

    def is_at_goal(self, state, agent_name):
        """ Checks if a state's location matches the agent's goal location. """
        # ... (remains the same) ...
        if not isinstance(state, State): return False
        try:
            return state.location == self.agent_dict[agent_name]["goal"].location
        except KeyError:
             print(f"Warning: Agent '{agent_name}' not found in agent_dict during goal check.")
             return False

    def get_first_conflict(self, solution):
        """ Finds the first vertex or edge conflict in a solution dictionary. """
        # ... (remains the same) ...
        if not solution: return None
        max_t = 0
        for path in solution.values():
            if path: max_t = max(max_t, path[-1].time)

        for t in range(max_t + 1):
            loc_at_t = {}
            for agent_name, path in solution.items():
                state_t = self.get_state(agent_name, path, t)
                if state_t is None: continue
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

            if t < max_t:
                for agent_1, agent_2 in combinations(solution.keys(), 2):
                    path1 = solution.get(agent_1)
                    path2 = solution.get(agent_2)
                    if not path1 or not path2: continue

                    state1_t = self.get_state(agent_1, path1, t)
                    state1_t_plus_1 = self.get_state(agent_1, path1, t + 1)
                    state2_t = self.get_state(agent_2, path2, t)
                    state2_t_plus_1 = self.get_state(agent_2, path2, t + 1)

                    if state1_t and state1_t_plus_1 and state2_t and state2_t_plus_1 and \
                       state1_t.location == state2_t_plus_1.location and \
                       state1_t_plus_1.location == state2_t.location:
                        conflict = Conflict()
                        conflict.time = t + 1
                        conflict.type = Conflict.EDGE
                        conflict.agent_1 = agent_1
                        conflict.agent_2 = agent_2
                        conflict.location_1 = state1_t.location
                        conflict.location_2 = state1_t_plus_1.location
                        return conflict
        return None


    def create_constraints_from_conflict(self, conflict):
        """ Creates constraint dictionaries {agent_name: Constraints} from a conflict. """
        # ... (remains the same) ...
        constraints_dict = {}
        if conflict is None: return constraints_dict

        agent1 = conflict.agent_1
        agent2 = conflict.agent_2

        if conflict.type == Conflict.VERTEX:
            vc = VertexConstraint(conflict.time, conflict.location_1)
            c1 = Constraints(); c1.vertex_constraints.add(vc)
            c2 = Constraints(); c2.vertex_constraints.add(vc)
            constraints_dict[agent1] = c1
            constraints_dict[agent2] = c2

        elif conflict.type == Conflict.EDGE:
            constraint_time = conflict.time - 1
            ec1 = EdgeConstraint(constraint_time, conflict.location_1, conflict.location_2)
            ec2 = EdgeConstraint(constraint_time, conflict.location_2, conflict.location_1)
            c1 = Constraints(); c1.edge_constraints.add(ec1)
            c2 = Constraints(); c2.edge_constraints.add(ec2)
            constraints_dict[agent1] = c1
            constraints_dict[agent2] = c2
        return constraints_dict


    def get_state(self, agent_name, path, t):
        """ Safely gets an agent's state at time t, holding goal state if t exceeds path length. """
        # ... (remains the same) ...
        if not path:
             start_state = self.agent_dict.get(agent_name, {}).get("start")
             if start_state:
                  return State(t, start_state.location)
             else:
                  print(f"Error in get_state: Agent '{agent_name}' has no path and no registered start state.")
                  return None

        if t < len(path):
            return path[t]
        else:
            last_state = path[-1]
            return State(t, last_state.location)


    def compute_solution(self):
        """ Computes paths for all agents using A* based on constraints in self.constraint_dict. """
        # ... (remains the same) ...
        solution = {}
        for agent_name in self.agent_dict.keys():
            # Set self.constraints for the A* call for this agent
            self.constraints = self.constraint_dict.get(agent_name, Constraints())
            local_solution = self.a_star.search(agent_name)
            if not local_solution:
                return None # Infeasible
            solution[agent_name] = local_solution
        # Reset self.constraints after planning for all agents? Maybe not necessary.
        # self.constraints = Constraints()
        return solution


    def compute_solution_cost(self, solution):
        """ Computes the Sum of Costs (SoC) for a given solution dictionary. """
        # ... (remains the same) ...
        if not solution: return float('inf')
        total_cost = 0
        for path in solution.values():
            if path and len(path) > 1:
                total_cost += len(path) - 1
        return total_cost


class HighLevelNode(object):
    """ Represents a node in the CBS Constraint Tree (CT). """
    # ... (remains the same as previous version, including __lt__) ...
    def __init__(self):
        self.solution = {}
        self.constraint_dict = {}
        self.cost = float('inf')

    def __lt__(self, other):
        if not isinstance(other, HighLevelNode): return NotImplemented
        num_constraints_self = sum(len(c.vertex_constraints) + len(c.edge_constraints) for c in self.constraint_dict.values())
        num_constraints_other = sum(len(c.vertex_constraints) + len(c.edge_constraints) for c in other.constraint_dict.values())
        if self.cost != other.cost:
            return self.cost < other.cost
        else:
            return num_constraints_self < num_constraints_other

    def __eq__(self, other):
         if not isinstance(other, HighLevelNode): return NotImplemented
         # Equality check might involve constraints for closed list
         return self.cost == other.cost and self.constraint_dict == other.constraint_dict

    def __hash__(self):
         # Hashing based on constraints
         return hash((self.cost, frozenset(self.constraint_dict.items())))


class CBS(object):
    """ Conflict-Based Search High-Level Solver using A* for low-level search. """
    # ... (init remains the same) ...
    def __init__(self, environment, verbose=True):
        if not isinstance(environment, Environment):
             raise TypeError("CBS requires an Environment object.")
        self.env = environment
        self.verbose = verbose
        self.open_list = [] # Use list as min-heap (priority queue) managed by heapq
        # self.closed_list = set() # Optional: Store hashes/keys of visited constraint sets

    def search(self):
        """ Performs the high-level CBS search. """
        # ... (start node initialization remains the same) ...
        start_node = HighLevelNode()
        start_node.constraint_dict = {agent: Constraints() for agent in self.env.agent_dict.keys()}
        self.env.constraint_dict = start_node.constraint_dict
        start_node.solution = self.env.compute_solution()

        if start_node.solution is None:
            if self.verbose: print("CBS Error: Initial solution could not be computed.")
            return {}

        start_node.cost = self.env.compute_solution_cost(start_node.solution)
        heapq.heappush(self.open_list, start_node)

        if self.verbose: print(f"Initializing CBS search. Start node cost: {start_node.cost}")
        nodes_generated = 0
        nodes_expanded = 0

        while self.open_list:
            nodes_expanded += 1
            # ... (verbose printing) ...
            if self.verbose and nodes_expanded % 100 == 0:
                print(f"CBS search... Open list size: {len(self.open_list)}, Nodes expanded: {nodes_expanded}")

            current_node = heapq.heappop(self.open_list)

            first_conflict = self.env.get_first_conflict(current_node.solution)

            if first_conflict is None:
                if self.verbose: print(f"CBS Success: Solution found after expanding {nodes_expanded} nodes. Final Cost: {current_node.cost}")
                return current_node.solution # Return internal solution format

            if self.verbose > 1:
                 print(f"  Node {nodes_expanded}: Cost={current_node.cost}, Conflict={first_conflict}")

            new_constraint_pairs = self.env.create_constraints_from_conflict(first_conflict)

            for agent_involved, added_constraint in new_constraint_pairs.items():
                nodes_generated += 1
                if self.verbose > 1: print(f"    Branching {nodes_generated} for {agent_involved} with constraint: {added_constraint}")

                successor_constraints_dict = deepcopy(current_node.constraint_dict)
                successor_constraints_dict[agent_involved].add_constraint(added_constraint)

                # --- Re-plan ---
                self.env.constraint_dict = successor_constraints_dict
                # Call A* via the environment object
                new_path_agent = self.env.a_star.search(agent_involved) # CORRECTED CALL

                if new_path_agent is None:
                    if self.verbose > 1: print(f"      Pruning branch: No path found for {agent_involved}.")
                    continue

                successor_node = HighLevelNode()
                successor_node.constraint_dict = successor_constraints_dict
                successor_node.solution = deepcopy(current_node.solution)
                successor_node.solution[agent_involved] = new_path_agent
                successor_node.cost = self.env.compute_solution_cost(successor_node.solution)

                heapq.heappush(self.open_list, successor_node)
                if self.verbose > 1: print(f"      Added successor node {nodes_generated}. New cost: {successor_node.cost}")
                # --- End Re-plan ---

        if self.verbose: print(f"CBS Failure: Open list empty after expanding {nodes_expanded} nodes. No solution found.")
        return {}

    # ... (generate_plan_from_solution remains the same) ...
    def generate_plan_from_solution(self, solution_internal):
        """ Converts the internal solution (list of State objects) to the output YAML format. """
        plan = {}
        if not solution_internal: return plan
        for agent, path in solution_internal.items():
            if not path:
                 plan[agent] = []
                 continue
            path_dict_list = [{"t": state.time, "x": state.location.x, "y": state.location.y} for state in path]
            plan[agent] = path_dict_list
        return plan


# ... (main function remains the same) ...
def main():
    parser = argparse.ArgumentParser(description="Run Conflict-Based Search MAPF solver.")
    parser.add_argument("param", help="Input YAML file containing map and agents definition.")
    parser.add_argument("output", help="Output YAML file to write the schedule.")
    args = parser.parse_args()
    try:
        with open(args.param, "r") as param_file:
            param = yaml.safe_load(param_file)
            if param is None: raise ValueError("Input file is empty or invalid YAML.")
        if "map" not in param or "dimensions" not in param["map"]: raise KeyError("Missing 'map/dimensions' in input.")
        if "agents" not in param: raise KeyError("Missing 'agents' list in input.")
        dimension = param["map"]["dimensions"]
        obstacles_yaml = param["map"].get("obstacles", [])
        agents_yaml = param["agents"]
        obstacles_list = [list(obs) for obs in obstacles_yaml] if obstacles_yaml else []
        env = Environment(dimension, agents_yaml, obstacles_list)
        cbs_solver = CBS(env, verbose=True)
        solution_internal = cbs_solver.search()
        if not solution_internal:
            print(" CBS search completed: Solution not found.")
            with open(args.output, "w") as output_yaml:
                 yaml.safe_dump({"schedule": {}, "cost": -1, "status": "No Solution"}, output_yaml)
            return
        solution_output_format = cbs_solver.generate_plan_from_solution(solution_internal)
        final_cost = env.compute_solution_cost(solution_internal)
        output_data = dict()
        output_data["schedule"] = solution_output_format
        output_data["cost"] = final_cost
        output_data["status"] = "Success"
        with open(args.output, "w") as output_yaml:
            yaml.safe_dump(output_data, output_yaml, default_flow_style=None, sort_keys=False)
        print(f"\nSolution found (Cost: {final_cost}) and written to {args.output}")
    except FileNotFoundError: print(f"Error: Input file not found at {args.param}")
    except (KeyError, ValueError, TypeError) as e: print(f"Error processing input file or setting up environment: {e}")
    except yaml.YAMLError as exc: print(f"Error parsing YAML input file {args.param}: {exc}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()