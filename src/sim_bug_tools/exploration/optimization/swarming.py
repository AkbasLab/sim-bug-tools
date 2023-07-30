"""
Swarming Optimization Algorithms
"""

import numpy as np
import time
import random
from numpy import ndarray
from itertools import cycle
from sim_bug_tools.structs import Point, Domain, Grid
from sim_bug_tools.simulation.simulation_core import Scorable, Graded
from sim_bug_tools.exploration.brute_force import brute_force_grid_search, true_envelope_finding_alg, true_boundary_algorithm
import matplotlib.pyplot as plt
from sim_bug_tools.graphics import Grapher

class SwarmAgent():
    """
    - ParticleSwarmAgent
        Creates a single agent for partical swarm optimization
        - Inputs:
            - `intial_point: ndarray` - The initial position for the agent
            - `swarm_constants: dict` - Dictionary containing the swarm constant values:
                - `agentC: float` - aka c1, The constant value to multiply with the distance to personal best
                        Note: agentC > globalC results in high exploration, low exploitation.
                - `globalC: float` - aka c2, The constant value to multiply with the distance to gloabal best
                        Note: globalC > agentC results in low exploration, high exploitation.
                - `w: float` - The interia weight to multiply the velocity by
    """
    def __init__(self, scorable: Scorable, space_shape: ndarray, swarm_constants: dict):
        self.scorable = scorable
        self.space_shape = space_shape
        
        self.agent_stats = {
            "best_points": [],
            "num_teleports": 0,
            "num_of_steps": 0,
            "still_moving": True,
            "move_towards_score": [],
            "prev_points": [],
        }

        initial_position = [random.uniform(0,i) for i in self.space_shape]
        self.point = Point(initial_position)
        self.best_pt = self.point
        self.velocity = [0.0 for _ in self.space_shape]
        self.velocity = np.array(self.velocity)
        # "green"=moved to higher score, "red"=moved to lower score
        # "orange"=did not move, "blue"=teleported, "black"=found best score
        self.movement_color: str = "black"
        self.swarm_constants = swarm_constants
        if "teleporting" not in swarm_constants.keys():
            self.swarm_constants["teleporting"] = False
        
    def random_position(self) -> ndarray:
        """Returns an ndarray of a position between 0 and i for each position in space_shape"""
        return [random.uniform(0,i) for i in self.space_shape]

    def teleport(self):
        """ Resets attributes and point using a random position for teleporting agents """
        new_position = self.random_position()
        self.point = Point(new_position)
        self.agent_stats["best_points"].append(self.best_pt)
        self.best_pt = self.point
        self.agent_stats["still_moving"] = True
        self.agent_stats["num_teleports"] += 1
        self.velocity = np.zeros(self.space_shape)
        return
    
    def calculate_velocity(self) -> ndarray:
        print("!!! Parent calculate velocity !!!")
        pass

    def compare_score(self, pt1: Point, pt2: Point) -> bool:
        """ Takes 2 points and returns true if the second point has a higher score, false if it is the same or lower"""
        # Getting the current score and the new score
        pt1_score = self.scorable.score(pt1)
        pt2_score = self.scorable.score(pt2)
        if pt1_score < pt2_score:
            return True
        else:
            return False

    
    def update_point(self, new_point: Point):
        """ Given a new point, the score is compared to the current point. 
         if the new point is better, best point and current point is updated.
          If the current point is better and the 'teleporting' constant is True, 
           the agent will teleport using the teleport method """
        # Saving current point as a previous point
        self.agent_stats["prev_points"].append(self.point)
        # If the current score is less than the new score, best point is updated
        if self.compare_score(self.point, new_point):
            if self.compare_score(self.best_pt, new_point): self.best_pt = new_point
            self.agent_stats["move_towards_score"].append(True)
            self.movement_color = "green"
        else:
            self.agent_stats["move_towards_score"].append(False)
            self.movement_color = "red"
        # Setting point to new point
        self.point = new_point
        return

    def update_agent(self, new_v):
        """ Takes the new velocity calculated in the child agent class and updates the agent """
        # Limiting the velocity if there is a max_v set:
        if self.swarm_constants["max_v"] is not None:
            new_v[new_v > self.swarm_constants["max_v"]] = self.swarm_constants["max_v"]
        # If there are no nonzero elements in the new velocity, the agent is not moving and nothing is updated
        if np.count_nonzero(new_v) == 0:
            if self.swarm_constants["teleporting"]:
                # teleporting agent to new random location
                self.movement_color = "blue"
                self.teleport()
            else:
                self.agent_stats["still_moving"] = False
                self.agent_stats["move_towards_score"].append(False)
                self.movement_color = "orange"
            return
        
        # Updating the agents attributes based on new movement
        self.agent_stats["still_moving"] = True
        new_p = self.point.array + new_v
        self.velocity = new_v
        new_point = Point(new_p)
        self.update_point(new_point)
        self.index_in_bounds()
        self.agent_stats["num_of_steps"] += 1
        return

    # Determines if the new point is in the space shape, if not, puts it in just inside the limits
    def index_in_bounds(self):
        index = self.point.array
        for i, idx in enumerate(index):
            if idx < 0:
                index[i] = 0
            if idx >= self.space_shape[i]:
                index[i] = self.space_shape[i] - 0.1
        self.point = Point(index)
        return


class StandardPSOAgent(SwarmAgent):

    def __init__(self, scorable: Scorable, initial_point: ndarray, swarm_constants: dict):
        super().__init__(scorable, initial_point, swarm_constants)
    
    def calculate_velocity(self) -> ndarray:
            current_p = self.point.array
            # Caluclating distances:
            dist_to_agent_best = (self.best_pt.array - current_p)
            dist_to_global_best = (self.global_best_position - current_p)
            # Caluclating Inertia: 
            inertia = self.swarm_constants["w"] * self.velocity
            # Calculating the personal_best vector:
            personal_best = (self.swarm_constants["agentC"] * random.uniform(0,1)) * dist_to_agent_best
            # Calculating the global best vector:
            global_best = (self.swarm_constants["globalC"] * random.uniform(0,1) * dist_to_global_best)
            # New velocity:
            new_v = (inertia + personal_best + global_best)
            return new_v

    def update_agent(self):
        # Calculating the new velocity using the new global best position
        new_v = self.calculate_velocity()
        super().update_agent(new_v)
        return

   
class AntSwarmAgent(SwarmAgent):

    def __init__(self, scorable: Scorable, initial_point: ndarray, swarm_constants: dict):
        super().__init__(scorable, initial_point, swarm_constants)
        if "search_radius" not in self.swarm_constants.keys():
            self.swarm_constants["search_radius"] = len(initial_point) * 0.01
        if "search_scatter" not in self.swarm_constants.keys():
            self.swarm_constants["search_scatter"] = len(initial_point)
        # Attempts at a higher score in the area
        self.attempts: int = 0
        return

    def calculate_velocity(self) -> ndarray:
        curr_pos = self.point.array
        r = self.swarm_constants["search_radius"]
        best_pos = []
        best_pt: Point = None
        best_score = -1
        for i in range(self.swarm_constants["search_scatter"]):
            new_pos = [random.uniform(i-r,i+r) for i in curr_pos]
            new_pt = Point(new_pos)
            new_score = self.scorable.score(new_pt)
            if new_score > best_score:
                best_score = new_score
                best_pos = new_pos
        new_v = best_pos - curr_pos
        if best_score < self.scorable.score(self.point) and self.attempts < 10:
            self.attempts += 1
            new_v = self.calculate_velocity()
        elif best_score < self.scorable.score(self.point):
            new_v = [0 for _ in curr_pos]
            self.attempts = 0
        return new_v
    
    def update_agent(self):
        new_v = self.calculate_velocity()
        return super().update_agent(new_v)


class ParticleSwarmOptimization():
    """
    - PSO_SingleObjective - Particle Swarm Optimization finding the global maximum
        Creates num_agent number of agents for the swam starting them at random points in the given score array.
        - Inputs:
            - `scorable: Scorable` - Scorable object
            - `score_array: ndarray` - The score matrix
            - `num_agents: int` - the number of agents to add to the swarm
            - `swarm_constants: dict` - Dictionary containing constants for swarm with the following values:
                - `agentC: float` - aka c1, The constant value to multiply with the distance to personal best
                        Note: agentC > globalC results in high exploration, low exploitation.
                - `globalC: float` - aka c2, The constant value to multiply with the distance to gloabal best
                        Note: globalC > agentC results in low exploration, high exploitation.
                - `w: float` - The interia weight to multiply the velocity by
                - `max_v: float` - The max velocity an agent can move by. If no max, set to None
            - `max_iter: int` - The maximum number of iterations to run
        - Outputs:
            - `ndarray` - The best global estimation
    """
    def __init__(self, scorable: Scorable, space_shape: ndarray, num_agents: int, swarm_constants: dict, max_iter: int, agent_class=None) -> None:
        self.scorable: Scorable = scorable
        self.space_shape = space_shape
        self.num_agents: int = num_agents
        self.swarm_constants: dict = swarm_constants
        self.max_iter: int = max_iter
        self.agent_points: list[Point] = []
        self.agent_velocities: list[ndarray] = []
        self.agent_colors: list[str] = []
        self.global_best_point: Point
        if isinstance(agent_class, SwarmAgent):
            self.agent_class = agent_class
        else:
            self.agent_class = AntSwarmAgent
        self.swarm_stats = {
            "total_steps": int,
            "final_score": float,
            "final_class": bool,
            "avg_dir_to_score": bool,
            "average_steps": float,
            "iterations": int,
            "global_best_position": ndarray,
            "total_steps": int,
        }
        self.create_swarm()

    def create_swarm(self):
        index = [random.uniform(0,i) for i in self.space_shape]
        start_point = Point(index)
        self.global_best_point = start_point
        self.initialize_agents()
        return

    def initialize_agents(self):
        """ Initializing the agents of the agent_class parameter type for the swarm """
        self.agents = []
        start_points = []
        # For the input number of agents,
        for _ in range(self.num_agents):
            # Create an agent of agent_class
            a = self.agent_class(self.scorable, self.space_shape, self.swarm_constants)
            # Checking agent score vs. global best score
            self.check_score(a)
            # Saving the agent and their start point
            self.agents.append(a)
            start_points.append(a.point)
        # Adding start points of all agents to the agent points list    
        self.agent_points.append([start_points])
        return
        
    def check_score(self, agent: SwarmAgent):
        """ Given a SwarmAgent agent, sets global best point to agent point if the agent score is higher than global best score"""
        a_score = self.scorable.score(agent.point)
        g_score = self.scorable.score(self.global_best_point)
        if a_score > g_score:
            agent.movement_color = "black"
            self.global_best_point = agent.point
        return


    def single_iteration(self) -> bool:
        """ Moves all agents in swarm a single time. Saves the agents iteration points and velocities used for graphing and final data """
        iteration_points = []
        iteration_velocities = []
        iteration_colors = []
        still_moving = False
        for agent in self.agents:
            # Setting agents global best position to the global best point position
            agent.global_best_position = self.global_best_point.array
            # Updating the agent (all depends on the child agent classes to calculate velocity and new position)
            agent.update_agent()
            self.check_score(agent)
            if agent.agent_stats["still_moving"]:
                still_moving = True
            iteration_points.append(agent.point)
            iteration_velocities.append(agent.velocity)
            iteration_colors.append(agent.movement_color)
            
        self.agent_colors.append([iteration_colors])
        self.agent_points.append([iteration_points])
        self.agent_velocities.append(iteration_velocities)
        return still_moving
        
    def run_swarm(self, max_iter: int = None):
        """ Running the entire swarm for max_iter number of iterations or until all agents have stopped moving """
        if max_iter is None:
            max_iter = self.max_iter
        start_time = time.perf_counter()
        # For given max number of iterations...
        i = 0
        cont = True
        while (i < max_iter) and cont:
            cont = self.single_iteration()
            i += 1
        
        end_time = time.perf_counter()
        self.swarm_runtime = end_time - start_time
        self.swarm_stats["iterations"] = i
        # Setting the final stats and returning the global best position
        self.set_final_stats()    
        return
    
    def graph_swarm_3D(self, g: Grapher, colors=["black"]):
        """ Graphs 3D swarm with points """
        for iter_pts, iter_vels, iter_colors in zip(self.agent_points, self.agent_velocities, self.agent_colors):
            for point, velocity, c in zip(iter_pts, iter_vels, iter_colors):
                _element = g.plot_all_points(point, color=c)    
                # _arrows = g.add_all_arrows(locs=point, directions=velocity, color=color, length=0.2)
                plt.pause(0.1)
                _element.remove()
                # _arrows.remove()

        # Graphs all final points and keeps graph open    
        g.plot_all_points(self.agent_points[-1], color="purple")
        # plt.show()
        return

    def set_final_stats(self):
        """ Sets all the final stats for sacing to file or printing """
        total_steps = 0
        all_move = 0
        self.swarm_stats["final_score"] = self.scorable.score(self.global_best_point)
        self.swarm_stats["final_class"] = self.scorable.classify_score(self.swarm_stats["final_score"])
        self.swarm_stats["global_best_position"] = self.global_best_point.array
        best_scores = []
        best_pos = []
        for agent in self.agents:
            

            total_steps += agent.agent_stats["num_of_steps"]
            if agent.agent_stats["move_towards_score"].count(True) > agent.agent_stats["move_towards_score"].count(False):
                all_move += 1
            else:
                all_move -= 1
        unique_best_scores = best_scores
        unique_best_pos = best_pos
        self.swarm_stats["unique_best_positions"] = unique_best_pos
        self.swarm_stats["unique_best_scores"] = unique_best_scores
        self.swarm_stats["avg_dir_to_score"] = (all_move >= 0)
        self.swarm_stats["total_steps"] = total_steps
        self.swarm_stats["average_steps"] = total_steps / self.num_agents

    def print_swarm_stats(self):
        print("Swarm final stats:")
        print("- Constants:", self.swarm_constants)
        print("- Number of agents:", self.num_agents)
        print("- Swarm runtime:", self.swarm_runtime)
        for key, val in zip(self.swarm_stats.keys(), self.swarm_stats.values()):
            print("-", key, ":", str(val))
        return

    def store_to_file(self, filename="PSO-Test", mode='w'):
        # Creating and writing to a new file
        file = open(filename, mode=mode)
        file.write("Swarm final stats:")
        file.write("\n- Constants: " + str(self.swarm_constants))
        file.write("\n- Number of agents: " + str(self.num_agents))
        file.write("\n- Swarm runtime: " + str(self.swarm_runtime))
        for key, val in zip(self.swarm_stats.keys(), self.swarm_stats.values()):
            file.write("\n- " + key + ": " + str(val))
        file.close()
        return

            
def set_spheres_scorable(g: Grapher) -> Scorable:
    ndims = 3
    pt1 = Point(0.5, 0.2, 0.3)
    pt2 = Point([0.5] * ndims)
    pt3 = Point(0, 0.1, 0.3)
    sphere1 = ProbilisticSphere(pt1, 0.5, 0.3)
    sphere2 = ProbilisticSphere(pt2, 0.4, 0.25)
    sphere3 = ProbilisticSphere(pt3, 0.7, 0.80)
    g._draw_3d_sphere(pt1, 0.5)
    g._draw_3d_sphere(pt2, 0.4)
    g._draw_3d_sphere(pt3, 0.7)
    scoreable = ProbilisticSphereCluster([sphere1, sphere2, sphere3])
    return scoreable

def random_sphere_scorable(g: Grapher, num_of_spheres=3) -> Scorable:
    spheres = []
    for i in range(num_of_spheres):
        pt = Point([random.random(),random.random(),random.random()])
        radius = random.random()*.5
        sphere = ProbilisticSphere(pt, radius, random.random())
        g._draw_3d_sphere(pt, radius)
        spheres.append(sphere)
    scoreable = ProbilisticSphereCluster(spheres)
    return scoreable

def test_particle_swarm_point():
    # scoreable: Scorable, domain: Domain, score_matrix: ndarray
    ndims = 3
    domain = Domain.normalized(ndims)
    space_shape = domain.dimensions
    grid = Grid([0.05]*ndims)
    g = Grapher(True, domain)
    
    

    # scoreable = random_sphere_scorable(g, num_of_spheres=2)
    scoreable = set_spheres_scorable(g)
    
    score_matrix = brute_force_grid_search(scoreable, domain, grid)
    
    swarm_constants = {"agentC": 2.05, "globalC": 2.05, "w": 0.72984, "max_v": None, "teleporting": False}
    swarm = ParticleSwarmOptimization(scoreable, space_shape, 60, swarm_constants, 100, AntSwarmAgent)
    swarm.run_swarm()
    swarm.store_to_file()
    swarm.print_swarm_stats()
    envelopes_list = true_envelope_finding_alg(scoreable, score_matrix)
    envelopes = map(
        lambda env: list(map(grid.convert_index_to_point, env)),
        envelopes_list,
    )
    # for env in envelopes:
    #     g.plot_all_points(env, color="gray") 
    colors = ["black"]
    swarm.graph_swarm_3D(g, colors=colors)
    
    bound = []
    for e in envelopes_list:
        b = true_boundary_algorithm(scoreable, score_matrix, e)
        # b1 = np.split(b, b.shape[0])
        bound.append(b)
    
    boundaries = map(
        lambda env2: list(map(grid.convert_index_to_point, env2)),
        bound,
    )
    colors = ["yellow", "cyan", "gray"]
    # plot for boundary points
    for env2, color in zip(boundaries, colors):
        g.plot_all_points(env2, color=color)
    plt.show()
    

     
    # max_score = np.max(score_matrix)
    # max_score_positions = []
    # for i, v in np.ndenumerate(score_matrix):
    #     if v == max_score or v >= 1:
    #         max_score_positions.append(i)
    # print("Max score possible:", max_score)
    # print("Rounded positions with score >= 1:", max_score_positions)

    
    
    
    
     
class ProbilisticSphere(Graded):
    def __init__(self, loc: Point, radius: float, lmbda: float):
        """
        Probability density is formed from the base function f(x) = e^-(x^2),
        such that f(radius) = lmbda and is centered around the origin with a max
        of 1.

        Args:
            loc (Point): Where the sphere is located
            radius (float): The radius of the sphere
            lmbda (float): The density of the sphere at its radius
        """
        self.loc = loc
        self.radius = radius
        self.lmda = lmbda
        self.ndims = len(loc)

        self._c = 1 / radius**2 * np.log(1 / lmbda)

    def score(self, p: Point) -> ndarray:
        "Returns between 0 (far away) and 1 (center of) envelope"
        dist = self.loc.distance_to(p)

        return np.array(1 / np.e ** (self._c * dist**2))

    def classify_score(self, score: ndarray) -> bool:
        return np.linalg.norm(score) > self.lmda

    def gradient(self, p: Point) -> np.ndarray:
        s = p - self.loc
        s /= np.linalg.norm(s)

        return s * self._dscore(p)

    def get_input_dims(self):
        return len(self.loc)

    def get_score_dims(self):
        return 1

    def generate_random_target(self):
        v = np.random.rand(self.get_input_dims())
        v = self.loc + Point(self.radius * v / np.linalg.norm(v) * np.random.rand(1))
        return v

    def generate_random_nontarget(self):
        v = np.random.rand(self.get_input_dims())
        v = self.loc + Point(
            self.radius * v / np.linalg.norm(v) * (1 + np.random.rand(1))
        )
        return v

    def boundary_err(self, b: Point) -> float:
        "Negative error is inside the boundary, positive is outside"
        return self.loc.distance_to(b) - self.radius

    def _dscore(self, p: Point) -> float:
        return -self._c * self.score(p) * self.loc.distance_to(p)

class ProbilisticSphereCluster(Graded):
    def __init__(self, spheres: list[ProbilisticSphere]):
        """
        Probability density is formed from the base function f(x) = e^-(x^2),
        such that f(radius) = lmbda and is centered around the origin with a max
        of 1.

        Args:
            loc (Point): Where the sphere is located
            radius (float): The radius of the sphere
            lmbda (float): The density of the sphere at its radius
        """
        self.spheres = spheres

    def score(self, p: Point) -> ndarray:
        "Returns between 0 (far away) and 1 (center of) envelope"
        return sum(map(lambda s: s.score(p), self.spheres))

    def classify_score(self, score: ndarray) -> bool:
        return any(map(lambda s: s.classify_score(score), self.spheres))

    def gradient(self, p: Point) -> np.ndarray:
        raise NotImplementedError()

    def get_input_dims(self):
        return len(self.spheres[0].loc)

    def get_score_dims(self):
        return 1

    def generate_random_target(self):
        raise NotImplementedError()

    def generate_random_nontarget(self):
        raise NotImplementedError()

    def boundary_err(self, b: Point) -> float:
        raise NotImplementedError()



if __name__ == "__main__":
    test_particle_swarm_point()
    
    

    
    
    
    



    
    




    