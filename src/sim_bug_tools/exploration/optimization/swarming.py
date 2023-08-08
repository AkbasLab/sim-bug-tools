"""
Swarming Optimization Algorithms
"""

import numpy as np
import time
import random
from numpy import ndarray
import itertools
from sim_bug_tools.structs import Point, Domain, Grid
from sim_bug_tools.simulation.simulation_core import Scorable, Graded
from sim_bug_tools.exploration.brute_force import brute_force_grid_search, true_envelope_finding_alg, true_boundary_algorithm
import matplotlib.pyplot as plt
from sim_bug_tools.graphics import Grapher
from rtree import index

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
            "num_samples": 0,
            "envelopes_found": []
        }

        initial_position = [random.uniform(0,i) for i in self.space_shape]
        self.point = Point(initial_position)
        self.best_pt = self.point
        self.velocity = [0.0 for _ in self.space_shape]
        self.velocity = np.array(self.velocity)
        self.teleported = True
        # "green"=moved to higher score, "red"=moved to lower score
        # "orange"=did not move, "blue"=teleported, "black"=found best score, "pink"=spawned
        self.movement_color: str = "pink"
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
        self.teleported = True
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
            if self.compare_score(self.best_pt, new_point): 
                self.best_pt = new_point
            self.agent_stats["move_towards_score"].append(True)
            self.movement_color = "green"
        else:
            self.agent_stats["move_towards_score"].append(False)
            self.movement_color = "red"
        if self.teleported: 
            self.movement_color = "pink" 
            self.teleported = False      
        
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
        # If it reaches the envelope, dont update num of steps
        if self.scorable.classify(self.point):
            self.movement_color = "cyan"
        else:
            self.agent_stats["num_of_steps"] += 1
        return
       
    def index_in_bounds(self):
        """ Determines if the new point is in the space shape, if not, puts it in just inside the limits """
        index = self.point.array
        for i, idx in enumerate(index):
            if idx < 0:
                index[i] = 0
            if idx >= self.space_shape[i]:
                index[i] = self.space_shape[i] - 0.1
        self.point = Point(index)
        return
    
    def set_final_stats(self):
        self.agent_stats["best_points"].append(self.best_pt)
        a_scores = self.scorable.v_score(np.array(self.agent_stats["best_points"]))
        self.agent_stats["best_scores"] = a_scores


            
        return self.agent_stats


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
            self.agent_stats["num_samples"] += 1
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
        if "scatter_retries" not in self.swarm_constants.keys():
            self.swarm_constants["scatter_retries"] = 10

        # Attempts at a higher score in the area
        self.attempts: int = 0
        self.velocity = self.scatter_search()
        return
    
    def update_agent(self):
        new_pos = self.point.array + self.velocity
        if self.scorable.score(Point(new_pos)) <= self.scorable.score(self.point):
            self.velocity = self.scatter_search()

        # Velocity is determined by a scatter search each time
        # self.velocity = self.scatter_search()

        return super().update_agent(self.velocity)
    
    def scatter_search(self):
        curr_pos = self.point.array
        curr_score = self.scorable.score(self.point)
        r = self.swarm_constants["search_radius"]
        best_pos = []
        best_pt: Point = None
        best_score = -1
        for i in range(self.swarm_constants["search_scatter"]):
            new_pos = [random.uniform(i-r,i+r) for i in curr_pos]
            new_pt = Point(new_pos)
            new_score = self.scorable.score(new_pt)
            self.agent_stats["num_samples"] += 1
            
            if new_score > best_score:
                best_score = new_score
                best_pos = new_pos
            
        new_v = best_pos - curr_pos
        if best_score < curr_score and self.attempts < 10:
            self.attempts += 1
            new_v = self.scatter_search()
        elif best_score < curr_score:
            new_v = [0 for _ in curr_pos]
            self.attempts = 0
        return new_v

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
    def __init__(self, scorable: Scorable, space_shape: ndarray, swarm_constants: dict=None, num_agents: int=50, max_iter: int=100, agent_class=AntSwarmAgent) -> None:
        self.scorable: Scorable = scorable
        self.space_shape = space_shape
        self.num_agents: int = num_agents
        if swarm_constants is None:
            swarm_constants = {"agentC": 2.05, "globalC": 2.05, "w": 0.72984, "max_v": None}
        self.swarm_constants: dict = swarm_constants
        self.max_iter: int = max_iter
        self.agent_points: list[Point] = []
        self.agent_velocities: list[ndarray] = []
        self.agent_colors: list[str] = []
        self.global_best_point: Point
        self.agent_class = agent_class
        self.swarm_stats = {
            "total_steps": int,
            "global_best_score": float,
            "global_best_classification": bool,
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
        # Setting the final stats and calculating runtime to do so
        s_time = time.perf_counter()
        self.set_final_stats()
        e_time = time.perf_counter()
        self.swarm_stats["post_process_runtime"] = e_time - s_time
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
        self.swarm_stats["global_best_score"] = self.scorable.score(self.global_best_point)
        self.swarm_stats["global_best_classification"] = self.scorable.classify_score(self.swarm_stats["global_best_score"])
        self.swarm_stats["global_best_position"] = self.global_best_point.array
        best_scores = []
        best_points = []
        total_samples = 0
        for agent in self.agents:
            a_stats = agent.set_final_stats()
            best_scores.append(a_stats["best_scores"])
            best_points.append(a_stats["best_points"])
            total_steps += agent.agent_stats["num_of_steps"]
            total_samples += agent.agent_stats["num_samples"]
            if a_stats["move_towards_score"].count(True) > a_stats["move_towards_score"].count(False):
                all_move += 1
            else:
                all_move -= 1
        
        spheres_found = []
        sphere_num = 1
        for sphere in self.scorable.spheres:
            samples = 0
            for pt in itertools.chain(*best_points):
                if sphere.classify(pt):
                    samples += 1
            spheres_found.append([sphere_num, samples])
            sphere_num += 1
                
        self.swarm_stats["total_samples"] = total_samples
        self.swarm_stats["spheres_found"] = spheres_found
        self.swarm_stats["best_scores_range"] = [min(best_scores), max(best_scores)]
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

    def store_to_file(self, filename="PSO-Test", mode='w', other_stats: dict=None):
        # Creating and writing to a new file
        file = open(filename, mode=mode)
        file.write("Swarm final stats:")
        file.write("\n- Constants: " + str(self.swarm_constants))
        file.write("\n- Number of agents: " + str(self.num_agents))
        file.write("\n- Swarm runtime: " + str(self.swarm_runtime))
        for key, val in zip(self.swarm_stats.keys(), self.swarm_stats.values()):
            file.write("\n- " + key + ": " + str(val))
        if other_stats is not None:
            for key, val in zip(other_stats.keys(), other_stats.values()):
                file.write("\n- " + key + ": " + str(val))
        file.close()
        return

# class MultiSwarm():

#     DEFAULT_GRID_SCALE = 0.1
#     DEFAULT_NDIMS = 3
#     DEFAULT_NUM_SWARMS = 2
#     DEFAULT_DOMAIN = Domain.normalized(DEFAULT_NDIMS)
#     DEFAULT_GRID = Grid([DEFAULT_GRID_SCALE]*DEFAULT_NDIMS)
#     GRAPH_3D = Grapher(True, DEFAULT_DOMAIN)
#     GRAPH = False
#     DEFAULT_ITER = 100

#     def __init__(self, scorable: Scorable, domain: Domain, max_iter=DEFAULT_ITER, graph_3d: bool=False, grid_scale: float=0.1, num_swarms=2, *args):
#         self.scorable = scorable
#         self.domain = domain
#         self.grid = Grid([grid_scale]*len(domain.dimensions))
#         MultiSwarm.GRAPH = graph_3d
#         self.num_swarms = num_swarms
#         self.max_iter = max_iter
#         VALID_INPUTS = (dict, int, SwarmAgent)
#         if len(args) == 0:
#             self.default_swarms() 
#         elif len(args) > 3 or not isinstance(args, VALID_INPUTS):
#             raise ValueError(
#                 f"{__class__.__name__}.__init__: Invalid arguments (args = {args})."
#             )
#         else:
#             self.initialize_swarms(args)

#         return


#     def __init__(self, graph_3d: bool=False, num_swarms=DEFAULT_NUM_SWARMS, max_iter=DEFAULT_ITER):
#         self.num_swarms = num_swarms
#         self.max_iter = max_iter
#         MultiSwarm.GRAPH = graph_3d
#         self.default_scorable()

    
#     def default_swarms(self):
#         """ Randomized and default values for swarms """
#         space_shape = self.domain.dimensions
#         self.swarms: list[ParticleSwarmOptimization] = []
#         for i in range(self.num_swarms):
#             new_swarm = ParticleSwarmOptimization(scorable=self.scorable, space_shape=space_shape)
#             self.swarms.append(new_swarm)
#         return
    
#     def set_swarms(self, *args):
#         """ Set swarms using arguments passed in """
#         space_shape = self.domain.dimensions
#         self.swarms: list[ParticleSwarmOptimization] = []
#         for i in range(self.num_swarms):
#             new_swarm = ParticleSwarmOptimization(scorable=self.scorable, space_shape=space_shape, *args)
#             self.swarms.append(new_swarm)
#         return

#     def default_scorable(self):
#         """ Completely default swarms including scorable, domain, and grid """
#         if self.GRAPH:
#             self.scorable, _ = random_sphere_scorable(self.GRAPH_3D)
#         else:
#             self.scorable, _ = random_sphere_scorable()
#         self.domain = self.DEFAULT_DOMAIN
#         self.grid = self.DEFAULT_GRID
#         self.default_swarms()
#         return

#     def run_swarms(self):
#         cont = True
#         i = 0
#         while (i < self.max_iter) and cont:
#             cont = False
#             for s in self.swarms: 
#                 if s.single_iteration(): cont = True
#             i += 1
#         self.iterations = i
#         self.set_final_stats()
#         return
    
#     def set_final_stats(self):
#         swarm_best_points: list[Point] = []
#         swarm_best_scores: list[float] = []
#         swarm_iter_points: list[Point] = []
#         swarm_iter_colors: list[str] = []
#         for s in self.swarms:
#             s.set_final_stats()
#             s_stats = s.swarm_stats
#             swarm_best_points.append(s_stats["global_best_position"])
#             swarm_best_scores.append(s_stats["global_best_score"])
#             swarm_iter_points = list(zip(swarm_iter_points, s.agent_points))
#             # swarm_iter_points.append(s.agent_points)
#             swarm_iter_colors.append(s.agent_colors)
#         self.multi_swarm_stats = {
#             "best_points": swarm_best_points,
#             "best_scores_range": [min(swarm_best_scores), max(swarm_best_scores)]
#         }
#         self.iter_points = swarm_iter_points
#         self.iter_colors = swarm_iter_colors
#         # self.print_multi_swarm()
#         return
    
#     def print_multi_swarm(self):
#         print(vars(self))
#         return
    
#     def graph_multi_swarms(self, colors=None):
#         print(self.iter_points)
#         for pts in self.iter_points:
#             # for p, c in zip(pts, colors):
#             _elements = self.GRAPH_3D.plot_all_points(pts)
#             plt.pause(0.1)
#             _elements.remove()
#         self.GRAPH_3D.plot_all_points(self.multi_swarm_stats["best_points"], color="purple")
#         plt.show()    
#         pass


def set_spheres_scorable(g: Grapher=None, ndims=3, lam=0.1) -> ndarray:
    pt1 = Point([0.0 for _ in range(ndims)])
    pt2 = Point([0.1 for _ in range(ndims)])
    pt3 = Point([0.55 for _ in range(ndims)])
    pt4 = Point([0.75 for _ in range(ndims)])
    pt5 = Point([0.95 for _ in range(ndims)])
    points = [pt1, pt2, pt3, pt4, pt5]
    i = 1
    radius = ndims*0.05
    sphere_info = {}
    spheres: list[ProbilisticSphere] = []
    for pt in points:
        sphere_info[("Sphere #"+str(i))] = ("\n  - " + str(pt) + "\n  - radius: " + str(radius) + "\n  - lambda: " + str(lam))
        i += 1
        spheres.append(ProbilisticSphere(pt, radius, lam, height=1000))
    # sphere1 = ProbilisticSphere(pt1, 0.15, lam)
    # sphere2 = ProbilisticSphere(pt2, 0.15, lam)
    # sphere3 = ProbilisticSphere(pt3, 0.15, lam)
    if g is not None:
        for s in spheres:
            g._draw_3d_sphere(s.loc, s.radius)
    scoreable = ProbilisticSphereCluster(spheres)
    return scoreable, sphere_info

def random_sphere_scorable(g: Grapher=None, ndims:int=3, num_of_spheres=3, intersect=False):
    spheres = []
    sphere_info = {}
    lam = random.random()
    for i in range(num_of_spheres):
        index = [random.random() for _ in range(ndims)]
        pt = Point(index)
        # pt = Point([random.random(),random.random(),random.random()])

        radius = random.random()
        if abs(lam-radius) >= 0.2:
            radius = lam - 0.25
        sphere_info[("Sphere #"+str(i))] = ("\n  - " + str(pt) + "\n  - radius: " + str(radius) + "\n  - lambda: " + str(lam))
        sphere = ProbilisticSphere(pt, radius, lam)
        if g is not None and ndims < 4:
            g._draw_3d_sphere(pt, radius)
        spheres.append(sphere)
    scoreable = ProbilisticSphereCluster(spheres)
    return scoreable, sphere_info

def test_PSO_no_graph(ndims: int=3, test_num=None):
    
    domain = Domain.normalized(ndims)
    space_shape = domain.dimensions
    # scorable, run_info = random_sphere_scorable(num_of_spheres=5, ndims=ndims)
    scorable, run_info = set_spheres_scorable(ndims=ndims)
    p: str = "C:/Users/User/OneDrive/AI Validation Research/Swarming/PSO Tests"

    run_info["ndims"] = ndims
    # swarm_constants = {"agentC": 2.05, "globalC": 2.05, "w": 0.72984, "max_v": None, "teleporting": False}
    agent_type = AntSwarmAgent
    run_info["Agent_type"] = agent_type
    # swarm = ParticleSwarmOptimization(scoreable, space_shape, swarm_constants, 60, 100, agent_type)
    swarm = ParticleSwarmOptimization(scorable, space_shape, num_agents=200, agent_class=agent_type)
    swarm.run_swarm()
    if test_num is not None:
        filename = p+ "\Run "+ str(test_num) + "_" + str(ndims) + "D.txt"
        swarm.store_to_file(filename=filename, other_stats=run_info)
    swarm.print_swarm_stats()




def test_particle_swarm_point(test_num: int):
    # scoreable: Scorable, domain: Domain, score_matrix: ndarray
    ndims = 3
    domain = Domain.normalized(ndims)
    space_shape = domain.dimensions
    scale = 0.05
    grid = Grid([scale]*ndims)
    g = Grapher(True, domain)
    p: str = "C:/Users/User/OneDrive/AI Validation Research/Swarming/PSO Tests"
    # scoreable, run_info = random_sphere_scorable(g, num_of_spheres=2)
    scoreable, run_info = set_spheres_scorable(g)
    run_info["ndims"] = ndims
    run_info["Domain"] = domain
    run_info["grid"] = "Grid([" + str(scale) + "]*ndims)"
    # score_matrix = brute_force_grid_search(scoreable, domain, grid)
    
    swarm_constants = {"agentC": 2.05, "globalC": 2.05, "w": 0.72984, "max_v": None, "teleporting": False}
    agent_type = AntSwarmAgent
    run_info["Agent_type"] = agent_type
    swarm = ParticleSwarmOptimization(scoreable, space_shape, swarm_constants, 60, 100, agent_type)
    swarm.run_swarm()
    filename = p+ "\Run "+ str(test_num) + ".txt"
    # swarm.store_to_file(filename=filename, other_stats=run_info)
    swarm.print_swarm_stats()
    # envelopes_list = true_envelope_finding_alg(scoreable, score_matrix)
    # envelopes = map(
    #     lambda env: list(map(grid.convert_index_to_point, env)),
    #     envelopes_list,
    # )
    # for env in envelopes:
    #     g.plot_all_points(env, color="gray") 
    colors = ["black"]
    swarm.graph_swarm_3D(g)
    graph_name = p + "\Run "+ str(test_num) + ".png"
    
    # bound = []
    # for e in envelopes_list:
    #     b = true_boundary_algorithm(scoreable, score_matrix, e)
    #     # b1 = np.split(b, b.shape[0])
    #     bound.append(b)
    
    # boundaries = map(
    #     lambda env2: list(map(grid.convert_index_to_point, env2)),
    #     bound,
    # )
    # colors = ["yellow", "cyan", "gray"]
    # # plot for boundary points
    # for env2, color in zip(boundaries, colors):
    #     g.plot_all_points(env2, color=color)
    # g.save(path=graph_name)
    # plt.pause(20)
    # plt.close()
    plt.show()
    return

     
    # max_score = np.max(score_matrix)
    # max_score_positions = []
    # for i, v in np.ndenumerate(score_matrix):
    #     if v == max_score or v >= 1:
    #         max_score_positions.append(i)
    # print("Max score possible:", max_score)
    # print("Rounded positions with score >= 1:", max_score_positions)


class ProbilisticSphere(Graded):
    def __init__(self, loc: Point, radius: float, lmbda: float, height: float = 1):
        self.loc = loc
        self.radius = radius
        self.lmda = lmbda
        self.ndims = len(loc)
        self.height = height

        self._c = 1 / radius**2 * np.log(height / lmbda)

    @property
    def const(self) -> float:
        return self._c

    def score(self, p: Point) -> ndarray:
        "Returns between 0 (far away) and 1 (center of) envelope"
        dist = self.loc.distance_to(p)

        return self.height * np.array(1 / np.e ** (self._c * dist**2))

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
        return abs(self.loc.distance_to(b) - self.radius)

    def _dscore(self, p: Point) -> float:
        return -self._c * self.score(p) * self.loc.distance_to(p)

class ProbilisticSphereCluster(Graded):
    def __init__(self, spheres: list[ProbilisticSphere]):
        self.spheres = spheres
        self._ndims = self.spheres[0].ndims
        p = index.Property()
        p.set_dimension(self._ndims)
        self._index = index.Index(properties=p)

        self._id = 0
        lmbda = spheres[0].lmda
        height = spheres[0].height
        self._lmbda = lmbda
        self._height = height
        # self.construct_cluster(p0, r0, num_points_per_point, depth)

        self._sph_radii = np.array([sph.radius for sph in self.spheres])
        self._sph_lmbda = (
            np.ones(self._sph_radii.shape) * lmbda
        )  # np.array([sph.lmda for sph in self.spheres])

        self._gradient_coef = -2 / self._sph_radii**2 * np.log(1 / lmbda)

        # self._base = np.array(
        #     [
        #         1 / np.e ** (1 / r**2 * np.log(1 / l))
        #         for r, l in zip(self._sph_radii, self._sph_lmbda)
        #     ]
        # )  # np.array([1 / np.e**sph.const for sph in self.spheres])
        # print(self._base)
        self._base = np.e ** (
            -self._sph_radii ** (-2) * np.log(height / self._sph_lmbda)
        )
        self._exp = -self._sph_radii ** (-2) * np.log(height / self._sph_lmbda)
        self._sph_locs = np.array([sph.loc.array for sph in self.spheres])

    @property
    def ndims(self):
        return self._ndims

    @property
    def lmbda(self):
        return self._lmbda

    @property
    def min_dist_b_perc(self):
        return self._min_dist_b_perc

    @property
    def max_dist_b_perc(self):
        return self._max_dist_b_perc

    @property
    def min_rad_perc(self):
        return self._min_rad_perc

    @property
    def max_rad_perc(self):
        return self._max_rad_perc

    def construct_cluster(self, p0: Point, r0: float, n: int, k: int):
        queue = [(p0, r0)]
        self.spheres: list[ProbilisticSphere] = []

        remaining = n**k
        while len(queue) > 0 and remaining > 0:
            p, r = queue.pop()

            self._index.insert(len(self.spheres), p)
            self.spheres.append(ProbilisticSphere(p, r, self.lmbda, self._height))
            remaining -= 1

            queue = [self.create_point_from(p, r) for i in range(n)] + queue

    def create_point_from(self, p: Point, r: float) -> tuple[Point, float]:
        ## dist
        # r1 * (1 + min_dist_b_perc) < d < r1 * (1 + max_dist_b_perc)
        # min = 0, d must be beyond the border
        # min = -1, d must be beyond the r1's center
        # max = 0, d must be before the border
        # max = 1, d must be before twice the radius

        i = 0
        valid = False
        while not valid:
            if i > 100:
                raise Exception(
                    "100 failed attempts for generating a sphere in bounds of domain."
                )

            # pick random direction and distance to find location
            v = np.random.rand(self.ndims)
            v = v * 2 - 1
            v /= np.linalg.norm(v)
            d = self._random_between(
                r * (1 + self.min_dist_b_perc), r * (1 + self.max_dist_b_perc)
            )
            p2 = p + Point(v * d)

            # pick a radius that touches the parent sphere
            min_r = (1 + self.min_rad_perc) * (d - r)
            max_r = (1 + self.max_rad_perc) * r
            r2 = self._random_between(min_r, max_r)

            c = np.ones(p.array.shape)
            c = c / np.linalg.norm(c) * r2

            # if domain specified, do not leave domain.
            valid = (
                self._domain is None
                or (p2 + c) in self._domain
                and (p2 - c) in self._domain
            )
            i += 1

        return (p2, r2)

    def v_score(self, v: ndarray) -> ndarray:
        dif2 = np.linalg.norm(v[:, None] - self._sph_locs, axis=-1) ** 2
        return np.sum(
            self._height * np.e ** (self._exp * dif2), axis=-1
        )  # sum(self._base**dif2)

    def score(self, p: Point) -> ndarray:
        dif2 = np.linalg.norm(p.array - self._sph_locs, axis=1) ** 2
        # return sum(self._base**dif2)
        # closest_index = min(
        #     enumerate(
        #         abs(np.linalg.norm(p - self._sph_locs, axis=1) - self._sph_radii).T
        #     ),
        #     key=lambda pair: pair[1],
        # )[0]
        return sum(self._height * np.e ** (self._exp * dif2))  # sum(self._base**dif2)
        # return np.array([sum(map(lambda sph: sph.score(p), self.spheres))])

    def classify_score(self, score: ndarray) -> bool:
        return np.linalg.norm(score) > self.lmbda

    def get_input_dims(self):
        return self.ndims

    def get_score_dims(self):
        return 1

    def generate_random_target(self):
        sph_index = random.randint(0, len(self.spheres) - 1)
        return self.spheres[sph_index].generate_random_target()

    def generate_random_nontarget(self):
        raise NotImplementedError()

    def _nearest_sphere(self, b: Point) -> ProbilisticSphere:
        nearest_err = self.spheres[0].boundary_err(b)
        nearest = self.spheres[0]

        for sph in self.spheres[1:]:
            if err := sph.boundary_err(b) < nearest_err:
                nearest_err = err
                nearest = sph

        return nearest

    def boundary_err(self, b: Point) -> float:
        "euclidean distance from boundary"
        # return min(abs(np.linalg.norm(b - self._sph_locs, axis=1) - self._sph_radii))

        # return self._nearest_sphere(b).boundary_err(b)
        # return min(self.spheres, key=lambda sph:
        # sph.boundary_err(b)).boundary_err(b)

        # linearization approach - led to high-error :(

        # err_v[err_v > 1] = 0  # get rid of inf due to axis alignment
        # return (self.score(b) - self.lmbda) / np.linalg.norm(self.gradient(b))

        # nearest sphere's
        # id = self._index.nearest(b, 1)
        # p = self._sph_locs[id]
        # r = self._sph_radii[id]
        # return abs(np.linalg.norm(p - b.array) - r)
        return min(
            abs(np.linalg.norm(self._sph_locs - b.array, axis=1) - self._sph_radii)
        )

    def true_osv(self, b: Point) -> ndarray:
        sph = self._nearest_sphere(b)
        return self.normalize((b - sph.loc).array)

    def osv_err(self, b: Point, n: ndarray) -> float:
        return self.angle_between(self.true_osv(b), n)

    def gradient(self, p: Point) -> ndarray:
        # return sum(self.spheres, key=lambda sph: sph.gradient(p))
        return self._gradient_coef[None].T * (p.array - self._sph_locs) * self.score(p)

    @staticmethod
    def _random_between(a, b) -> float:
        return random.random() * (b - a) + a

    @staticmethod
    def normalize(v: ndarray) -> ndarray:
        return v / np.linalg.norm(v)

    @classmethod
    def angle_between(cls, u, v):
        u, v = cls.normalize(u), cls.normalize(v)
        return np.arccos(np.clip(np.dot(u, v), -1, 1.0))

# distance/score, >1 increase distance, <1 decrease distance

if __name__ == "__main__":
    test_particle_swarm_point(1)
    # test_PSO_no_graph(ndims=20, test_num=3)
    # m = MultiSwarm()
    # m.run_swarms()
    # m.graph_multi_swarms()
    # point_matrix = np.zeros((11,11))
    # point_matrix = [Point(np.array(index)/10) for index, _ in np.ndenumerate(point_matrix)]
    # print(point_matrix)
    
    # s = ProbilisticSphere(Point([0.2,0.2]), 0.2, 0.2)
    # g = Grapher(domain=Domain.normalized(2))
    # grid = Grid([0.1]*2)
    # g._draw_2d_circle(Point([0.2, 0.2]), 0.2, color="green")
    # g.plot_all_points(point_matrix)
    
    # gradient_matrix = [s.gradient(pt) for pt in point_matrix]
    # for p, gr in zip(point_matrix, gradient_matrix):
    #     print(p, "\n- Gradient:", gr, "\n- Score:", (s.score(p)), "\n- Classification:", (s.classify(p)))
    #     if s.classify(p):
    #         g.plot_point(p, color="red")
        
    # # g.plot_all_points(gradient_matrix, color="red")
    # # plt.scatter(gradient_matrix[0], gradient_matrix[1])
    # plt.show()
    # prev_p = Point([0.8,0.8])
    # prev_r = 0.1
    # prev_score = s.score(prev_p)
    # print(s.gradient(prev_p))
    
    # _cir = g._draw_2d_circle(prev_p, prev_r, color="red")
    # _p = g.plot_point(prev_p, color="purple")
    # for _ in range(10):
    #     plt.pause(1)
    #     _cir.remove()
    #     _p.remove()
    #     prev_pos = prev_p.array
    #     new_p = Point([random.uniform(i-prev_r,i+prev_r) for i in prev_pos])
    #     print(s.gradient(new_p))
    #     new_score = s.score(new_p)
    #     rate = (new_score-prev_score)/prev_score
    #     if rate > 1:
    #         new_r = prev_r - (prev_r*0.2)
    #     else:
    #         new_r = prev_r + (prev_r*0.2)
    #     print(rate, "  ", new_r)
    #     _cir = g._draw_2d_circle(new_p, new_r, color="red")
    #     _p = g.plot_all_points([new_p], color="purple")
    #     plt.pause(1)
    #     prev_p = new_p
    #     prev_r = new_r    
    
    # plt.show()
    
    

    
    
    
    



    
    




    