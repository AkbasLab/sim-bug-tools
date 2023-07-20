"""
Particle Swarm Optimization
"""

import numpy as np
import time
import random
from numpy import ndarray
from sim_bug_tools.structs import Point, Domain, Grid
from sim_bug_tools.simulation.simulation_core import Scorable, Graded
from sim_bug_tools.exploration.brute_force import brute_force_grid_search, true_envelope_finding_alg
import matplotlib.pyplot as plt
from sim_bug_tools.graphics import Grapher



class ParticleSwarmAgent():
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

    def __init__(self, initial_point: ndarray, swarm_constants: dict):
        self.position = np.array(initial_point)
        self.velocity = np.zeros_like(initial_point)
        self.best_position = np.array(initial_point)
        self.best_score: float
        self.best_fitness = float('inf')
        self.num_of_steps = 0
        self.ndims = len(initial_point)
        self.swarm_constants = {"agentC": 0.0, "globalC": 0.0, "w": 0.0}
        self.swarm_constants = swarm_constants
        self.move_towards_score = []
        self.still_moving = True
        self.prev_position = self.position

    def update_agent(self, global_best_position, max_v):
        # Caluclating distances:
        dist_to_agent_best = (self.best_position - self.position)
        dist_to_global_best = (global_best_position - self.position)
        # Caluclating Inertia: 
        inertia = self.swarm_constants["w"] * self.velocity
        # Calculating the personal_best vector:
        personal_best = (self.swarm_constants["agentC"] * random.uniform(0,1)) * dist_to_agent_best
        # Calculating the global best vector:
        global_best = (self.swarm_constants["globalC"] * random.uniform(0,1) * dist_to_global_best)
        # New velocity:
        new_v = (inertia + personal_best + global_best)
        # Limiting the velocity:
        if max_v is not None:
            new_v[new_v > max_v] = max_v

        # Rounding the values and changing the array to an integer array
        new_v_int = np.rint(new_v).astype(int)
        # print("non rounded",new_v,"--rounded--",new_v_int,"--non-zero--",np.count_nonzero(new_v_int))

        if np.count_nonzero(new_v) == 0:
            self.still_moving = False
            return
        self.still_moving = True
        self.position += new_v_int
        self.velocity = new_v
        self.num_of_steps += 1

    def index_in_bounds(self, shape):
        index = self.position
        for i, idx in enumerate(index):
            if idx < 0:
                index[i] = 0
            if idx >= shape[i]:
                index[i] = shape[i] - 1
        self.position = index

class ParticleSwarmOptimization():
    """
    - Particle Swarm Optimization
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
            - `max_iter: int` - The maximum number of iterations to run
        - Outputs:
            - `ndarray` - The best global estimation
    """
    def __init__(self, scorable: Scorable, score_array: ndarray, num_agents: int, swarm_constants: dict, max_iter: int, v_max: float=None):
        self.scorable = scorable
        self.score_array = score_array
        self.num_agents = num_agents
        self.swarm_constants = swarm_constants
        self.v_max = v_max
        self.max_iter = max_iter
        self.agent_positions = []
        self.agent_velocities = []
        self.create_swarm()
        self.initialize_agents()

    def create_swarm(self):
        self.dimensions = self.score_array.ndim
        start_index = [np.random.randint(0, size) for size in np.shape(self.score_array)]
        self.global_best_position = start_index
        self.max_score = np.max(self.score_array)
        self.global_best_fitness = float('inf')
        return

    def initialize_agents(self):
        self.agents = []
        start_positions = []
        for i in range(self.num_agents):
            # Generating a random start index
            r = []
            for i in np.shape(self.score_array):
                r.append(random.randint(0, i))

            # r = [random.randint(0, size) for size in np.shape(self.score_array)]
            a = ParticleSwarmAgent(r, self.swarm_constants)
            self.agents.append(a)
            start_positions.append(tuple(r))
            # Setting global best fitness and position to agents if agents best is better
            if a.best_fitness < self.global_best_fitness:
                self.global_best_fitness = a.best_fitness
                self.global_best_position = a.best_position
        self.agent_positions.append([start_positions])
        return
        
    def check_fitness(self, agent) -> ParticleSwarmAgent:
        max_score = np.max(self.score_array)
        new_score = self.score_array[tuple(agent.position)]
        new_fitness = max_score - new_score

        if new_fitness < agent.best_fitness:
            agent.best_position = agent.position
            agent.best_fitness = new_fitness
            agent.move_towards_score.append(True)
        elif agent.still_moving:
            agent.move_towards_score.append(False)

        if new_fitness < self.global_best_fitness:
            self.global_best_fitness = new_fitness
            self.global_best_position = agent.best_position
        return agent

    def single_iteration(self) -> bool:
        iteration_positions = []
        iteration_velocities = []
        
        for agent in self.agents:
            agent.update_agent(self.global_best_position, self.v_max)
            agent.index_in_bounds(np.shape(self.score_array))
            self.check_fitness(agent)
            iteration_positions.append(tuple(agent.position))
            iteration_velocities.append(tuple(agent.velocity))
            agent.prev_position = agent.position
        
        self.agent_positions.append([iteration_positions])
        self.agent_velocities.append([iteration_velocities])
        if len(set(iteration_velocities)) == 1:
            # All Agents at same position
            return True
        else:
            # Agents in different positions
            return False
        
    def run_swarm(self, max_iter: int = None):
        if max_iter is None:
            max_iter = self.max_iter
        start_time = time.perf_counter()
        # For given max number of iterations...
        
        for i in range(0, max_iter):
            # Updating each agents velocity and position
            agent_vel_0 = self.single_iteration()
            if agent_vel_0:
                print("All positions same")
                # Take out this line
                break
        
        end_time = time.perf_counter()
        self.swarm_runtime = end_time - start_time
        self.iterations = i
        self.set_final_stats()    
        return self.global_best_position
    
    def graph_swarm_3D(self, g: Grapher, grid: Grid, colors=["red"]):
        
        for row, vel, color in zip(self.agent_positions, self.agent_velocities, colors):
            iter_points = map(
                lambda ipts: list(map(grid.convert_index_to_point, ipts)),
                row,
            )

            for point, velocity in zip(iter_points, vel):
                _element = g.plot_all_points(point, color=color)
                _arrows = g.add_all_arrows(locs=point, directions=velocity, color=color, length=0.05)
                plt.pause(1)
                _element.remove()
                _arrows.remove()

        g.plot_point(grid.convert_index_to_point(self.global_best_position), color=colors[0])
        plt.show()

    def set_final_stats(self):
        self.total_steps = 0
        all_move = 0
        self.final_score = self.score_array[tuple(self.global_best_position)]
        self.final_class = self.scorable.classify_score(self.final_score)
        for agent in self.agents:
            self.total_steps += agent.num_of_steps
            agent_true_move = agent.move_towards_score.count(True)
            agent_false_move = agent.move_towards_score.count(False)
            if agent_true_move > agent_false_move:
                all_move += 1
            else:
                all_move -= 1
        self.avg_dir_to_score = (all_move >= 0)
        self.average_steps = self.total_steps / self.num_agents
        self.absolute_error = abs(self.max_score - self.final_score)

    def print_swarm_stats(self):
        print("Swarm final stats:")
        print("- Constants:", self.swarm_constants)
        print("- Number of iterations:", self.iterations)
        print("- Number of agents:", self.num_agents)
        print("- Best swarm position:", self.global_best_position)
        print("- Best swarm score:", self.final_score)
        print("- Best score classification:", self.final_class)
        print("- Average number of agent steps:", self.average_steps)
        print("- Total number of agent steps:", self.total_steps)
        print("- Average direction towards score:", self.avg_dir_to_score)
        print("- Swarm runtime:", self.swarm_runtime)
        print("- Absolute Error:", self.absolute_error)
        return

    def store_to_file(self, filename="PSO-Test", mode='w'):
        # Creating and writing to a new file
        file = open(filename, mode=mode)
        file.write("Swarm final stats:")
        file.write("\n- Constants: " + str(self.swarm_constants))
        file.write("\n- Number of iterations: " + str(self.iterations))
        file.write("\n- Number of agents: " + str(self.num_agents))
        file.write("\n- Best swarm position: " + str(self.global_best_position))
        file.write("\n- Best swarm score: " + str(self.final_score))
        file.write("\n- Best score classification: " + str(self.final_class))
        file.write("\n- Average number of agent steps: " + str(self.average_steps))
        file.write("\n- Total number of agent steps: " + str(self.total_steps))
        file.write("\n- Average direction towards score: " + str(self.avg_dir_to_score))
        file.write("\n- Swarm runtime: " + str(self.swarm_runtime))
        file.write("\n- Absolute Error: " + str(self.absolute_error))
        file.close()
        return
    
def set_spheres_scorable(g: Grapher) -> Scorable:
    ndims = 3
    pt1 = Point(0, 5, 2)
    pt2 = Point([0.5] * ndims)
    pt3 = Point(0, 2, 0.8)
    sphere1 = ProbilisticSphere(pt1, 0.5, 0.3)
    sphere2 = ProbilisticSphere(pt2, 0.4, 0.25)
    sphere3 = ProbilisticSphere(pt3, 0.7, 0.80)
    g._draw_3d_sphere(pt1, 0.2)
    g._draw_3d_sphere(pt2, 0.4)
    g._draw_3d_sphere(pt3, 0.7)
    scoreable = ProbilisticSphereCluster([sphere1, sphere2, sphere3])
    return scoreable

def random_sphere_scorable(g: Grapher, num_of_spheres=3) -> Scorable:
    spheres = []
    for i in range(num_of_spheres):
        pt = Point(random.random(),random.random(),random.random())
        radius = random.random()
        sphere = ProbilisticSphere(pt, radius, random.random())
        g._draw_3d_sphere(pt, radius)
        spheres.append(sphere)
    scoreable = ProbilisticSphereCluster(spheres)
    return scoreable

def test_particle_swarm():
    # scoreable: Scorable, domain: Domain, score_matrix: ndarray
    ndims = 3
    domain = Domain.normalized(ndims)
    grid = Grid([0.5]*ndims)
    g = Grapher(True, domain)

    scoreable = random_sphere_scorable(g, num_of_spheres=2)
    # scoreable = set_spheres_scorable(g)

    score_matrix = brute_force_grid_search(scoreable, domain, grid)
        
    max_score = np.max(score_matrix)
    max_score_positions = []
    for i, v in np.ndenumerate(score_matrix):
        if v == max_score or v >= 1:
            max_score_positions.append(i)
    print("Max score possible:", max_score)
    print("Positions with score >= 1:", max_score_positions)

    swarm_constants = {"agentC": 2.05, "globalC": 2.05, "w": 0.72984}
    swarm = ParticleSwarmOptimization(scoreable, score_matrix, 50, swarm_constants, v_max=2, max_iter=100)
    swarm.run_swarm()
    # swarm.store_to_file()
    swarm.print_swarm_stats()
    colors = ["black", "green", "red", "blue", "pink", "brown"]
    swarm.graph_swarm_3D(g, grid, colors=colors)
    
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
    # Maximize the local max for spheres, not just global
    # If the score is increasing, keep going, if its decreasing, try different direction
    # Group agents to find different local maximums
    # Don't go to one max that youve already found
    # How many envelopes that have been found to te4st for higher definitions, check if an agent is in that envelope
    # ant colony algorithms, sebastian log - this is seperate algorithm
    # prof Akbas out of town, only meet with John for 2 weeks
    test_particle_swarm()
    

    
    
    
    



    
    




    