# Have manager create agents, define optional agent start point
# using scorable, domain
# look for optimizing swarm
# Goal is to maximize the score 

import numpy as np
import time
from numpy import ndarray
from sim_bug_tools.structs import Point, Domain, Grid
from sim_bug_tools.simulation.simulation_core import Scorable, Graded
from sim_bug_tools.exploration.brute_force import brute_force_grid_search

class Agent():
    def __init__(self, initial_point):
        self.position = np.array(initial_point)
        self.velocity = np.zeros_like(initial_point)
        self.best_position = np.array(initial_point)
        self.best_fitness = float('inf')

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
    - Partical Swarm Optimization
        Creates num_agent number of agents for the swam starting them at random points in the given score array.
        - Inputs:
            - `score_array: ndarray` - The score matrix
            - `num_agents: int` - the number of agents to add to the swarm
            - `agentC: float` - The constant value to multiply with the distance to personal best
                    Note: agentC > globalC results in high exploration, low exploitation.
            - `globalC: float` - The constant value to multiply with the distance to gloabal best
                    Note: globalC > agentC results in low exploration, high exploitation.
            - `w: float` - The interia weight to multiply the velocity by
            - `max_iter: int` - The maximum number of iterations to run
        - Outputs:
            - `ndarray` - The best global estimation
    """
    def __init__(self, score_array: ndarray, num_agents: int, agentC: float, globalC: float, w: float, v_max: float, max_iter: int):
        self.score_array = score_array
        self.num_agents = num_agents
        self.agentC = agentC
        self.globalC = globalC
        self.w = w
        self.v_max = v_max
        self.max_iter = max_iter
        self.create_swarm()

    def create_swarm(self):
        self.dimensions = self.score_array.ndim
        start_index = [np.random.randint(0, size) for size in np.shape(self.score_array)]
        self.global_best_position = start_index
        self.global_best_fitness = float('inf')
        self.max_score = np.max(self.score_array)
        self.initialize_agents()
        return

    def initialize_agents(self):
        self.agents = []
        for _ in range(self.num_agents):
            # Generating a random start index
            r = [np.random.randint(0, size) for size in np.shape(self.score_array)]
            a = Agent(r)
            self.agents.append(a)

            # Setting global best fitness and position to agents if agents best is better
            if a.best_fitness < self.global_best_fitness:
                self.global_best_fitness = a.best_fitness
                self.global_best_position = a.best_position
        return
    
    def update_agent(self, agent: Agent) -> Agent:
        dist_to_agent_best = (agent.best_position - agent.position)
        dist_to_global_best = (self.global_best_position - agent.position)
        agent.velocity = ((self.w * agent.velocity) +
                        (self.agentC * np.random.rand(self.dimensions) * dist_to_agent_best) + 
                        (self.globalC * np.random.rand(self.dimensions) * dist_to_global_best))
        agent.position += agent.velocity.astype(int)
        agent.index_in_bounds(np.shape(self.score_array))
        return agent

    
    def single_iteration(self):
        for agent in self.agents:
            agent = self.update_agent(agent)
            new_score = self.score_array[tuple(agent.position)]
            new_fitness = self.max_score - new_score

            if new_fitness < agent.best_fitness:
                agent.best_position = agent.position
                agent.best_fitness = new_fitness

            if new_fitness < self.global_best_fitness:
                self.global_best_fitness = new_fitness
                self.global_best_position = agent.best_position
        return
    
    def run_swarm(self, max_iter: int = None):
        if max_iter is None:
            max_iter = self.max_iter
        # For given max number of iterations...
        for i in range(max_iter):
            # Updating each agenst velocity and position
            self.single_iteration()
        return self.global_best_position

def particle_swarm_split_functions(score_array: ndarray, num_agents: int, agentC: float, globalC: float, w: float, v_max: float, max_iter: int = 10) -> ndarray:
    """
    - Partical Swarm Optimization split functions
        Creates num_agent number of agents for the swam starting them at random points in the given score array.
        - Inputs:
            - `score_array: ndarray` - The score matrix
            - `num_agents: int` - the number of agents to add to the swarm
            - `agentC: float` - The constant value to multiply with the distance to personal best
                    Note: agentC > globalC results in high exploration, low exploitation.
            - `globalC: float` - The constant value to multiply with the distance to gloabal best
                    Note: globalC > agentC results in low exploration, high exploitation.
            - `w: float` - The interia weight to multiply the velocity by
            - `max_iter: int` - The maximum number of iterations to run
        - Outputs:
            - `ndarray` - The best global estimation
    """
    dimensions = score_array.ndim
    start_index = [np.random.randint(0, size) for size in np.shape(score_array)]
    global_best_position = start_index
    global_best_fitness = float('inf')
    max_score = np.max(score_array)
    agents = []
    for _ in range(num_agents):
        a = create_agent(score_array)
        agents.append(a)
        # Setting global best fitness and position to agents if agents best is better
        if a.best_fitness < global_best_fitness:
            global_best_fitness = a.best_fitness
            global_best_position = a.best_position
    # For given max number of iterations...
    for i in range(max_iter):
        # Updating each agenst velocity and position
        for agent in agents:
            updated_agent = update_agent(agent, global_best_position, w, agentC, globalC, dimensions)
            agent = updated_agent
            agent.index_in_bounds(np.shape(score_array))
            new_score = score_array[tuple(agent.position)]
            new_fitness = max_score - new_score

            if new_fitness < agent.best_fitness:
                agent.best_position = agent.position
                agent.best_fitness = new_fitness

            if new_fitness < global_best_fitness:
                global_best_fitness = new_fitness
                global_best_position = agent.best_position
    return global_best_position

def create_agent(score_array: ndarray) -> ndarray:
    # Generating a random start index
    r = [np.random.randint(0, size) for size in np.shape(score_array)]
    agent = Agent(r)
    return agent

def update_agent(agent: Agent, global_best_position, w, agentC, globalC, dimensions) -> Agent:
    dist_to_agent_best = (agent.best_position - agent.position)
    dist_to_global_best = (global_best_position - agent.position)
    agent.velocity = ((w * agent.velocity) +
                        (agentC * np.random.rand(dimensions) * dist_to_agent_best) + 
                        (globalC * np.random.rand(dimensions) * dist_to_global_best))
    agent.position += agent.velocity.astype(int)
    return agent

def particle_swarm_single_function(score_array: ndarray, num_agents: int, agentC: float, globalC: float, w: float, v_max: float, max_iter: int = 10) -> ndarray:
    """
    - Partical Swarm Optimization in a single function
        Creates num_agent number of agents for the swam starting them at random points in the given score array.
        - Inputs:
            - `score_array: ndarray` - The score matrix
            - `num_agents: int` - the number of agents to add to the swarm
            - `agentC: float` - The constant value to multiply with the distance to personal best
                    Note: agentC > globalC results in high exploration, low exploitation.
            - `globalC: float` - The constant value to multiply with the distance to gloabal best
                    Note: globalC > agentC results in low exploration, high exploitation.
            - `w: float` - The interia weight to multiply the velocity by
            - `max_iter: int` - The maximum number of iterations to run
        - Outputs:
            - `ndarray` - The best global estimation
    """
    dimensions = score_array.ndim
    start_index = [np.random.randint(0, size) for size in np.shape(score_array)]
    global_best_position = start_index
    global_best_fitness = float('inf')
    max_score = np.max(score_array)
    agents = []
    for _ in range(num_agents):
        # Generating a random start index
        r = [np.random.randint(0, size) for size in np.shape(score_array)]
        a = Agent(r)
        agents.append(a)
        # Setting global best fitness and position to agents if agents best is better
        if a.best_fitness < global_best_fitness:
            global_best_fitness = a.best_fitness
            global_best_position = a.best_position

    # For given max number of iterations...
    for i in range(max_iter):
        # Updating each agenst velocity and position
        for agent in agents:
            dist_to_agent_best = (agent.best_position - agent.position)
            dist_to_global_best = (global_best_position - agent.position)
            agent.velocity = ((w * agent.velocity) +
                              (agentC * np.random.rand(dimensions) * dist_to_agent_best) + 
                              (globalC * np.random.rand(dimensions) * dist_to_global_best))
            agent.position += agent.velocity.astype(int)
            agent.index_in_bounds(np.shape(score_array))
            new_score = score_array[tuple(agent.position)]
            new_fitness = max_score - new_score

            if new_fitness < agent.best_fitness:
                agent.best_position = agent.position
                agent.best_fitness = new_fitness

                if new_fitness < global_best_fitness:
                    global_best_fitness = new_fitness
                    global_best_position = agent.best_position

    return global_best_position



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
    ndims = 3
    domain = Domain.normalized(ndims)
    grid = Grid([0.1] * ndims)

    sphere1 = ProbilisticSphere(Point(0, 0, 0), 0.2, 0.25)
    sphere2 = ProbilisticSphere(Point([0.5] * ndims), 0.3, 0.25)
    sphere3 = ProbilisticSphere(Point(0, 0, 0.8), 0.2, 0.25)
    scoreable = ProbilisticSphereCluster([sphere1, sphere2, sphere3])
    score_matrix = brute_force_grid_search(scoreable, domain, grid)

    max_score = np.max(score_matrix)
    max_score_positions = []
    for i, v in np.ndenumerate(score_matrix):
        if v == max_score:
            max_score_positions.append(i)
    print("Max score:", max_score)
    print("Max score positions:", max_score_positions)

    start_time = time.perf_counter()
    g_pos = particle_swarm_single_function(score_matrix, 10, 0, 2, 0.8, 0.5, 100)
    end_time = time.perf_counter()
    print("\nClosest position: ", g_pos)
    print("Value:", score_matrix[tuple(g_pos)])
    runtime = end_time - start_time
    print("Runtime =", runtime, "seconds")

    start_time2 = time.perf_counter()
    swarm = ParticleSwarmOptimization(score_matrix, 10, 0, 2, 0.8, 0.5, 100)
    g_pos2 = swarm.run_swarm()
    end_time2 = time.perf_counter()
    print("\nClosest position:", g_pos2)
    print("Value:", score_matrix[tuple(g_pos2)])
    runtime2 = end_time2 - start_time2
    print("Runtime =", runtime2, "seconds")

    start_time3 = time.perf_counter()
    g_pos3 = particle_swarm_split_functions(score_matrix, 10, 0, 2, 0.8, 0.5, 100)
    end_time3 = time.perf_counter()
    print("\nClosest position: ", g_pos3)
    print("Value:", score_matrix[tuple(g_pos3)])
    runtime3 = end_time3 - start_time3
    print("Runtime =", runtime3, "seconds")




    