"""
Swarming Optimization Algorithms
"""

import numpy as np
import time
import random
from numpy import ndarray
from tkinter import *
from sim_bug_tools.structs import Point, Domain, Grid
from sim_bug_tools.simulation.simulation_core import Scorable, Graded
import matplotlib.pyplot as plt
from sim_bug_tools.graphics import Grapher
from rtree import index
import abc


class SwarmAgent():
    
    DEFAULT_AGENT_PARAMS = { 
        "max_v": None, 
        "teleporting_agent": False,
        "max_teleports": 5,
        "stop_at_envelope": True
    }
    
    """
    - ParticleSwarmAgent
        Creates a single agent for partical swarm optimization
        - Inputs:
            - `scorable: Scorable` - The scorable object for the agent to use
            - `domain_shape: ndarray` - The shape of the agent search space
            - `agent_params: dict=None` - Dictionary with values of:
                - `"max_v": float=None` - The maximum velocity the agent can move by. Defaulted to None.
                - `"teleporting": bool=False` - determines if the agent will teleport if the velocity is 0. Defaulted to not teleport. 
                - `"max_teleports": int=5` - Maximum number of teleports an agent can do.
                - `"stop_at_envelope": bool=True` - True means agent stops moving once envelope is reached. False means agent will continue exploration into envelope attempting to maximize score.
    """
    def __init__(self, scorable: Scorable, domain_shape: ndarray, agent_params: dict=None):
        self.scorable = scorable
        self.domain_shape = domain_shape

        self.agent_stats = {
            "best_points": [],
            "num_teleports": 0,
            "num_of_steps": 0,
            "still_moving": True,
            "move_towards_score": [],
            "all_points": [],
            "num_samples": 0,
            "teleport_at": []
        }
        """ 
        agent_stats: Dictionary containing variables that affect the agents
        - `"best_points": list[Point]` - Best points are stored here if the agent teleports or no longer moves. 
        - `"num_teleports": int` - Keeps track of the number of teleports made by the agent
        - `"num_of_steps": int` - Keeps track of the number of steps (number of times they moved) made by the agent.
        - `"still_moving": bool` - True means agent is still moving. False means the agent is no longer moving.
        - `"move_towards_score": list[bool]` - Tracks each move if the agent moved to a higher score (True), or towards a lower score (False). Tracks each move.
        - `"all_points": list[Point]` - New agent point is added each time agent is updated
        - `"num_samples": int` - The number of samples the agent takes. Incremented each time a new potential point is evaluated with the score() method
        - `"teleport_at": list[int]` - What number point in all_points a teleport happened
        """
        
        # Setting the agent parameters
        self.agent_params = self.DEFAULT_AGENT_PARAMS
        if agent_params is not None:
            self.agent_params.update(agent_params)

        initial_position = self.random_position()
        self.point = Point(initial_position)
        self.best_pt = self.point
        self.velocity = np.array([0.0 for _ in self.domain_shape])
        self.s = True
        return
        
    def random_position(self) -> ndarray:
        """Returns an ndarray of a position between 0 and i for each position in space_shape"""
        return [random.uniform(0,i) for i in self.domain_shape]

    def teleport(self):
        """ Resets attributes and point using a random position for teleporting agents """
        if not self.agent_stats["num_teleports"] < self.agent_params["max_teleports"]:
            return            
        new_position = self.random_position()
        self.point = Point(new_position)
        if self.scorable.classify(self.best_pt) or len(self.agent_stats["best_points"]) == 0:
            self.agent_stats["best_points"].append(self.best_pt)
        self.best_pt = self.point
        self.agent_stats["still_moving"] = True
        self.agent_stats["num_teleports"] += 1
        self.agent_stats["teleport_at"].append(len(self.agent_stats["all_points"]))
        self.velocity = np.array([0.0 for _ in self.domain_shape])
        self.s = True
        return
    
    def calculate_velocity(self) -> ndarray:
        print("!! PARENT calculate_velocity FUNCTION CALLED FOR SWARM AGENT !!")
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

    def update_agent(self, new_v: ndarray):
        """ Takes the new velocity calculated in the child agent class and updates the agent """
        # If the new velocity is None, default to Swarm Agent velocity
        self.agent_stats["all_points"].append(self.point)
        # Limiting the velocity if there is a max_v set:
        if self.agent_params["max_v"] is not None:
            new_v[new_v > self.agent_params["max_v"]] = self.agent_params["max_v"]
        
        if self.scorable.classify(self.point) and self.agent_params["stop_at_envelope"]:
            if self.agent_params["teleporting_agent"]:
                self.teleport()
            return
        # If check_velocity returns false or the agent is in an envelope, do not update agent
        if not self.check_velocity(new_v): return

        # Updating the agents attributes based on new movement
        self.agent_stats["still_moving"] = True
        new_p = self.point.array + new_v
        self.velocity = new_v
        new_point = Point(new_p)
        # Setting point to new point while ensuring its in bounds
        self.point = self.point_in_bounds(new_point)

        if self.compare_score(self.best_pt, new_point): 
            self.best_pt = new_point

        self.agent_stats["num_of_steps"] += 1        
        return
    
    def check_velocity(self, new_v) -> bool:
        """ If the new velocity is 0, update as necessary """
        # If there are no nonzero elements in the new velocity, the agent is not moving and nothing is updated
        if np.count_nonzero(new_v) == 0:
            self.agent_stats["still_moving"] = False
            if self.s:
                self.s = False
            return False
        
        return True


       
    def point_in_bounds(self, point: Point) -> Point:
        """ Determines if the new point is in the space shape, if not, puts it in just inside the limits """
        index = point.array
        for i, idx in enumerate(index):
            if idx < 0:
                index[i] = 0
            if idx >= self.domain_shape[i]:
                index[i] = self.domain_shape[i] - 0.1
        point = Point(index)
        return point
    
    def set_final_stats(self):
        """ Setting the final stats of the agent. Returns the updated agent_stats dictionary """
        if self.scorable.classify(self.best_pt) or len(self.agent_stats["best_points"]) == 0 :
            self.agent_stats["best_points"].append(self.best_pt)
        a_scores = self.scorable.v_score(np.array(self.agent_stats["best_points"]))
        self.agent_stats["best_scores"] = a_scores

        move_colors: list[str] = ["black"]
        move_towards_score: list[bool] = [True]
        all_pts = self.agent_stats["all_points"]
        prev_pt = all_pts[0]
        for i, pt in enumerate(all_pts):
            if i == 0:
                continue
            curr_higher = self.compare_score(prev_pt, pt)
            if curr_higher:
                move_colors.append("green")
                move_towards_score.append(True)
            else:
                move_colors.append("red")
                move_towards_score.append(False)
            if (pt.array == prev_pt.array).all():
                move_colors.append("orange")
            if self.agent_params["teleporting_agent"]:
                if i in self.agent_stats["teleport_at"]:
                    move_colors.append("blue")
        
        self.agent_stats["movement_colors"] = move_colors
        self.agent_stats["move_towards_score"] = move_towards_score

        return self.agent_stats

class StandardPSOAgent(SwarmAgent):
    """
    StandardPSOAgent - Moves a a standard Particle Swarm Optimization particle. 
    Uses the equation: vi = w*vi-1 + c1*r1(agent_best - current_pos) + c2*r2(global_best - current_pos)
    
    Parameters:
    - `scorable: Scorable` - Scorable object being evaluated
    - `domain_shape: ndarray` - Shape of the domain. aka length of each dimension. 
    - `agent_params: dict` - See SwarmAgent for general default values. 
        - Default StandardPSOAgent Params:
            - `agentC: float = 2.05` - Aka c1. Constant multiplied by the distance to agents best position.
            - `globalC: float = 2.05` - Aka c2. Constant multiplied by the distance from agent position to global best position
            - `w: float = 0.72984` - Multiplied by current agent velocity to calculate inertia.
        - Values were chosen based on optimal values cited in this article: https://towardsdatascience.com/particle-swarm-optimization-visually-explained-46289eeb2e14
    """

    DEFAULT_SPO_PARAMS = {
        "agentC": 2.05, 
        "globalC": 2.05, 
        "w": 0.72984,
    }

    def __init__(self, scorable: Scorable, domain_shape: ndarray, agent_params: dict):        
        super().__init__(scorable, domain_shape, agent_params)
        self.agent_params.update(self.DEFAULT_SPO_PARAMS)
    
    def calculate_velocity(self) -> ndarray:
        """ Calculates the velocity for the next point using the Particle Swarm Optimization equation """
        current_p = self.point.array
        # Caluclating distances:
        dist_to_agent_best = (self.best_pt.array - current_p)
        dist_to_global_best = (self.global_best_position - current_p)
        # Caluclating Inertia: 
        inertia = self.agent_params["w"] * self.velocity
        # Calculating the personal_best vector:
        personal_best = (self.agent_params["agentC"] * random.uniform(0,1)) * dist_to_agent_best
        # Calculating the global best vector:
        global_best = (self.agent_params["globalC"] * random.uniform(0,1) * dist_to_global_best)
        # New velocity:
        new_v = (inertia + personal_best + global_best)
        self.agent_stats["num_samples"] += 1
        return new_v

    def update_agent(self):
        """ Sends the velocity caluclated in `calculate_velocity()` to the `update_agent()` method in SwarmAgent """
        # Calculating the new velocity using the new global best position
        new_v = self.calculate_velocity()
        super().update_agent(new_v)
        return

   
class AntSwarmAgent(SwarmAgent):
    
    def __init__(self, scorable: Scorable, domain_shape: ndarray, agent_params: dict=None):
        """
        Ant Swarm Agent: 
            Moves independently from other agents. Child class of SwarmAgent
        Parameters:
        - `scorable: Scorable` - See `SwarmAgent` details.
        - `domain_shape: ndarray` - See SwarmAgent details.
        - `agent_params: dict=None` - Has same defaulted values as SwarmAgent plus the following:
            - `search_radius: float = len(domain_shape)*0.01` - Search radius around the current point to take sample points from
            - `scatter_samples: int = len(domain_shape)` - Number of sample points to take in search radius
            - `scatter_retries: int = 10` - Number of scatter sample retires to get a higher score
            - `scatter_always: bool = False` - If True, agent scatters for better score/point each move. Else, agent velocity
            will remain the same until the new position score is <= current score
        """
        default_ant_params = {
            "search_radius": len(domain_shape) * 0.01,
            "scatter_samples": len(domain_shape),
            "scatter_retries": 10,
            "scatter_always": False
        }
        if agent_params is not None:
            default_ant_params.update(agent_params)
        super().__init__(scorable, domain_shape, default_ant_params)

        # Attempts at a higher score in the area
        self.attempts: int = 0
        self.velocity = self.scatter_search()
        return
    
    def update_agent(self):
        """ 
        If the agent_param `"scatter_always"` is set to true, the agent will scatter everytime it is updated.
         Otherwise, the new velocity remains the same until the new point score is lower than the current score.
         The new velocity is sent to the SwarmAgent `update_agent()` method  """
        if self.agent_params["scatter_always"]:
            # Velocity is determined by a scatter search each time
            self.velocity = self.scatter_search()
        else:
            # Velocity is only updated if the new point is less than or equal to the current point
            new_pos = self.point.array + self.velocity
            if self.scorable.score(Point(new_pos)) <= self.scorable.score(self.point):
                self.velocity = self.scatter_search()
        return super().update_agent(self.velocity)
    
    def scatter_search(self):
        """
        Samples `"search_scatter"` number of samples in `"search_radius"` for `scatter_retries` until 
        a higher score than the current point is found
        """
        best_score = -1
        curr_score = self.scorable.score(self.point)
        while (best_score < curr_score) and self.attempts < self.agent_params["scatter_retries"]:
            best_pt, best_score = self.new_scatter()
            new_v = best_pt.array - self.point.array
            self.attempts += 1
        if best_score < curr_score:
            new_v = [0 for _ in self.point.array]
            self.attempts = 0
        return new_v
    
    def new_scatter(self):
        """ Randmoizes `"scatter_samples"` number of points in `"search_radius"`. Returns the best point and best score from the scatter. """
        r = self.agent_params["search_radius"]
        curr_pos = self.point.array
        best_pt = None
        best_score = -1
        for _ in range(self.agent_params["scatter_samples"]):
            new_pt = Point([random.uniform(i-r,i+r) for i in curr_pos])
            new_score = self.scorable.score(new_pt)
            self.agent_stats["num_samples"] += 1
            if new_score > best_score:
                best_pt = new_pt
                best_score = new_score
        return best_pt, best_score



class ParticleSwarmOptimization():    
    """
    ParticleSwarmOptimization - Particle Swarm Optimization finding the best scores in given domain
        Creates num_agent number of agents for the swam starting them at random points in the given score array.
    Parameters:
    - `scorable: Scorable` - Scorable object
    - `domain: Domain` - Domain of the search area
    - `swarm_params: dict = None` - Dictionary for the swarm constants containing:
        - `num_agents: int = 100 or ndims*10` - The number of agents in the swarm
        - `max_iter: int = 100` - Maximum number of iterations
        - `agent_type: SwarmAgent = AntSwarmAgent` - The type of agents in the swamr. Defaulted to AntSwarmAgent
    - `agent_params: dict = None` - Dictionary containing adjustable agent constants.
    
    Agent parameters (agent_params) vary per agent type. 
    - All agents (SwarmAgent) parameters:
        - `"max_v": float = None` - Max velocity an agent can move by
        - `"teleporting_agent": bool = False` - True means agent will teleport if velocity is 0
        - `"max_teleports": int = 3` - Maximum number of teleports an agent can make.
        - `"stop_at_envelope": bool = True` - True means agents will stop exploration once envelope is reached. False means they will continue exploration to find max score.
        Keep this set to True when using a StandardPSOAgent to catch more envelopes.
        
    - StandardPSOAgent Parameters:
        - `"agentC": float` - *Constant multiplied by the agent distance to personal best position.
        - `"globalC": float` - **Constant multiplied by the agent distance to global best position.
        - `"w": float` - Constant multiplied by velocity to obtain inertia.
        * agentC > globalC -> Exploration is higher, exploitation is lower.
        ** globalC = 0 -> Agents will not communicate. 

    - AntSwarmAgent Parameters:
        - `"search_radius": float` - Radius of the scatter search area to obtain a new point
        - `"scatter_samples": int` - Number of samples to take in new scatter search
        - `"scatter_retries": int` - Number of scatter retries to obtain better score
        - `"scatter_always": bool` - True means agent will scatter for better point every move. False means agent will use same velocity until there is a decrease in the score.

    The following are tracked in `swarm_stats` dictionary:
    - `"total_steps": int` - total number of steps taken by all agents
    - `"global_best_point": Point` - The global best point across all agents
    - `"global_best_score": float` - The best score at the global best point
    - `"global_best_classification": bool` - Classification at the global best point
    - `"avg_dir_to_score": bool` - average direction towards score across all agents. True is towards higher score, False is towards lower score.
    - `"average_steps": float` - Average number of steps taken per agents
    - `"iterations": int` - Number of iterations of the entire swarm
    - `"swarm_runtime": float` - How long the actual movement of the agents in the swarm took in seconds
    - `"total_samples": int` - The total number of samples taken. Each time a new point or potential point was evaluated.
    - `"spheres_found": list[sphere#, #ofSamples]` - A list of the sphere number and how many samples were in that sphere
    - `"best_scores_range": list[min_score, max_score]` - The minimum and maximum of the best scores found across all agents
    - `"post_process_runtime": float` - How long setting the final stats took in seconds.
    - `"total_teleports": int` - Total number of teleports made if teleporting agents exist.
    - `"best_points": list[Points]" - The best points found by all agents
            
    """
    def __init__(self, scorable: Scorable, domain: Domain, swarm_params: dict=None, agent_params: dict=None) -> None:
        
        self.scorable: Scorable = scorable
        self.domain_shape = domain.dimensions

        self.agents: list[SwarmAgent] = []
        self.swarm_params = {
            "num_agents": 100, 
            "max_iter": 100, 
            "agent_type": AntSwarmAgent
        }
        if swarm_params is not None:
            self.swarm_params.update(swarm_params)
        elif len(domain.dimensions) >= 10:
            self.swarm_params["num_agents"] = len(domain.dimensions)*10

        self.agent_params = agent_params
        """
        Agent parameters (agent_params) vary per agent type. 
        SwarmAgent parameters:
        - `"max_v": float` - Max velocity an agent can move by
        - `"teleporting_agent": bool` - True means agent will teleport if velocity is 0
        - `"max_teleports": int` - Maximum number of teleports an agent can make.
        
        StandardPSOAgent Parameters:
        - `"agentC": float` - Constant multiplied by the agent distance to personal best position
        - `"globalC": float` - Constant multiplied by the agent distance to global best position
        - `"w": float` - Constant multiplied by velocity to obtain inertia
        
        AntSwarmAgent Parameters:
        - `"search_radius": float` - Radius of the scatter search area to obtain a new point
        - `"scatter_samples": int` - Number of samples to take in new scatter search
        - `"scatter_retries": int` - Number of scatter retries to obtain better score
        - `"scatter_always": bool` - True means agent will scatter for better point every move. False means agent will use same velocity until there is a decrease in the score.
        """

        self.global_best_point: Point
        self.swarm_stats = {
            "total_steps": int, # total number of steps taken by all agents
            "global_best_point": Point, # The global best point across all agents
            "global_best_score": float, # The best score at the global best point
            "global_best_classification": bool, # Classification at the global best point
            "avg_dir_to_score": bool, # average direction towards score across all agents. True is towards higher score, False is towards lower score.
            "average_steps": float, # Average number of steps taken per agents
            "iterations": int, # Number of iterations of the entire swarm
        }
        """
        swarm_stats dictionary:
        - `"total_steps": int` - total number of steps taken by all agents
        - `"global_best_point": Point` - The global best point across all agents
        - `"global_best_score": float` - The best score at the global best point
        - `"global_best_classification": bool` - Classification at the global best point
        - `"avg_dir_to_score": bool` - average direction towards score across all agents. True is towards higher score, False is towards lower score.
        - `"average_steps": float` - Average number of steps taken per agents
        - `"iterations": int` - Number of iterations of the entire swarm
        - `"swarm_runtime": float` - How long the actual movement of the agents in the swarm took in seconds
        - `"total_samples": int` - The total number of samples taken. Each time a new point or potential point was evaluated.
        - `"spheres_found": list[sphere#, #ofSamples]` - A list of the sphere number and how many samples were in that sphere
        - `"best_scores_range": list[min_score, max_score]` - The minimum and maximum of the best scores found across all agents
        - `"post_process_runtime": float` - How long setting the final stats took in seconds.
        - `"total_teleports": int` - Total number of teleports made if teleporting agents exist.
        - `"best_points": list[Points]" - The best points found by all agents
        """
        
        self.initialize_agents()
        self.agent_params = self.agents[0].agent_params

    def initialize_agents(self):
        """ Initializing the agents of the agent_class parameter type for the swarm """
        # Setting global best point:
        index = [random.uniform(0,i) for i in self.domain_shape]
        start_point = Point(index)
        self.global_best_point = start_point

        # For the input number of agents,
        for _ in range(self.swarm_params["num_agents"]):
            # Create an agent of agent_class
            a = self.swarm_params["agent_type"](self.scorable, self.domain_shape, self.agent_params)
            # Checking agent score vs. global best score
            self.check_score(a)
            # Saving the agent
            self.agents.append(a)
        return
        
    def check_score(self, agent: SwarmAgent):
        """ Given a SwarmAgent agent, sets global best point to agent point if the agent score is higher than global best score"""
        a_score = self.scorable.score(agent.point)
        g_score = self.scorable.score(self.global_best_point)
        if a_score > g_score:
            self.global_best_point = agent.point
        return


    def single_iteration(self) -> bool:
        """ Moves all agents in swarm a single time. Saves the agents iteration points and velocities used for graphing and final data """
        still_moving = False
        for agent in self.agents:
            # Setting agents global best position to the global best point position
            agent.global_best_position = self.global_best_point.array
            # Updating the agent (all depends on the child agent classes to calculate velocity and new position)
            agent.update_agent()
            self.check_score(agent)
            if agent.agent_stats["still_moving"] and not self.scorable.classify(agent.point):
                still_moving = True
            if not self.scorable.classify(agent.point):
                still_moving = True

        return still_moving
        
    def run_swarm(self, max_iter: int = None):
        """ Running the entire swarm for max_iter number of iterations or until all agents have stopped moving """
        if max_iter is None:
            max_iter = self.swarm_params["max_iter"]
        start_time = time.perf_counter()
        
        # For given max number of iterations...
        i = 0
        cont = True
        while (i < max_iter) and cont:
            # cont = False when all agents are not moving
            cont = self.single_iteration()
            i += 1
        
        end_time = time.perf_counter()
        self.swarm_stats["swarm_runtime"] = end_time - start_time
        self.swarm_stats["iterations"] = i
        # Setting the final stats and calculating runtime to do so
        s_time = time.perf_counter()
        self.set_final_stats()
        e_time = time.perf_counter()
        self.swarm_stats["post_process_runtime"] = e_time - s_time
        return
    
    def graph_swarm_3D(self, g: Grapher):
        """ Graphs 3D swarm with points """
        agent_points = []
        for i in range(self.swarm_stats["iterations"]):
            i_pts = []
            i_clrs = []
            for agent in self.agents:
                i_clrs.append(agent.agent_stats["movement_colors"][i])
                i_pts.append(agent.agent_stats["all_points"][i])
            _element = g.plot_all_points(i_pts, color=i_clrs)
            plt.pause(0.1)
            _element.remove()
            agent_points = i_pts
                
        # Graphs best points  
        g.plot_all_points(self.swarm_stats["best_points"], color="purple")
        plt.show()

    def set_final_stats(self):
        """ Sets all the final stats for sacing to file or printing """
        total_steps = 0
        all_move = 0
        self.swarm_stats["global_best_score"] = self.scorable.score(self.global_best_point)
        self.swarm_stats["global_best_classification"] = self.scorable.classify_score(self.swarm_stats["global_best_score"])
        best_scores = []
        best_points = []
        total_samples = 0
        total_teleports = 0
        for agent in self.agents:
            a_stats = agent.set_final_stats()
            for a in a_stats["best_scores"]:
                best_scores.append(a)
            for b in a_stats["best_points"]:
                best_points.append(b)
            total_steps += agent.agent_stats["num_of_steps"]
            total_samples += agent.agent_stats["num_samples"]
            if agent.agent_params["teleporting_agent"]: 
                total_teleports += agent.agent_stats["num_teleports"]
            if a_stats["move_towards_score"].count(True) > a_stats["move_towards_score"].count(False):
                all_move += 1
            else:
                all_move -= 1
        
        ############# ONLY WORKS WHEN SCORABLE IS MADE OF UP SPHERES #################
        try:
            spheres_found = []
            sphere_num = 1
            for sphere in self.scorable.spheres:
                samples = 0
                for pt in best_points:
                    if sphere.classify(pt):
                        samples += 1
                spheres_found.append([sphere_num, samples])
                sphere_num += 1
        except Exception as e:
            print(e)
        #############################################################################
        if agent.agent_params["teleporting_agent"]: 
            self.swarm_stats["total_teleports"] = total_teleports
        else:
            self.swarm_stats["total_teleports"] = "N/A"
        self.swarm_stats["total_samples"] = total_samples
        self.swarm_stats["spheres_found"] = spheres_found
        self.swarm_stats["best_scores_range"] = [min(best_scores), max(best_scores)]
        self.swarm_stats["avg_dir_to_score"] = (all_move >= 0)
        self.swarm_stats["total_steps"] = total_steps
        self.swarm_stats["average_steps"] = total_steps / self.swarm_params["num_agents"]
        self.swarm_stats["best_points"] = best_points

    def print_swarm_stats(self):
        """ Printing swarm dictionaries containing parameters and stats to console. """
        print("Swarm final stats:")
        print("- Swarm Parameters:", self.swarm_params)
        print("- Agent Parameters:", self.agent_params)
        for key, val in zip(self.swarm_stats.keys(), self.swarm_stats.values()):
            print("-", key, ":", str(val))
        return

    def store_to_file(self, filename="PSO-Test", mode='w', other_stats: dict=None):
        """ Storing swarm parameters, agent parameters, swarm stats, and other stats (optional input). to `filename` txt file. """
        try:    
            # Creating and writing to a new file
            file = open(filename, mode=mode)
            file.write("Swarm final stats:")
            file.write("\n- Swarm parameters: " + str(self.swarm_params))
            file.write("\n- Agent parameters: " + str(self.agent_params))
            for key, val in zip(self.swarm_stats.keys(), self.swarm_stats.values()):
                file.write("\n- " + key + ": " + str(val))
            if other_stats is not None:
                for key, val in zip(other_stats.keys(), other_stats.values()):
                    file.write("\n- " + key + ": " + str(val))
            file.close()
        except Exception as e:
            print(e)
        return

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

        radius = random.random()*(0.1)
        if radius > lam:
            radius = lam
        sphere_info[("Sphere #"+str(i))] = ("\n  - " + str(pt) + "\n  - radius: " + str(radius) + "\n  - lambda: " + str(lam))
        sphere = ProbilisticSphere(pt, radius, lam)
        if g is not None and ndims < 4:
            g._draw_3d_sphere(pt, radius)
        spheres.append(sphere)
    scoreable = ProbilisticSphereCluster(spheres)
    return scoreable, sphere_info

def test_PSO_no_graph(ndims: int=3, test_num=None, filename:str=None):
    
    domain = Domain.normalized(ndims)
    # scorable, run_info = random_sphere_scorable(num_of_spheres=5, ndims=ndims)
    scorable, run_info = set_spheres_scorable(ndims=ndims)
    p: str = "C:/Users/User/OneDrive/AI Validation Research/Swarming/PSO Tests - Personal"

    run_info["ndims"] = ndims
    agent_p = {"globalC": 0, "teleporting_agent": True}
    agent_type = StandardPSOAgent
    run_info["Agent_type"] = agent_type
    swarm_p = {"agent_type": agent_type, "num_agents": ndims*20}
    start_time = time.perf_counter()
    swarm = ParticleSwarmOptimization(scorable, domain, swarm_params=swarm_p, agent_params=agent_p)
    # swarm = ParticleSwarmOptimization(scorable, domain, swarm_params=swarm_params)
    # swarm = ParticleSwarmOptimization(scorable, domain)
    swarm.run_swarm()
    end_time = time.perf_counter()
    run_info["total_runtime"] = end_time - start_time
    if test_num is not None:
        filename = p+ "\Run "+ str(test_num) + "_" + str(ndims) + "D_StandardPSOAgent_gC0_teleporting_setSpheres.txt"
        swarm.store_to_file(filename=filename, other_stats=run_info)
    print("\n\nDimensions:", ndims)
    print("Runtime:", swarm.swarm_stats["swarm_runtime"])
    print("Total runtime:", run_info["total_runtime"])
    print("Spheres found:", swarm.swarm_stats["spheres_found"])
    # print("Sphere info:", str(run_info))
    # swarm.print_swarm_stats()

def test_particle_swarm_point():
    # scoreable: Scorable, domain: Domain, score_matrix: ndarray
    ndims = 3
    domain = Domain.normalized(ndims)
    scale = 0.05
    grid = Grid([scale]*ndims)
    g = Grapher(True, domain)

    # Getting scorable from random spheres or from set spheres
    # scorable, run_info = random_sphere_scorable(g, num_of_spheres=3)
    scorable, run_info = set_spheres_scorable(g)

    run_info["ndims"] = ndims
    run_info["Domain"] = domain
    run_info["grid"] = "Grid([" + str(scale) + "]*ndims)"

    agent_params = {"globalC": 0, "teleporting_agent": True}
    agent_type = StandardPSOAgent
    run_info["Agent_type"] = agent_type
    swarm_p = {"agent_type": agent_type}
    # swarm = ParticleSwarmOptimization(scorable, domain, swarm_params=swarm_params, agent_params=agent_params)
    # swarm = ParticleSwarmOptimization(scorable, domain, agent_params=agent_params)
    swarm = ParticleSwarmOptimization(scorable, domain, swarm_params=swarm_p, agent_params=agent_params)
    swarm.run_swarm()
    swarm.store_to_file(other_stats=run_info)
    swarm.print_swarm_stats()
    swarm.graph_swarm_3D(g)
    return


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


if __name__ == "__main__":
    # test_particle_swarm_point()
    test_PSO_no_graph()
    
    
    

    
    
    
    



    
    




    