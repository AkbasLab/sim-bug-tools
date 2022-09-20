from abc import abstractmethod
import sim_bug_tools.simulator as simulator
import sim_bug_tools.rng.lds.sequences as sequences
from sim_bug_tools.sumo import TraCIClient
import sim_bug_tools.utils as utils
import sim_bug_tools.structs as structs
from sim_bug_tools.rng.rrt import RapidlyExploringRandomTree

import numpy as np
import json

import traci
import os

FILE_DIR = os.path.dirname(os.path.abspath(__file__))


class TrafficLightRace(simulator.Simulator):

    def __init__(self, 
        sequence_generator : sequences.Sequence, 
        seed : int = 500,
        **kwargs):
        

        # Setup sequence
        self._vehicle_attribute_indices = [1, self.n_vehicles * self.n_veh_params + 1]
        self._tls_attribute_indices = [self.vehicle_attribute_indices[1]+1, self.vehicle_attribute_indices[1]+4]
        self._n_parameters = self.tls_attribute_indices[1]

        self._seq = sequence_generator(
            domain = structs.Domain([(0,1) for n in range(self.n_parameters)]),
            axes_names = ["dim%d" % n for n in range(self.n_parameters)],
            seed = seed
        )
        

        kwargs["domain"] = self.seq.domain
        super().__init__(**kwargs)



        # SUMO configuration
        map_dir = "%s" % FILE_DIR
        self._config = {
            # "gui" : False,
            "gui" : True,

            # Street network
            "--net-file" : "%s/tl-race.net.xml" % map_dir,

            # Logging
            "--error-log" : "%s/error-log.txt" % map_dir,

            # Traci Connection
            "--num-clients" : 1,
            "--remote-port" : 5522,

            # GUI Options
            "--delay" : 100,
            # "--start" : "--quit-on-end",

            # RNG
            "--seed" : 333
        }


        # Parameter Data
        self._parameter_names = []
        self._parameter_uom = []
        self._parameter_range = []
        self._parameter_granularity = []
        return

    @property
    def config(self) -> dict:
        """
        SUMO Configuration setting.
        """
        return self._config


    @property
    def client(self) -> TraCIClient:
        """
        TraCI CLient
        """
        return self._client

    @property
    def n_vehicles(self) -> int:
        """
        Number of vehicles in the race.
        """
        return 10

    @property
    def n_parameters(self) -> int:
        """
        Number of input parameters
        """
        return self._n_parameters

    @property
    def n_dim(self) -> int:
        """
        Alias for n_parameters.
        """
        return self._n_parameters

    @property
    def vehicle_id_prefix(self) -> str:
        """Vehicle prefix ID"""
        return "veh_"

    @property
    def route_id(self) -> str:
        """
        Route ID
        """
        return "r0"

    @property
    def seq(self) -> sequences.Sequence:
        """
        Sequence for sampling points
        """
        return self._seq

    @property
    def n_vehicles_added(self) -> int:
        """
        Number of vehicles added to simulation.
        """
        return self._n_vehicles_added

    @property
    def vehicles(self) -> list[str]:
        """
        List of vehicle ids in the sim.
        """
        return self._vehicles

    @property
    def tlsid(self) -> str:
        """
        TraCI Traffic Light ID
        """
        return self._tlid

    @property
    def n_veh_params(self) -> int:
        """
        Number of vehicle parameters
        """
        return 8

    @property
    def vehicle_attribute_indices(self) -> list[int, int]:
        """
        Indcies of the point used to generate the vehicle attributes.
        """
        return self._vehicle_attribute_indices

    @property
    def tls_attribute_indices(self) -> list[int, int]:
        """
        Indices of the point used to generate the tls attribute indices
        """
        return self._tls_attribute_indices

    @property
    def parameter_names(self) -> list[str]:
        """
        Names of each parameter.
        """
        return self._parameter_names    
    
    @property
    def parameter_uom(self) -> list[str]:
        """
        Unit of measurement for each parameter.
        """
        return self._parameter_uom
    
    @property
    def parameter_range(self) -> list[list[float,float]]:
        """
        Min, max of each parameter
        """
        return self._parameter_range

    @property
    def parameter_granularity(self) -> list[float]:
        """
        Granularity of each parameter
        """
        return self._parameter_granularity
    

    @abstractmethod
    def as_dict(self):
        d = {
            "config" : json.dumps(self.config),
            "n_vehicles" : self.n_vehicles,
            "n_parameters" : self.n_parameters,
            "n_dim" : self.n_dim,
            "vehicle_id_prefix" : self.vehicle_id_prefix,
            "route_id" : self.route_id,
            "seq" : self.seq.as_dict(),
            "n_vehicles_added" : self.n_vehicles_added,
            "vehicles" : self.vehicles,
            "tlsid" : self.tlsid,
            "n_vehicle_params" : self.n_veh_params,
            "vehicle_attribute_indices" : self.vehicle_attribute_indices,
            "tls_attribute_indices" : self.tls_attribute_indices,
            "parameter_names" : self.parameter_names,
            "parameter_uom" : self.parameter_uom,
            "parameter_range" : self.parameter_range,
            "parameter_granularity" : self.parameter_granularity
        }
        return utils.flatten_dicts([d, super().as_dict()])

    @staticmethod
    @abstractmethod
    def from_dict(d : dict, sim = None):
        """
        Create a simulator instance from a dictionary

        -- Parameters --
        d : dict
            Instance properties
        sim : Simulator (default = None)
            simulator to inherit properties. Will construct a new instance
            if None.
        """
        if sim is None:
            sim = TrafficLightRace(
                sequence_generator = sequences.RandomSequence,
                file_name = d["file_name"]
            )
        sim._seq = sequences.from_dict(d["seq"])
        

        return simulator.Simulator.from_dict(d, sim)



    def _add_vehicles(self, point : structs.Point) -> np.ndarray:
        """
        Adds vehicles to the client.

        -- Parameters --
        point : np.ndarray
            Point of shape (n_vehicles, n_vehicle_parameters)
        
        -- Return --
        np.ndarray
            Concrete values of the shape (n_vehicles, n_vehicle_parameters)
        """
        n_veh_params = self.n_veh_params
        concrete_params = np.array(
            [ self._add_veh(point[i*n_veh_params : i*n_veh_params+n_veh_params]) \
            for i in range(self.n_vehicles) ]
        ).flatten()
        return concrete_params
    
    def _add_veh(self, point : np.ndarray ) -> list[float]:
        """
        Add a vehicle

        --- Parameters --
        point : np.ndarray
            normal array of size (1,8)

        -- Return --
        np.ndarray
            Concrete values in a list of size (1,8)
        """
        vid = "%s%d" % (self.vehicle_id_prefix, self.n_vehicles_added)
        self.vehicles.append(vid)
        self._n_vehicles_added += 1

        traci.vehicle.add(vid, self.route_id)
        
        setters = [
            traci.vehicle.setLength,
            traci.vehicle.setWidth,
            traci.vehicle.setMinGap,
            traci.vehicle.setAccel,
            traci.vehicle.setDecel,
            traci.vehicle.setMaxSpeed,
        ]

        # These ranges are taken from the default SUMO vTypes
        attributes = np.array([
            [.215, 16.5, .25], # Length    (m)
            [.478, 2.55, .25], # Width     (m)
            [.25, 2.5, .25],   # Min Gap   (m)
            [1.1, 6., 0.00007716049382716],     # Accel     (m/s^2) by (1km/h^2)
            [2., 10., 0.00007716049382716],     # Decel     (m/s^2) by (1km/h^2)
            [5.4, 200, 1/3.6]                   # Max Speed (m/s) by (1 kmh)
        ])

        names_uom = [
            ["length", "m"],
            ["width", "m"],
            ["min_gap", "m"],
            ["accel", "m/s^2"],
            ["decel", "m/s^2"],
            ["max_speed", "m/s"]
        ]

        for i, attrib in enumerate(attributes):
            a,b,g = attrib
            name, uom = names_uom[i]
            self._parameter_names.append("%s_%s" % (vid, name))
            self._parameter_uom.append(uom)
            self._parameter_range.append([a,b])
            self._parameter_granularity.append(g)


        #  Set concrete Values
        concrete_values = [utils.project(
                attrib[0], attrib[1], point[i], by=attrib[2]) \
            for i, attrib in enumerate(attributes)]

        [setters[i](vid, val) for i, val in enumerate(concrete_values)]

        
        # Emergency decel must be greater than decel. 
        a = concrete_values[-2]
        b = 10.
        g = 0.00007716049382716
        emergency_decel = utils.project(a, b, point[6], by=g)
        traci.vehicle.setEmergencyDecel(vid, emergency_decel)
        concrete_values.append(emergency_decel)

        self._parameter_names.append("%s_emergency_decel" % vid)
        self._parameter_uom.append("m/s^2")
        self._parameter_range.append(["%s_decel" % vid,b])
        self._parameter_granularity.append(g)

        
        # Set initial speed between 0 and the max speed.
        a = 0 
        b = concrete_values[-1]
        g = 1/3.6
        initial_speed = utils.project(a, b, point[7], by=g)
        traci.vehicle.setSpeed(vid, initial_speed)
        concrete_values.append(initial_speed)

        self._parameter_names.append("%s_initial_speed" % vid)
        self._parameter_uom.append("m/s")
        self._parameter_range.append([a,"%s_max_speed" % vid])
        self._parameter_granularity.append(g)


        return np.array(concrete_values)


    def _init_traffic_light(self, point : structs.Point) -> np.ndarray:
        """
        Initilize traffic light state

        -- Parameters --
        point : Point
            Point of shape (1,3)

        -- Return --
        np.ndarray of shape (1,3)
            Concrete values
        """
        durations = [
            [1., 30.], # Green to Yellow (s) 
            [0., 10.], # Yellow to Red (s)
            [1., 30.]  # Red to Gree (s)
        ]
        by = 1. #1 s
        names = ["GY","YR","RG"]

        for i, dur in enumerate(durations):
            a,b = dur
            g = by
            name = "tls_%s" % (names[i])
            uom = "s"
            self._parameter_names.append(name)
            self._parameter_uom.append(uom)
            self._parameter_range.append([a,b])
            self._parameter_granularity.append(g)
            

        concrete_durations = [utils.project(*dur, point[i], by=by) \
            for i, dur in enumerate(durations)]

        states = ["GG", "YY", "rr"]

        phases = [traci.trafficlight.Phase(
            dur, states[i], minDur = dur, maxDur = dur) \
                for i, dur in enumerate(concrete_durations)]

        traci.trafficlight.setProgramLogic(
            self.tlsid,
            traci.trafficlight.Logic(
                "tls0", 
                type=0, 
                currentPhaseIndex=0, 
                phases = phases
            )
        )
        return np.array(concrete_durations)

    def _run_until_emergency_stop(self) -> bool:
        return 

    def _run_sumo_scenario(self, 
            point : structs.Point) -> list[bool, structs.Point]:
        """
        Generate and run a single SUMO scenario

        -- Parameter --
        point : structs.Point
            Normal point, which is used to select the concrete parameters
        
        -- Return --
        list[bool, structs.Point]
            pos 0 : if a collision was observed in the scene
            pos 1 : concrete parameters as a point
        """
        # Clear parameter field
        self._parameter_names = []
        self._parameter_uom = []
        self._parameter_range = [] 
        self._parameter_granularity = []

        # Start the client
        self._client = TraCIClient(self.config)
        self._tlid = traci.trafficlight.getIDList()[0]
        
        # Add the route
        traci.route.add(self.route_id, ["before_tl","after_tl"])

        # Add vehicles
        self._vehicles = []
        self._n_vehicles_added = 0

        # Add Vehicles
        concrete_vehicle_params = self._add_vehicles(
            point[self.vehicle_attribute_indices[0] : self.vehicle_attribute_indices[1]])
        
        # Traffic Light
        concrete_tl_params = self._init_traffic_light(
            point[self.tls_attribute_indices[0] : self.tls_attribute_indices[1]])
        
        # Combine into the concrete parameters
        concrete_params = structs.Point(
            np.concatenate([concrete_vehicle_params, concrete_tl_params])
        )

        # Run simulation and observe an emergency stop
        collision_observed = False
        while traci.simulation.getMinExpectedNumber() > 0:
            if traci.simulation.getCollidingVehiclesNumber():
                collision_observed = True
                break
            traci.simulationStep()

        # Sim complete
        self.client.close()
        return collision_observed, concrete_params

    def long_walk_on_update(self):
        point = self.seq.get_points(1)[0]
        is_collision, params = self._run_sumo_scenario(point)
        
        return super().long_walk_on_update(
            point_normal = point, 
            point_concrete = params, 
            is_bug = is_collision)




class TrafficLightRaceRRT(TrafficLightRace):
    """
    A simple simulator with known bugs.
    Samples from a normal domain.
    RRT Version
    """
    def __init__(self, 
        sequence_generator : sequences.Sequence,
        rrt : RapidlyExploringRandomTree,
        n_branches : int,
        seed : int = 500,
        **kwargs
    ):
        super().__init__(sequence_generator, seed, **kwargs)

        self._rrt = rrt
        self._n_branches = n_branches
        self._n_branches_remaining = n_branches
        return

    @property
    def rrt(self) -> RapidlyExploringRandomTree:
        """
        A Rapidly Exploring Random Tree
        """
        return self._rrt

    @property
    def n_branches(self) -> int:
        """
        The number of branches the RRT will grow before resetting to 0 branches.
        """
        return self._n_branches

    @property
    def n_branches_remaining(self) -> int:
        """
        The number of branches left for the RRT to grow.
        """
        return self._n_branches_remaining

    def as_dict(self) -> dict:
        d = {
            "rrt" : self.rrt.as_dict(),
            "n_branches" : self.n_branches,
            "n_branches_remaining" : self.n_branches_remaining
        }
        return utils.flatten_dicts([d, super().as_dict()])

    @staticmethod
    @abstractmethod
    def from_dict(d : dict, sim = None):
        """
        Create a simulator instance from a dictionary

        -- Parameters --
        d : dict
            Instance properties
        sim : Simulator (default = None)
            simulator to inherit properties. Will construct a new instance
            if None.
        """
        if sim is None:
            sim = TrafficLightRaceRRT(
                sequence_generator = sequences.RandomSequence,
                rrt = RapidlyExploringRandomTree.from_dict(d["rrt"]),
                n_branches = int(d["n_branches"])
            )
        sim._n_branches_remaining = int(d["n_branches_remaining"])
        return TrafficLightRace.from_dict(d, sim)
    

    def long_walk_to_local_search_on_enter(self):
        """
        The RRT is reset and centered on the last observed point.
        """
        # Reset the RRT
        self.rrt.reset(self.last_observed_point_normal)
        self._n_branches_remaining = self.n_branches
        return

    def local_search_on_update(self):
        """
        Local Exploration using the RRT for point selection, until the
        specified amount of branches are grown.
        """
        # Generate the next point
        point = self.rrt.step()[2]

        # Check if it's in the bug profile
        is_collision, params = self._run_sumo_scenario(point)

        # Reduce branches remaining.
        self._n_branches_remaining -= 1

        #  Call parent function
        return super().local_search_on_update(
            point_normal = point, 
            point_concrete = params, 
            is_bug = is_collision)
        
    
    def local_search_to_long_walk_trigger(self) -> bool:
        """
        The local search ends when the RRT does not need to grow anymore
        branches.
        """
        return self.n_branches_remaining <= 0

    def local_search_to_paused_trigger(self) -> bool:
        """
        When the sim is complete (implied in base class), and there are no
        branches remaining. Transition to Paused State.
        """
        return self.n_branches_remaining <= 0

    def long_walk_to_local_search_trigger(self) -> bool:
        """
        Move to local search when a bug is observed
        """
        return self.last_observed_point_is_bug
    
