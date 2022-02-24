import sim_bug_tools.simulator as simulator
import sim_bug_tools.rng.lds.sequences as sequences
from sim_bug_tools.sumo import TraCIClient
import sim_bug_tools.utils as utils
import sim_bug_tools.structs as structs
import numpy as np

import traci
import os

FILE_DIR = os.path.dirname(os.path.abspath(__file__))


class TrafficLightRace(simulator.Simulator):

    def __init__(self, sequence_generator : sequences.Sequence, **kwargs):
        super().__init__(**kwargs)



        # Setup sequence
        self._vehicle_attribute_indices = [1, self.n_vehicles * self.n_veh_params + 1]
        self._tls_attribute_indices = [self.vehicle_attribute_indices[1]+1, self.vehicle_attribute_indices[1]+4]
        self._n_parameters = self.tls_attribute_indices[1]
        self._seq = sequence_generator(
            domain = [(0,1) for n in range(self.n_parameters)],
            axes_names = ["dim%d" % n for n in range(self.n_parameters)]
        )

        # SUMO configuration
        map_dir = "%s" % FILE_DIR
        self._config = {
            "gui" : False,
            # "gui" : True,

            # Street network
            "--net-file" : "%s/tl-race.net.xml" % map_dir,

            # Logging
            "--error-log" : "%s/error-log.txt" % map_dir,

            # Traci Connection
            "--num-clients" : 1,
            "--remote-port" : 5522,

            # GUI Options
            "--delay" : 100,
            "--start" : "--quit-on-end",

            # RNG
            "--seed" : 333
        }
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

        concrete_values = [utils.project(
                attrib[0], attrib[1], point[i], by=attrib[2]) \
            for i, attrib in enumerate(attributes)]

        [setters[i](vid, val) for i, val in enumerate(concrete_values)]

        
        # Emergency decel must be greater than decel. 
        emergency_decel = utils.project(concrete_values[-2], 10., point[6], by=.1)
        traci.vehicle.setEmergencyDecel(vid, emergency_decel)
        concrete_values.append(emergency_decel)
        
        # Set initial speed between 0 and the max speed.
        initial_speed = utils.project(concrete_values[-1], 200, point[7], 1/3.6)
        traci.vehicle.setSpeed(vid, initial_speed)
        concrete_values.append(initial_speed)

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

    def _run_sumo_scenario(self):
        # Start the client
        self._client = TraCIClient(self.config)
        self._tlid = traci.trafficlight.getIDList()[0]
        
        # Add the route
        traci.route.add(self.route_id, ["before_tl","after_tl"])

        # Add vehicles
        self._vehicles = []
        self._n_vehicles_added = 0

        # Add Vehicles
        point = self.seq.get_points(1)[0]
        concrete_vehicle_params = self._add_vehicles(
            point[self.vehicle_attribute_indices[0] : self.vehicle_attribute_indices[1]])
        
        # Traffic Light
        concrete_tl_params = self._init_traffic_light(
            point[self.tls_attribute_indices[0] : self.tls_attribute_indices[1]])
        
        # Combine into the concrete parameters
        concrete_params = np.concatenate([concrete_vehicle_params, concrete_tl_params])

        # TODO:
        # Check if the concrete params are duplicate

        # Run simulation and observe an emergency stop
        collision_observed = False
        while traci.simulation.getMinExpectedNumber() > 0:
            if traci.simulation.getCollidingVehiclesNumber():
                collision_observed = True
                break
            traci.simulationStep()

        # Sim complete
        self.client.close()
        return collision_observed