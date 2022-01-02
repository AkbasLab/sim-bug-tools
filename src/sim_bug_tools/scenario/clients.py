import traci
import numpy as np
import traci.constants as tc
import sim_bug_tools.scenario.actors as actors
import sim_bug_tools.structs as structs
import warnings

class Client:
    def __init__(self, config : dict, priority : np.int32 = 1):
        """
        Barebones TraCI client.

        --- Parameters ---
        priority : int
            Priority of clients. MUST BE UNIQUE
        config : dict
            SUMO arguments stored as a python dictionary.
        """
        self._priority = priority
        self._config = config
        self.connect()
        return

    @property
    def priority(self) -> np.int32:
        return self._priority

    @property
    def config(self) -> dict:
        return self._config

    def test(self):
        self.run_to_end()
        self.close()


    def run_to_end(self):
        """
        Runs the client until the end.
        """
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            # more traci commands
        return

    def close(self):
        """
        Closes the client.
        """
        traci.close()
        return


    def connect(self):
        """
        Start or initialize the TraCI connection.
        """
        warnings.simplefilter("ignore", ResourceWarning)
        # Start the traci server with the first client
        if self.priority == 1:
            cmd = []

            for key, val in self.config.items():
                if key == "gui":
                    sumo = "sumo"
                    if val: sumo +="-gui"
                    cmd.append(sumo)
                    continue
                
                if key == "--remote-port":
                    continue

                cmd.append(key)
                cmd.append(str(val))
                continue

            traci.start(cmd,port=self.config["--remote-port"])
            traci.setOrder(self.priority)
            return
        
        # Initialize every client after the first.
        traci.init(port=self.config["--remote-port"])
        traci.setOrder(self.priority)
        return    




class VehiclePopulation(Client):

    def __init__(self, config : dict, id_prefix : str = "veh_", priority : np.int32 = 1):
        """
        Vehicle population client

        --- Parameters ---
        id_prefix : str
            Prefix for incrementing vehicle IDs when a vehicle is added.
        priority : numpy.int32
            Priority of clients. MUST BE UNIQUE
        config : dict
            SUMO arguments stored as a python dictionary.

        -- Return --
        Traci vehicle ID : str
        """
        super().__init__(config,priority)
        self._id_prefix = id_prefix
        self._size = np.int32(0)
        self._contents = []
        return

    @property
    def id_prefix(self) -> str:
        return self._id_prefix
    
    @property
    def size(self) -> np.int32:
        return self._size

    @property
    def contents(self) -> list[str]:
        return self._contents

    def add_vehicle(self, route_id : str) -> str:
        """
        Adds a vehicle to the simulation

        -- Parameters --
        route_id : str
            Traci Route ID
        """
        # Give the vehicle a unique ID
        vehicle_id = "veh_%d" % self.size
        traci.vehicle.add(vehicle_id,route_id)

        # Update the client
        self._size += 1
        self._contents.append(vehicle_id)
        return vehicle_id


class Scenario(VehiclePopulation):

    YIELD = "YIELD"
    NO_YIELD = "NO_YIELD"

    def __init__(self, dut : actors.ActorConstantSpeed ,
                npc : actors.ActorConstantSpeed, config : dict):
        super().__init__(id_prefix="veh_",config=config)
        """
        Concrete Scenario client for TraCI

        -- Parameteres --
        DUT : Actor
            Device under test. Top road.
        NPC : Actor
            Other actor. Bottom road.
        """
        self._dut = dut
        self._npc = npc

        self.init_dut()
        self.init_npc()
        return

    @property
    def dut(self) -> actors.ActorConstantSpeed:
        return self._dut

    @property
    def npc(self) -> actors.ActorConstantSpeed:
        return self._npc

    # @property
    # def YIELD(self) -> str:
    #     return "YIELD"
    
    # @property
    # def NO_YIELD(self) -> str:
    #     return "NO_YIELD"


    def init_npc(self):
        # Add the route and place the vehicle
        route_id = "route_npc"
        start_edge = "E1"
        end_edge = "E3"
        traci.route.add(route_id,[start_edge,end_edge])
        self._npc.vehicle_id = self.add_vehicle(route_id)

        # Starting Position
        edge_length = 1000
        traci.vehicle.moveTo(self.npc.vehicle_id,
                            "%s_0" % start_edge,
                            edge_length-self.npc.distance_from_junction.meter)

        traci.vehicle.setSpeed(self.npc.vehicle_id,self.npc.speed.mps)
        traci.vehicle.setColor(self.npc.vehicle_id,(255,0,0,255))

        traci.vehicle.subscribe(
            self.npc.vehicle_id,
            [tc.VAR_LANE_ID,tc.VAR_LANEPOSITION,tc.VAR_SPEED])
        return


    def init_dut(self):
        # Add the route and place the vehicle
        route_id = "route_dut"
        start_edge = "E2"
        end_edge = "E3"
        traci.route.add(route_id,[start_edge,end_edge])
        self._dut.vehicle_id = self.add_vehicle(route_id)

        # Starting Position
        edge_length = 1000
        traci.vehicle.moveTo(self.dut.vehicle_id,
                            "%s_0" % start_edge,
                            edge_length-self.dut.distance_from_junction.meter)

        traci.vehicle.setSpeed(self.dut.vehicle_id,self.dut.speed.mps)
        traci.vehicle.setColor(self.dut.vehicle_id,(0,0,255,255))

        traci.vehicle.subscribe(
            self.dut.vehicle_id,
            [tc.VAR_LANE_ID,tc.VAR_LANEPOSITION,tc.VAR_SPEED])

        if self.config["gui"]:
            traci.gui.trackVehicle("View #0",self.dut.vehicle_id)
            traci.gui.setZoom("View #0",350)
            traci.vehicle.highlight(self.dut.vehicle_id,color=(0,255,0,255))
        return

    def is_yield(self, tau : np.float64) -> bool:
        """
        Checks if a yield is happening using Time to Junction.
        -- Parameters --
        tau : float
            Threshold where yield is when time to yield <= tau
            tau is in seconds.
        -- Return --
        bool
            Yield is happening or not.
        """
        sub = traci.vehicle.getAllSubscriptionResults()
        try:
            ttj_dut = self.time_to_junction(
                lane_id = sub[self.dut.vehicle_id][tc.VAR_LANE_ID],
                lane_position = np.float64(sub[self.dut.vehicle_id][tc.VAR_LANEPOSITION]),
                speed = np.float64(sub[self.dut.vehicle_id][tc.VAR_SPEED])
            )
            ttj_npc = self.time_to_junction(
                lane_id = sub[self.npc.vehicle_id][tc.VAR_LANE_ID],
                lane_position = np.float64(sub[self.npc.vehicle_id][tc.VAR_LANEPOSITION]),
                speed = np.float64(sub[self.npc.vehicle_id][tc.VAR_SPEED])
            )
        except KeyError:
            return False

        tty = np.abs(ttj_dut + ttj_npc)
        return tty <= tau

    def time_to_junction(self, lane_id : str, lane_position : np.float64, 
                        speed : np.float64, edge_length : np.int32 = np.int32(1000)):
        """
        Calculates time to junction
        -- Parameters --
        lane_id : string
            SUMO lane ID
        lane_position : float
            Distance of vehicle along lane in meters
        speed : float 
            Vehicle speed in meters/second
        edge_length : float (m)
            Edge length in meters
        -- Return --
        ttj : float
            Estimated arrival time to junction
        """
        warnings.simplefilter("ignore", RuntimeWarning)

        if ":J3" in lane_id:
            return 0
        try:
            # The edge past the junction
            if lane_id == "E3_0":
                return -lane_position / speed
            # Edge before the junction
            return (edge_length - lane_position) / speed
        # Speed is 0
        except ZeroDivisionError:
            return np.inf

    def run_until_yield(self, tau : np.float64):
        """
        Run the client until a yield or until the end.

        -- Parameters --
        tau : float
            Threshold where yield is when time to yield <= tau
            tau is in seconds.

        -- Return --
        str :
            Return completeion status.
            "YIELD" or "NO_YIELD" or "ERROR"
        """
        while traci.simulation.getMinExpectedNumber() > 0:
            if self.is_yield(tau):
                self.close()
                return self.YIELD
            traci.simulationStep()
        self.close()
        return self.NO_YIELD