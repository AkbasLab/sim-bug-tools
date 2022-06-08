import sim_bug_tools.simulator as simulator
from sim_bug_tools.sumo import TraCIClient
import sim_bug_tools.utils as utils
import sim_bug_tools.structs as structs

import traci
import os

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

class Simple(simulator.Simulator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # SUMO Configuration
        map_dir = "%s" % FILE_DIR
        self._config = {
            "gui" : False,
            # "gui" : True,

            # Street network
            "--net-file" : "%s/3-way-1k.net.xml" % map_dir,

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

        self._client = None
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

    def _run_sumo_scenario(self, 
        dist_from_stop : float, speed_start : float):

        # Start the client
        self._client = TraCIClient(self.config)

        # Add the route
        route_id = "route_1"
        traci.route.add(route_id, ["E0","E1"])

        # Add vehicle
        av_id = "av"
        traci.vehicle.add(
            av_id, 
            route_id, 
            departPos = 1000 - dist_from_stop,
        )

        traci.vehicle.setMaxSpeed(av_id, 100)
        traci.vehicle.setSpeed(av_id, speed_start)
        traci.vehicle.setAccel(av_id, speed_start)
        

        comfortable_deccel = 4.5
        is_comfy = True

        # Simulation loop
        i = 0
        while traci.simulation.getMinExpectedNumber() > 0:

            accel = traci.vehicle.getAcceleration(av_id)
            speed = traci.vehicle.getSpeed(av_id)
            print("%.2fkph %.2fm/s^2" % (speed, accel))

            i += 1
            if i > 20:
                break
            
            traci.simulationStep()
            continue

        # Sim complete
        self.client.close()

        print()
        print("is_comfortable: %s" % is_comfy)
        return