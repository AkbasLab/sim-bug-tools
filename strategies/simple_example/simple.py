import sim_bug_tools.simulator as simulator
from sim_bug_tools.sumo import TraCIClient
import sim_bug_tools.utils as utils
import sim_bug_tools.structs as structs

import traci
import os
import re
import numpy as np

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

class Simple(simulator.Simulator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # SUMO Configuration
        map_dir = "%s" % FILE_DIR
        self._error_log_fn = "%s/error-log.txt" % map_dir
        self._config = {
            "gui" : False,
            # "gui" : True,

            # Street network
            "--net-file" : "%s/3-way-1k.net.xml" % map_dir,

            # Logging
            "--error-log" : self._error_log_fn,

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
        dist_from_stop : float, speed_start : float) -> bool:
        """
        dist_from_stop : float
            Distance from stop line (m)
        speed_start : float
            Initial speed of AV (kph)
        """

        # Start the client
        self._client = TraCIClient(self.config)

        # Add the route
        route_id = "R0"
        traci.route.add(route_id, ["Ewarmup", "E0", "E1"])

        # Add vehicle
        av_id = "av"
        traci.vehicle.add(
            av_id, 
            route_id
        )

        # Convert speed from kps -> mps
        speed_start = speed_start/3.6

        # Get the vehicle up to speed.
        traci.vehicle.setSpeedMode(av_id, 0)

        traci.vehicle.setMaxSpeed(av_id, 100)
        traci.vehicle.setSpeed(av_id, speed_start)

        default_accel = traci.vehicle.getAccel(av_id)
        traci.vehicle.setAccel(av_id, speed_start)
        
        

        # Simulation loop
        score = 1
        is_warmup = True
        while traci.simulation.getMinExpectedNumber() > 0:
            if is_warmup:
                # Get the AV up to speed
                speed = traci.vehicle.getSpeed(av_id)
                # print(speed)
                if speed == speed_start:
                    # Transport to the new edge
                    traci.vehicle.moveTo(av_id, "E0_0", 0)
                    traci.vehicle.setSpeedMode(av_id, 100 - dist_from_stop)
                    traci.vehicle.setAccel(av_id, default_accel)
                    is_warmup = False
                traci.simulationStep()
                continue
            
            # Vehicle is up to speed
            e_stop = traci.simulation.getEmergencyStoppingVehiclesNumber()
            if e_stop:
                 break
            
            # AV stopped At traffic light
            speed = traci.vehicle.getSpeed(av_id)
            if not int(speed):
                break
            
            traci.simulationStep()
            continue

        # Sim complete
        self.client.close()


        
        if e_stop:
            # Update the score
            with open(self._error_log_fn, "r") as f:
                log = "".join(f.read())
            decel = -float(re.findall(r"decel=?-\d*\.\d*", log)[0] \
                .strip("decel="))
            
            if decel <= 4.5:
                score = 1
            elif decel > 4.5 and decel <= 9.:
                score = 1 - (9-decel)/4.5
            else:
                score = 0

        return np.round(score, decimals=3)