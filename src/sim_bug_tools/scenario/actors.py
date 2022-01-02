import sim_bug_tools.scenario.units as units
from dataclasses import dataclass


@dataclass
class ActorConstantSpeed:
    def __init__(self, speed : units.Speed, distance_from_junction : units.Distance):
        """
        Scenario actor

        --- Parameters ---
        speed : Speed
            Actor speed.
        distance_from_junction : Distance
            Starting distance from the junction.
        """
        if not isinstance(speed, units.Speed):
            raise TypeError("'speed' is not of type `Speed`.")
        if not isinstance(distance_from_junction, units.Distance):
            raise TypeError("`distance_from_junction` is not of type `Distance`")

        self._speed = speed
        self._distance_from_junction = distance_from_junction
        self._vehicle_id = str()
        return

    @property
    def speed(self) -> units.Speed:
        """Actor Speed"""
        return self._speed

    @property
    def distance_from_junction(self) -> units.Distance:
        """Starting distance from junction."""
        return self._distance_from_junction

    @property
    def vehicle_id(self) -> str:
        """SUMO vehicle ID."""
        return self._vehicle_id

    @vehicle_id.setter
    def vehicle_id(self, vid : str):
        self._vehicle_id = vid
        return