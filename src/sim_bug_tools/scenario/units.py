import numpy as np
from dataclasses import dataclass

@dataclass
class Distance:
    def __init__(self, meter : np.float64 = np.NaN):
        if not np.isnan(meter):
            self._nanometer = np.int64(meter * self.METER_TO_NANOMETER)
        else:
            raise ValueError("Distance not specified.")
        return

    
    @property
    def nanometer(self) -> np.int64:
        return self._nanometer
    
    @property
    def meter(self) -> np.float64:
        return np.float64(self.nanometer * self.NANOMETER_TO_METER)

    @property
    def METER_TO_NANOMETER(self) -> np.float64:
        return np.float64(1e9)

    @property
    def NANOMETER_TO_METER(self) -> np.float64:
        return np.float64(1./self.METER_TO_NANOMETER)



class Speed:

    def __init__(self, mps : np.float64 = np.NaN, kph : np.float64 = np.NaN,
                meters_per_second : np.float64 = np.NaN, kilometers_per_hour : np.float64 = np.NaN):
        if not np.isnan(mps):
            self._nanometers_per_second = np.int64(mps * self.MPS_TO_NMPS)
        elif not np.isnan(meters_per_second):
            self._nanometers_per_second = np.int64(meters_per_second * self.MPS_TO_NMPS)
        elif not np.isnan(kph):
            self._nanometers_per_second = np.int64(kph * self.KPH_TO_MPS * self.MPS_TO_NMPS)
            pass
        elif not np.isnan(kilometers_per_hour):
            self._nanometers_per_second = np.int64(kilometers_per_hour * self.KPH_TO_MPS * self.MPS_TO_NMPS)
            pass
        else:
            raise ValueError("No speed given.")
        return


    @property
    def mps(self) -> np.float64:
        return self.meters_per_second

    @property
    def meters_per_second(self) -> np.float64:
        return np.float64(self._nanometers_per_second * self.NMPS_TO_MPS)

    @property
    def kph(self) -> np.float64:
        return self.kilometers_per_hour

    @property
    def kilometers_per_hour(self) -> np.float64:
        return np.float64(self._nanometers_per_second * self.NMPS_TO_KPH)

    @property
    def MPS_TO_NMPS(self) -> np.float64:
        return np.float64(1e9)
    
    @property
    def NMPS_TO_MPS(self) -> np.float64:
        return np.float64(1/self.MPS_TO_NMPS)

    @property
    def KPH_TO_MPS(self) -> np.float64:
        return np.float64(10./36.)

    @property
    def MPS_TO_KPH(self) -> np.float64:
        return np.float64(3.6)

    @property
    def NMPS_TO_KPH(self):
        return self.NMPS_TO_MPS * self.MPS_TO_KPH



class Time:

    def __init__(self, second : np.float64 = np.NaN, hour : np.float64 = np.NaN):
        if not np.isnan(second):
            self._nanosecond = np.int64(second * self.SECOND_TO_NANOSECOND)
        elif not np.isnan(hour):
            self._nanosecond = np.int64(hour * 3600 * self.SECOND_TO_NANOSECOND)
        else:
            raise ValueError("No time given.")
        return

    @property
    def nanosecond(self) -> np.int64:
        return self._nanosecond

    @property
    def second(self) -> np.float64:
        return np.float64(self._nanosecond * self.NANOSECOND_TO_SECOND)

    @property
    def SECOND_TO_NANOSECOND(self) -> np.float64:
        return np.float64(1e9)

    @property
    def NANOSECOND_TO_SECOND(self) -> np.float64:
        return np.float64(1./self.SECOND_TO_NANOSECOND)
    