import numpy as np
import sim_bug_tools.scenario.units as units
import sim_bug_tools.utils as utils
import sim_bug_tools.rng.lds.sequences as sequences
import sim_bug_tools.structs as structs
import sim_bug_tools.scenario.actors as actors
import sim_bug_tools.scenario.clients as clients
import sim_bug_tools.scenario.config as config
from sim_bug_tools.rng.rrt import RapidlyExploringRandomTree

class ControllerYieldConstantSpeed:
    def __init__(self, tau : np.float64, 
                scenarios_to_run : np.int32,
                local_searches : np.int32,
                offset : np.int32,
                speed : list[units.Speed], 
                distance_from_junction : list[units.Distance]):
        """
        Yield scenario with constant speed.
        
        -- Parameters --
        tau : np.float64 
            Time in seconds. When realtive arrival time of the two actors
            at the yield is below tau, a yield occurs.
        scenarios_to_run : np.int32
            Number of scenarios to run in total.
        local_searches : np.int32
            Number of local searches to perform before reverting to long walk.
        offset : np.int32
            The starting value for low discrepency sequences.
            Should be a prime number.
        speed : list[units.Speed]
            Min and max speed range of actors.
        distance_from_junction L list[units.Distance]
            Min and max distance from junction range
        """

        if not utils.is_prime(offset):
            raise ValueError("Offset >%d< must be a prime number." % offset)

        self._tau = tau
        self._scenarios_to_run = np.int32(scenarios_to_run)
        self._local_searches = np.int32(local_searches)
        self._offset = np.int32(offset)
        self._speed = speed 
        self._distance_from_junction = distance_from_junction
        self._domain = structs.Domain.normalized(self.n_dim)
        self._axes_names = ["dut.speed", "dut.dfj", "npc.speed", "npc.dfj"]
        return

    @property
    def tau(self) -> np.float64:
        return self._tau

    @property
    def scenarios_to_run(self) -> np.int32:
        return self._scenarios_to_run
    
    @property
    def offset(self) -> np.int32:
        return self._offset

    @property
    def speed(self) -> list[units.Speed]:
        return self._speed

    @property
    def distance_from_junction(self) -> list[units.Distance]:
        return self._distance_from_junction

    @property
    def n_dim(self) -> np.int32:
        return np.int32(4)

    @property
    def domain(self) -> structs.Domain:
        return self._domain

    @property
    def axes_names(self) -> list[str]:
        return self._axes_names

    @property
    def local_searches(self) -> np.int32:
        return self._local_searches

    def test_sequence(self, seq : sequences.Sequence, rrt : RapidlyExploringRandomTree):
        """
        Test a sequence.
        Performs long walks until yield event.
        The local search using RRT.

        --- Parameter --
        seq : sequences.Sequence
            Low discrepency sequence to use
        rrt : RapidlyExploringRandomTree
            RRT to use.

        -- Return --
        list[Point], list[str]
            Points and Statuses of test.
        """
        # Initialize sequence
        seq = seq(self.domain,self.axes_names)
        seq.seed = 3210123 # neccesary for Random to work
        seq.get_points(int(self.offset))
        n_long_walks = np.int32(0)
        n_local_searches = np.int32(0)
        points = []
        statuses = []

        while True:
            # Long walk
            n_long_walks += 1
            point = seq.get_points(1)[0]
            points.append(point)

            # Run scenario
            status = self.concrete_scenario(point)
            statuses.append(status)
            print(
                "%d: LONG %d %s"
                % (n_local_searches + n_long_walks - 1, n_long_walks, status)
            )

            # Exit condition
            if n_long_walks + n_local_searches >= self.scenarios_to_run:
                break
            
            # if yield start a local search
            if status == clients.Scenario.YIELD:
                # Local search
                rrt.reset(point)
                for i in range(self.local_searches):    
                    point = rrt.step()[2]

                    points.append(point)

                    # Run Scenario
                    n_local_searches += 1
                    status = self.concrete_scenario(point)
                    statuses.append(status)
                    print(
                        "%d: LOCAL %d %s"
                        % (n_local_searches + n_long_walks - 1, n_local_searches, status)
                    )
                    

                    # Exit condition
                    if n_long_walks + n_local_searches >= self.scenarios_to_run:
                        break
                # Exit condition
                if n_long_walks + n_local_searches >= self.scenarios_to_run:
                    break
                continue
            # Exit condition
            if n_long_walks + n_local_searches >= self.scenarios_to_run:
                break
            continue
        return points, statuses

    def concrete_scenario(self, point : structs.Point) -> str:
        """
        Run a concrete scenario.

        -- Parameters --
        point : structs.Point
            Parameter values.

        -- Return --
        str
            Run status. "YIELD" or "NO_YIELD"
        """
        dut = actors.ActorConstantSpeed(
            speed = units.Speed(
                kph = utils.project(
                    a = self.speed[0].kph,
                    b = self.speed[1].kph,
                    x = point[0]
                )
            ),
            distance_from_junction = units.Distance(
                meter = utils.project(
                    a = self.distance_from_junction[0].meter,
                    b = self.distance_from_junction[1].meter,
                    x = point[1]
                )
            )
        )

        npc = actors.ActorConstantSpeed(
            speed = units.Speed(
                kph = utils.project(
                    a = self.speed[0].kph,
                    b = self.speed[1].kph,
                    x = point[2]
                )
            ),
            distance_from_junction = units.Distance(
                meter = utils.project(
                    a = self.distance_from_junction[0].meter,
                    b = self.distance_from_junction[1].meter,
                    x = point[3]
                )
            )
        )

        return clients.Scenario(dut, npc, config.SUMO).run_until_yield(self.tau)
         

class KnownBugs:
    def __init__(self, 
        domain : structs.Domain,
        bug_profiles : list[structs.Domain],
        # tau : np.int32,
        scenarios_to_run : np.int32,
        local_searches : np.int32,
        offset : np.int32):
        

        if not utils.is_prime(offset):
            raise ValueError("Offset >%d< must be a prime number." % offset)

        # self._tau = tau
        self._scenarios_to_run = np.int32(scenarios_to_run)
        self._local_searches = np.int32(local_searches)
        self._offset = np.int32(offset)

        self._domain = domain
        self._n_dim = np.int32(len(domain))
        self._bug_profiles = bug_profiles
        self._axes_names = ["p%d" for i in range(self.n_dim)]
        # self._n_expected_bugs = np.int32(n_expected_bugs)
        return

    @property
    def domain(self) -> structs.Domain:
        return self._domain

    @property
    def bug_profiles(self) -> list[structs.Domain]:
        return self._bug_profiles

    # @property
    # def tau(self) -> np.float64:
    #     return self._tau

    @property
    def scenarios_to_run(self) -> np.int32:
        return self._scenarios_to_run
    
    @property
    def offset(self) -> np.int32:
        return self._offset

    @property
    def n_dim(self) -> np.int32:
        return self._n_dim

    @property
    def domain(self) -> structs.Domain:
        return self._domain

    @property
    def local_searches(self) -> np.int32:
        return self._local_searches
    
    @property
    def axes_names(self) -> list[str]:
        return self._axes_names

    # @property
    # def n_expected_bugs(self) -> np.int32:
    #     return self._n_expected_bugs


    def concrete_scenario(self, point):
        _point = self.domain.project(point)
        return any([_point in profile for profile in self.bug_profiles])

    def test_sequence(self, seq : sequences.Sequence, rrt : RapidlyExploringRandomTree):
        """
        Test a sequence.
        Performs long walks until yield event.
        The local search using RRT.

        --- Parameter --
        seq : sequences.Sequence
            Low discrepency sequence to use
        rrt : RapidlyExploringRandomTree
            RRT to use.

        -- Return --
        list[Point], list[str]
            Points and Statuses of test.
        """
        # Initialize sequence
        seq = seq(self.domain, self.axes_names)
        seq.seed = 3210123 # neccesary for Random to work
        seq.get_points(int(self.offset))
        n_long_walks = np.int32(0)
        n_local_searches = np.int32(0)
        points = []
        statuses = []

        while True:
            # Long walk
            n_long_walks += 1
            point = seq.get_points(1)[0]
            points.append(point)

            # Run scenario
            status = self.concrete_scenario(point)
            statuses.append(status)
            # print(
            #     "%d: LONG %d %s"
            #     % (n_local_searches + n_long_walks - 1, n_long_walks, status)
            # )

            # Exit condition
            if n_long_walks + n_local_searches >= self.scenarios_to_run:
                break
            
            # if yield start a local search
            if status:
                for i in range(self.local_searches):    
                    # Local search
                    rrt.reset(point)
                    point = rrt.step()[2]

                    points.append(point)

                    # Run Scenario
                    n_local_searches += 1
                    status = self.concrete_scenario(point)
                    statuses.append(status)
                    # print(
                    #     "%d: LOCAL %d %s"
                    #     % (n_local_searches + n_long_walks - 1, n_local_searches, status)
                    # )
                    

                    # Exit condition
                    if n_long_walks + n_local_searches >= self.scenarios_to_run:
                        break
                # Exit condition
                if n_long_walks + n_local_searches >= self.scenarios_to_run:
                    break
                continue
            # Exit condition
            if n_long_walks + n_local_searches >= self.scenarios_to_run:
                break
            continue
        return points, statuses
    