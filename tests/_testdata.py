from dataclasses import dataclass, field
from sim_bug_tools.structs import Point, Domain
import random


@dataclass
class PointTestData:
    seed: int
    uniform_arrays: list = field(default_factory=list)
    nonuniform_arrays: list = field(default_factory=list)
    invalid_nonuniform_arrays: list = field(default_factory=list)

    @property
    def uniform_points(self) -> list[Point]:
        return list(map(lambda p: Point(p), self.uniform_arrays))

    @property
    def nonuniform_points(self) -> list[Point]:
        return list(map(lambda p: Point(p), self.nonuniform_arrays))

    @property
    def invalid_nonuniform_points(self) -> list[Point]:
        return list(map(lambda p: Point(p), self.invalid_nonuniform_arrays))

    def generate_nonuniform_arrays(
        self,
        num: int,
        element_range: tuple[float, float] = (-50, 50),
        num_axes_range: tuple[int, int] = (3, 6),
    ) -> list[list]:

        random.seed(self.seed)
        lower, upper = element_range

        points = []
        for i in range(num):
            size = random.randint(num_axes_range[0], num_axes_range[1])
            points += [[random.random() * (upper - lower) + lower for x in range(size)]]

        return points

    def setup_nonuniform_arrays(
        self,
        num: int,
        element_range: tuple[float, float] = (-50, 50),
        num_axes_range: tuple[int, int] = (3, 6),
    ) -> None:
        """
        setup and append a series of points of varying # of axes and positions.
        """

        self.nonuniform_arrays += self.generate_nonuniform_arrays(
            num, element_range, num_axes_range
        )

    def setup_uniform_arrays(
        self,
        num_points: int,
        num_axes: int,
        element_range: tuple[float, float] = (-50, 50),
    ) -> None:
        """
        Generate and append a series of arrays with the equal # of axes and varying
        positions.
        """

        axes_range = (num_axes, num_axes)
        self.uniform_arrays = self.generate_nonuniform_arrays(
            num_points, element_range, axes_range
        )

    def setup_invalid_nonuniform_arrays(
        self,
        num: int,
        element_range: tuple[float, float] = (-50, 50),
        num_axes_range: tuple[int, int] = (3, 6),
    ) -> None:
        """
        Generate and append a series of arrays with varying # of axes, positions
        and element data types (valid and invalid).
        """
        ALL_TYPES = (str, int, float, bool)
        random.seed(self.seed)
        lower, upper = element_range

        points = []
        for i in range(num):
            size = random.randint(num_axes_range[0], num_axes_range[1])
            point = []

            for j in range(size):
                typ = ALL_TYPES[random.randint(0, len(ALL_TYPES) - 1)]
                if typ is bool:
                    point += [bool(random.randint(0, 1))]
                else:
                    point += [typ(random.random() * (upper - lower) + lower)]

            points += [point]

        self.invalid_nonuniform_arrays += points


@dataclass
class DomainTestData:
    DOMAIN = list[tuple[float, float]]
    seed: int
    domain_arrays: list = field(default_factory=list)
    invalid_domain_arrays: list = field(default_factory=list)

    @property
    def domains(self) -> list[Domain]:
        return list(map(lambda d: Domain(d, self.domain_arrays)))

    @property
    def invalid_domains(self) -> list[Domain]:
        return list(map(lambda d: Domain(d, self.invalid_domain_arrays)))

    def setup_valid_domain_arrays(
        self,
        num: int,
        num_axes_range: tuple[int, int],
        limits: tuple[float, float] = (0, 1),
    ) -> None:
        self.domain_arrays = self.generate_valid_domain_arrays(
            num, num_axes_range, limits
        )

    def setup_invalid_domain_arrays(
        self,
        num: int,
        num_axes_range: tuple[int, int],
        limits: tuple[float, float] = (0, 1),
    ) -> None:
        self.invalid_domain_arrays = self.generate_invalid_domain_arrays(
            num, num_axes_range, limits
        )

    def generate_valid_domain_arrays(
        self,
        num: int,
        num_axes_range: tuple[int, int],
        limits: tuple[float, float] = (0, 1),
    ) -> DOMAIN:
        random.seed(self.seed)

        low, high = limits

        domains = []
        for i in range(num):
            size = random.randint(num_axes_range[0], num_axes_range[1])
            domain = []

            for j in range(size):
                low_limit = random.random() * (high - low) + low
                upper_limit = random.random() * (high - low_limit) + low_limit
                domain += [(low_limit, upper_limit)]

            domains += [domain]

        return domains

    def generate_invalid_domain_arrays(
        self,
        num: int,
        num_axes_range: tuple[int, int],
        limits: tuple[float, float] = (0, 1),
    ) -> DOMAIN:
        random.seed(self.seed)

        low, high = limits

        domains = []
        for i in range(num):
            size = random.randint(num_axes_range[0], num_axes_range[1])
            domain = []

            for j in range(size):
                low_limit = random.random() * (high - low) + low
                upper_limit = random.random() * (low_limit - low) + low
                domain += [(low_limit, upper_limit)]

            domains += [domain]

        return domains
