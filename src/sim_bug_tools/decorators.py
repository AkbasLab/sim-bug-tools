"""
Contains a collection of decorators intended to provide common convenience functionality for both
classes and methods alike.
"""


from types import FunctionType
from numpy import int32, float64, ndarray


def is_valid_type(arg, typ, equiv, blacklist):
    result = not isinstance(arg, typ)
    if not result and arg in equiv:
        result = isinstance(arg, equiv[typ])

    return result


def typecheck(*blacklist):
    """
    Enforces function annotations at runtime. If an argument does not match its annotation,
    a ValueError will be thrown.

    Args:
        *blacklist: Names of the arguments that should be ignored from type checking
    """

    equivalent = {
        int: int32,
        int32: int,
        float: float64,
        float64: float,
        # Iterables
        list: ndarray,
        tuple: ndarray,
        ndarray: list,
    }

    def inner(f: FunctionType) -> FunctionType:

        return_type = (
            None if "return" not in f.__annotations__ else f.__annotations__["return"]
        )

        def wrapped(*args, **kwargs) -> return_type:
            keys = tuple(f.__annotations__.keys())
            for i, arg in enumerate(args):
                expected_type = f.__annotations__.get(keys[i])

                if not keys[i] in blacklist and not is_valid_type(
                    arg, expected_type, equivalent, blacklist
                ):
                    raise ValueError(
                        f"Datatype mismatch! For positional argument ({keys[i]}={arg}) expected {expected_type} and got {type(arg)} instead."
                    )

            for k, v in kwargs.items():
                expected_type = f.__annotations__.get(k)
                if not keys[i] in blacklist and not is_valid_type(
                    arg, expected_type, equivalent, blacklist
                ):
                    raise ValueError(
                        f"Datatype mismatch! For positional argument ({k}={v}) expected {expected_type} and got {type(v)} instead."
                    )

            return f(*args, **kwargs)

        return wrapped

    return inner
