from dataclasses import dataclass
from typing import Optional

import numpy


@dataclass
class QPOutput:
    """
    This is the generic Solver output object. This is the general extra infor return object for solvers. It contains\\
    all of the information you would need for the optimization solution including, optimal value, optimal solution, the \\
    active set, the value of the slack variables and the largange multipliers associated with every constraint.
    Parameters
    ----------
    obj: optimal objective \n
    sol: x*, numpy array \n
    Optional Parameters -> None or numpy.ndarray type
    slack: the slacks associated with every constraint \n
    equality_indices: the active set of the solution, including strongly and weakly active constraints \n
    dual: the lagrange multipliers associated with the problem\n
    """
    obj: float
    sol: numpy.ndarray

    slack: Optional[numpy.ndarray]
    active_set: Optional[numpy.ndarray]
    dual: Optional[numpy.ndarray]

    def __eq__(self, other):
        if not isinstance(other, QPOutput):
            return NotImplemented

        return numpy.allclose(self.slack, other.slack) and numpy.allclose(self.active_set,
                                                                          other.active_set) and numpy.allclose(
            self.dual, other.dual) and numpy.allclose(self.sol, other.sol) and numpy.allclose(self.obj, other.obj)
