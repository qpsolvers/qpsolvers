#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2022 Stéphane Caron and the qpsolvers contributors.
#
# This file is part of qpsolvers.
#
# qpsolvers is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# qpsolvers is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with qpsolvers. If not, see <http://www.gnu.org/licenses/>.

"""
Model for a quadratic program.
"""

from typing import Optional, Tuple, TypeVar, Union

import numpy as np
import scipy.sparse as spa

from .conversions import linear_from_box_inequalities
from .exceptions import ProblemError

VectorType = TypeVar("VectorType")


class Problem:

    """
    Model for a quadratic program defined as:

    .. math::

        \\begin{split}\\begin{array}{ll}
            \\underset{x}{\\mbox{minimize}} &
                \\frac{1}{2} x^T P x + q^T x \\\\
            \\mbox{subject to}
                & G x \\leq h                \\\\
                & A x = b                    \\\\
                & lb \\leq x \\leq ub
        \\end{array}\\end{split}

    This class provides sanity checks and metrics such as the condition number
    of a problem.

    Attributes
    ----------
    P :
        Symmetric cost matrix (most solvers require it to be definite
        as well).
    q :
        Cost vector.
    G :
        Linear inequality matrix.
    h :
        Linear inequality vector.
    A :
        Linear equality matrix.
    b :
        Linear equality vector.
    lb :
        Lower bound constraint vector.
    ub :
        Upper bound constraint vector.
    """

    P: Union[np.ndarray, spa.csc_matrix]
    q: np.ndarray
    G: Optional[Union[np.ndarray, spa.csc_matrix]] = None
    h: Optional[np.ndarray] = None
    A: Optional[Union[np.ndarray, spa.csc_matrix]] = None
    b: Optional[np.ndarray] = None
    lb: Optional[np.ndarray] = None
    ub: Optional[np.ndarray] = None

    @staticmethod
    def __check_matrix(
        M: Optional[Union[np.ndarray, spa.csc_matrix]]
    ) -> Optional[Union[np.ndarray, spa.csc_matrix]]:
        """
        Ensure a problem matrix has proper shape.

        Parameters
        ----------
        M :
            Problem matrix.
        name :
            Matrix name.

        Returns
        -------
        :
            Same matrix with proper shape.
        """
        if isinstance(M, np.ndarray) and M.ndim == 1:
            M = M.reshape((1, M.shape[0]))
        return M

    @staticmethod
    def __check_vector(v: VectorType, name: str) -> VectorType:
        """
        Ensure a problem vector has proper shape.

        Parameters
        ----------
        M :
            Problem matrix.
        name :
            Matrix name.

        Returns
        -------
        :
            Same matrix with proper shape.
        """
        if v is None or v.ndim <= 1:
            return v
        if v.shape[0] != 1 and v.shape[1] != 1 or v.ndim > 2:
            raise ProblemError(
                f"vector '{name}' should be flat "
                f"and cannot be flattened as its shape is {v.shape}"
            )
        return v.flatten()

    def __init__(
        self,
        P: Union[np.ndarray, spa.csc_matrix],
        q: np.ndarray,
        G: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
        h: Optional[np.ndarray] = None,
        A: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
        b: Optional[np.ndarray] = None,
        lb: Optional[np.ndarray] = None,
        ub: Optional[np.ndarray] = None,
    ) -> None:
        P = Problem.__check_matrix(P)
        q = Problem.__check_vector(q, "q")
        G = Problem.__check_matrix(G)
        h = Problem.__check_vector(h, "h")
        A = Problem.__check_matrix(A)
        b = Problem.__check_vector(b, "b")
        lb = Problem.__check_vector(lb, "lb")
        ub = Problem.__check_vector(ub, "ub")
        self.P = P
        self.q = q
        self.G = G
        self.h = h
        self.A = A
        self.b = b
        self.lb = lb
        self.ub = ub

    @property
    def has_sparse(self) -> bool:
        """
        Check whether the problem has sparse matrices.

        Returns
        -------
        :
            True if at least one of the :math:`P`, :math:`G` or :math:`A`
            matrices is sparse.
        """
        sparse_types = (spa.csc_matrix, spa.dia_matrix)
        return (
            isinstance(self.P, sparse_types)
            or isinstance(self.G, sparse_types)
            or isinstance(self.A, sparse_types)
        )

    def unpack(
        self,
    ) -> Tuple[
        Union[np.ndarray, spa.csc_matrix],
        np.ndarray,
        Optional[Union[np.ndarray, spa.csc_matrix]],
        Optional[np.ndarray],
        Optional[Union[np.ndarray, spa.csc_matrix]],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """
        Get problem matrices as a tuple.

        Returns
        -------
        :
            Tuple ``(P, q, G, h, A, b, lb, ub)`` of problem matrices.
        """
        return (
            self.P,
            self.q,
            self.G,
            self.h,
            self.A,
            self.b,
            self.lb,
            self.ub,
        )

    def check_constraints(self):
        """
        Check that problem constraints are properly specified.

        Raises
        ------
        ValueError
            If the constraints are not properly defined.
        """
        if self.G is None and self.h is not None:
            raise ValueError("incomplete inequality constraint (missing h)")
        if self.G is not None and self.h is None:
            raise ValueError("incomplete inequality constraint (missing G)")
        if self.A is None and self.b is not None:
            raise ValueError("incomplete equality constraint (missing b)")
        if self.A is not None and self.b is None:
            raise ValueError("incomplete equality constraint (missing A)")

    def cond(self):
        """
        Compute the condition number of the symmetric matrix representing the
        problem data:

        .. math::

            M =
            \\begin{bmatrix}
                P & G^T & A^T \\\\
                G & 0   & 0   \\\\
                A & 0   & 0
            \\end{bmatrix}

        Returns
        -------
        :
            Condition number of the problem.

        Raises
        ------
        ProblemError :
            If the problem is sparse.

        Notes
        -----
        Having a low condition number (say, less than 1e10) condition number is
        strongly tied to the capacity of numerical solvers to solve a problem.
        This is the motivation for preconditioning, as detailed for instance in
        Section 5 of [Stellato2020]_.
        """
        if self.has_sparse:
            raise ProblemError("This function is for dense problems only")
        P, A = self.P, self.A
        G, _ = linear_from_box_inequalities(
            self.G, self.h, self.lb, self.ub, use_sparse=False
        )
        if G is None and A is None:
            M = P
        elif A is None:  # G is not None
            M = np.vstack(
                [
                    np.hstack([P, G.T]),
                    np.hstack([G, np.zeros((G.shape[0], G.shape[0]))]),
                ]
            )
        else:  # G is not None and A is not None
            M = np.vstack(
                [
                    np.hstack([P, G.T, A.T]),
                    np.hstack(
                        [
                            G,
                            np.zeros((G.shape[0], G.shape[0])),
                            np.zeros((G.shape[0], A.shape[0])),
                        ]
                    ),
                    np.hstack(
                        [
                            A,
                            np.zeros((A.shape[0], G.shape[0])),
                            np.zeros((A.shape[0], A.shape[0])),
                        ]
                    ),
                ]
            )
        return np.linalg.cond(M)
