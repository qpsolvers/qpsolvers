#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2025 Stéphane Caron and the qpsolvers contributors

"""Unit tests for jaxopt.OSQP."""

import unittest
import warnings

try:
    import jax.numpy as jnp

    from qpsolvers import solve_qp

    class TestKVXOPT(unittest.TestCase):
        """Test fixture for the KVXOPT solver."""

        def setUp(self):
            """Prepare test fixture."""
            warnings.simplefilter("ignore", category=UserWarning)

        def test_jax_array_input(self):
            """We can call ``solve_qp`` with jax.Array matrices."""
            M = jnp.array([[1.0, 2.0, 0.0], [-8.0, 3.0, 2.0], [0.0, 1.0, 1.0]])
            P = M.T @ M  # this is a positive definite matrix
            q = jnp.array([3.0, 2.0, 3.0]) @ M
            G = jnp.array(
                [[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]]
            )
            h = jnp.array([3.0, 2.0, -2.0])
            A = jnp.array([1.0, 1.0, 1.0])
            b = jnp.array([1.0])
            x = solve_qp(P, q, G, h, A, b, solver="jaxopt_osqp")
            self.assertIsNotNone(x)

except ImportError as exn:  # in case the solver is not installed
    warnings.warn(f"Skipping jaxopt.OSQP tests: {exn}")
