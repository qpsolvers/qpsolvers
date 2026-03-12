import unittest
import warnings
from qpsolvers import available_solvers, solve_problem
from .problems import get_qpmad_demo_problem

class TestTimings(unittest.TestCase):
    def test_timings_recorded(self):
        problem = get_qpmad_demo_problem()
        
        for solver in available_solvers:
            with self.subTest(solver=solver):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        solution = solve_problem(problem, solver=solver)
                    
                    self.assertTrue(solution.found, f"Solver {solver} did not find a solution")
                    self.assertIsNotNone(solution.build_time)
                    self.assertIsNotNone(solution.solve_time)
                    self.assertGreaterEqual(solution.build_time, 0.0)
                    self.assertGreaterEqual(solution.solve_time, 0.0)
                    
                    print(f"[{solver}] build_time: {solution.build_time:.6f}s, solve_time: {solution.solve_time:.6f}s")
                except Exception as e:
                    print(f"[{solver}] Failed to solve: {e}")

if __name__ == "__main__":
    unittest.main()
