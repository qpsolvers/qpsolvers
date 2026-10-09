# Make sure Python treats the test directory as a package.

# PDHCG defaults to relative tolerances of 1e-4. Request higher accuracy in
# the shared regression suite, which also checks absolute errors near 1e-8.
SOLVER_OPTIONS = {
    "pdhcg": {
        "OptimalityTol": 1e-10,
        "FeasibilityTol": 1e-10,
        "InnerMinTol": 1e-12,
        "TimeLimit": 30.0,
    },
}
