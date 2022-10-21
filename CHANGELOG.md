# Changelog

All notable changes to this project will be documented in this file.

## [2.4.1] - 2022/10/21

### Changed

- Update ProxQP to version 0.2.2

## [2.4.0] - 2022/09/29

### Added

- New solver: [HiGHS](https://github.com/ERGO-Code/HiGHS)
- Raise error when there is no available solver

### Changed

- Make sure plot is shown in MPC example
- Print expected solutions in QP, LS and box-inequality examples
- Renamed ``starter_solvers`` optional deps to ``open_source_solvers``

### Fixed

- Correct documentation of ``R`` argument to ``solve_ls``

## [2.3.0] - 2022/09/06

### Added

- New solver: [ProxQP](https://github.com/Simple-Robotics/proxsuite)

### Changed

- Clean up unused dependencies in GitHub workflow
- Non-default solver parameters in unit tests to test their precision

### Fixed

- Configuration of `tox-gh-actions` for Python 3.7
- Enforce `USING_COVERAGE` in GitHub workflow configuration
- Remove redundant solver loop from ``test_all_shapes``

## [2.2.0] - 2022/08/15

### Added

- Add `lb` and `ub` arguments to all `<solver>_solve_qp` functions
- Internal ``qpsolvers.solvers.conversions`` submodule

### Changed

- Moved ``concatenate_bounds`` to internal ``conversions`` submodule
- Moved ``convert_to_socp`` to internal ``conversions`` submodule
- Renamed ``concatenate_bounds`` to ``linear_from_box_inequalities``
- Renamed internal ``convert_to_socp`` function to ``socp_from_qp``

## [2.1.0] - 2022/07/25

### Added

- Document how to add a new QP solver to the library
- Example with (box) lower and upper bounds
- Test case where `lb` XOR `ub` is set

### Changed

- SCS: use the box cone API when lower/upper bounds are set

## [2.0.0] - 2022/07/05

### Added

- Exception ``NoSolverSelected`` raised when the solver kwarg is missing
- Starter set of QP solvers as optional dependencies
- Test exceptions raised by `solve_ls` and `solve_qp`

### Changed

- **Breaking:** ``solver`` keyword argument is now mandatory for `solve_ls`
- **Breaking:** ``solver`` keyword argument is now mandatory for `solve_qp`
- Quadratic programming example now randomly selects an available solver

## [1.10.0] - 2022/06/25

### Changed

- qpSWIFT: Forward solver options as keywords arguments as with other solvers

## [1.9.1] - 2022/05/02

### Fixed

- OSQP: Pass extra keyword arguments properly (thanks to @urob)

## [1.9.0] - 2022/04/03

### Added

- Benchmark on model predictive control problem
- Model predictive control example
- qpSWIFT 0.0.2 solver interface

### Changed

- Compute colors automatically in benchmark example

### Fixed

- Bounds concatenation for CVXOPT sparse matrices

## [1.8.1] - 2022/03/05

### Added

- Setup instructions for Microsoft Visual Studio
- Unit tests where the problem is unbounded below

### Changed

- Minimum supported Python version is now 3.7

### Fixed

- Clear all Pylint warnings
- Disable Pylint false positives that are covered by mypy
- ECOS: raise a ValueError when the cost matrix is not positive definite

## [1.8.0] - 2022/01/13

### Added

- Build and test for Python 3.10 in GitHub Actions

### Changed

- Moved SCS to sparse solvers
- Re-run solver benchmark reported to the README
- Removed ``requirements2.txt`` and update Python 2 installation instructions
- Updated SCS to new 3.0 version

### Fixed

- Handle sparse matrices in ``print_matrix_vector``
- Match ``__all__`` in model and top-level ``__init__.py``
- Run unit tests in GitHub Actions
- Typing error in bound concatenation

## [1.7.2] - 2021/11/24

### Added

- Convenience function to prettyprint a matrix and vector side by side

### Changed

- Move old tests from the examples folder to the unit test suite
- Removed deprecated ``requirements.txt`` installation file
- Renamed ``solvers`` optional dependencies to ``all_pypi_solvers``

## [1.7.1] - 2021/10/02

### Fixed

- Make CVXOPT optional again (thanks to @adamoppenheimer)

## [1.7.0] - 2021/09/19

### Added

- Example script corresponding exactly to the README
- Handle lower and upper bounds with sparse matrices (thanks to @MeindertHH)
- SCS 2.0 solver interface
- Type annotations to all solve functions
- Unit tests: package coverage is now 94%

### Changed

- ECOS: simplify sparse matrix conversions
- Ignore warnings when running unit tests
- Inequality tolerance is now 1e-10 when validating solvers on README example
- Refactor QP to SOCP conversion to use more than one SOCP solver
- Rename "example problem" for testing to "README problem" (less ambiguous)
- Rename `sw` parameter of `solve_safer_qp` to `sr` for "slack repulsion"
- Reorganize code with a qpsolvers/solvers submodule
- quadprog: warning when `initvals is not None` is now verbose

### Fixed

- OSQP: forward keyword arguments to solver properly
- quadprog: forward keyword arguments to solver properly

## [1.6.1] - 2021/04/09

### Fixed

- Add quadprog dependency properly in `pyproject.toml`

## [1.6] - 2021/04/09

### Added

- Add `__version__` to main module
- First unit tests to check all solvers over a pre-defined set of problems
- GitHub Actions now make sure the project is built and tested upon updates
- Type hints now decorate all function definitions

### Changed

- Code formatting now applies [Black](https://github.com/psf/black)
- ECOS: refactor SOCP conversion to improve function readability
- Gurobi performance significantly improved by new matrix API (thanks to @DKenefake)

### Fixed

- CVXPY: properly return `None` on unfeasible problems
- Consistently warn when `initvals` is passed but ignored by solver interface
- ECOS: properly return `None` on unfeasible problems
- Fix `None` case in `solve_safer_qp` (found by static type checking)
- Fix warnings in repository-level `__init__.py`
- OSQP: properly return `None` on unfeasible problems
- Pass Flake8 validation for overall code style
- Reduce complexity of entry `solve_qp` via a module-level solve-function index
- Remove Python 2 compatibility line from examples
- quadprog: properly return `None` on unfeasible problems (thanks to @DKenefake)

## [1.5] - 2020/12/05

### Added

- Upgrade to Python 3 and deprecate Python 2
- Saved Python 2 package versions to `requirements2.txt`

### Fixed

- Deprecation warning in CVXPY

## [1.4.1] - 2020/11/29

### Added

- New ``solve_ls`` function to solve linear Least Squares problems

### Fixed

- Call to ``print`` in PyPI description
- Handling of quadprog ValueError exceptions

## [1.4] - 2020/07/04

### Added

- Solver settings can now by passed to ``solve_qp`` as keyword arguments
- Started an [API documentation](https://scaron.info/doc/qpsolvers/)

### Changed

- Made ``verbose`` an explicit keyword argument of all internal functions
- OSQP settings now match precision of other solvers (thanks to @Neotriple)

## [1.3.1] - 2020/06/13

### Fixed

- Equation of quadratic program on [PyPI page](https://pypi.org/project/qpsolvers/)

## [1.3] - 2020/05/16

### Added

- Lower and upper bound keyword arguments ``lb`` and ``ub``

### Fixed

- Check that equality/inequality matrices/vectors are provided consistently
- Relaxed offset check in [test\_solvers.py](examples/test_solvers.py)

## [1.2.1] - 2020/05/16

### Added

- cvxpy: verbose keyword argument
- ecos: verbose keyword argument
- gurobi: verbose keyword argument
- osqp: verbose keyword argument

### Fixed

- Ignore verbosity argument when solver is not available

## [1.2] - 2020/05/16

### Added

- cvxopt: verbose keyword argument
- mosek: verbose keyword argument
- qpoases: verbose keyword argument

## [1.1.2] - 2020/05/15

### Fixed

- osqp: handle both old and more recent versions

## [1.1.1] - 2020/05/15

### Fixed

- Avoid variable name clash in OSQP
- Handle quadprog exception to avoid confusion on cost matrix notation

## [1.1] - 2020/03/07

### Added

- ECOS solver interface (no need to go through CVXPY any more)
- Update ECOS performance in benchmark (much better than before!)

### Fixed

- Fix link to ECOS in setup.py
- Remove ned for IPython in solver test
- Update notes on P matrix

## [1.0.7] - 2019/10/26

### Changed

- Always reshape A or G vectors into one-line matrices

### Fixed

- cvxopt: handle case where G and h are None but not A and b
- osqp: handle case where G and h are None
- osqp: handle case where both G and A are one-line matrices
- qpoases: handle case where G and h are None but not A and b

## [1.0.6] - 2019/10/26

Thanks to Brian Delhaisse and Soeren Wolfers who contributed fixes to this
release!

### Fixed

- quadprog: handle case where G and h are None
- quadprog: handle cas where A.ndim == 1
- Make examples compatible with both Python 2 and Python 3

## [1.0.5] - 2019/04/10

### Added

- Equality constraint shown in the README example
- Installation file ``requirements.txt``
- Installation instructions for qpOASES
- OSQP: automatic CSC matrix conversions (with performance warnings)
- This change log

### Fixed

- CVXOPT: case where A is one-dimensional
- qpOASES: case where both G and A are not None
- quadprog: wrapper for one-dimensional A matrix (thanks to @nvitucci)

### Changed

- CVXOPT version is now 1.1.8 due to [this issue](https://github.com/urinieto/msaf-gpl/issues/2)
- Examples now in a [separate folder](examples)

## [1.0.4] - 2018/07/05

### Added

- Let's take this change log from there.
