# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [4.7.1] - 2025-06-03

### Added

- `py.typed` file to indicate tools like `mypy` to use type annotations (thanks to @ValerianRey)

## [4.7.0] - 2025-05-13

### Added

- New solver: [SIP](https://github.com/joaospinto/sip_python) (thanks to @joaospinto)
- warnings: Add `SparseConversionWarning` to filter the corresponding warning
- warnings: Base class `QPWarning` for all qpsolvers-related warnings
- warnings: Recall solver name when issuing conversion warnings

### Changed

- Add solver name argument to internal `ensure_sparse_matrices` function
- CICD: Update Python version to 3.10 in coverage job

### Fixed

- docs: Add jaxopt.OSQP to the list of supported solvers

### Removed

- OSQP: Remove pre-1.0 version pin
- OSQP: Update interface after relase of v1.0.4
- Warning that was issued every time an unsupported solver is available

## [4.6.0] - 2025-04-17

### Added

- New solver: [KVXOPT](https://github.com/sanurielf/kvxopt/) (thanks to @agroudiev)
- jaxopt.OSQP: Support JAX array inputs when jaxopt.OSQP is the selected solver

## [4.5.1] - 2025-04-10

### Changed

- CICD: Update checkout action to v4

### Fixed

- OSQP: Temporary fix in returning primal-dual infeasibility certificates

## [4.5.0] - 2025-03-04

### Added

- HPIPM: Document new `tol_dual_gap` parameter
- New solver: [jaxopt.OSQP](https://jaxopt.github.io/stable/_autosummary/jaxopt.OSQP.html)
- Support Python 3.12

### Changed

- Bump minimum Python version to 3.8
- CICD: Remove Python 3.8 from continuous integration
- Fix output datatypes when splitting linear-box dual multipliers
- OSQP: version-pin to < 1.0.0 pending an interface update
- Warn when solving unconstrained problem by SciPy's LSQR rather than QP solver

### Fixed

- Fix mypy error in `Solution.primal_residual`

## [4.4.0] - 2024-09-24

### Added

- HPIPM: Link to reference paper for details on solver modes
- New solver: [qpax](https://github.com/kevin-tracy/qpax) (thanks to @lvjonok)

## [4.3.3] - 2024-08-06

### Changed

- CICD: Remove Gurobi from macOS continuous integration
- CICD: Remove Python 3.7 from continuous integration
- CICD: Update ruff to 0.4.3

### Fixed

- CICD: Fix coverage and licensed-solver workflows
- CICD: Install missing dependency in licensed solver test environment
- Clarabel: Catch pyO3 panics that can happen when building a problem
- Default arguments to active set dataclass to `None` rather than empty list
- PIQP: Warning message about CSC matrix conversions (thanks to @itsahmedkhalil)
- Update all instances of `np.infty` to `np.inf`

## [4.3.2] - 2024-03-25

### Added

- Optional dependency: `wheels_only` for solvers with pre-compiled binaries

### Changed

- Update developer notes in the documentation
- Update some solver tolerances in unit tests
- Warn rather than raise when there is no solver detected

### Fixed

- CICD: Update micromamba setup action

## [4.3.1] - 2024-02-06

### Fixed

- Gurobi: sign of inequality multipliers (thanks to @563925743)

## [4.3.0] - 2024-01-23

### Added

- Extend continuous integration to Python 3.11
- Function to get the CUTE classification string of the problem
- Optional dependencies for all solvers in the list available on PyPI

### Changed

- **Breaking:** no default QP solver installed along with the library
- NPPro: update exit flag value to match new solver API (thanks to @ottapav)

### Fixed

- Documentation: Add Clarabel to the list of supported solvers (thanks to @ogencoglu)
- Documentation: Correct note in `solve_ls` documentation (thanks to @ogencoglu)
- Documentation: Correct output of LS example (thanks to @ogencoglu)

## [4.2.0] - 2023-12-21

### Added

- Example: [lasso regularization](https://scaron.info/blog/lasso-regularization-in-quadratic-programming.html)
- `Problem.load` function
- `Problem.save` function

## [4.1.1] - 2023-12-05

### Changed

- Mark QPALM as a sparse solver only

## [4.1.0] - 2023-12-04

### Added

- New solver: [QPALM](https://kul-optec.github.io/QPALM/Doxygen/)
- Unit test for internal linear-box inequality combination

### Changed

- Internal: refactor linear-box inequality combination function
- Renamed main branch of the repository from `master` to `main`

### Fixed

- Fix combination of box inequalities with empty linear inequalities
- Gurobi: Account for a slight regression in QPSUT01 performance

## [4.0.1] - 2023-11-01

### Added

- Allow installation of a subset of QP solvers from PyPI

## [4.0.0] - 2023-08-30

### Added

- New solver: [PIQP](https://github.com/PREDICT-EPFL/piqp) (thanks to @shaoanlu)
- Type for active set of equality and inequality constraints

### Changed

- **Breaking:** condition number requires an active set (thanks to @aescande)

## [3.5.0] - 2023-08-16

### Added

- New solver: [HPIPM](https://github.com/giaf/hpipm) (thanks to @adamheins)

### Changed

- MOSEK: Disable CI test on QPSUT03 due to regression with 10.1.8
- MOSEK: Relax test tolerances as latest version is less accurate with defaults

## [3.4.0] - 2023-04-28

### Changed

- Converted THANKS file to [CFF](https://citation-file-format.github.io/)
- ECOS: raise a ProblemError if inequality vectors contain infinite values
- ECOS: raise a ProblemError if the cost matrix is not positive definite
- MOSEK is now a supported solver (thanks to @uricohen and @aszekMosek)

### Fixed

- Residual and duality gap computations when solution is not found
- Update OSQP version to 0.6.2.post9 for testing

## [3.3.1] - 2023-04-12

### Fixed

- DAQP: Update to 0.5.1 to fix installation of arm64 wheels

## [3.3.0] - 2023-04-11

### Added

- New sample problems in `qpsolvers.problems`
- New solver: [DAQP](https://darnstrom.github.io/daqp/) (thanks to @darnstrom)

### Changed

- Dual multipliers are empty arrays rather than None when no constraint
- Store solver results even when solution is not found
- Switch to `Solution.found` as solver success status (thanks to @rxian)

### Fixed

- Unit test on actual solution to QPSUT03 problem

## [3.2.0] - 2023-03-29

### Added

- Sparse strategy to convert LS problems to QP (thanks to @bodono)
- Start `problems` submodule to collect sample test problems

### Fixed

- Clarabel: upstream handling of infinite values in inequalities
- CVXOPT: option passing

## [3.1.0] - 2023-03-07

### Added

- New solver: NPPro

### Changed

- Documentation: separate support and unsupported solver lists
- Exclude unsupported solvers from code coverage report
- Move unsupported solvers to a separate submodule
- Remove CVXOPT from dependencies as it doesn't have arm64 wheels
- Remove quadprog from dependencies as it doesn't have arm64 wheels

## [3.0.0] - 2023-02-28

### Added

- Exception `ParamError` for incorrect solver parameters
- Exception `SolverError` for solver failures

### Changed

- All functions throw only qpsolvers-owned exceptions
- CVXOPT: rethrow `ValueError` as either `ProblemError` or `SolverError`
- Checking `Solution.is_empty` becomes `not Solution.found`
- Install open source solvers with wheels by default
- Remove `solve_safer_qp`
- Remove `sym_proj` parameter

## [2.8.1] - 2023-02-28

### Changed

- Expose `solve_unconstrained` function from main module

### Fixed

- Clarabel: handle unconstrained problems
- README: correct and improve FAQ on non-convex problems (thanks to @nrontsis)

## [2.8.0] - 2023-02-27

### Added

- New solver: [Clarabel](https://github.com/oxfordcontrol/Clarabel.rs)

### Changed

- Move documentation to [GitHub Pages](https://qpsolvers.github.io/qpsolvers/)
- Remove Python 2 installation instructions

## [2.7.4] - 2023-01-31

### Fixed

- Check vector shapes in problem constructor

## [2.7.3] - 2023-01-16

### Added

- qpOASES: return number of WSR in solution extra info

### Fixed

- CVXOPT: fix domain errors when some bounds are infinite
- qpOASES: fix missing lower bound when there is no equality constraint
- qpOASES: handle infinite bounds
- qpOASES: segmentation fault with conda feedstock

## [2.7.2] - 2023-01-02

### Added

- ECOS: handle two more exit flags
- Exception `ProblemError` for problem formulation errors
- Exception `QPError` as a base class for exceptions
- Property to check if a Problem has sparse matrices
- qpOASES: raise a ProblemError when matrices are not dense
- qpSWIFT: raise a ProblemError when matrices are not dense
- quadprog: raise a ProblemError when matrices are not dense

### Changed

- Add `use_sparse` argument to internal linear-from-box conversion
- Restrict condition number calculation to dense problems for now

## [2.7.1] - 2022-12-23

### Added

- Document problem conversion functions in developer notes
- ECOS: handle more exit flags

### Changed

- quadprog: use internal `split_dual_linear_box` conversion function

### Fixed

- SCS: require at least version 3.2
- Solution: duality gap computation under infinite box bounds

## [2.7.0] - 2022-12-15

### Added

- Continuous integration for macOS
- CVXOPT: return dual multipliers
- ECOS: return dual multipliers
- Example: dual multipliers
- Gurobi: return dual multipliers
- HiGHS: return dual multipliers
- MOSEK: return dual multipliers
- OSQP: return dual multipliers
- Problem class with utility metrics on quadratic programs
- Problem: condition number
- ProxQP: return dual multipliers
- qpOASES: return dual multipliers
- qpOASES: return objective value
- qpSWIFT: return dual multipliers
- qpSWIFT: return objective value
- quadprog: return dual multipliers
- SCS: return dual multipliers

### Changed

- Code: move `solve_safer_qp` to a separate source file
- Code: refactor location of internal conversions submodule
- ProxQP: bump minimum supported version to 0.2.9

### Fixed

- qpOASES: eliminate redundant equality constraints

## [2.6.0] - 2022-11-14

### Added

- Example: constrained linear regression
- Example: sparse linear least squares
- Gurobi: forward keyword arguments as solver parameters
- Handle diagonal matrices when combining linear and box inequalities
- qpOASES: pre-defined options parameter
- qpOASES: time limit parameter

### Changed

- CVXOPT: forward all keyword arguments as solver options
- Deprecate `solve_safer_qp` and warn about future removal
- Example: disable verbose output in least squares example
- HiGHS: forward all keyword arguments as solver options
- OSQP: drop support for versions <= 0.5.0
- OSQP: streamline stacking of box inequalities
- ProxQP: also consider constraint matrices to select backend
- qpOASES: forward all keyword arguments as solver options
- qpOASES: forward box inequalities directly
- Remove CVXPY which is not a solver
- SCS: `SOLVED_INACCURATE` is now considered a failure

### Fixed

- Dot product bug in `solve_ls` with sparse matrices
- MOSEK: restore CVXOPT options after calling MOSEK
- ProxQP: fix box inequality shapes when combining bounds
- qpOASES: non-persistent solver options between calls
- qpOASES: return failure on `RET_INIT_FAILED*` return codes

## [2.5.0] - 2022-11-04

### Added

- CVXOPT: absolute tolerance parameter
- CVXOPT: feasibility tolerance parameter
- CVXOPT: limit maximum number of iterations
- CVXOPT: refinement parameter
- CVXOPT: relative tolerance parameter
- Documentation: reference solver papers
- ECOS: document additional parameters
- Gurobi: time limit parameter
- HiGHS: dual feasibility tolerance parameter
- HiGHS: primal feasibility tolerance parameter
- HiGHS: time limit parameter

### Changed

- CVXOPT matrices are not valid types for qpsolvers any more
- CVXOPT: improve documentation
- CVXOPT: solver is now listed as sparse as well
- ECOS: type annotations allow sparse input matrices
- OSQP: don't override default solver tolerances
- Remove internal CVXOPT-specific type annotation
- Restrict matrix types to NumPy arrays and SciPy CSC matrices
- SCS: don't override default solver tolerances
- Simplify intermediate internal type annotations

### Fixed

- CVXOPT: pass warm-start primal properly
- ECOS: forward keyword arguments
- OSQP: dense arrays for vectors in type annotations
- SCS: fix handling of problems with only box inequalities

## [2.4.1] - 2022-10-21

### Changed

- Update ProxQP to version 0.2.2

## [2.4.0] - 2022-09-29

### Added

- New solver: [HiGHS](https://github.com/ERGO-Code/HiGHS)
- Raise error when there is no available solver

### Changed

- Make sure plot is shown in MPC example
- Print expected solutions in QP, LS and box-inequality examples
- Renamed starter solvers optional deps to `open_source_solvers`

### Fixed

- Correct documentation of `R` argument to `solve_ls`

## [2.3.0] - 2022-09-06

### Added

- New solver: [ProxQP](https://github.com/Simple-Robotics/proxsuite)

### Changed

- Clean up unused dependencies in GitHub workflow
- Non-default solver parameters in unit tests to test their precision

### Fixed

- Configuration of `tox-gh-actions` for Python 3.7
- Enforce `USING_COVERAGE` in GitHub workflow configuration
- Remove redundant solver loop from `test_all_shapes`

## [2.2.0] - 2022-08-15

### Added

- Add `lb` and `ub` arguments to all `<solver>_solve_qp` functions
- Internal `conversions` submodule

### Changed

- Moved `concatenate_bounds` to internal `conversions` submodule
- Moved `convert_to_socp` to internal `conversions` submodule
- Renamed `concatenate_bounds` to `linear_from_box_inequalities`
- Renamed internal `convert_to_socp` function to `socp_from_qp`

## [2.1.0] - 2022-07-25

### Added

- Document how to add a new QP solver to the library
- Example with (box) lower and upper bounds
- Test case where `lb` XOR `ub` is set

### Changed

- SCS: use the box cone API when lower/upper bounds are set

## [2.0.0] - 2022-07-05

### Added

- Exception `NoSolverSelected` raised when the solver kwarg is missing
- Starter set of QP solvers as optional dependencies
- Test exceptions raised by `solve_ls` and `solve_qp`

### Changed

- **Breaking:** `solver` keyword argument is now mandatory for `solve_ls`
- **Breaking:** `solver` keyword argument is now mandatory for `solve_qp`
- Quadratic programming example now randomly selects an available solver

## [1.10.0] - 2022-06-25

### Changed

- qpSWIFT: forward solver options as keywords arguments as with other solvers

## [1.9.1] - 2022-05-02

### Fixed

- OSQP: pass extra keyword arguments properly (thanks to @urob)

## [1.9.0] - 2022-04-03

### Added

- Benchmark on model predictive control problem
- Model predictive control example
- qpSWIFT 0.0.2 solver interface

### Changed

- Compute colors automatically in benchmark example

### Fixed

- Bounds concatenation for CVXOPT sparse matrices

## [1.8.1] - 2022-03-05

### Added

- Setup instructions for Microsoft Visual Studio
- Unit tests where the problem is unbounded below

### Changed

- Minimum supported Python version is now 3.7

### Fixed

- Clear all Pylint warnings
- Disable Pylint false positives that are covered by mypy
- ECOS: raise a ValueError when the cost matrix is not positive definite

## [1.8.0] - 2022-01-13

### Added

- Build and test for Python 3.10 in GitHub Actions

### Changed

- Moved SCS to sparse solvers
- Re-run solver benchmark reported to the README
- Removed `requirements2.txt` and update Python 2 installation instructions
- Updated SCS to new 3.0 version

### Fixed

- Handle sparse matrices in `print_matrix_vector`
- Match `__all__` in model and top-level `__init__.py`
- Run unit tests in GitHub Actions
- Typing error in bound concatenation

## [1.7.2] - 2021-11-24

### Added

- Convenience function to prettyprint a matrix and vector side by side

### Changed

- Move old tests from the examples folder to the unit test suite
- Removed deprecated `requirements.txt` installation file
- Renamed `solvers` optional dependencies to `all_pypi_solvers`

## [1.7.1] - 2021-10-02

### Fixed

- Make CVXOPT optional again (thanks to @adamoppenheimer)

## [1.7.0] - 2021-09-19

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

## [1.6.1] - 2021-04-09

### Fixed

- Add quadprog dependency properly in `pyproject.toml`

## [1.6.0] - 2021-04-09

### Added

- Add `__version__` to main module
- First unit tests to check all solvers over a pre-defined set of problems
- GitHub Actions now make sure the project is built and tested upon updates
- Type hints now decorate all function definitions

### Changed

- Code formatting now applies [Black](https://github.com/psf/black)
- ECOS: refactor SOCP conversion to improve function readability
- Gurobi: performance significantly improved by new matrix API (thanks to @DKenefake)

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

## [1.5.0] - 2020-12-05

### Added

- Upgrade to Python 3 and deprecate Python 2
- Saved Python 2 package versions to `requirements2.txt`

### Fixed

- Deprecation warning in CVXPY

## [1.4.1] - 2020-11-29

### Added

- New `solve_ls` function to solve linear Least Squares problems

### Fixed

- Call to `print` in PyPI description
- Handling of quadprog ValueError exceptions

## [1.4.0] - 2020-07-04

### Added

- Solver settings can now by passed to `solve_qp` as keyword arguments
- Started an [API documentation](https://scaron.info/doc/qpsolvers/)

### Changed

- Made `verbose` an explicit keyword argument of all internal functions
- OSQP settings now match precision of other solvers (thanks to @Neotriple)

## [1.3.1] - 2020-06-13

### Fixed

- Equation of quadratic program on [PyPI page](https://pypi.org/project/qpsolvers/)

## [1.3.0] - 2020-05-16

### Added

- Lower and upper bound keyword arguments `lb` and `ub`

### Fixed

- Check that equality/inequality matrices/vectors are provided consistently
- Relaxed offset check in [test\_solvers.py](examples/test_solvers.py)

## [1.2.1] - 2020-05-16

### Added

- CVXPY: verbose keyword argument
- ECOS: verbose keyword argument
- Gurobi: verbose keyword argument
- OSQP: verbose keyword argument

### Fixed

- Ignore verbosity argument when solver is not available

## [1.2.0] - 2020-05-16

### Added

- cvxopt: verbose keyword argument
- mosek: verbose keyword argument
- qpoases: verbose keyword argument

## [1.1.2] - 2020-05-15

### Fixed

- osqp: handle both old and more recent versions

## [1.1.1] - 2020-05-15

### Fixed

- Avoid variable name clash in OSQP
- Handle quadprog exception to avoid confusion on cost matrix notation

## [1.1.0] - 2020-03-07

### Added

- ECOS solver interface (no need to go through CVXPY any more)
- Update ECOS performance in benchmark (much better than before!)

### Fixed

- Fix link to ECOS in setup.py
- Remove ned for IPython in solver test
- Update notes on P matrix

## [1.0.7] - 2019-10-26

### Changed

- Always reshape A or G vectors into one-line matrices

### Fixed

- cvxopt: handle case where G and h are None but not A and b
- osqp: handle case where G and h are None
- osqp: handle case where both G and A are one-line matrices
- qpoases: handle case where G and h are None but not A and b

## [1.0.6] - 2019-10-26

Thanks to Brian Delhaisse and Soeren Wolfers who contributed fixes to this
release!

### Fixed

- quadprog: handle case where G and h are None
- quadprog: handle cas where A.ndim == 1
- Make examples compatible with both Python 2 and Python 3

## [1.0.5] - 2019-04-10

### Added

- Equality constraint shown in the README example
- Installation file `requirements.txt`
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

## [1.0.4] - 2018-07-05

### Added

- A changelog :)

[unreleased]: https://github.com/qpsolvers/qpsolvers/compare/v4.7.1...HEAD
[4.7.1]: https://github.com/qpsolvers/qpsolvers/compare/v4.7.0...v4.7.1
[4.7.0]: https://github.com/qpsolvers/qpsolvers/compare/v4.6.0...v4.7.0
[4.6.0]: https://github.com/qpsolvers/qpsolvers/compare/v4.5.1...v4.6.0
[4.5.1]: https://github.com/qpsolvers/qpsolvers/compare/v4.5.0...v4.5.1
[4.5.0]: https://github.com/qpsolvers/qpsolvers/compare/v4.4.0...v4.5.0
[4.4.0]: https://github.com/qpsolvers/qpsolvers/compare/v4.3.3...v4.4.0
[4.3.3]: https://github.com/qpsolvers/qpsolvers/compare/v4.3.2...v4.3.3
[4.3.2]: https://github.com/qpsolvers/qpsolvers/compare/v4.3.1...v4.3.2
[4.3.1]: https://github.com/qpsolvers/qpsolvers/compare/v4.3.0...v4.3.1
[4.3.0]: https://github.com/qpsolvers/qpsolvers/compare/v4.2.0...v4.3.0
[4.2.0]: https://github.com/qpsolvers/qpsolvers/compare/v4.1.1...v4.2.0
[4.1.1]: https://github.com/qpsolvers/qpsolvers/compare/v4.1.0...v4.1.1
[4.1.0]: https://github.com/qpsolvers/qpsolvers/compare/v4.0.1...v4.1.0
[4.0.1]: https://github.com/qpsolvers/qpsolvers/compare/v4.0.0...v4.0.1
[4.0.0]: https://github.com/qpsolvers/qpsolvers/compare/v3.5.0...v4.0.0
[3.5.0]: https://github.com/qpsolvers/qpsolvers/compare/v3.4.0...v3.5.0
[3.4.0]: https://github.com/qpsolvers/qpsolvers/compare/v3.3.1...v3.4.0
[3.3.1]: https://github.com/qpsolvers/qpsolvers/compare/v3.3.0...v3.3.1
[3.3.0]: https://github.com/qpsolvers/qpsolvers/compare/v3.2.0...v3.3.0
[3.2.0]: https://github.com/qpsolvers/qpsolvers/compare/v3.1.0...v3.2.0
[3.1.0]: https://github.com/qpsolvers/qpsolvers/compare/v3.0.0...v3.1.0
[3.0.0]: https://github.com/qpsolvers/qpsolvers/compare/v2.8.1...v3.0.0
[2.8.1]: https://github.com/qpsolvers/qpsolvers/compare/v2.8.0...v2.8.1
[2.8.0]: https://github.com/qpsolvers/qpsolvers/compare/v2.7.3...v2.8.0
[2.7.3]: https://github.com/qpsolvers/qpsolvers/compare/v2.7.2...v2.7.3
[2.7.2]: https://github.com/qpsolvers/qpsolvers/compare/v2.7.1...v2.7.2
[2.7.1]: https://github.com/qpsolvers/qpsolvers/compare/v2.7.0...v2.7.1
[2.7.0]: https://github.com/qpsolvers/qpsolvers/compare/v2.6.0...v2.7.0
[2.6.0]: https://github.com/qpsolvers/qpsolvers/compare/v2.5.0...v2.6.0
[2.5.0]: https://github.com/qpsolvers/qpsolvers/compare/v2.4.0...v2.5.0
[2.4.0]: https://github.com/qpsolvers/qpsolvers/compare/v2.3.0...v2.4.0
[2.3.0]: https://github.com/qpsolvers/qpsolvers/compare/v2.2.0...v2.3.0
[2.2.0]: https://github.com/qpsolvers/qpsolvers/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/qpsolvers/qpsolvers/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/qpsolvers/qpsolvers/compare/v1.10.0...v2.0.0
[1.10.0]: https://github.com/qpsolvers/qpsolvers/compare/v1.9.1...v1.10.0
[1.9.1]: https://github.com/qpsolvers/qpsolvers/compare/v1.9.0...v1.9.1
[1.9.0]: https://github.com/qpsolvers/qpsolvers/compare/v1.8.1...v1.9.0
[1.8.1]: https://github.com/qpsolvers/qpsolvers/compare/v1.8.0...v1.8.1
[1.8.0]: https://github.com/qpsolvers/qpsolvers/compare/v1.7.2...v1.8.0
[1.7.2]: https://github.com/qpsolvers/qpsolvers/compare/v1.7.1...v1.7.2
[1.7.1]: https://github.com/qpsolvers/qpsolvers/compare/v1.7.0...v1.7.1
[1.7.0]: https://github.com/qpsolvers/qpsolvers/compare/v1.6.1...v1.7.0
[1.6.1]: https://github.com/qpsolvers/qpsolvers/compare/v1.6.0...v1.6.1
[1.6.0]: https://github.com/qpsolvers/qpsolvers/compare/v1.5.0...v1.6.0
[1.5.0]: https://github.com/qpsolvers/qpsolvers/compare/v1.4.1...v1.5.0
[1.4.1]: https://github.com/qpsolvers/qpsolvers/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/qpsolvers/qpsolvers/compare/v1.3.1...v1.4.0
[1.3.1]: https://github.com/qpsolvers/qpsolvers/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/qpsolvers/qpsolvers/compare/v1.2.1...v1.3.0
[1.2.1]: https://github.com/qpsolvers/qpsolvers/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/qpsolvers/qpsolvers/compare/v1.1.2...v1.2.0
[1.1.2]: https://github.com/qpsolvers/qpsolvers/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/qpsolvers/qpsolvers/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/qpsolvers/qpsolvers/compare/v1.0.7...v1.1.0
[1.0.7]: https://github.com/qpsolvers/qpsolvers/compare/v1.0.6...v1.0.7
[1.0.6]: https://github.com/qpsolvers/qpsolvers/compare/v1.0.5...v1.0.6
[1.0.5]: https://github.com/qpsolvers/qpsolvers/compare/v1.0.4...v1.0.5
[1.0.4]: https://github.com/qpsolvers/qpsolvers/releases/tag/v1.0.4
