# Changelog

All notable changes to this project will be documented in this file.

## [1.2.1] - 2020/05/16

### Added

- ecos: verbose keyword argument
- gurobi: verbose keyword argument

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

- CVXOPT version is not 1.1.8 due to [this issue](https://github.com/urinieto/msaf-gpl/issues/2)
- Examples now in a [separate folder](examples)

## [1.0.4] - 2018/07/05

### Added
- Let's take this change log from there.
