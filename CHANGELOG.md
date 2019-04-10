# Changelog

All notable changes to this project will be documented in this file.

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
