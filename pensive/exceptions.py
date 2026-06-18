"""Exceptions raised by pensive."""


class PensiveError(Exception):
    """Base class for pensive errors."""


class PensiveValidationError(PensiveError):
    """Raised when a model fails structural or semantic validation."""


class NonDeterministicError(PensiveValidationError):
    """Raised when a DFA determinism invariant is violated."""


class StochasticValidationError(PensiveValidationError):
    """Raised when probability masses are invalid."""


class UnifilarityError(PensiveValidationError):
    """Raised when a unifilarity invariant is violated."""


class QuasiStochasticValidationError(PensiveValidationError):
    """Raised when quasi-stochastic invariants are violated."""
