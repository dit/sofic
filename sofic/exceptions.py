"""Exceptions raised by sofic."""


class SoficError(Exception):
    """Base class for sofic errors."""


class SoficValidationError(SoficError):
    """Raised when a model fails structural or semantic validation."""


class NonDeterministicError(SoficValidationError):
    """Raised when a DFA determinism invariant is violated."""


class StochasticValidationError(SoficValidationError):
    """Raised when probability masses are invalid."""


class UnifilarityError(SoficValidationError):
    """Raised when a unifilarity invariant is violated."""


class QuasiStochasticValidationError(SoficValidationError):
    """Raised when quasi-stochastic invariants are violated."""


class LumpabilityError(SoficValidationError):
    """Raised when a partition is not strongly lumpable for a model."""


class InfiniteTransductionError(SoficError):
    """Raised when a finite input has infinitely many transducer outputs."""
