"""Exceptions raised by sofic."""


class SoficError(Exception):
    """Base class for sofic errors."""


class SoficValidationError(SoficError):
    """Raised when a model fails structural or semantic validation."""


class NonDeterministicError(SoficValidationError):
    """Raised when a DFA determinism invariant is violated."""


class StochasticValidationError(SoficValidationError):
    """Raised when probability masses are invalid."""


class MixedStateExplosionError(StochasticValidationError):
    """Raised when the mixed-state presentation does not close within ``max_states``.

    This is distinct from a malformed model: the belief set may be genuinely
    infinite. A process can have a finite forward ε-machine and infinitely many
    retrodictive causal states, in which case the reverse mixed-state
    presentation never closes no matter how large the cap
    (see :cite:`Crutchfield2009` for :math:`\\Delta C_\\mu`).
    """


class UnifilarityError(SoficValidationError):
    """Raised when a unifilarity invariant is violated."""


class QuasiStochasticValidationError(SoficValidationError):
    """Raised when quasi-stochastic invariants are violated."""


class LumpabilityError(SoficValidationError):
    """Raised when a partition is not strongly lumpable for a model."""


class InfiniteTransductionError(SoficError):
    """Raised when a finite input has infinitely many transducer outputs."""
