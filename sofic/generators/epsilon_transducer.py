"""Epsilon-transducers: minimal unifilar presentations of channels.

Computational mechanics quantifies structure in a single stochastic process via
its causal states, yielding the minimal optimal predictor -- the ε-machine.
Barnett & Crutchfield (*Computational Mechanics of Input-Output Processes:
Structured Transformations and the ε-Transducer*, J. Stat. Phys. 161:2 (2015)
404-451, doi:10.1007/s10955-015-1327-5) extend this to communication channels
coupling two processes, obtaining the ε-transducer: the tuple
``(X, Y, S, T)`` of input alphabet, output alphabet, causal states, and
conditional-symbol transition probabilities ``T(y, s' | s, x)``.

Structurally the ε-transducer is a unifilar, input-driven stochastic Mealy
machine, so :class:`EpsilonTransducer` extends the probability-aware
:class:`~sofic.automata.transducers.MealyMachine` with a causal-state initial
distribution, causal-state minimization, transCSSR reconstruction, and channel
information measures.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from sofic.automata.transducers import MealyMachine
from sofic.exceptions import StochasticValidationError, UnifilarityError

if TYPE_CHECKING:
    from sofic.generators.base import HiddenMarkovModel


class EpsilonTransducer(MealyMachine):
    """Unifilar stochastic Mealy machine: minimal causal presentation of a channel.

    The channel law lives on edges as ``T(y, s' | s, x)``: each outgoing edge of
    causal state ``s`` carries an input symbol ``x``, an output symbol ``y``, and
    a probability, with the probabilities of all ``(y, s')`` sharing a given
    ``(s, x)`` summing to one. Unifilarity means the observed pair ``(x, y)``
    determines the successor causal state.

    Examples
    --------
    >>> from sofic.examples.processes import BinaryChannel
    >>> from sofic import EpsilonTransducer
    >>> channel = BinaryChannel(0.1, 0.2)
    >>> eps = EpsilonTransducer.from_channel(channel)
    >>> eps.is_unifilar()
    True
    """

    initial_distribution: dict[Hashable, float]

    def __init__(
        self,
        input_alphabet: frozenset[Any] | None = None,
        output_alphabet: frozenset[Any] | None = None,
        initial_states: frozenset[Hashable] | None = None,
        initial_distribution: Mapping[Hashable, float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            input_alphabet=input_alphabet,
            output_alphabet=output_alphabet,
            initial_states=initial_states,
            **kwargs,
        )
        self.initial_distribution = dict(initial_distribution or {})

    def validate(self) -> None:
        super().validate()
        self.validate_stochastic()
        self._check_unifilar()
        self._validate_initial_distribution()

    def _validate_initial_distribution(self) -> None:
        if not self.initial_distribution:
            return
        total = float(sum(self.initial_distribution.values()))
        if not np.isclose(total, 1.0):
            raise StochasticValidationError(f"initial distribution sums to {total}, not 1")
        for state, mass in self.initial_distribution.items():
            if mass < 0:
                raise StochasticValidationError(f"negative initial probability at {state!r}")
            self._require(self.graph.has_state(state), f"unknown initial state {state!r}")

    def _check_unifilar(self) -> None:
        if self.is_unifilar():
            return
        raise UnifilarityError("epsilon-transducer must be unifilar on (state, input, output)")

    def is_unifilar(self) -> bool:
        """Return whether ``(state, input, output)`` determines the successor state."""
        from sofic.properties import is_unifilar_transducer

        return is_unifilar_transducer(self)

    def causal_states(self) -> list[Hashable]:
        """Return the causal states (an alias for :meth:`states`)."""
        return list(self.states())

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_channel(cls, channel: MealyMachine, **kwargs: Any) -> EpsilonTransducer:
        """Minimize a (joint-unifilar) stochastic transducer to causal states."""
        from sofic.generators.epsilon_transducer_construction import build_epsilon_transducer

        return build_epsilon_transducer(channel, **kwargs)

    @classmethod
    def from_iohmm(cls, channel: MealyMachine, **kwargs: Any) -> EpsilonTransducer:
        """Alias for :meth:`from_channel` (an input-output HMM is a stochastic Mealy machine)."""
        return cls.from_channel(channel, **kwargs)

    @classmethod
    def from_joint_generator(cls, generator: HiddenMarkovModel, **kwargs: Any) -> EpsilonTransducer:
        """Build the ε-transducer from a generator emitting ``(input, output)`` pairs."""
        from sofic.generators.epsilon_transducer_construction import from_joint_generator

        return from_joint_generator(generator, **kwargs)

    @classmethod
    def from_paired_sequences(
        cls,
        inputs: Sequence[Any],
        outputs: Sequence[Any],
        **kwargs: Any,
    ) -> EpsilonTransducer:
        """Reconstruct an ε-transducer from paired input/output sequences via transCSSR."""
        from sofic.generators.epsilon_transducer_inference import transcssr

        return transcssr(inputs, outputs, **kwargs)

    # -- channel measures -----------------------------------------------------

    def statistical_complexity(self, input_process: HiddenMarkovModel) -> float:
        """Channel statistical complexity ``H[S]`` when driven by ``input_process``."""
        from sofic.generators.channel_measures import channel_statistical_complexity

        return channel_statistical_complexity(self, input_process)

    def driven_entropy_rate(self, input_process: HiddenMarkovModel) -> float:
        """Entropy rate of the output process induced by ``input_process``."""
        from sofic.generators.channel_measures import driven_entropy_rate

        return driven_entropy_rate(self, input_process)

    def directed_information(self, input_process: HiddenMarkovModel, *, length: int = 1) -> float:
        """Directed information ``I(X^n -> Y^n)`` when driven by ``input_process``."""
        from sofic.generators.channel_measures import directed_information

        return directed_information(self, input_process, length=length)

    def transfer_entropy(self, input_process: HiddenMarkovModel, *, history: int = 1) -> float:
        """Transfer entropy from input to output when driven by ``input_process``."""
        from sofic.generators.channel_measures import transfer_entropy

        return transfer_entropy(self, input_process, history=history)

    # -- conversions ----------------------------------------------------------

    @classmethod
    def from_wfst(cls, wfst: Any) -> EpsilonTransducer:
        """Build an ε-transducer from a probability-semiring weighted transducer."""
        if getattr(wfst, "semiring", "probability") != "probability":
            raise ValueError("from_wfst requires a probability-semiring WFST")
        return cls.from_channel(MealyMachine.from_networkx(wfst.to_networkx(), initial_states=wfst.initial_states))
