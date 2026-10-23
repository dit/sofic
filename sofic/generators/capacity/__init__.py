"""Capacity bounds for finite unifilar channels (ε-transducers), with and without feedback."""

from sofic.generators.capacity._channel import ChannelTensors, channel_tensors
from sofic.generators.capacity.feedback import (
    duality_upper_bound,
    encoder_rate_bounds,
    feedback_capacity,
    qgraph_bounds,
    qgraph_lower_bound,
    qgraph_upper_bound,
    value_iteration_estimate,
)
from sofic.generators.capacity.no_feedback import (
    capacity,
    duality_upper_bound_no_feedback,
    markov_input_lower_bound,
)
from sofic.generators.capacity.qgraph import QGraph
from sofic.generators.capacity.results import CapacityBounds

__all__ = [
    "CapacityBounds",
    "ChannelTensors",
    "QGraph",
    "capacity",
    "channel_tensors",
    "duality_upper_bound",
    "duality_upper_bound_no_feedback",
    "encoder_rate_bounds",
    "feedback_capacity",
    "markov_input_lower_bound",
    "qgraph_bounds",
    "qgraph_lower_bound",
    "qgraph_upper_bound",
    "value_iteration_estimate",
]
