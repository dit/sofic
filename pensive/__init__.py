"""
pensive is a Python package for hidden Markov models, symbolic dynamics,
finite state machines, and other stochastic symbol generators.
"""

try:
    from importlib.metadata import version

    __version__ = version("pensive")
except Exception:
    __version__ = "0.0.0"
