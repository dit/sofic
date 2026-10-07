"""
Configuration for tests.

Select a Hypothesis profile with the ``HYPOTHESIS_PROFILE`` environment
variable: ``sofic`` (default), ``ci``, or ``nightly``.
"""

import os

from hypothesis import settings

settings.register_profile("sofic", deadline=None)
settings.register_profile("ci", deadline=None, max_examples=50)
settings.register_profile("nightly", deadline=None, max_examples=1000)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "sofic"))
