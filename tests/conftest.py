"""
Configuration for tests.
"""

from hypothesis import settings

settings.register_profile("sofic", deadline=None)
settings.load_profile("sofic")
