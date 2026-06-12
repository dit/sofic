"""
Configuration for tests.
"""

from hypothesis import settings

settings.register_profile("pensive", deadline=None)
settings.load_profile("pensive")
