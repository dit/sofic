General Information
===================

Installation
------------

Install from PyPI::

   pip install pensive

For development, clone the repository and sync dependencies with `uv`::

   git clone https://github.com/dit/pensive.git
   cd pensive
   uv sync --extra dev

Testing
-------

Run the test suite::

   uv run pytest

Linting and type checking::

   uv run ruff check .
   uv run ruff format --check .
   uv run ty check

Building documentation
----------------------

::

   uv run sphinx-build -W --keep-going -b html docs docs/_build/html
