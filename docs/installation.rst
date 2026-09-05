Installation
============

Prerequisites
-------------

Python 3.13 or newer and Git are required. The project uses ``uv`` to create environments and
install dependencies. Install ``uv`` by following the `official installation guide
<https://docs.astral.sh/uv/getting-started/installation/>`_.

User installation
-----------------

Create an isolated environment and install the package:

.. code-block:: bash

    git clone https://github.com/j-moralejo-pinas/bp-k-means.git
    cd bp-k-means
    uv python install 3.13
    uv sync

``uv sync`` creates the project environment in ``.venv`` and installs the locked runtime
dependencies. Activate it for direct command usage:

.. code-block:: bash

    source .venv/bin/activate

You can also run commands without activating the environment by prefixing them with ``uv run``.

Verify the installation and public API:

.. code-block:: bash

    uv run python -c "import bp_k_means; print(bp_k_means.__version__)"
    uv run python -c "from bp_k_means import bp_kmeans; print(bp_kmeans)"

Development installation
------------------------

Install the development and documentation dependencies:

.. code-block:: bash

    uv sync --extra dev --extra docs

Run the validation commands:

.. code-block:: bash

    uv run pytest
    uv run ruff check .
    uv run pyright
    uv run sphinx-build -W -b html docs docs/_build/html
