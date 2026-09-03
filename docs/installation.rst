Installation
============

Prerequisites
-------------

Python 3.13 or newer is required. Git is needed when installing directly from the repository.

User installation
-----------------

Create an isolated environment and install the package:

.. code-block:: bash

    git clone https://github.com/j-moralejo-pinas/bp-k-means.git
    cd bp-k-means
    python3.13 -m venv .venv
    source .venv/bin/activate
    python -m pip install .

Verify the installation and public API:

.. code-block:: bash

    python -c "import bp_k_means; print(bp_k_means.__version__)"
    python -c "from bp_k_means import bp_kmeans; print(bp_kmeans)"

Development installation
------------------------

Install the development and documentation dependencies:

.. code-block:: bash

    python -m pip install -e ".[dev,docs]"

Run the validation commands:

.. code-block:: bash

    pytest
    ruff check .
    pyright
    sphinx-build -W -b html docs docs/_build/html

Benchmark reproduction
----------------------

The reproducibility configuration is stored in ``experiments/default.toml``. Paths in that
file are resolved relative to the configuration file. Datasets must be available under
``data/datasets`` before running the benchmark:

.. code-block:: bash

    bp-k-means-benchmark --config experiments/default.toml
    bp-k-means-analyze

The benchmark records its effective configuration under ``output/benchmark/experiment_config.json`` and
stores the seed in each run's metadata. Dataset construction uses the source URLs and processing
steps documented in :mod:`bp_k_means.tools.create_dataset`; data providers' terms and attribution
requirements apply to any redistribution.

COP-KMeans is available as an optional comparator. Set ``include_cop_kmeans = true`` in the TOML
configuration to include it in the benchmark stages.
