============
bp-k-means
============

.. image:: https://img.shields.io/badge/python-3.13+-blue.svg
    :target: https://www.python.org/downloads/
    :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
    :alt: License

Project Description
-------------------

Boundary Preserving K-Means (BP-KMeans) is a label-constrained clustering
algorithm. It preserves the boundaries defined by pre-existing labels while
allocating additional clusters until the requested number of clusters is reached.

The project includes configurable ranking and centroid initialization strategies,
along with standard k-means, COP-KMeans, hierarchical agglomerative clustering,
and optimized bisecting k-means implementations for comparison and benchmarking.

Installation
------------

The package requires Python 3.13 or newer:

.. code-block:: bash

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install .

Core API
--------

Run boundary-preserving clustering with a pre-existing label for each point:

.. code-block:: python

    import numpy as np

    from bp_k_means import bp_kmeans

    X = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0], [6.0, 5.0]])
    y = np.array(["left", "left", "right", "right"])
    clusters = bp_kmeans(X, y, target_k=4, seed=42)

The returned array contains one cluster assignment per row of ``X``. Clusters are
label-pure: points with different values in ``y`` are never assigned to the same cluster.

Reproducing the benchmark
--------------------------

Datasets are expected in ``data/datasets``. The version-controlled benchmark configuration
is ``experiments/default.toml``; its paths are relative to the configuration file. After
obtaining the documented datasets, run:

.. code-block:: bash

    bp-k-means-benchmark --config experiments/default.toml

The command writes benchmark outputs to ``output/benchmark`` and records the effective
configuration in ``output/benchmark/experiment_config.json``. Benchmark results include elapsed
time, WCSS statistics, cluster assignments, and the random seed used for the run.

Set ``skip_existing = true`` to resume a partial benchmark: combinations that already have a
``metadata.json`` output are skipped. It defaults to ``false``.

COP-KMeans is available as an opt-in comparator; set ``include_cop_kmeans = true`` in the TOML
configuration when it should be included.

The two bisecting implementations can be controlled independently with
``include_bisecting_kmeans`` and ``include_precomputed_bisecting_kmeans``; both default to
``true``.

To generate figures and aggregate tables from those outputs, which are written to
``output/analysis``:

.. code-block:: bash

    bp-k-means-analyze

The analysis compares BP-KMeans initialized with k-means++ against standard Bisecting KMeans;
other benchmark algorithms and initialization variants are excluded from the analysis outputs.

Data provenance and licensing must be reviewed before redistributing datasets generated from
OpenStreetMap or INE sources. See the documentation for the source URLs and processing steps.

Documentation
-------------

- 📦 `Installation Guide <docs/installation.rst>`_ - Setup instructions and requirements
- 📚 `API Reference <docs/api.rst>`_ - Public functions and classes
- 🤝 `Contributing Guidelines <CONTRIBUTING.rst>`_ - Development standards and contribution process
- 📄 `License <LICENSE.txt>`_ - License terms and usage rights
- 👥 `Authors <AUTHORS.rst>`_ - Project contributors and maintainers
- 📜 `Changelog <CHANGELOG.rst>`_ - Project history and version changes
- 📜 `Code of Conduct <CODE_OF_CONDUCT.rst>`_ - Guidelines for participation and conduct
