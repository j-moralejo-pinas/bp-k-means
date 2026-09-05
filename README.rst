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

The project includes configurable ranking and centroid initialization strategies, standard
k-means, COP-KMeans, Ward hierarchical clustering, and optimized bisecting k-means implementations.

If you use this project in research, please cite the associated work. Citation metadata is
provided in `CITATION.cff <CITATION.cff>`_.

Quick Start
-----------

Install ``uv`` using the `official installation guide
<https://docs.astral.sh/uv/getting-started/installation/>`_, then run the paper benchmark suite
from a fresh checkout:

.. code-block:: bash

    git clone https://github.com/j-moralejo-pinas/bp-k-means.git
    cd bp-k-means
    uv python install 3.13
    uv sync

Download the benchmark datasets into ``data/datasets``, then run:

.. code-block:: bash

    uv run bp-k-means-download-zenodo
    uv run bp-k-means-benchmark --config experiments/default.toml
    uv run bp-k-means-analyze --config experiments/default.toml

The default configuration is the shortest path through the paper benchmark workflow. It resumes
completed cases, writes raw results to ``output/benchmark``, and writes tables and figures to
``output/analysis``.

Key Features
------------

* Label-pure clustering that never assigns points from different original labels to one cluster.
* BP-KMeans with configurable ranking metrics, initialization strategies, and initialization
  algorithms.
* K-Means, COP-KMeans, Ward HAC, standard Bisecting K-Means, and Bisecting K-Means M_RL
  comparison algorithms.
* Reproducible benchmark configurations with deterministic seeds, multiple initializations, and
  resumable output generation.
* CSV tables, Parquet assignments, metadata, publication plots, relative WCSS, and runtime
  analysis.

See the `full documentation <docs/index.rst>`_ for installation, algorithm usage, benchmark
configuration, and analysis output details.

Documentation
-------------

* `Installation Guide <docs/installation.rst>`_ - Set up the project with ``uv``.
* `Project Usage <docs/usage.rst>`_ - Use the base algorithms and common interfaces.
* `Benchmark Reproduction <docs/benchmarks.rst>`_ - Download data, run benchmarks, and configure
  additional cases.
* `Benchmark Analysis <docs/analysis.rst>`_ - Understand generated tables and figures.
* `API Reference <docs/index.rst>`_ - Browse the generated API documentation.
* `Contributing Guidelines <CONTRIBUTING.rst>`_ - Development standards and contribution process.
