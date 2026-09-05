Reproducing benchmarks
======================

Dataset setup
-------------

The benchmark reads Parquet files from ``data/datasets``. Each dataset must contain the numeric
columns ``x_utm`` and ``y_utm`` plus the configured label column, ``CUSEC`` by default. The
repository includes a downloader for the configured Zenodo record. Run it from the repository
root:

.. code-block:: bash

    uv run bp-k-means-download-zenodo

The script downloads every file in the record into ``data/datasets`` and skips files already
present. Check that the resulting filenames match the datasets expected by the benchmark before
starting a long run.

Running the benchmark
---------------------

Run the checked-in paper configuration with:

.. code-block:: bash

    uv run bp-k-means-benchmark --config experiments/default.toml

The command runs the selected regular and special benchmark stages. Set ``skip_existing = true``
to resume an interrupted run; a case is skipped when its ``metadata.json`` already exists.

Configuration file
------------------

The TOML file contains a ``[benchmark]`` table. Relative paths are resolved relative to the TOML
file, not the current working directory.

``experiments/default.toml`` uses the following settings:

* ``datasets_dir``: directory containing input Parquet datasets.
* ``benchmark_output_dir``: directory for per-run benchmark results.
* ``analysis_output_dir``: directory used by the analysis command.
* ``seed``: random seed passed to algorithms.
* ``k``: positive multipliers used by regular and HAC-strength stages.
* ``n_inits``: positive numbers of initialization attempts to benchmark.
* ``subsample_size``: sample size used by subsampled k-means++ initialization.
* ``run_regular``: run the regular benchmark over ordinary datasets.
* ``run_hac_strength``: run the HAC-strength stage, where requested clusters are based on node
  count.
* ``run_special``: run the two fixed, distance-to-representative-node benchmarks.
* ``include_cop_kmeans``, ``include_hac``, ``include_bisecting_kmeans``,
  ``include_bisecting_kmeans_m_rl``, and ``include_bp_kmeans``: include or omit each algorithm
  family.
* ``skip_existing``: reuse completed cases instead of overwriting them.

For regular benchmarks, each value in ``k`` requests approximately ``k * n_labels`` clusters,
where ``n_labels`` is the number of unique input labels. HAC-strength cases use approximately
``k * n_instances`` clusters, capped at the number of input rows. The special stage uses the
fixed datasets and target cluster counts defined by the benchmark runner.

Outputs
-------

Each run is stored below ``benchmark_output_dir`` using the dataset and sanitized algorithm name.
Every run directory contains:

* ``metadata.json``: dataset, algorithm, target, seed, duration, WCSS, and distribution statistics.
* ``centroids.csv``: one centroid row per resulting cluster.
* ``instances.parquet``: coordinates, original labels, and cluster assignments.

The effective configuration is saved to ``output/benchmark/experiment_config.json`` by default.
Use a separate output directory when comparing experiments with different settings.
