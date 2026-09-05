Benchmark analysis
==================

Run the analysis after the benchmark has produced metadata:

.. code-block:: bash

    uv run bp-k-means-analyze --config experiments/default.toml

Use ``--show-titles`` to include plot titles. The command reads the benchmark metadata, loads
dataset sizes from the configured dataset directory, computes relative metrics, writes CSV
summaries, and generates PNG figures.

Regular benchmark analysis
--------------------------

The base analysis compares standard ``Bisecting KMeans`` with BP-KMeans runs initialized using
k-means++. Other algorithm rows remain available in the raw benchmark output but are excluded
from this comparison. Relative WCSS and relative runtime are normalized within each dataset and
cluster-multiplier case against the best result in that case; a value of ``1.0`` is the best
observed value.

Results are written under ``analysis_output_dir / base``:

* ``relative_metrics.csv``: one normalized row per dataset, algorithm, initialization count, and
  cluster multiplier.
* ``overall_avg.csv``: averages grouped by algorithm and ``n_init``.
* ``by_k_multiplier.csv``: averages split by requested cluster multiplier.
* ``by_size_bin.csv``: averages split by inferred dataset size bin.
* PNG files: bar charts, lines, WCSS/runtime scatter plots, and Pareto-front plots for each view.

HAC-strength and special analyses
----------------------------------

When their benchmark stages have data, HAC-strength results are written under
``analysis_output_dir / hac_strength`` with the same relative-metric tables and plot families.

The special stage writes absolute distance metrics for Community of Madrid and Castile and León
under ``com_madrid_distance_metrics`` and ``castile_leon_distance_metrics``. These outputs
include ``overall_avg.csv``, one plot per distance metric, scatter and Pareto plots, and time
comparison tables. Special analyses use the metric values directly rather than normalizing them
as relative WCSS and runtime.

If no matching metadata is available for a stage, the command logs a warning or leaves that
stage's output directory unchanged. Existing benchmark outputs are not modified by analysis.
