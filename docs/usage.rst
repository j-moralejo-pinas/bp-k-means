Using the algorithms
====================

Algorithm interface
-------------------

Every algorithm class implements ``BaseAlgo`` and follows the same fitted-model workflow:

.. code-block:: python

    model.fit(X, y, target_k)
    training_clusters = model.labels_
    new_clusters = model.predict(new_X, new_y)

The interface uses three inputs:

* ``X`` is a numeric array with shape ``(n_samples, n_features)``.
* ``y`` is either a one-dimensional array containing one source label per sample or ``None``.
* ``target_k`` is the number of clusters to produce during fitting.

Except for ``KMeans``, the implemented algorithms require ``y`` when fitting. They preserve the
source-label boundaries: samples with different source labels cannot belong to the same cluster.
Consequently, ``target_k`` must be at least the number of distinct source labels and cannot exceed
the number of samples. Cluster IDs are integers and should be treated as identifiers rather than
ordered values.

Fitting and predicting
~~~~~~~~~~~~~~~~~~~~~~

``fit(X, y, target_k)``
    Fits the selected algorithm, stores its learned state, and returns the model itself. The
    assignments for the training samples are available in ``labels_`` and the centroids at the
    fitted cut are available in ``centroids_``.

``fit_predict(X, y, target_k)``
    Performs the same fitting operation and returns the training assignments directly. The model
    remains fitted, so it can subsequently predict assignments for other samples.

``predict(X, y=None)``
    Assigns samples using the state retained during fitting. It does not fit the algorithm again
    and does not change ``labels_``. Each implementation applies its own assignment rule, described
    in `Implemented algorithms`_.

When ``y`` is passed to ``predict``, each sample is considered only for fitted clusters associated
with that source label. Omitting ``y`` allows each sample to be considered for every fitted
cluster. This only changes assignment constraints; it does not create a new clustering.

The following example fits BP-KMeans and then assigns both labelled and unlabelled samples:

.. code-block:: python

    import numpy as np

    from bp_k_means.algos.bp_kmeans import BPKMeans

    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
            [10.0, 0.0],
            [11.0, 0.0],
            [14.0, 0.0],
            [15.0, 0.0],
        ]
    )
    y = np.array(["left"] * 4 + ["right"] * 4)

    model = BPKMeans(seed=42, n_init=10, subsample_size=4)
    training_clusters = model.fit_predict(X, y, target_k=4)

    new_X = np.array([[0.5, 0.0], [14.5, 0.0]])
    labelled_clusters = model.predict(new_X, np.array(["left", "right"]))
    unrestricted_clusters = model.predict(new_X)

Calling ``predict`` before ``fit`` or ``fit_predict`` raises ``RuntimeError``. Labelled prediction
also raises ``ValueError`` when a supplied label was not present during fitting.

Implemented algorithms
----------------------

BP-KMeans
~~~~~~~~~

``BPKMeans`` is the main boundary-preserving algorithm. It begins with one cluster for every
source label, then repeatedly assigns an additional cluster to the selected label until
``target_k`` is reached. Its constructor exposes three aspects of that process:

* ``ranking_metric`` controls how the next source label is selected. ``M_L`` uses the label's
  total WCSS, ``M_C`` uses its largest single-cluster WCSS, ``M_ERL`` estimates the reduction from
  adding a cluster, and ``M_RL`` evaluates the reduction using a precomputed trial split.
* ``init_strategy`` controls how the centroids for the expanded label are initialized. ``I_LRI``
  reinitializes every centroid for the label; ``I_ACL`` retains the existing centroids and adds
  one; ``I_CRI`` replaces the highest-WCSS centroid with two; and ``I_ACC`` retains that centroid
  while adding another centroid within its cluster.
* ``init_algorithm`` selects ``KMEANS_PLUS_PLUS``, ``SUBSAMPLING_KMEANS_PLUS_PLUS``, or
  ``RANDOM_SAMPLING`` through ``InitAlgorithm``.

``n_init`` sets the number of k-means attempts made for each expansion. ``subsample_size`` is used
by the subsampled k-means++ initializer. Prediction assigns a sample to the nearest fitted
BP-KMeans centroid that is compatible with its source label, or to the nearest centroid overall
when no label is supplied.

.. code-block:: python

    from bp_k_means.algos.bp_kmeans import (
        BPKMeans,
        InitAlgorithm,
        InitStrategy,
        RankingMetric,
    )

    model = BPKMeans(
        ranking_metric=RankingMetric.M_ERL,
        init_strategy=InitStrategy.I_CRI,
        init_algorithm=InitAlgorithm.KMEANS_PLUS_PLUS,
        subsample_size=1000,
        seed=42,
        n_init=10,
    )

K-Means
~~~~~~~

``KMeans`` implements unconstrained Lloyd k-means and serves as the standard baseline. Pass
``y=None`` to ``fit`` or ``fit_predict``; labels supplied to this wrapper are ignored. ``n_init``
controls the number of independent initializations and ``max_iter`` limits each Lloyd run.
Prediction assigns every sample to its nearest fitted centroid.

.. code-block:: python

    from bp_k_means.algos.k_means import KMeans

    model = KMeans(seed=42, n_init=10, max_iter=300)
    clusters = model.fit_predict(X, None, target_k=4)

COP-KMeans
~~~~~~~~~~

``COPKMeans`` performs centroid-based clustering while enforcing the source-label constraint
during assignment. It tries ``n_init`` independent runs and retains the feasible result with the
lowest within-cluster sum of squares. ``max_iter`` limits each run, while ``init_ensure_label``
controls whether initialization selects at least one centroid from every source label.

Prediction assigns a sample to the nearest fitted centroid that is feasible for its supplied
source label. Without a supplied label, prediction uses the nearest fitted centroid.

.. code-block:: python

    from bp_k_means.algos.cop_k_means import COPKMeans

    model = COPKMeans(seed=42, n_init=10, max_iter=300, init_ensure_label=True)
    clusters = model.fit_predict(X, y, target_k=4)

Bisecting K-Means
~~~~~~~~~~~~~~~~~

Four optimized, label-constrained bisecting implementations are available:

.. list-table::
   :header-rows: 1
   :widths: 34 20 46

   * - Class
     - Ranking
     - Split behavior
   * - ``BisectingKMeans``
     - WCSS priority
     - Splits a selected cluster and refines all clusters belonging to that source label.
   * - ``BisectingKMeansNoRefine``
     - WCSS priority
     - Splits only the selected cluster and retains the fitted binary hierarchy.
   * - ``BisectingKMeansMRL``
     - Exact WCSS reduction (``M_RL``)
     - Splits a selected cluster and refines all clusters belonging to that source label.
   * - ``BisectingKMeansMRLNoRefine``
     - Exact WCSS reduction (``M_RL``)
     - Splits only the selected cluster and retains the fitted binary hierarchy.

All four classes use ``seed`` and ``n_init``. ``BisectingKMeans`` additionally accepts
``use_wcss_per_cluster`` to control its split-priority calculation.

The refined variants predict using their final refined centroids. Refinement can move samples
across earlier split boundaries, so those boundaries no longer describe the final clustering. The
non-refined variants retain the actual split hierarchy: prediction chooses the source-label root
and follows the nearest child at each split until it reaches a fitted leaf. Without ``y``, it first
selects the nearest source-label root.

.. code-block:: python

    from bp_k_means.algos.bisecting_k_means_optimized import (
        BisectingKMeans,
        BisectingKMeansNoRefine,
    )
    from bp_k_means.algos.bisecting_k_means_m_rl_optimized import (
        BisectingKMeansMRL,
        BisectingKMeansMRLNoRefine,
    )

    refined = BisectingKMeans(seed=42, n_init=10)
    hierarchical = BisectingKMeansNoRefine(seed=42, n_init=10)

Ward hierarchical clustering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``HACWard`` performs label-constrained agglomerative clustering using Ward's merge criterion.
``HACWardNNC`` implements the same clustering objective using a nearest-neighbor-chain hierarchy
construction. Both start with one cluster per training sample and merge only clusters associated
with the same source label until ``target_k`` clusters remain.

Prediction evaluates the Ward cost of inserting a sample into each compatible cluster at the
fitted hierarchy cut. This calculation uses both the fitted centroid and cluster size; it is not
ordinary nearest-centroid assignment.

.. code-block:: python

    from bp_k_means.algos.hac import HACWard, HACWardNNC

    ward = HACWard(seed=42)
    ward_nnc = HACWardNNC(seed=42)
    clusters = ward_nnc.fit_predict(X, y, target_k=4)

Choosing an implementation
--------------------------

Use ``BPKMeans`` for the algorithm proposed by this project. Use ``KMeans`` as an unconstrained
baseline and ``COPKMeans`` as a constrained centroid-based baseline. The bisecting classes are
divisive alternatives, with the non-refined variants preserving a prediction hierarchy and the
``MRL`` variants ranking splits by exact WCSS reduction. Use ``HACWardNNC`` for the optimized Ward
hierarchical baseline; ``HACWard`` provides the direct implementation for comparison.

The complete constructor and method signatures are available in the :doc:`API reference
<autoapi/index>`.
