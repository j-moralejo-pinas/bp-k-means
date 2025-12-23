import numpy as np
from heapq import heappush, heappop


def hac_ward_by_label(X, y, target_k):
    """
    Hierarchical agglomerative clustering using Ward's criterion,
    restricted to merging only clusters of the same label.

    X: (n, d) array of features
    y: (n,) array of labels (e.g. CUSEC)
    target_k: desired number of clusters

    Returns:
        labels_final: (n,) array with cluster ids in [0, target_k - 1]
    """

    X = np.asarray(X)
    y = np.asarray(y)
    n, d = X.shape

    # Safety: cannot have fewer clusters than unique labels
    unique_labels = np.unique(y)
    min_clusters = len(unique_labels)
    if target_k < min_clusters:
        raise ValueError(
            f"target_k={target_k} is smaller than the number of labels={min_clusters}. "
            f"With the 'same-label only' constraint you cannot go below {min_clusters} clusters."
        )

    # ---------------------------------------------------
    # INITIAL CLUSTERS: each point is its own cluster
    # ---------------------------------------------------
    clusters = {i: [i] for i in range(n)}           # cluster_id -> list of point indices
    cluster_label = {i: y[i] for i in range(n)}     # cluster_id -> label
    active = set(range(n))                          # active cluster ids
    sizes = {i: 1 for i in range(n)}                # cluster_id -> size
    centroids = {i: X[i].copy() for i in range(n)}  # cluster_id -> centroid

    # ---------------------------------------------------
    # PRIORITY QUEUE: (cost, cluster_a, cluster_b)
    # Only same-label initial pairs get into the queue
    # ---------------------------------------------------
    pq = []

    # For speed: group indices by label first
    label_to_indices = {}
    for idx, lbl in enumerate(y):
        label_to_indices.setdefault(lbl, []).append(idx)

    for lbl, idxs in label_to_indices.items():
        m = len(idxs)
        for ii in range(m):
            i = idxs[ii]
            for jj in range(ii + 1, m):
                j = idxs[jj]
                mu_i = centroids[i]
                mu_j = centroids[j]
                diff = mu_i - mu_j
                # Ward cost for two singletons: (1*1)/(1+1) * ||diff||^2 = 0.5 * ||diff||^2
                cost = 0.5 * np.sum(diff * diff)
                heappush(pq, (cost, i, j))

    next_cid = n  # new cluster ids start after the original n

    # ---------------------------------------------------
    # HAC MERGING LOOP (Ward)
    # ---------------------------------------------------
    while len(active) > target_k:
        if not pq:
            # No more possible same-label merges but still > target_k clusters
            raise ValueError(
                f"No feasible merges left for label-restricted Ward HAC. "
                f"active_clusters={len(active)}, target_k={target_k}."
            )

        cost, a, b = heappop(pq)

        # Skip outdated clusters
        if a not in active or b not in active:
            continue

        # Just in case, enforce same-label constraint
        if cluster_label[a] != cluster_label[b]:
            continue

        # -------------------------------
        # Merge clusters a and b into new_id
        # -------------------------------
        new_id = next_cid
        next_cid += 1

        members_a = clusters[a]
        members_b = clusters[b]
        new_members = members_a + members_b

        clusters[new_id] = new_members
        cluster_label[new_id] = cluster_label[a]  # label is the same for both
        active.remove(a)
        active.remove(b)
        active.add(new_id)

        nA = sizes[a]
        nB = sizes[b]
        sizes[new_id] = nA + nB

        muA = centroids[a]
        muB = centroids[b]
        centroids[new_id] = (nA * muA + nB * muB) / (nA + nB)

        # Clean old entries
        del clusters[a]
        del clusters[b]
        del centroids[a]
        del centroids[b]
        del sizes[a]
        del sizes[b]

        # Add new merge candidates new_id <-> other (same label only)
        for other in list(active):
            if other == new_id:
                continue
            if cluster_label[other] == cluster_label[new_id]:
                nO = sizes[other]
                muO = centroids[other]
                muN = centroids[new_id]
                diff = muN - muO
                cost = (sizes[new_id] * nO) / (sizes[new_id] + nO) * np.sum(diff * diff)
                heappush(pq, (cost, new_id, other))

    # Sanity check: we must have exactly target_k active clusters
    if len(active) != target_k:
        raise RuntimeError(
            f"Internal error: len(active)={len(active)} but target_k={target_k}"
        )

    # ---------------------------------------------------
    # OUTPUT FINAL ASSIGNMENTS (RENORMALIZED)
    # ---------------------------------------------------
    active_list = sorted(active)
    new_id_map = {old: i for i, old in enumerate(active_list)}

    labels_final = np.zeros(n, dtype=int)
    for old_cid in active_list:
        new_cid = new_id_map[old_cid]
        for idx in clusters[old_cid]:
            labels_final[idx] = new_cid

    return labels_final
