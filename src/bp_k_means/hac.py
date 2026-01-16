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
    clusters = {i: [i] for i in range(n)}  # cluster_id -> list of point indices
    cluster_label = {i: y[i] for i in range(n)}  # cluster_id -> label
    active = set(range(n))  # active cluster ids
    sizes = {i: 1 for i in range(n)}  # cluster_id -> size
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
        raise RuntimeError(f"Internal error: len(active)={len(active)} but target_k={target_k}")

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


def hac_ward_nnc_by_label(X, y, target_k):
    """
    Hierarchical Agglomerative Clustering using Ward linkage
    and Nearest-Neighbor Chain (NNC), restricted to same-label merges.

    Parameters
    ----------
    X : (n, d) array
        Feature matrix
    y : (n,) array
        Labels (merges allowed only within same label)
    target_k : int
        Desired total number of clusters

    Returns
    -------
    labels_final : (n,) array
        Cluster labels in [0, target_k-1]
    """

    X = np.asarray(X)
    y = np.asarray(y)
    n, d = X.shape

    labels = np.unique(y)
    if target_k < len(labels):
        raise ValueError(
            "target_k cannot be smaller than number of labels under label-restricted HAC."
        )

    # ------------------------------------------------------------
    # INITIALIZE CLUSTERS
    # ------------------------------------------------------------
    # Global cluster storage
    centroids = {}
    sizes = {}
    active = set()

    # Initial clusters: one per point
    for i in range(n):
        centroids[i] = X[i].copy()
        sizes[i] = 1
        active.add(i)

    next_cid = n

    # Per-label active clusters
    label_clusters = {}
    for lbl in labels:
        label_clusters[lbl] = set(np.where(y == lbl)[0])

    total_active = n

    # ------------------------------------------------------------
    # WARD DISTANCE
    # ------------------------------------------------------------
    def ward_distance(a, b):
        na, nb = sizes[a], sizes[b]
        diff = centroids[a] - centroids[b]
        return (na * nb) / (na + nb) * np.dot(diff, diff)

    # ------------------------------------------------------------
    # NNC PER LABEL
    # ------------------------------------------------------------
    for lbl in labels:
        chain = []
        clusters = label_clusters[lbl]

        while total_active > target_k and len(clusters) > 1:
            if not chain:
                chain.append(next(iter(clusters)))

            while True:
                a = chain[-1]

                # Find nearest neighbor of a
                best_b = None
                best_dist = np.inf
                for b in clusters:
                    if b == a:
                        continue
                    d_ab = ward_distance(a, b)
                    if d_ab < best_dist:
                        best_dist = d_ab
                        best_b = b

                # Extend chain
                if len(chain) >= 2 and best_b == chain[-2]:
                    # Reciprocal nearest neighbors -> merge
                    break
                else:
                    chain.append(best_b)

            # ----------------------------------------------------
            # MERGE last two in chain
            # ----------------------------------------------------
            b = chain.pop()
            a = chain.pop()

            new_id = next_cid
            next_cid += 1

            na, nb = sizes[a], sizes[b]
            mu_a, mu_b = centroids[a], centroids[b]

            centroids[new_id] = (na * mu_a + nb * mu_b) / (na + nb)
            sizes[new_id] = na + nb

            # Update active sets
            clusters.remove(a)
            clusters.remove(b)
            clusters.add(new_id)

            active.remove(a)
            active.remove(b)
            active.add(new_id)

            total_active -= 1

            # Cleanup
            del centroids[a], centroids[b]
            del sizes[a], sizes[b]

            # Reset chain
            chain.clear()

            if total_active <= target_k:
                break

    # ------------------------------------------------------------
    # ASSIGN FINAL LABELS
    # ------------------------------------------------------------
    active_list = sorted(active)
    cid_map = {cid: i for i, cid in enumerate(active_list)}

    labels_final = np.empty(n, dtype=int)

    # Track final membership via nearest centroid
    # (safe because HAC never mixes labels)
    for i in range(n):
        best = None
        best_dist = np.inf
        for cid in label_clusters[y[i]]:
            if cid in active:
                diff = X[i] - centroids[cid]
                d = np.dot(diff, diff)
                if d < best_dist:
                    best_dist = d
                    best = cid
        labels_final[i] = cid_map[best]

    return labels_final
