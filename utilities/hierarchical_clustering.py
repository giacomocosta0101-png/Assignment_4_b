from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch


def hierarchical_clustering(
    matrix: Union[np.ndarray, pd.DataFrame],
    linkage_method: str = "single",
    distance_metric: str = "euclidean",
    plot_dendrogram: bool = False,
) -> pd.DataFrame:
    """
    Compute hierarchical clustering on the input matrix according to linkage_method and
    distance_metric.

    Parameters:
        matrix (Union[np.ndarray, pd.DataFrame]): Input matrix.

    Returns:
        pd.DataFrame: An (n - 1) by 4 DataFrame Z is returned. At the ith iteration, clusters with
            indices Z[i, 0] and Z[i, 1] are combined to form cluster n+i. A cluster with an index
            less than n corresponds to one of the original observations. The distance between
            clusters Z[i, 0] and Z[i, 1] is given by Z[i, 2]. The fourth value Z[i, 3] represents
            the number of original observations in the newly formed cluster.
    """

    data = matrix.values if isinstance(matrix, pd.DataFrame) else np.asarray(matrix)

    linkage_matrix = sch.linkage(
        data,
        method=linkage_method,
        metric=distance_metric,
    )

    if plot_dendrogram:
        plt.figure()
        sch.dendrogram(
            linkage_matrix,
            labels=matrix.index if isinstance(matrix, pd.DataFrame) else None,
            color_threshold=0,
            leaf_rotation=90,
        )
        plt.title(f"Dendrogram - Linkage: {linkage_method} - Metric: {distance_metric}")
        plt.show()

    linkage_matrix = pd.DataFrame(
        data=linkage_matrix,
        index=range(linkage_matrix.shape[0]),
        columns=["cluster_idx1", "cluster_idx2", "cluster_distance", "n_obs"],
    )
    linkage_matrix[["cluster_idx1", "cluster_idx2", "n_obs"]] = linkage_matrix[
        ["cluster_idx1", "cluster_idx2", "n_obs"]
    ].astype(int)

    return linkage_matrix
