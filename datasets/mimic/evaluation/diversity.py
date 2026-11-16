import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from .data import matrix


def diversity(train, real, generated, seed, *, pca=True, pca_dim=32):
    training, left, right = [matrix(batch) for batch in (train, real, generated)]
    size, neighbors = len(left), 5
    if len(right) != size:
        raise ValueError("Diversity requires equal real and generated test row counts")
    if size <= neighbors or (pca and len(training) < 2):
        return {
            "Coverage": None,
            "APDR": None,
            "rows": size,
            "reason": "Insufficient rows for the fixed neighborhood or PCA",
        }
    dimension = min(pca_dim, training.shape[1], len(training) - 1)
    if pca and not np.any(np.std(training, axis=0) > 0):
        return {
            "Coverage": None,
            "APDR": None,
            "rows": size,
            "reason": "Constant training representation",
        }
    if pca:
        projection = PCA(n_components=dimension, svd_solver="randomized", random_state=int(seed)).fit(training)
        left, right = (left - projection.mean_) @ projection.components_.T, (right - projection.mean_) @ projection.components_.T
    radii = (
        NearestNeighbors(n_neighbors=neighbors + 1, n_jobs=1).fit(left).kneighbors(left)[0][:, -1]
    )
    nearest = NearestNeighbors(n_neighbors=1, n_jobs=1).fit(right).kneighbors(left)[0][:, 0]
    rng = np.random.default_rng(seed)
    first = rng.integers(0, size, size=size)
    second = rng.integers(0, size - 1, size=size)
    second += second >= first
    real_apd = float(np.linalg.norm(left[first] - left[second], axis=1).mean(dtype=np.float64))
    generated_apd = float(np.linalg.norm(right[first] - right[second], axis=1).mean(dtype=np.float64))
    return {
        "Coverage": float(np.mean(nearest <= radii)),
        "APDR": generated_apd / real_apd if real_apd > 1e-12 else None,
        "real_APD": real_apd,
        "generated_APD": generated_apd,
        "rows": size,
        "pca": pca,
        "pca_dim": dimension if pca else None,
        "pca_fit": "real training partition only" if pca else None,
        "distance_dim": left.shape[1],
        "neighbors": neighbors,
    }
