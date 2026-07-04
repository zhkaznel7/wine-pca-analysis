"""Load the UCI Wine dataset, standardize it, and run PCA from scratch
using only NumPy (no scikit-learn, no Pandas)."""

import numpy as np


def load_wine(path: str = "wine.data") -> tuple[np.ndarray, np.ndarray]:
    """Load wine.data. Returns (X, y) where X has shape (178, 13) of
    floats and y has shape (178,) of integer class labels."""
    raw = np.genfromtxt(path, delimiter=",")
    y = raw[:, 0].astype(int)
    X = raw[:, 1:]
    return X, y


def standardize(X: np.ndarray) -> np.ndarray:
    """Return X with each column having zero mean and unit variance."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std


def pca(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Run PCA on X, returning (projection, explained_variance_ratio).

    projection: shape (n, k) - X projected onto the top k components.
    explained_variance_ratio: shape (k,) - fraction of variance each
    of the top k components explains, in descending order.
    """
    # 1. center the data
    X_centered = X - X.mean(axis=0)

    # 2. covariance matrix
    cov = np.cov(X_centered, rowvar=False, ddof=0)

    # 3. eigendecomposition (cov is symmetric -> eigh)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # eigh returns eigenvalues ascending -> reverse to descending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # 4. keep top k
    top_eigvals = eigvals[:k]
    top_eigvecs = eigvecs[:, :k]

    # 5. project centered data onto top k eigenvectors
    projection = X_centered @ top_eigvecs

    # 6. explained variance ratio (over ALL eigenvalues, not just top k)
    explained_variance_ratio = top_eigvals / eigvals.sum()

    return projection, explained_variance_ratio


def main() -> None:
    X, y = load_wine("wine.data")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    classes, counts = np.unique(y, return_counts=True)
    print(f"classes: {classes}")
    print(f"counts:  {counts}")

    print("\nper-feature mean:")
    print(np.round(X.mean(axis=0), 3))
    print("per-feature std:")
    print(np.round(X.std(axis=0), 3))

    # --- standardize ---
    X_std = standardize(X)
    print("\nX_std per-column mean (approx 0):")
    print(np.round(X_std.mean(axis=0), 6))
    print("X_std per-column std (approx 1):")
    print(np.round(X_std.std(axis=0), 6))

    cov_matrix = np.cov(X_std, rowvar=False, ddof=0)
    print(f"\ncovariance matrix shape: {cov_matrix.shape}")
    print(f"covariance matrix trace: {np.trace(cov_matrix):.4f}")

    corr_matrix = np.corrcoef(X, rowvar=False)
    print(
        "cov(X_std) == corr(X_raw):",
        np.allclose(cov_matrix, corr_matrix),
    )

    # --- PCA with all 13 components ---
    _, evr_all = pca(X_std, k=13)
    print("\nexplained variance ratio (all 13, descending):")
    print(np.round(evr_all, 4))
    print(f"sum of explained variance ratios: {evr_all.sum():.4f}")

    cumulative = np.cumsum(evr_all)
    print("\ncumulative variance explained:")
    print(np.round(cumulative, 4))

    k_80 = int(np.argmax(cumulative >= 0.80) + 1)
    k_95 = int(np.argmax(cumulative >= 0.95) + 1)
    print(f"\ncomponents needed for >=80% variance: {k_80}")
    print(f"components needed for >=95% variance: {k_95}")

    # --- PCA with 2 components, per-class means ---
    projection_2d, _ = pca(X_std, k=2)
    print(f"\nprojection shape: {projection_2d.shape}")
    for cls in classes:
        class_mean = projection_2d[y == cls].mean(axis=0)
        print(f"class {cls} mean: {np.round(class_mean, 4)}")


if __name__ == "__main__":
    main()