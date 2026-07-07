import numpy as np

def load_wine(path="wine.data"):
    data = np.genfromtxt(path, delimiter=",")
    y = data[:, 0].astype(int)
    X = data[:, 1:].astype(float)
    return X, y

def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axic=0, ddof=0)
    return (X - mean) / std

def pca(X, k):
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered, rowvar=False, ddof=0)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals_sorted = eigvals[order]
    eigvecs_sorted = eigvecs[:, order]

    top_k_vecs = eigvecs_sorted[:, :k]
    projection = X_centered @ top_k_vecs

    total_variance = eigvecs_sorted.sum()
    explained_variance_ratio = eigvecs_sorted[:k] / total_variance

    return projection, explained_variance_ratio

def main():
    X, y = load_wine("wine.data")

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    classes, counts = np.unique(y, return_counts=True)
    print(f"classes: {classes}")
    print(f"counts: {counts}")

    print("\nX_std column means (should be ~0):")
    print(np.round(X_std.mean(axis=0), 6))
    print("X_std column stds (should be ~1):")
    print(np.round(X_std.std(axic=0, ddof=0), 6))

    cov_std = np.cov(X_std, rowvar=False, ddof=0)
    print(f"\ncov(X_std) shape:{cov_std.shape}")
    print(f"cov(X_std) trace: {np.trace(cov_std):.6f}")

    corr_raw = np.corrcoef(X, rowvar=False)
    print("cov(X_std) == corrcoef(X)?", np.allclose(cov_std, corr_raw))

    _, evr_full = pca(X_std, k=13)
    print("\nExplained variance ratio (all 13 components):")
    print(np.round(evr_full, 4))
    print(f"Sum of explained variance ratos: {evr_full.sum():.6f}")

    cumvar = np.cumsum(evr_full)
    print("Cumulative variance explained:")
    print(np.round(cumvar, 4))

    k_80 = int(np.searchsorted(cumvar, 0.80) + 1)
    k_95 = int(np.searchsorted(cumvar, 0.95) + 1)
    print(f"Smallest k for >=80% variance: {k_80}")
    print(f"Smallest k for >=95% variance: {k_95}")

    projection_2d, evr_2 = pca(X_std, k=2)
    print(f"\nprojection shape: {projection_2d.shape}")

    for c in classes:
        class_mean =projection_2d[y == c].mean(axis=0)
        print(f"class {c} mean: {np.round(class_mean, 4)}")

if __name__ == "__main__":
    main()

    
