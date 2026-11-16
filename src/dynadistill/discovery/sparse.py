
"""Sequential Thresholded Least Squares for sparse equation discovery."""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def stlsq(
    x,
    y,
    degree=3,
    threshold=0.1,
    ridge=1e-5,
    max_iter=30,
):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D arrays")

    if len(x) != len(y):
        raise ValueError("Sample counts must match")

    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Inputs must contain finite values")

    if not (0 <= threshold < 1):
        raise ValueError(
            "Relative threshold must be between 0 and 1"
        )

    library = PolynomialFeatures(
        degree=degree,
        include_bias=True,
    )
    theta = library.fit_transform(x)

    # Normalize columns by RMS magnitude.
    scales = np.sqrt(np.mean(theta**2, axis=0))
    scales = np.maximum(scales, 1e-12)

    A = theta / scales

    n_features = A.shape[1]
    n_outputs = y.shape[1]

    coefficients = np.zeros(
        (n_features, n_outputs),
        dtype=np.float64,
    )

    for j in range(n_outputs):
        target = y[:, j]

        beta = np.linalg.solve(
            A.T @ A + ridge * np.eye(n_features),
            A.T @ target,
        )

        for _ in range(max_iter):
            max_coefficient = np.max(np.abs(beta))

            if max_coefficient < 1e-12:
                beta[:] = 0.0
                break

            cutoff = threshold * max_coefficient
            support = np.abs(beta) >= cutoff

            if not support.any():
                beta[:] = 0.0
                break

            updated = np.zeros_like(beta)
            updated[support] = np.linalg.lstsq(
                A[:, support],
                target,
                rcond=None,
            )[0]

            new_cutoff = threshold * np.max(
                np.abs(updated)
            )
            new_support = np.abs(updated) >= new_cutoff

            beta = updated

            if np.array_equal(support, new_support):
                break

        # Final selection and refit.
        if np.any(beta):
            cutoff = threshold * np.max(np.abs(beta))
            support = np.abs(beta) >= cutoff

            final_beta = np.zeros_like(beta)
            final_beta[support] = np.linalg.lstsq(
                A[:, support],
                target,
                rcond=None,
            )[0]

            beta = final_beta

        coefficients[:, j] = beta / scales

    return library, coefficients

