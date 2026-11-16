"""Analytic Hierarchy Process (Saaty) weights and consistency check."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# Saaty's random consistency index by matrix size.
RANDOM_INDEX = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
CONSISTENCY_THRESHOLD = 0.10


@dataclass(frozen=True)
class AHPResult:
    weights: list[float]
    lambda_max: float
    consistency_index: float
    consistency_ratio: float
    consistent: bool

    def as_dict(self) -> dict:
        return asdict(self)


def ahp_weights(matrix: list[list[float]]) -> AHPResult:
    """Return principal-eigenvector weights of a reciprocal pairwise matrix."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] < 1:
        raise ValueError("La matrice AHP doit être carrée.")
    n = values.shape[0]
    if n > max(RANDOM_INDEX):
        raise ValueError(f"AHP limité à {max(RANDOM_INDEX)} critères.")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Les jugements AHP doivent être des nombres strictement positifs.")
    if np.any(values > 9 + 1e-9) or np.any(values < 1 / 9 - 1e-9):
        raise ValueError("Les jugements AHP doivent être sur l'échelle de Saaty (1/9 à 9).")
    if not np.allclose(np.diag(values), 1.0):
        raise ValueError("La diagonale de la matrice AHP doit valoir 1.")
    if not np.allclose(values * values.T, 1.0, rtol=1e-3, atol=1e-3):
        raise ValueError("La matrice AHP doit être réciproque (a_ji = 1 / a_ij).")

    eigenvalues, eigenvectors = np.linalg.eig(values)
    index = int(np.argmax(eigenvalues.real))
    vector = np.abs(eigenvectors[:, index].real)
    weights = vector / vector.sum()
    lambda_max = float(eigenvalues[index].real)
    ci = (lambda_max - n) / (n - 1) if n > 2 else 0.0
    ri = RANDOM_INDEX[n]
    cr = ci / ri if ri > 0 else 0.0
    return AHPResult(
        weights=[round(float(w), 6) for w in weights],
        lambda_max=round(lambda_max, 6),
        consistency_index=round(ci, 6),
        consistency_ratio=round(cr, 6),
        consistent=cr <= CONSISTENCY_THRESHOLD,
    )
