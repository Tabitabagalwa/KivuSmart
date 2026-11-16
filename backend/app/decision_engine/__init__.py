"""Multi-criteria restoration decision engine (RNA, reboisement, agroforesterie).

The original transparent rule engine in :mod:`backend.engine` is kept unchanged
and reported alongside the AHP-weighted scores for comparison.
"""

from .ahp import ahp_weights
from .scoring_engine import ENGINE_VERSION, evaluate_parcel, summarize_portfolio

__all__ = ["ENGINE_VERSION", "ahp_weights", "evaluate_parcel", "summarize_portfolio"]
