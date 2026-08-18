"""Independent metric target grids for epidemiological tensors."""

from .selection import CandidateGrid, build_grid, evaluate_candidate

__all__ = ["CandidateGrid", "build_grid", "evaluate_candidate"]
