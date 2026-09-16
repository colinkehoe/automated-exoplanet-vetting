"""exovet: automated vetting of TESS transiting exoplanet candidates."""

from exovet.candidate import Candidate
from exovet.features import compute_features

__all__ = ["Candidate", "compute_features"]
__version__ = "0.1.0"
