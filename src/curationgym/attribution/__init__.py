"""Attribution module for CurationGym."""

from curationgym.attribution.composition import (
    CompositionVector,
    extract_composition_from_stats,
    extract_composition_from_manifest,
    collect_compositions,
)
from curationgym.attribution.ridge import (
    RidgeAttribution,
    RidgeAttributionResult,
    AttributionCoefficient,
    run_ridge_attribution,
)
from curationgym.attribution.ablation import (
    AblationAttribution,
    AblationResult,
    AblationStudyResult,
    create_ablation_table,
)
from curationgym.attribution.sanity_checks import (
    AttributionSanityChecker,
    SanityCheckResult,
)

__all__ = [
    "CompositionVector", "extract_composition_from_stats", "extract_composition_from_manifest", "collect_compositions",
    "RidgeAttribution", "RidgeAttributionResult", "AttributionCoefficient", "run_ridge_attribution",
    "AblationAttribution", "AblationResult", "AblationStudyResult", "create_ablation_table",
    "AttributionSanityChecker", "SanityCheckResult",
]