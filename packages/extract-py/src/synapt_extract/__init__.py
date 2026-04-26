"""synapt-extract: SynaptExtraction IL v1 schema, validation, and finalization."""

from synapt_extract.schema import (
    SynaptExtraction,
    SynaptEntity,
    SynaptGoal,
    SynaptFact,
    SynaptRelation,
    SynaptSourceRef,
    SynaptEmbedding,
    SynaptAssertionSignals,
    SynaptTemporalRef,
)
from synapt_extract.validate import validate_extraction, ValidationResult, ValidationError
from synapt_extract.finalize import finalize_extraction, FinalizeContext, FinalizeResult

__all__ = [
    "SynaptExtraction",
    "SynaptEntity",
    "SynaptGoal",
    "SynaptFact",
    "SynaptRelation",
    "SynaptSourceRef",
    "SynaptEmbedding",
    "SynaptAssertionSignals",
    "SynaptTemporalRef",
    "validate_extraction",
    "ValidationResult",
    "ValidationError",
    "finalize_extraction",
    "FinalizeContext",
    "FinalizeResult",
]
