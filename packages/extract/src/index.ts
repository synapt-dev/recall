export type {
  SynaptExtraction,
  SynaptEntity,
  SynaptGoal,
  SynaptFact,
  SynaptRelation,
  SynaptSourceRef,
  SynaptEmbedding,
  SynaptAssertionSignals,
  SynaptTemporalRef,
  ExtractionCapability,
} from "./schema.js";

export { validateExtraction } from "./validate.js";
export type { ValidationResult, ValidationError } from "./validate.js";

export { finalizeExtraction } from "./finalize.js";
export type { FinalizeContext, FinalizeResult } from "./finalize.js";
