import type { SynaptExtraction, ExtractionCapability } from "./schema.js";

export interface ValidationError {
  path: string;
  message: string;
}

export interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
}

const VALID_CAPABILITIES: Set<string> = new Set([
  "entities", "entity_state", "entity_context", "entity_ids",
  "goals", "goal_timing", "goal_entity_refs",
  "themes", "summary", "sentiment", "facts",
  "temporal_refs", "temporal_classes",
  "relations", "relation_origin",
  "assertion_signals", "evidence_anchoring",
]);

const VALID_ENTITY_TYPES: Set<string> = new Set([
  "person", "place", "event", "concept", "organization", "object",
]);

const VALID_GOAL_STATUSES: Set<string> = new Set([
  "open", "resolved", "abandoned", "in_progress",
]);

const VALID_TEMPORAL_TYPES: Set<string> = new Set([
  "point", "range", "duration", "unresolved",
]);

function validateSourceRef(obj: unknown, path: string, errors: ValidationError[]): void {
  if (typeof obj !== "object" || obj === null) {
    errors.push({ path, message: "must be an object" });
    return;
  }
  const ref = obj as Record<string, unknown>;
  if (ref.version !== "1") {
    errors.push({ path: `${path}.version`, message: "must be \"1\"" });
  }
  if (ref.snippet !== undefined && typeof ref.snippet !== "string") {
    errors.push({ path: `${path}.snippet`, message: "must be a string" });
  }
}

function validateSignals(obj: unknown, path: string, errors: ValidationError[]): void {
  if (typeof obj !== "object" || obj === null) {
    errors.push({ path, message: "must be an object" });
    return;
  }
  const sig = obj as Record<string, unknown>;
  if (sig.version !== "1") {
    errors.push({ path: `${path}.version`, message: "must be \"1\"" });
  }
  if (sig.confidence !== undefined) {
    if (typeof sig.confidence !== "number" || sig.confidence < 0 || sig.confidence > 1) {
      errors.push({ path: `${path}.confidence`, message: "must be a number between 0.0 and 1.0" });
    }
  }
  if (sig.negated !== undefined && typeof sig.negated !== "boolean") {
    errors.push({ path: `${path}.negated`, message: "must be a boolean" });
  }
  if (sig.hedged !== undefined && typeof sig.hedged !== "boolean") {
    errors.push({ path: `${path}.hedged`, message: "must be a boolean" });
  }
  if (sig.condition !== undefined && typeof sig.condition !== "string") {
    errors.push({ path: `${path}.condition`, message: "must be a string" });
  }
}

function validateEmbedding(obj: unknown, path: string, errors: ValidationError[]): void {
  if (typeof obj !== "object" || obj === null) {
    errors.push({ path, message: "must be an object" });
    return;
  }
  const emb = obj as Record<string, unknown>;
  if (emb.version !== "1") {
    errors.push({ path: `${path}.version`, message: "must be \"1\"" });
  }
  if (!Array.isArray(emb.vector)) {
    errors.push({ path: `${path}.vector`, message: "required array" });
  }
  if (typeof emb.model !== "string") {
    errors.push({ path: `${path}.model`, message: "required string" });
  }
  if (typeof emb.input !== "string") {
    errors.push({ path: `${path}.input`, message: "required string" });
  }
  if (typeof emb.dimensions !== "number" || !Number.isInteger(emb.dimensions) || emb.dimensions < 1) {
    errors.push({ path: `${path}.dimensions`, message: "required positive integer" });
  }
}

function validateRelation(obj: unknown, path: string, errors: ValidationError[]): void {
  if (typeof obj !== "object" || obj === null) {
    errors.push({ path, message: "must be an object" });
    return;
  }
  const rel = obj as Record<string, unknown>;
  if (typeof rel.target !== "string") {
    errors.push({ path: `${path}.target`, message: "required string" });
  }
  if (typeof rel.type !== "string") {
    errors.push({ path: `${path}.type`, message: "required string" });
  }
  if (rel.signals !== undefined) {
    validateSignals(rel.signals, `${path}.signals`, errors);
  }
}

function validateEntity(obj: unknown, path: string, errors: ValidationError[]): void {
  if (typeof obj !== "object" || obj === null) {
    errors.push({ path, message: "must be an object" });
    return;
  }
  const ent = obj as Record<string, unknown>;
  if (typeof ent.name !== "string") {
    errors.push({ path: `${path}.name`, message: "required string" });
  }
  if (typeof ent.type !== "string") {
    errors.push({ path: `${path}.type`, message: "required string" });
  }
  if (ent.source !== undefined) {
    validateSourceRef(ent.source, `${path}.source`, errors);
  }
  if (ent.signals !== undefined) {
    validateSignals(ent.signals, `${path}.signals`, errors);
  }
  if (ent.relations !== undefined) {
    if (!Array.isArray(ent.relations)) {
      errors.push({ path: `${path}.relations`, message: "must be an array" });
    } else {
      for (let i = 0; i < ent.relations.length; i++) {
        validateRelation(ent.relations[i], `${path}.relations[${i}]`, errors);
      }
    }
  }
}

function validateGoal(obj: unknown, path: string, errors: ValidationError[]): void {
  if (typeof obj !== "object" || obj === null) {
    errors.push({ path, message: "must be an object" });
    return;
  }
  const goal = obj as Record<string, unknown>;
  if (typeof goal.text !== "string") {
    errors.push({ path: `${path}.text`, message: "required string" });
  }
  if (typeof goal.status !== "string" || !VALID_GOAL_STATUSES.has(goal.status)) {
    errors.push({ path: `${path}.status`, message: "must be one of: open, resolved, abandoned, in_progress" });
  }
  if (!Array.isArray(goal.entity_refs)) {
    errors.push({ path: `${path}.entity_refs`, message: "required array of strings" });
  }
  if (goal.source !== undefined) {
    validateSourceRef(goal.source, `${path}.source`, errors);
  }
  if (goal.signals !== undefined) {
    validateSignals(goal.signals, `${path}.signals`, errors);
  }
}

function validateFact(obj: unknown, path: string, errors: ValidationError[]): void {
  if (typeof obj !== "object" || obj === null) {
    errors.push({ path, message: "must be an object" });
    return;
  }
  const fact = obj as Record<string, unknown>;
  if (typeof fact.text !== "string") {
    errors.push({ path: `${path}.text`, message: "required string" });
  }
  if (fact.source !== undefined) {
    validateSourceRef(fact.source, `${path}.source`, errors);
  }
  if (fact.signals !== undefined) {
    validateSignals(fact.signals, `${path}.signals`, errors);
  }
}

function validateTemporalRef(obj: unknown, path: string, errors: ValidationError[]): void {
  if (typeof obj !== "object" || obj === null) {
    errors.push({ path, message: "must be an object" });
    return;
  }
  const ref = obj as Record<string, unknown>;
  if (ref.version !== "1") {
    errors.push({ path: `${path}.version`, message: "must be \"1\"" });
  }
  if (typeof ref.raw !== "string") {
    errors.push({ path: `${path}.raw`, message: "required string" });
  }
  if (ref.type !== undefined && (typeof ref.type !== "string" || !VALID_TEMPORAL_TYPES.has(ref.type))) {
    errors.push({ path: `${path}.type`, message: "must be one of: point, range, duration, unresolved" });
  }
}

export function validateExtraction(obj: unknown): ValidationResult {
  const errors: ValidationError[] = [];

  if (typeof obj !== "object" || obj === null) {
    return { valid: false, errors: [{ path: "", message: "must be an object" }] };
  }

  const doc = obj as Record<string, unknown>;

  if (doc.version !== "1") {
    errors.push({ path: "version", message: "must be \"1\"" });
  }

  if (typeof doc.extracted_at !== "string") {
    errors.push({ path: "extracted_at", message: "required string (ISO 8601)" });
  }

  if (typeof doc.produced_by !== "string") {
    errors.push({ path: "produced_by", message: "required string (provider URI)" });
  }

  if (!Array.isArray(doc.entities)) {
    errors.push({ path: "entities", message: "required array" });
  } else {
    for (let i = 0; i < doc.entities.length; i++) {
      validateEntity(doc.entities[i], `entities[${i}]`, errors);
    }
  }

  if (!Array.isArray(doc.goals)) {
    errors.push({ path: "goals", message: "required array" });
  } else {
    for (let i = 0; i < doc.goals.length; i++) {
      validateGoal(doc.goals[i], `goals[${i}]`, errors);
    }
  }

  if (!Array.isArray(doc.themes)) {
    errors.push({ path: "themes", message: "required array" });
  }

  if (!Array.isArray(doc.capabilities)) {
    errors.push({ path: "capabilities", message: "required array" });
  } else {
    for (let i = 0; i < doc.capabilities.length; i++) {
      if (typeof doc.capabilities[i] !== "string") {
        errors.push({ path: `capabilities[${i}]`, message: "must be a string" });
      } else if (!VALID_CAPABILITIES.has(doc.capabilities[i] as string)) {
        errors.push({ path: `capabilities[${i}]`, message: `unknown capability: "${doc.capabilities[i]}"` });
      }
    }
  }

  if (doc.facts !== undefined) {
    if (!Array.isArray(doc.facts)) {
      errors.push({ path: "facts", message: "must be an array" });
    } else {
      for (let i = 0; i < doc.facts.length; i++) {
        validateFact(doc.facts[i], `facts[${i}]`, errors);
      }
    }
  }

  if (doc.temporal_refs !== undefined) {
    if (!Array.isArray(doc.temporal_refs)) {
      errors.push({ path: "temporal_refs", message: "must be an array" });
    } else {
      for (let i = 0; i < doc.temporal_refs.length; i++) {
        validateTemporalRef(doc.temporal_refs[i], `temporal_refs[${i}]`, errors);
      }
    }
  }

  if (doc.embeddings !== undefined) {
    if (!Array.isArray(doc.embeddings)) {
      errors.push({ path: "embeddings", message: "must be an array" });
    } else {
      for (let i = 0; i < doc.embeddings.length; i++) {
        validateEmbedding(doc.embeddings[i], `embeddings[${i}]`, errors);
      }
    }
  }

  return { valid: errors.length === 0, errors };
}
