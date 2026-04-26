import type {
  SynaptExtraction,
  SynaptEntity,
  SynaptGoal,
  SynaptFact,
  SynaptEmbedding,
  SynaptTemporalRef,
  ExtractionCapability,
} from "./schema.js";
import { validateExtraction, type ValidationResult } from "./validate.js";

export interface FinalizeContext {
  produced_by: string;
  user_id?: string;
  source_id?: string;
  source_type?: string;
  kind?: string;
  extensions?: Record<string, unknown>;
  embeddings?: Omit<SynaptEmbedding, "version">[];
  capabilities_hint?: ExtractionCapability[];
}

export interface FinalizeResult {
  extraction: SynaptExtraction;
  validation: ValidationResult;
  warnings: string[];
}

const CAPABILITY_DEPS: Record<string, string[]> = {
  entity_state: ["entities"],
  entity_context: ["entities"],
  entity_ids: ["entities"],
  goal_timing: ["goals"],
  goal_entity_refs: ["goals", "entity_ids"],
  temporal_classes: ["temporal_refs"],
  relations: ["entities", "entity_ids"],
  relation_origin: ["relations"],
};

function injectSubSchemaVersions(obj: Record<string, unknown>): void {
  if (obj && typeof obj === "object") {
    obj.version = "1";
  }
}

function hasPayloadBeyondVersion(obj: Record<string, unknown>): boolean {
  return Object.keys(obj).some((k) => k !== "version");
}

function detectCapabilities(doc: Record<string, unknown>): ExtractionCapability[] {
  const caps: ExtractionCapability[] = [];
  const entities = doc.entities as Record<string, unknown>[] | undefined;
  const goals = doc.goals as Record<string, unknown>[] | undefined;

  if (Array.isArray(entities) && entities.length > 0) {
    caps.push("entities");
    if (entities.some((e) => e.state !== undefined)) caps.push("entity_state");
    if (entities.some((e) => e.context !== undefined || e.date_hint !== undefined)) caps.push("entity_context");
    if (entities.some((e) => e.id !== undefined)) caps.push("entity_ids");
    if (entities.some((e) => Array.isArray(e.relations) && (e.relations as unknown[]).length > 0)) {
      caps.push("relations");
      const allRelations = entities.flatMap((e) => (e.relations as Record<string, unknown>[]) || []);
      if (allRelations.some((r) => r.origin !== undefined)) caps.push("relation_origin");
    }
    if (entities.some((e) => e.source !== undefined)) caps.push("evidence_anchoring");
    if (entities.some((e) => e.signals !== undefined)) caps.push("assertion_signals");
  }

  if (Array.isArray(goals) && goals.length > 0) {
    caps.push("goals");
    if (goals.some((g) => g.stated_at !== undefined || g.resolved_at !== undefined)) caps.push("goal_timing");
    if (goals.some((g) => Array.isArray(g.entity_refs) && (g.entity_refs as unknown[]).length > 0)) {
      caps.push("goal_entity_refs");
    }
    if (!caps.includes("evidence_anchoring") && goals.some((g) => g.source !== undefined)) {
      caps.push("evidence_anchoring");
    }
    if (!caps.includes("assertion_signals") && goals.some((g) => g.signals !== undefined)) {
      caps.push("assertion_signals");
    }
  }

  if (Array.isArray(doc.themes) && (doc.themes as unknown[]).length > 0) caps.push("themes");
  if (typeof doc.summary === "string") caps.push("summary");
  if (typeof doc.sentiment === "string") caps.push("sentiment");

  if (Array.isArray(doc.facts) && (doc.facts as unknown[]).length > 0) {
    caps.push("facts");
    const facts = doc.facts as Record<string, unknown>[];
    if (!caps.includes("evidence_anchoring") && facts.some((f) => f.source !== undefined)) {
      caps.push("evidence_anchoring");
    }
    if (!caps.includes("assertion_signals") && facts.some((f) => f.signals !== undefined)) {
      caps.push("assertion_signals");
    }
  }

  if (Array.isArray(doc.temporal_refs) && (doc.temporal_refs as unknown[]).length > 0) {
    caps.push("temporal_refs");
    const refs = doc.temporal_refs as Record<string, unknown>[];
    if (refs.some((r) => r.type !== undefined || r.resolved_end !== undefined)) caps.push("temporal_classes");
  }

  return caps;
}

export function finalizeExtraction(
  llmOutput: Record<string, unknown>,
  context: FinalizeContext,
): FinalizeResult {
  const warnings: string[] = [];
  const doc = { ...llmOutput };

  // Stage 2: inject client context
  doc.version = "1";
  doc.produced_by = context.produced_by;
  if (context.user_id !== undefined) doc.user_id = context.user_id;
  if (context.source_id !== undefined) doc.source_id = context.source_id;
  if (context.source_type !== undefined) doc.source_type = context.source_type;
  if (context.kind !== undefined) doc.kind = context.kind;
  if (context.extensions !== undefined) doc.extensions = context.extensions;

  // Stage 2: inject embeddings with version
  if (context.embeddings !== undefined) {
    doc.embeddings = context.embeddings.map((emb) => {
      const full: Record<string, unknown> = { version: "1", ...emb };
      if (full.dimensions === undefined && Array.isArray(full.vector)) {
        full.dimensions = (full.vector as number[]).length;
      }
      return full;
    });
  }

  // Stage 3: inject sub-schema versions on source refs and signals
  if (Array.isArray(doc.entities)) {
    for (const ent of doc.entities as Record<string, unknown>[]) {
      if (ent.source && typeof ent.source === "object") {
        const src = ent.source as Record<string, unknown>;
        if (hasPayloadBeyondVersion(src)) {
          injectSubSchemaVersions(src);
        } else {
          delete ent.source;
        }
      }
      if (ent.signals && typeof ent.signals === "object") {
        const sig = ent.signals as Record<string, unknown>;
        if (hasPayloadBeyondVersion(sig)) {
          injectSubSchemaVersions(sig);
        } else {
          delete ent.signals;
        }
      }
      if (Array.isArray(ent.relations)) {
        for (const rel of ent.relations as Record<string, unknown>[]) {
          if (rel.signals && typeof rel.signals === "object") {
            const sig = rel.signals as Record<string, unknown>;
            if (hasPayloadBeyondVersion(sig)) {
              injectSubSchemaVersions(sig);
            } else {
              delete rel.signals;
            }
          }
        }
      }
    }
  }

  if (Array.isArray(doc.goals)) {
    for (const goal of doc.goals as Record<string, unknown>[]) {
      if (goal.source && typeof goal.source === "object") {
        const src = goal.source as Record<string, unknown>;
        if (hasPayloadBeyondVersion(src)) {
          injectSubSchemaVersions(src);
        } else {
          delete goal.source;
        }
      }
      if (goal.signals && typeof goal.signals === "object") {
        const sig = goal.signals as Record<string, unknown>;
        if (hasPayloadBeyondVersion(sig)) {
          injectSubSchemaVersions(sig);
        } else {
          delete goal.signals;
        }
      }
    }
  }

  if (Array.isArray(doc.facts)) {
    for (const fact of doc.facts as Record<string, unknown>[]) {
      if (fact.source && typeof fact.source === "object") {
        const src = fact.source as Record<string, unknown>;
        if (hasPayloadBeyondVersion(src)) {
          injectSubSchemaVersions(src);
        } else {
          delete fact.source;
        }
      }
      if (fact.signals && typeof fact.signals === "object") {
        const sig = fact.signals as Record<string, unknown>;
        if (hasPayloadBeyondVersion(sig)) {
          injectSubSchemaVersions(sig);
        } else {
          delete fact.signals;
        }
      }
    }
  }

  if (Array.isArray(doc.temporal_refs)) {
    for (const ref of doc.temporal_refs as Record<string, unknown>[]) {
      injectSubSchemaVersions(ref);
    }
  }

  // Stage 3: compute capabilities from observed payload
  const observed = detectCapabilities(doc);
  if (context.capabilities_hint) {
    for (const hinted of context.capabilities_hint) {
      if (!observed.includes(hinted)) {
        warnings.push(`capabilities_hint includes "${hinted}" but payload does not contain it; using observed`);
      }
    }
  }
  doc.capabilities = observed;

  // Stage 3: capability implication enforcement
  if (observed.includes("goal_entity_refs") && !observed.includes("entity_ids")) {
    warnings.push("goal_entity_refs present but entity_ids missing; entity ref resolution will fall back to name matching");
  }

  // Stage 3: validate
  const validation = validateExtraction(doc);

  return {
    extraction: doc as unknown as SynaptExtraction,
    validation,
    warnings,
  };
}
