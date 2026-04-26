export interface SynaptSourceRef {
  version: "1";
  snippet?: string;
  offset_start?: number;
  offset_end?: number;
  sentence_index?: number;
}

export interface SynaptEmbedding {
  version: "1";
  vector: number[];
  model: string;
  input: "source" | "summary" | "entities" | string;
  dimensions: number;
  space?: string;
  computed_at?: string;
}

export interface SynaptAssertionSignals {
  version: "1";
  confidence?: number;
  negated?: boolean;
  hedged?: boolean;
  condition?: string;
}

export interface SynaptTemporalRef {
  version: "1";
  raw: string;
  type?: "point" | "range" | "duration" | "unresolved";
  resolved?: string;
  resolved_end?: string;
  context?: string;
}

export interface SynaptRelation {
  target: string;
  type: string;
  origin?: string;
  signals?: SynaptAssertionSignals;
}

export interface SynaptEntity {
  id?: string;
  name: string;
  type: "person" | "place" | "event" | "concept" | "organization" | "object" | string;
  state?: string;
  context?: string;
  date_hint?: string;
  source?: SynaptSourceRef;
  signals?: SynaptAssertionSignals;
  relations?: SynaptRelation[];
}

export interface SynaptGoal {
  text: string;
  status: "open" | "resolved" | "abandoned" | "in_progress";
  entity_refs: string[];
  stated_at?: string;
  resolved_at?: string;
  source?: SynaptSourceRef;
  signals?: SynaptAssertionSignals;
}

export interface SynaptFact {
  text: string;
  category?: string;
  source?: SynaptSourceRef;
  signals?: SynaptAssertionSignals;
}

export type ExtractionCapability =
  | "entities"
  | "entity_state"
  | "entity_context"
  | "entity_ids"
  | "goals"
  | "goal_timing"
  | "goal_entity_refs"
  | "themes"
  | "summary"
  | "sentiment"
  | "facts"
  | "temporal_refs"
  | "temporal_classes"
  | "relations"
  | "relation_origin"
  | "assertion_signals"
  | "evidence_anchoring";

export interface SynaptExtraction {
  version: "1";
  extracted_at: string;
  source_id?: string;
  source_type?: string;
  user_id?: string;
  produced_by: string;
  kind?: string;
  entities: SynaptEntity[];
  goals: SynaptGoal[];
  themes: string[];
  sentiment?: string;
  summary?: string;
  facts?: SynaptFact[];
  temporal_refs?: SynaptTemporalRef[];
  capabilities: ExtractionCapability[];
  embeddings?: SynaptEmbedding[];
  extensions?: Record<string, unknown>;
}
