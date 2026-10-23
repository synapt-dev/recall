# Changelog

All notable changes to synapt are documented here.

## [Unreleased]

## [0.30.0] - 2026-10-23

Minor release. A new command, `synapt recall channel watch`, follows one or
more rooms and prints each new post as it lands, and a room map lets several
gripspaces on one machine share one room. An idle MCP server now gives its
memory back: it embeds through a child process that exits when idle, and it
drops its cached index after ten minutes with no tool call. Knowledge nodes
are stored whole instead of cut to 300 characters, `recall_contradict` lists
five pending rows at a time, and recall no longer deletes a `recall.db` it
cannot open. The new command and the new arguments on `recall_save`,
`recall_search`, `recall_quick`, `recall_contradict` and `recall_channel` are
why this is a minor release rather than a patch.

Seventy-eight files changed across `v0.29.0..dev`, excluding this entry: 10329
insertions, 391 deletions, carrying the 47 pull requests cited below plus the
version bump. As in earlier entries, the figure includes the version files and
excludes this changelog entry.

### Added

- **`synapt recall channel watch <room>...` follows rooms as they grow.** It
  prints each new post as `from [id]: body`, flushed per line, until SIGINT or
  SIGTERM (both exit 0). It resolves each room's log the same way `post` and
  `read` do and writes nothing while it watches: no cursor, membership or
  presence. It skips your own posts unless `--include-mine`, `--all` adds every
  room in the store and the room map, and bodies are capped at 300 characters
  with control characters escaped so a post cannot forge a second output line.
  `--since` is compared as an instant, so an offset, a sub-second stamp or a
  naive (UTC) value orders correctly, and a value that is not a timestamp is
  refused with exit 2 before any file is opened. A log rewritten in place past
  the read position, or rewritten at the same size, is read again from the
  start instead of dropping the new post. A watched room name is held to the
  same containment check as post and read. (#1302, #1304, #1305, #1307, #1312,
  #1313, #1316, #1317)
- **A room map places a shared room in one store.** `~/.synapt/channels/rooms.toml`
  (or the file named by `SYNAPT_ROOM_MAP`) maps a room to a store under
  `~/.synapt/channels/`, and that room's log is then read and written there from
  every gripspace. Only the log moves; presence, cursors and pins stay local. A
  mapping that cannot be honoured (a missing store, a malformed value, a room
  mapped twice, a map that does not parse) is refused by name and never falls
  back to the caller's store. `SYNAPT_SHARED_CHANNELS_DIR` still pins every
  room. (#1291)
- **`recall_search`, `recall_quick` and `synapt search` can skip contradiction
  detection.** A new `detect_contradictions` argument (default true) and
  `--no-detect-contradictions` flag let background callers search without
  queueing pending contradictions for review. Existing callers are unchanged.
  (#1293)
- **Channel search results carry `cited_by`.** It counts how many posts cite
  that post by id, across all channels; an id that cannot be resolved to one
  post is never counted. A `set_search_ranker` hook may reorder or drop results,
  never add; no ranker ships. (#1295)
- **The wake card names where each line came from.** GOAL, NEXT, RULED OUT and
  CONSTRAINTS lines name their journal entry or memory file, and a line the card
  cannot trace says `[source: unknown]`. (#1297)
- **The wake card and the session-start text name a newer journal written from
  another directory.** When a sibling store under the same recall root holds a
  newer journal entry than the one shown, both add one line:
  `newer journal at <path> (<timestamp>)`. The read is bounded at 8 MiB and says
  so when it stops early. (#1318)

### Changed

- **The MCP server embeds through a child process and releases its index when
  idle.** The server never imports torch; the child exits after
  `RECALL_EMBED_IDLE_S` seconds with no request (default 600), or at once when
  the server goes away, and the next search starts a new one. After
  `RECALL_IDLE_RELEASE_S` seconds (default 600) with no tool call, the cached
  index is dropped and the next search reloads it. Measured on Linux with a
  10,000-chunk synthetic store: server memory after one search went from
  667 MB to 212 MB, and to 53 MB after the release. (#1301, #1303)
- **Knowledge nodes are stored whole.** They were cut to their first 300
  characters with no notice. Content up to 4,000 characters is now kept, and
  anything longer is refused with nothing written, by `recall_save`,
  `recall_contradict`, `recall_correct` and the ADK memory service (which now
  drops an over-limit event with a warning instead of cutting it).
  Consolidation keeps facts up to 1,500 characters. Existing node ids do not
  move. (#1292)
- **`recall_save` updates keep what you did not pass.** An update now keeps the
  node's existing tags, sources, confidence, status, successor and
  contradiction note when they are omitted, instead of resetting them. An update
  to a contradicted or superseded node is refused and names the successor;
  `restore=true` revives it on purpose. An explicit value, such as `tags=[]`,
  still wins. (#1292, #1309)
- **`recall_contradict(action="list")` shows 5 rows, oldest first.** New `limit`
  (default 5, `0` for all) and `order` (`oldest` or `newest`) arguments; each
  side prints on one line, and a footer says how many rows were withheld. The
  reference page and the tool's docstring now list every action and argument,
  and which action uses and requires each one. (#1320, #1321, #1324, #1325)
- **Channel reads are bounded by size.** A rendered body is capped at 8,000
  characters with a marker naming the offset to resume from, and a whole read
  at 32,000, with older messages dropping to a one-line reference.
  `read_message` returns one chunk, with `offset` and `length` on the MCP tool
  and a new `synapt recall channel read_message <channel> <id>` CLI verb.
  (#1288)
- **`recall_channel` says its store preamble and each mention once.** The store
  line prints on the first call of a session and again only when the store
  changes; an unread @mention prints in full once, then as one count line. A
  mention clears once the session reads it with `read_message` or posts a reply
  naming its id. (#1287, #1315)
- **A leading `#` on a channel name is dropped everywhere.** `#dev` and `dev`
  now name the same room at every entry point. Existing `#`-prefixed logs are
  not migrated. (#1287)
- **The channel CLI refuses inputs an action does not use.** `post --name`
  exits 1 and posts nothing (use `join --name`), and any flag or positional an
  action ignores is refused with exit 1. `search` filters by its channel
  argument, `chat` takes `--name`, and `broadcast "message"` works. `who`,
  `list`, `unread` and `heartbeat` given a channel now exit 1. (#1287)
- **recall finds Claude Code transcripts under `~/.claude-accounts/*/projects`.**
  Each account directory is read after `~/.claude/projects`, so a session kept
  under an account is found without `--source`. (#1323)
- **The cross-encoder reranker loads on cpu unless told otherwise.**
  `SYNAPT_RERANKER_DEVICE` can name `cpu`, `mps` or `cuda`. On one Apple-silicon
  host the default model's load took 21 MB on cpu against 1058 MB on MPS.
  (#1299)
- **Search access records live in their own `access.db`.** The access tables
  move out of `index.db` into `access.db` beside it, and opening a store moves
  any existing rows there, so an access write no longer makes another server
  reload its index. (#1311)

### Fixed

- **recall never deletes a `recall.db` it cannot open.** A locked store was
  deleted at load and an empty index started. Now a lock waits and then raises,
  a file that is not SQLite at all is moved aside to
  `recall.db.corrupt-<UTC stamp>` with its `-wal` and `-shm`, and a damaged
  sharded store raises with nothing moved. The MCP server answers with an error
  naming the path instead of an empty index. `synapt recall build` on a non-SQLite
  `recall.db` stops with a named error and exit 2 and leaves the file as it
  was. (#1329, #1330)
- **`build` and `init` say keyword-only when the store has no embeddings.**
  They printed "Computing embeddings" for a store that never computes them. The
  check is read-only. (#1327)
- **A server's own writes no longer reload its index.** Its access writes, and
  the empty WAL its own load creates, made the next call reload the whole index,
  which took 9 to 50 seconds on a 275,000-chunk index. A change by another
  writer still reloads. (#1298, #1306)
- **A channel post that landed reports success.** When the state store was
  locked after the post was written, the post came back as `database is locked`
  and callers posted again. It now returns its id with a note that channel state
  was not updated, and a pin that was not recorded is not reported as pinned.
  (#1294, #1296)
- **A posted body is stored as given.** Literal backslash-n and backslash-t
  pairs were rewritten to a newline and a tab before writing. (#1288, #1319)
- **A room name cannot reach outside the store.** A room named `../x` wrote one
  directory above the store; `/`, NUL and a backslash are now refused too, by
  post, read and watch alike, with exit 2 from the CLI. Names such as `a..b`
  that post accepts are no longer refused by watch. (#1308)
- **Channel time comparisons use instants.** `--since` on reads and unread
  counts, and the check for unread @mentions, compared timestamps as text, so an
  offset or a sub-second stamp misordered posts. A `--since` that is not a
  timestamp is refused with exit 2; a stored cursor that does not parse reads as
  the epoch. (#1310, #1314)
- **`recall_contradict(action="forget")` stops the node being served.** It
  archived the node only in `knowledge.jsonl`, so search kept returning it.
  (#1322)
- **Sharded stores read their chunks from the shards.** The timeline, turn
  context, consolidation's source resolver and sample, and cluster text, search
  fallback and promotion read `index.db`'s empty or frozen copy, so sessions
  added since the split were missing. (#1285, #1286)
- **The flat shard mirror opens read-only and stays current.** The `data_NNN.db`
  copies kept for older recall versions failed a read-only open, and could hold
  old rows after `save_embeddings` or a failed copy; a mirror that cannot be
  trusted is now removed with a warning. (#1290, #1326, #1328)

### Tests

- **Test and CI changes.** The two packaging tests that reach PyPI run in their
  own `network-tests` job, and a failed command in that helper names its output;
  a lock-wait row counts lock cycles instead of wall time; startup test
  docstrings describe behaviour. (#1284, #1289, #1300)

## [0.29.0] - 2026-10-12

Minor release. A new command, `synapt orient`, prints a short wake card from
the user's own journal and memory, so an agent starting a session knows where
it is before it acts. `recall_sync_memory` now syncs only this project's own
memory and names every file it could not sync. Two things a first-time user
hit are fixed: the printed install lines use `uv tool install`, and `resume` no
longer tells a caught-up reader to build. The new command is why this
is a minor release rather than a patch.

Nineteen files changed across `v0.28.0..dev`, excluding this entry: 1035
insertions, 91 deletions, carrying #1275, #1276, #1277, #1278 and #1279 plus the
version bump. As in earlier entries, the figure includes the version files and
excludes this changelog entry.

### Added

- **`synapt orient` (also `synapt recall orient`): a wake card from your own
  records.** It prints GOAL (the newest meaningful journal entry's focus, with
  its age), NEXT (that entry's first next step), CARRIED (next steps still open
  across recent entries), RULED OUT (that entry's decisions) and CONSTRAINTS
  (the user's feedback memories from the session project's Claude Code memory
  directory). `--json` prints the same card with `fields_filled` and
  `tokens_est`. Every source fails soft: a missing journal or memory directory
  is one named NOT READ line. STALE says when a finished session transcript is
  more than an hour newer than the journal; the running session is skipped by
  its runtime id (`CLAUDE_CODE_SESSION_ID`, `CODEX_THREAD_ID` or
  `SYNAPT_SESSION_ID`), and by a two-minute window only when no id is set. The
  card stays under a 4,000-token estimate, and every capped list says how many
  it withheld. (#1275)

### Changed

- **`recall_sync_memory` syncs this project's own memory directory.** It read
  `*/memory` under every Claude Code root, so one project's sync wrote every
  other project's memory into its store. It now reads
  `<root>/<slug>/memory` under each Claude Code root, where the slug is the
  directory the session runs in, not a shared `GRIPSPACE_ROOT`.
  `memory_dirs` syncs exactly the directories it names. (#1276)
- **Printed install lines use `uv tool install synapt`.** `pip install --user`
  is refused on a system Python that marks itself externally managed (PEP 668).
  A hint for a missing extra names `uv tool install 'synapt[<extra>]'` first and
  pip in a virtualenv second; the integration sections keep pip, since they
  install synapt as a library.
  (#1277)
- **`synapt resume` no longer tells a caught-up reader to build.** When the
  store's chunks carry byte offsets, its build note compares them with the
  transcripts by bytes, and a caught-up reader sees no note; the turn-timestamp
  test remains the fallback. Session ids print
  whole unless they are UUIDs. (#1277)

### Fixed

- **`recall_sync_memory` names every file it could not sync.** A failed file
  was a bare error count, and the run still said "complete". The good files are
  synced first; then, if any failed, the call raises with the counts and one
  line per file (path and reason), so an MCP caller sees an error. (#1276)
- **A cold read no longer starts an unchecked background
  build.** The detached refresh a cold `synapt resume` queues now checks host
  memory first (a refusal is recorded in `build-deferrals.jsonl`), takes the
  host-wide build lock, and records which process started it. (#1278)

## [0.28.0] - 2026-10-11

Minor release. An agent that wakes after an unclean end now sees what was typed
to it before it acts: `synapt resume` and the session-start context read the gap
from the live transcripts, which the index can lag by hours. recall now finds
Claude Code transcripts wherever Claude Code writes them (`CLAUDE_CONFIG_DIR`,
including a comma-separated list, and `~/.config/claude`), MCP servers embed on
cpu by default, and `recall_code` answers from a gripspace root. The new
resume and build flags are why this is a minor release rather than a patch.

Forty-four files changed across `v0.27.0..dev`, excluding this entry: 4415
insertions, 179 deletions, carrying #1255, #1256, #1258, #1259, #1260, #1261,
#1262, #1263, #1264, #1265, #1266, #1267, #1268, #1269, #1270, #1271 and #1272
plus the version bump. As in earlier entries, the figure includes the version files
and excludes this changelog entry.

### Added

- **The transcript gap, in `synapt resume` and in the session-start wake.**
  `synapt resume` ends with a TRANSCRIPT GAP block: every message typed to this
  agent in each session written since the newest authored journal (or
  `--since`, an ISO time or `Nh`), read from the live transcript files, with the
  files each session wrote. The live session is excluded by identity
  (`--live-session`, else `$CLAUDE_CODE_SESSION_ID`); with neither, nothing is
  excluded and the header says so. `SYNAPT_GAP_HIDE` takes the reader's own
  pattern for machine-written pastes. The session-start wake carries the same
  gap when the hook names the starting session and something was typed in the
  gap: newest session first, six messages per session, inside its own
  2,500-byte share of the wake. The read is proportional to the size of the
  sessions in the gap. Codex rollouts are not read for it. (#1270, #1271)
- **`recall_code` searches each declared member of a gripspace root.** Asked
  from a root whose manifest declares its members, it indexes and searches each
  cloned member under its own tag instead of refusing; a repository the
  manifest does not declare is not searched. (#1265)
- **`synapt build --check-floor`** runs only the build's host-memory check and
  answers with an exit code and one line: 0 when the floor is met, 2 when the
  build would refuse, and 3 when it cannot measure the host. (#1266)
- **Build progress after the clustering line.** The steps after clustering
  print time-based progress, so a long build no longer goes silent. (#1256)

- **`synapt recall consolidate` says how many nodes its content filters turned away.**
  A run printed `3 created` whether the model had returned three nodes or seven. It now
  also prints, on stderr, how many nodes the model returned, how many the content filters
  rejected, and the first words of each rejected node (at most five, 48 characters each).
  The preview is one printable line. What is kept is decided exactly as before.
  (#1264)

### Changed

- **recall reads Claude Code transcripts and memory under every root.** Transcripts are found under
  `$CLAUDE_CONFIG_DIR` (a comma-separated list is read in order) and
  `~/.config/claude` (when it holds `projects`) as well as `~/.claude`, so sessions written under a
  configured directory are no longer invisible to `resume`, `build` and
  `transcript`. The bulk transcript discovery, `recall_sync_memory` and the
  memory-file source read the same roots. A listed directory that is not
  absolute is relative to the project Claude Code ran in: commands that read
  one project resolve it there, and the walkers over every project skip it.
  (#1267, #1269, #1272)
- **MCP servers embed on cpu by default, and the embedding device and torch
  load at first use.** A long-lived server no longer holds about 1 GB of
  graphics memory on Apple Silicon; `SYNAPT_EMBED_DEVICE` still overrides it,
  and batch builds keep the device the probe picks. A sentence-transformers
  that is installed but fails to import now surfaces at the first embed
  instead of falling back to Ollama. (#1268)
- **`recall_channel` read bounds its pins by `limit`, and search honours
  `limit` and `detail`.** A search that names no limit returns up to 20 hits
  instead of 10. (#1261, #1262)
- **The low-specificity filter reads a named-rule identifier such as `STAMP-PARITY-91`
  (two or more ALLCAPS words, then a number) as a project-specific signal**, as it already
  does a path, a version or a CamelCase name, so a short node naming one is no longer
  dropped as generic advice. Standard identifiers (`UTF-8`, `SHA-256`, `HTTP-2`, `CI-CD`),
  a bare ticket number (`ABC-123`), hyphenated lowercase words and numbers with units are
  not signals, because generic advice carries all of them. (#1264)

### Fixed

- `synapt recall journal` honours `--path` in every mode and refuses `--write`
  beside another mode or the repair-only flags, instead of writing to the live
  journal. (#1258)
- The journal's carry-forward line counts this write's own retirements, not
  the session's. (#1259)
- `synapt recall build` prints the store it wrote, not the one the environment
  names. (#1260)
- The dashboard's pane output keeps streaming across log rotation. (#1263)

### Removed

- The GitHub Pages deploy workflow, which could not succeed because Pages is
  not enabled for the repository. (#1255)

## [0.27.0] - 2026-10-01

Minor release. Index builds now ask the kernel whether the host is under memory
pressure, and keep asking while they run; the build's free-memory floor becomes a
config key that `synapt maintenance status` explains; and a journal step retired
earlier in a session stays retired when `recall_journal action=pending` reads it.
The new config key and status command are why this is a minor release rather than
a patch.

Thirty-two files changed across `v0.26.0..dev`, excluding this entry: 3898
insertions, 191 deletions, carrying #1238, #1240, #1241, #1242, #1243, #1244,
#1245, #1247, #1248, #1249, #1250 and #1251 plus the version bump. As in the
0.26.0 entry, the figure includes the version files and excludes this changelog
entry.

### Added

- **`memory.build_min_free_gb`, and `synapt maintenance status`.** The build's
  free-memory floor was a constant behind one environment variable. It is now a
  key in the global config, resolved environment, then global config, then the
  default, and `synapt maintenance status` prints the floor beside its live
  reading, where its value came from, and the decision. A project config is not
  consulted for `memory.*`, and a project file that sets one is named as ignored:
  the floor is a property of the host, not of the directory a build starts in. A
  value the gate cannot use falls back to the default and says what was asked
  for and what was used; a boolean is refused, and a value below 0.5 GB is
  clamped to it with a note (#1247).

### Changed

- **The build gate asks the kernel, and keeps watching while the build runs.**
  On macOS, where the gate applies, it used to read free and inactive pages once,
  at admission. It now also reads the kernel's own memory-pressure level and
  refuses above normal, and it keeps checking while a build runs. Each refusal
  prints the free-memory reading beside its floor and the kernel's pressure
  level. On a loaded host this means builds defer more often, by design: heavy
  maintenance belongs in a quiet window. A pressure reading that cannot be taken does not stop a build (#1251).

### Fixed

- **A journal step retired earlier in the session stays retired.**
  `recall_journal action=pending` built its retirement set from a view that kept
  only the newest entry per session, so a step retired by an earlier write in the
  same session came back on the next read. The retirement set now comes from
  every entry of the sessions in view (#1249).
- **A memory-floor refusal of an automatic build leaves a trace.** A refused
  catchup or precompact rebuild used to be printed to a stream its own spawn
  discarded, so days without a build looked like a build never attempted. Each
  refusal is now one line in `build-deferrals.jsonl` beside the index, and
  `synapt resume` and the session-start banner print "index last built <date>;
  automatic build deferred <n> times (memory floor); run synapt build" once
  refusals have piled up since the last build. The default floor is unchanged
  (#1243).
- **A `built` row names the caller who started the build.** Each row now carries
  the build's `argv`, `pid`, `ppid`, and its parent's command line (read from
  `ps`), so a build can be attributed after the fact instead of only while its
  parent process is alive. A build whose parent has already exited has been
  reparented, so its row records `ppid` 1 and a parent command line naming the
  init process rather than the caller; the `pid` and `argv` are still the
  build's own. The background cold-refresh build now records a row too. Because
  a row carries two unbounded command lines, `build-deferrals.jsonl` is written
  owner-only (0600), on creation and on any file an earlier version left
  world-readable (#1245). The mode is tightened before each append rather than
  after it, so a ledger an operator set read-only is repaired instead of
  silently ending the record (#1248).
- **`resume` reads only the query-tail rows it was asked for.** Two loaders that
  promised a bounded set of sessions read the whole query-tail overlay and
  filtered afterwards; on a large store that was most of a `resume` that ran
  well past its budget. Both now bound the read in SQL (#1242).
- **Query tails retire on content, not on the source file's key.** A transcript
  rewritten in place by a compaction got a new inode, so its overlay's stored key
  never matched again and the overlay grew without draining. An overlay now
  retires when every one of its turns is present in the base index, compared by
  a digest of the turn's text and timestamp; an empty overlay never counts as
  covered (#1244).
- **A healthy save heals knowledge vectors a failed save left empty.** A save
  whose embedding provider was briefly unreachable lands without a vector, and on
  a store with no chunk embeddings nothing refilled it. A later save whose
  provider call succeeds now backfills up to 32 such rows, oldest first, after
  its own write (#1241).
- **One cosine for the package, refusing a width mismatch.** Three copies of the
  cosine computed the dot product over the shorter of two vectors and the norms
  over both, so a mismatched pair returned a plausible number with no meaning.
  There is now one implementation, and it refuses vectors of different widths
  (#1240).
- **`recall stats` no longer advertises `SYNAPT_EMBEDDING_MODEL`.** The variable
  changed the model `stats` displayed but not the model recall loads, so the
  display was wrong whenever it was set. It is no longer read (#1238).

### Tests

- The guard that keeps a test run from taking the real host build lock now
  refuses any call that resolves the real `~/.synapt` directory, not only a
  write to one named lock file (#1250).

## [0.26.0] - 2026-09-28

Minor release. A new user whose machine runs Ollama with a local embedding model
can install and save knowledge: recall refuses an embedding provider whose
vector width is not the store's, and a save never leaves a half-written node. One
new command, `comms send`, is why this is a minor release rather than a patch.

Sixteen files changed across `v0.25.4..dev`, excluding this entry: 1276
insertions, 32 deletions, carrying #1229, #1230, #1231, #1232, #1233, #1234,
#1235 and #1236 plus the version bump. As in the 0.25.4 entry, the figure
includes the version files and excludes this changelog entry.

### Added

- **`comms send`, a declared-pane communication proof.** It records what one
  declared tmux-pane transport observed next to the durable direct-message
  ledger, rather than guessing delivery. A guessed time label and an empty body
  are refused before anything is written. A missing pane is recorded as
  undeliverable, and a capture that cannot be read is recorded as unknown
  rather than raising (#1231).

### Fixed

- **An embedding provider whose width is not the store's is refused.** On a
  machine where a local Ollama embedding model is present, recall could pick a
  1024-wide model for a 384-wide store, and every knowledge save then failed.
  The mismatch is now refused visibly and recall stays on the store's width
  (#1233).

- **A knowledge save computes its embedding before the node is committed.** A
  failed embedding used to leave the node written but unsearchable by vector.
  The vector is now packed before the write, so a failing pack leaves nothing
  behind (#1234).

- **One `pack_embedding()` serves every write path.** The store's vector format
  is packed in one place, with a named width error, so no write path can pack a
  vector of the wrong width by a side door (#1235).

- **A transient provider failure keeps the fact.** When the embedding provider
  errors for a moment, the node is saved and reported as saved without its
  search vector, instead of the fact being lost. A wrong-width vector still
  refuses the whole save (#1236).

- **Session start defers the incremental index build.** Session-start catchup
  still archives transcripts and compacts the journal, but no longer launches
  the build; explicit maintenance commands still run it (#1230).

- **The overview cache prune survives a file that is not a database.** The
  prune caught `sqlite3.OperationalError`, but a non-database file raises its
  parent, `sqlite3.DatabaseError`; both the prune and the write accessor now
  catch that (#1232).

- **Release binaries build on macOS again.** The macOS release build's lookup
  of the bundled Python distribution was unauthenticated and hit the shared
  anonymous rate limit, so v0.25.1 and v0.25.2 carry no binary assets, and
  v0.25.4's were attached only by a later run. The lookup is now authenticated
  (#1229).

## [0.25.4] - 2026-09-25

Patch release. Six fixes, all in the paths a session start, a compaction and a
knowledge write run through. A session-start catchup build on a low-memory host
now defers instead of running, and only one build runs per host.

Sixteen files changed across `v0.25.3..dev`, excluding this entry: 916
insertions, 64 deletions, carrying #1221, #1222, #1223, #1224, #1225 and #1226
plus the version bump. The figure covers the whole range including the version
files' own two-line change, and excludes this changelog entry on purpose, the
same way the 0.25.2 entry does: a count that includes its own document changes
every time the document is edited.

### Fixed

- **A step retired under `done` no longer returns on the session's own second
  journal write.** The previous-entry lookup skipped the session's own writes;
  `session_done_items` unions the session's own retirements into the one
  predicate, on both write paths (#1221).

- **The consolidation model override reaches inference.** `get_model
  ("consolidation")` returned the configured value while the pipeline handed a
  hardcoded constant to the client, so `SYNAPT_CONSOLIDATION_MODEL` and the
  config file changed what was reported and nothing that ran. The model is
  resolved where it is consumed, and the CLI's `--model` default is no longer a
  caller that always chooses (#1222).

- **MCP guidance is kept within the delivery cap.** The instruction text
  measured past the 2,048-character default cap Claude Code applies, so the tail
  of it was not delivered. It is now 1,954 characters, with the budget and pin
  guidance moved onto the `detail` and `pin` parameters they describe (#1223).

- **A session start's catchup build is gated on host memory, and there is one
  per host.** The build ran unconditionally on every session start on every
  desk, and one was measured at 1.52 GB in two and a half minutes. It now defers on
  a refusal from the free+inactive gate (6 GB floor for a build,
  `SYNAPT_BUILD_MIN_FREE_GB` to override) and takes a lock in `~/.synapt/`, so a
  second session start joins or skips instead of spawning its own. Swap
  percentage is deliberately not a gate: measured the same morning, free+inactive
  rose 4.89 to 6.50 GB while swap also rose 80.1 to 84.3 percent, so a swap rule
  would have refused a host that had just become usable. The override is
  validated in the same release: `0`, `-1` and `nan` no longer become the floor,
  and the test seam a suite injects through carries this repository's name so the
  host gate's own variable cannot force a pass on a real build (#1224, #1225).

- **A compaction's rebuild is gated on host memory, and shares the one build per
  host.** The PreCompact hook rebuilt the index in-process on every compaction
  with no memory gate and no host-wide limit; its per-project lock bounded
  concurrency within one project and waited up to 60 seconds, which is the wrong
  instrument for a host running several desks. One such rebuild was measured at
  1.58 GB in two and a half minutes, taking free+inactive to 3.63 GB. The hook
  now consults the same free+inactive gate and takes the same host-wide lock, so
  one build runs across both hooks rather than one per hook; a refusal skips the
  rebuild and the sync, and an unreadable instrument rebuilds with a line saying
  so. The journal write and the heartbeat stay outside both gates, because
  crash-recovery state is why the hook exists (#1226).

## [0.25.3] - 2026-09-23

Patch release. Four fixes in the storage and resolution paths an agent's memory
runs through, and one documentation change. Fifteen files changed across
`v0.25.2..v0.25.3`: 1,448 insertions, 33 deletions, carrying #1214, #1215,
#1216, #1217 and #1218 plus the version bump.

### Fixed

- A read no longer replaces a live gripspace-root marker: the first binding
  wins, and a rebind is an explicit act (#1214).
- A retraction is written to both stores, so the next sync cannot put the
  retracted node back into search (#1215).
- A supersession is written to both stores, and dedup orders by a per-record
  `revision` rather than a timestamp, so a record stamped in the future can no
  longer revert it (#1218).
- The source-index parser ceiling defaults to 5,000 units and is settable from
  outside the process with `SYNAPT_SOURCE_PARSER_UNITS` (#1216).

### Docs

- Codex MCP config now documents its env block: Codex does not pass
  shell-inherited environment variables to MCP server processes (#1217).

## [0.25.2] - 2026-09-22

Patch release. Two substantive changes, both in the MCP server surface, both
about what an agent is told it can do. Five files of code and tests changed
across `v0.25.1..v0.25.2`: 298 insertions, 10 deletions, carrying #1209 and
#1210 plus the version bump.

That figure excludes this changelog entry on purpose. A count that includes its
own document changes every time the document is edited, so it cannot be made
correct by writing it - only by writing it last and measuring afterwards. The
figure above is stable under any further edit to this entry.

### Changed

- **An unconfigured server no longer advertises the X/Twitter tools.**
  `register_tools` now returns early unless all four X credentials
  (`X_API_KEY`, `X_API_SECRET`, `X_ACCESS_TOKEN`, `X_ACCESS_TOKEN_SECRET`) are
  present, so a server without them lists none of the eight `x_*` tools
  instead of all eight (#1209).

  **This is visible to anyone upgrading.** Before 0.25.2 the tools were
  advertised whether or not they could work, and a listed tool tells an agent
  it has that capability - it would attempt a post and fail at call time.
  After 0.25.2 an unconfigured install does not offer them at all. If your
  agents were seeing the `x_*` tools without credentials configured, they will
  stop seeing them; set the four variables to get them back.

  Known gap, stated rather than left to be found: the design called for gating
  on the `x` extra being installed *and* credentials being present. The
  implementation gates on credentials only, and the `tweepy` import is lazy, so
  the extra half is not enforced. The observable consequence, measured on both
  0.25.1 and 0.25.2: a server with all four credentials set but without the `x`
  extra installed still advertises all eight tools, and every call comes back as
  an ordinary successful result whose text reports the failure, for example
  `Failed to post: tweepy is required for X/Twitter tools: pip install tweepy`.
  The tools catch the exception, so a caller sees no error and no traceback -
  only a result that says it did not work. Install the extra with
  `pip install "synapt[x]"`. This behaviour is unchanged by this release - the
  only thing 0.25.2 changes is that an UNCONFIGURED server no longer advertises
  the tools at all.

### Features

- **Server text for a first-time agent:** the MCP server's tool instructions
  now name the knowledge lifecycle explicitly - save a durable fact and keep
  the returned node id, update in place with that id rather than saving a
  second node for the same fact, and retract on request so the node is hidden
  from retrieval (#1209). Until now the text described how to *search* memory
  and said nothing about maintaining it, so an agent reading only the tool
  instructions would save a second node for a fact it already held, and had no
  way to learn that retraction exists. A contract test asserts the text names
  each operation, so the instructions cannot silently drift from the tools they
  describe.

### Fixes

- The 0.25.1 changelog count now states the measured final state (#1210).

## [0.25.1] — 2026-09-21

Patch release. 47 commits across 19 first-parent merges since 0.25.0 (counted
with `git log --first-parent --merges 2ece8b3..dev` at the final dev tip:
18 pull-request merges, #1186 through #1206 less #1194 and #1197, which never
landed on dev, and #1202, which landed on main directly, plus the
version-bump merge #1207). The headline is search quality: saved knowledge
searches semantically (#1203), `recall_code` ranks by module path (#1200) and
stems inflected queries (#1191), and the co-retrieval contradiction detector
gates its queue on a measured similarity floor (#1206). Several fixes make
first-run behavior honest on Windows and on a bare install.

### Features

- **Semantic knowledge search:** a knowledge-only store searches its saved
  knowledge semantically when a provider resolves; base installs stay BM25-only
  and say so (#1203).
- **`synapt pack`:** seal closed transcript sessions into a content-addressed
  pack (#1189), with push/fetch to a directory remote (#1192).
- **Agent quickstart:** the README names when to save, update, correct, and
  forget (#1201).

### Fixes

- **Windows:** drive separators normalize while deriving project slugs
  (`C:\Code\fp-loom` resolves to `C--Code-fp-loom`, not the nonexistent
  `C:-Code-fp-loom`, #1204); the CURRENT-pointer replace retries on Windows
  PermissionError instead of failing (#1186).
- **Co-retrieval contradictions:** one similarity floor (embedding cosine
  ≥ 0.40) gates both the pending queue and the banner, so unrelated
  low-overlap pairs stop flagging; within-batch dedup keeps overlapping nodes
  from queueing one contradiction twice (#1206).
- **Silent un-retract refused:** updating a retracted knowledge node answers
  with the restore path instead of silently reviving a buried fact; `stats`
  lists only models whose stack resolves in this process; the README names all
  eleven knowledge categories plus the silent default (#1205).
- **Code search:** module-path matches outrank shallower hits (#1200); the
  stemmer strips a trailing silent "e" so inflected query words meet a
  symbol's bare form (#1191); the ambiguity guard checks any depth, not just
  direct children (#1190); an ambiguous multi-repo root is refused in one
  sentence instead of guessing (#1188).
- **Source parsing:** parser-limit refusals name what to do (#1199); source
  render honors its token budget (#1198).
- **Stats performance:** tool and file counts come from SQL aggregates,
  ending the triple full pass over the store (#1196).
- **Schema setup:** setup is serialized and stale schema-lock companions are
  cleaned (#1195).
- **Memory sources:** `claude_memory_source` registers per-root, not on a
  fixed name (#1187).
- **Scrubbing:** bare-value coverage for Supabase, npm, and RunPod token
  prefixes (#1193).
- **Build receipts:** tolerate index objects that predate optional skip and
  warning metadata instead of failing the build after indexing completes
  (#1204).

## [0.25.0] — 2026-09-10

Minor release. 164 commits across 68 first-parent merges since 0.24.2 (66 pull
requests plus a topology-repair sync from main and one direct branch merge).
This is a **minor**, not a patch, because the
range adds new surface a patch would not: the `recall_code` tool — a code-symbol
index and plain-language search (#1150) — and new store-resolution refusals that
change what a call is allowed to do (#1132), alongside candidate-merge
distinctiveness (#1148), a save create-path helper (#1129), and a review-install
hint (#1180). Today's sender-attribution and gripspace-root marker fixes ride in
this range.

### Highlights by area

- **Features:** `recall_code` code index and search (#1150); candidate-merge
  distinctiveness (#1148); resolver refusals for an unpopulated or
  out-of-gripspace root (#1132); a save create-path helper (#1129); a
  review-install hint for gr2 review runs (#1180).
- **Store resolution and identity:** a gripspace-root marker refuses an
  unrelated target; the inverted parent→child marker is neither written nor
  followed, and its dangling/pruned-worktree over-block (write) and under-block
  (read) are split by liveness; a deterministic per-workspace coordinate (#974);
  cwd-inference no longer strands a store; Codex and Google-ADK project
  roots resolve (#1097); channel sender attribution and journal agent-id
  attribution name the calling session.
- **Channels:** sender attribution keyed on the session; UTC timestamps render
  with their zone; bracketed-paste for `speak_to_agent`; the
  knowledge-sync path no longer doubles.
- **Journal:** meaningful auto-stubs (#937); carry-forward "done" matching;
  agent-id attribution; the wake read reports truncation instead of hiding it
  (#856); compaction boilerplate is stripped.
- **Enrich and sleep cycle:** per-entry enrich bound; working-memory
  second-pull boost; recluster maintenance and refusals (#435); signature
  distinctiveness and tiebreak determinism;
  candidate-side containment and merge-image floors; sharded save-clusters
  receipts.
- **Code index and search:** grammar-hash refresh on change; path and definition
  preference in ranking; batch chunk hydrate; chunk-header date text.
- **CI and tests:** extras-matrix smoke coverage; pytest-asyncio dependency;
  Windows path normalization and assertions (#1136); crewai skipped without an
  LLM (#1137); test isolation; cold-refresh contract-drift coverage.
- **Performance:** Codex cwd and session caches; O(1) session update on
  materialize-all-chunks; a session-overview covering index.

The detailed entries below were authored as changes landed and are grouped under
this release.

### Fixed
- **A session that ended without a handoff is named as such at the next wake.**
  A host crash, kill, or forced shutdown runs no SessionEnd, so no checkpoint
  is written, and nobody journals a session they did not know was ending. The
  session-start hook then rendered whatever `checkpoint.json` held, which can be
  another session's clean SessionEnd (measured after a crash: a subagent's tail
  under "LAST CHECKPOINT", twelve hours of the main session's work with no
  record, and nothing saying so). `synapt.recall.resume.detect_unclean_end`
  judges the newest previous caller transcript against the authored journals
  and a checkpoint bound to that session's id. A handoff is session-bound
  evidence: a journal bound to the session covers it when written within the
  fifteen-minute grace before its last activity or any time after; a legacy
  sessionless journal covers only inside the symmetric window; a journal bound
  to another session never covers, so later work by other sessions cannot erase
  the verdict. Exclusion of the starting
  session is by identity alone: recency is not liveness evidence, since a
  crash followed by a fast restart sits inside any recency window;
  when neither covers it, the wake leads with an `UNCLEAN END` block that names
  the session, the gap, the foreign checkpoint if any, and the session's own
  bounded recovered tail, and `synapt resume` carries the same verdict in its
  header. The starting session is excluded by the hook payload's `session_id`
  (or `CLAUDE_CODE_SESSION_ID` / `CODEX_THREAD_ID` on the generic
  `synapt recall startup` path); a caller that cannot name its own session does
  not publish the verdict, so a wake never reports itself.
- **`LAST CHECKPOINT` names the session it belongs to**, so a checkpoint from
  one session is not read as the bridge for another.
- **The recovered tail skips runtime-authored user lines.** `<task-notification>`,
  `<system-reminder>` and the slash-command echo family are written with the
  user role but are not the operator's words; the checkpoint reader now keeps
  the last human line instead (the crashed session's real last request, not the
  background-task notice that followed it). Prose that merely quotes a tag keeps
  its author.
- **A #dev backlog is read one line per message, up to thirty.** Five messages
  at full detail is the right read after a quiet night and the wrong one after a
  gap: with eleven unread, three rendered inside the channel byte cap and the
  two that mattered were withheld. Above five unread the wake switches to the
  one-line form so it covers what happened rather than the three newest posts.
- **Cross-gripspace direct messages no longer silo.** `send_message` reported
  "Delivered" on a cross-org send while the recipient saw nothing: both the JSONL
  inbox and the SQLite delivery-state store resolve through the gripspace-local
  `_channels_dir(project_dir)`, so a send from gripspace A landed in A's store
  while a read from gripspace B queried B's store and found nothing. The fix adds
  a project-independent, org-canonical cross-org root
  (`~/.synapt/channels/<org>/_cross-org/direct/<recipient>.jsonl`, honoring
  `SYNAPT_SHARED_CHANNELS_DIR`): `send_message` dual-writes the local inbox and
  the canonical inbox, and `read_inbox` unions the local SQLite delivered set
  with the canonical messages, deduped by `message_id`, urgent-then-oldest, with
  a local tracking row written for first-read cross-org messages so re-reads
  dedup and `ack` transitions them. Forward-fix (dual-write + union-read), no
  destructive migration of existing inboxes.
- **Messages the operator queued mid-turn are indexed.** Typing while a turn is
  running does not write a `type: "user"` line: Claude Code emits a
  `queue-operation` line, which the parser skips as unstructured noise, plus an
  `attachment` line whose `attachment.prompt` carries the text. Neither was read,
  so the operator's own words never reached `user_text` and a query for something
  they said mid-turn could return the assistant's later paraphrase and never the
  message itself.

  `origin.kind` is the safety boundary and there are at least three cohorts, so
  it also defines the measurement cohort: `human` (the operator, the only user
  turn), `peer` (another agent's queued message), and entries with no `origin`
  key at all (`commandMode == "task-notification"`, machine background events).
  Presence of an `origin` cannot distinguish authorship, since both `human` and
  `peer` have one; only `origin.kind == "human"` is accepted.

  Measured on one workspace at 2026-08-30T13:44Z, and every count is a property
  of that moment because these files are appended to continuously: of 510
  `queued_command` attachments, 352 were `human` against 1075 ordinary user
  turns — about a quarter of what the operator said — and 349 of the 352 appear
  nowhere else. The remaining 158 are task-notifications, not empty text.

  `prompt` arrives as a string or as a list of content blocks; both are read,
  and only text blocks within a list are the operator's words.

  **A rebuild is required for the fix to reach existing chunks.**
  `--incremental` decides what to re-parse from an mtime-and-size manifest, so
  transcripts that have not changed since their last index are skipped no matter
  what the parser now understands. Run a full build, or invalidate the manifest.

## [0.21.0] — 2026-08-26

**Scope.** This release promotes `v0.20.0..b8f22c4`: 7 commits and 2 first-parent
units, measured with `git rev-list --count` and `git rev-list --first-parent --count`
over that exact range.

### Added
- **Bounded SessionEnd recovery checkpoints.** Claude Code and Codex hooks capture a
  scrubbed continuity checkpoint inside the three-second SessionEnd ceiling without
  importing SQLite, subprocess, or the recall indexing stack.
- **Runtime compaction handoff indexing.** Recall locates trustworthy compaction
  summaries while indexing sessions and can surface the newest handoff on a later
  fresh start.
- **Standalone release binaries.** Release automation now builds hermetic macOS arm64
  and Linux x86_64 executables with checksums for package-manager distribution.

### Changed
- **SessionStart continuity is configurable.** `off`, `explicit`, `automatic`, and
  `always` modes let users choose how much startup recovery they want. Automatic is
  the default.
- **Compaction-triggered starts do no resume work.** Plugin and global SessionStart
  matchers exclude `compact`, with an early CLI guard as a backstop. The active
  conversation already carries its context across compaction.
- **Agent plugin versions advance to 0.3.0.** Both the Claude Code and Codex bundles
  include the SessionEnd checkpoint and compact-start exclusion.

## [0.20.0] — 2026-08-25

**Scope.** This release promotes `v0.19.1..5b891d0`: 7 commits and 2 first-parent
units, measured with `git rev-list --count` and `git rev-list --first-parent --count`
over that exact range.

### Added
- **Automatic bounded session continuity for Codex and Claude Code.** The repository
  plugin bundles now own SessionStart hooks that inject a time-budgeted wake
  without making transcript resume the normal launch path. The bundles are distributed
  from the repository rather than inside the Python sdist or wheel, and their versions
  advance from 0.1.0 to 0.2.0.
- **`synapt recall catchup` runs deferred session maintenance explicitly.** The command
  archives newly available transcripts and rebuilds stale recall state outside the
  synchronous SessionStart hook.

### Changed
- **Session listing and resume hydrate only selected sessions.** Routing metadata is
  read first, then full event payloads are loaded only for the bounded result set.
- **SessionStart defers transcript catchup.** The hook budgets its wake and records the
  outcome. Journal compaction and startup-context generation remain synchronous.

### Fixed
- **Windows build locking.** Windows can read the build-lock stamp while the lock is held.

## [0.19.0] — 2026-08-21

**Counting rule, stated so it can be checked.** Range `ca62880..bc594f3` — from the 0.18.0
promotion merge to the `dev` tip at this release. Over that range: **35 commits total, 17 of
them merge commits and 18 direct. 17 first-parent commits.**

Composition of those 17, by path: **12 touch `src/` when each unit is diffed against its first
parent**; 5 touch only tests, docs, or fixtures; and one of those 5 (`116ffcc`) is a
post-promotion topology-repair merge carrying no content by its own message. "Reviewed units
landed" is therefore 16 if you mean units that changed something, and 17 if you mean merges in
the range.

**The diff method is part of the claim, not pedantry.** A merge commit has no canonical diff.
Against first parent — or `git log --first-parent --diff-merges=first-parent` — the answer is
12. But `git show <merge> --name-only`, which is the command most readers will reach for first,
returns **0**: a combined diff omits everything that was not a conflict resolution, and it does
so silently rather than as an error. A count offered for checking is worth only as much as the
method that reproduces it.

**Why 0.19.0 and not 0.18.1.** 0.18.0 was published to PyPI at `2026-08-07T18:01:21Z`. **Every
one of the 17 units above landed after that instant** — measured against each commit's date,
not inferred. The published 0.18.0 artifact contains none of this work. PyPI forbids
re-uploading a version, so this is a new minor rather than a patch to the existing one.

An earlier draft of this section said "11 of the units above landed after that date." **The
number 11 was real but attached to the wrong category** — it counts units carrying product
changes, which the prose never named, while the sentence was making a claim about *timing*,
where the true answer is all 17. Caught in review before this shipped. Worth recording rather than silently
correcting, because this section's heading invites a reader to check the numbers, and a reader
who finds one that fails has no way to know the other four are exact.

### Fixed
- **Windows test collection, broken on every pull request since 2026-08-07.**
  `tests/recall_store_isolation.py` imported the POSIX-only `pwd` module at module scope, and
  `conftest.py` imports that module at *root* scope. The `ModuleNotFoundError` therefore fired
  before any collection at all — earlier than skip markers exist, so no marker could have
  applied — taking down the entire Windows run rather than one module. The import is now made
  at call time. POSIX behaviour is byte-identical.

  **What is recorded here is the fact, without a theory attached.** Across all 19 pull-request
  runs since 0.18.0 published, three Windows jobs failed in **every single one**, without
  exception. That is measured, not sampled.

  Why it went unaddressed for two weeks is **not something this entry can answer honestly.** An
  earlier draft offered an explanation — that the rest of the matrix stayed green, so a
  partially-red result read as a flaky lane rather than a broken one. Measurement does not
  support it: that pattern holds in 8 of the 19 runs and is false in the other 11, where the
  entire matrix was red for reasons this fix does not address and this entry does not diagnose.
  The explanation was removed rather than softened.

  Deliberately **not** replaced with `Path.home()`: that function reads `$HOME`, which is
  precisely the value a test fixture can move, and the protected boundary must not be
  derivable from the value under test or the guarantee becomes circular. On Windows the
  function still raises, loudly and at the point of use, which is the honest outcome for a
  POSIX-only guarantee.

### Added
- **Incremental builds by default**, with a new `maintain` command and change-detection
  idempotence.

### Changed
- **Summary work moved out of `build` and into `maintain`.** This is a user-visible behaviour
  change, not only an internal one: a `build` that previously produced summaries no longer
  does, and `maintain` is where that work now happens.

### Improved
- **Store and data-root isolation**, including a resolution fix so that membership takes
  precedence over locality.
- **Journal correctness**, and a root-resolution fix across the archive verbs — export and
  import, and also the archive, CLI, and server paths.
- **`code_git` hardening** and a session-start prompt fix.

### Documentation
- Fixture provenance is now declared for the identify test fixtures.

### Note on the changelog gap
**0.15.2, 0.15.3, 0.16.0, 0.17.0, and 0.18.0 shipped without changelog entries** — the gap is
wider than the three versions an earlier draft named. Rather than reconstruct them after the
fact from commit archaeology, it is recorded here honestly.

Their content **is** recoverable from the git history between the corresponding tags. **All five
were checked on the public remote** with `ls-remote` — `v0.15.2`, `v0.15.3`, `v0.16.0`,
`v0.17.0`, and `v0.18.0` — not merely the three an earlier draft vouched for while naming five.
A recovery instruction is worth only as much as the refs it names.

## [0.15.1] — 2026-05-12

### Fixed
- Fixed `recall_search` falsely reporting a missing index after the cached
  embedding-enabled index was loaded. `_get_index()` now correctly updates the
  embedding-cache flag instead of swallowing an `UnboundLocalError` and
  returning `None`.
- Fixed `recall_reload()` for editable installs so it restarts the MCP server
  even when the package version string has not changed.

## [0.11.0] — 2026-04-09

### Added
- **Agent-attributed recall** — `TranscriptChunk` now carries an optional `agent_id` field, auto-populated from `SYNAPT_AGENT_ID` env var. Scoped search via `lookup(agent_id="opus")` returns only that agent's transcripts plus legacy chunks and shared knowledge nodes. Wildcard `agent_id="*"` searches all. (#618)
- **ActionRegistry for plugin-aware channel dispatch** — new `synapt.recall.actions` module replaces the monolithic if/elif dispatcher in `recall_channel()`. OSS registers 13 base actions; premium plugin can register additional actions or override existing ones at import time. Three-tier status model (available/locked/unknown) for action discovery. (#621, #622)
- **Structured channel message types** — messages can carry a `msg_type` field (status, claim, pr, code, message) for filtering on read. (#444)

### Improved
- **Channel dispatch** — `recall_channel()` now routes through the shared ActionRegistry instead of a 150-line hard-coded switch. Net -128 lines in server.py.
- **SQLite schema migration** — `agent_id TEXT` column added transparently to existing databases via `_migrate_chunks_table()`.

## [0.6.1] — 2026-03-13

### Added
- **Secret scrubbing** — API keys, tokens, passwords, JWTs, and connection strings are scrubbed from transcripts at index time with deterministic `[REDACTED:hash]` placeholders. New `synapt recall rescrub` CLI command retroactively cleans existing archives (#65)
- **X/Twitter MCP plugin** — read timelines, search, post, reply, and thread via `synapt.plugins` entry point with prompt injection safeguards (#62)
- **MCP server setup guide** — step-by-step configuration docs for Claude Code, Cursor, and Windsurf (#66)
- **Advanced search/config reference** — documentation for all search parameters, intent types, and configuration options (#66)
- **Windows & cross-platform support** — platform-aware paths, optional MLX/ONNX, graceful fallbacks (#45)

### Improved
- **Content-aware adaptive filtering** — conversations classified as code/personal/mixed; personal content gets relaxed consolidation but `max_knowledge=0` in retrieval (#54)
- **Proactive recall** — `recall_quick` tool for fast, speculative memory checks; improved MCP server instructions (#54)
- **Blog & website** — index page, mobile responsive, cross-links, SEO, OG tags, token efficiency section (#55, #56, #57)

### Benchmarks
- **LOCOMO J-Score: 73.38%** (v0.5.1, Ministral 3B local) — unchanged from v0.6.0; now documented with Full-Context comparison (#56)
- **CodeMemo: 94% J-Score** — new benchmark for code-project memory (50 questions, 1 project)

## [0.6.0] — 2026-03-12

### Added
- **Aggregation-aware entity search** — reduced tier discounts and wider search limits for aggregation queries, plus entity-only knowledge FTS to surface scattered facts about a person across sessions
- **Category-intent alignment** — knowledge nodes whose category matches the query intent get a 1.5x boost (e.g., decision nodes for decision queries)
- **Inferential multi-hop patterns** — aggregation classifier now handles "would X enjoy", "based on the conversation", "who is [Name]" style queries
- **Decision intent** — new intent category for surfacing past decisions, with dedicated patterns and journal decision boost
- **MCP server --dev mode** — auto-reload on source changes via `synapt server --dev` (requires `watchfiles`)
- **Plugin backend registry** — extensible model routing via `synapt.backends` entry points
- **CLI subcommand discovery** — plugins can register CLI commands via `synapt.commands` entry points
- **Tool result enrichment** — enrichment summaries now include tool output content (config values, URLs, command outputs)
- **Entity-anchored FTS** — supplementary FTS search using extracted entities for better multi-hop retrieval
- **Source session IDs** — knowledge nodes display their source session for provenance tracking

### Improved
- **Embedding-based inline dedup** — cosine similarity (≥0.80) fallback after Jaccard for knowledge node deduplication
- **Generic knowledge filter** — tool-tautology patterns and specificity signals remove low-value knowledge nodes
- **Cluster summary hallucination detection** — novel entity check prevents fabricated summaries
- **Word-aware truncation** — prevents mid-word corruption in knowledge node content
- **Garbled knowledge rejection** — filters corrupt nodes with section prefixes or malformed content
- **3B model robustness** — improved handling of smaller model outputs (JSON repair, truncated dict repair, text fallback parser)
- **Intent classification expanded** — broader factual, decision, and aggregation pattern coverage with tightened patterns to reduce false positives
- **Aggregation knowledge_boost** — increased from 1.5 to 2.5 after A/B testing showed higher boost improves retrieval
- **Default decoder model** — switched from Llama-3.2-3B to Ministral-3-3B-Instruct-2512-4bit

### Fixed
- **Intent threading** — intent parameter now threaded through both global and progressive lookup paths
- **Chunk ID collision** — fixed for short session IDs
- **Consolidation** — fixed producing 0 knowledge nodes for non-code data
- **Timestamp truncation** — preserve full timestamp for free-text date formats
- **Knowledge interleaving** — interleave by relevance instead of always prepending

### Benchmarks
- **LOCOMO J-Score: 73.38%** (Ministral 3B local enrichment) — beats Full-Context upper bound (72.90%), Mem0+Graph (68.44%), Mem0 (66.88%), Zep (65.99%)
- Open-domain 80.14% — best of all systems tested
- Multi-hop 70.21% — best of all systems tested

## [0.5.0] — 2026-03-10

### Added
- **Hybrid RRF search** — reciprocal rank fusion combining FTS5, BM25, and semantic embeddings
- **Intent classification** — routes queries to adjust embedding weight, recency decay, and knowledge boost
- **Knowledge graph** — LLM-powered enrichment and consolidation pipeline
- **Cross-encoder reranking** — ms-marco-MiniLM-L-6-v2 for result re-ranking
- **ONNX Runtime inference** — 6.6x faster T5 enrichment on CPU
- **Configurable model selection** — global and project-level config with env var overrides
- **Result deduplication** — Jaccard-based near-duplicate filtering
- **Confidence-weighted knowledge boost** — higher-confidence nodes rank higher
- **Query result cache** — repeated lookups return cached results
- **Temporal date extraction** — parse date ranges from search queries
- **Source expansion** — knowledge nodes include source_turns for chunk-level provenance
- **Proactive MCP instructions** — server instructions tell Claude to search before answering
- **GitHub Pages site** — landing page at synapt.dev
- **CLA bot** — contributor license agreement enforcement

### Fixed
- **Consolidation model routing** — fixed T5 being used instead of decoder-only
- **Markdown response parser** — handle markdown-wrapped JSON from LLMs
- **Non-project path filtering** — filter irrelevant paths from journal file lists
- **Enrichment prompt** — generalized for non-code sessions

## [0.3.0] — 2026-03-08

Initial public release with core recall functionality:
- Session transcript indexing and search
- BM25 + semantic embedding retrieval
- Journal entries and cross-session reminders
- MCP server with 13 tools
- Timeline and session listing
- Plugin architecture
