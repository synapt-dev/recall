"""CodeMemo competitor benchmark evaluation.

Run the CodeMemo benchmark against competitor memory systems (Mem0, Memobase)
using the same questions, judge, and scoring as the synapt eval.

Usage:
    # Mem0 (open-source, local qdrant):
    python -m evaluation.codememo.competitor_eval --system mem0

    # Mem0 on a single project:
    python -m evaluation.codememo.competitor_eval --system mem0 --project project_01_cli_tool

    # Memobase (requires running server — see notes below):
    python -m evaluation.codememo.competitor_eval --system memobase

    # Retrieval-only (no answer generation / judging):
    python -m evaluation.codememo.competitor_eval --system mem0 --retrieval-only

Notes on systems:
    - mem0: Uses the open-source `mem0ai` package with local Qdrant vector store.
      Requires: pip install mem0ai
      Uses OpenAI for LLM (fact extraction) and embeddings (OPENAI_API_KEY).
    - memobase: Requires a running Memobase server (Docker). The Python client
      talks to the server over HTTP. Set MEMOBASE_URL and MEMOBASE_API_KEY.
      See reference/memobase/src/server/readme.md for server setup.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from evaluation.codememo.eval import (
    DATA_DIR,
    SystemUnderTest,
    discover_projects,
    run_evaluation,
)
from evaluation.codememo.schema import ALL_CATEGORIES, CATEGORY_NAMES


# ---------------------------------------------------------------------------
# Transcript conversion utilities
# ---------------------------------------------------------------------------

def _parse_session_to_messages(session_path: Path) -> list[dict[str, str]]:
    """Convert a Claude Code JSONL session transcript to a flat message list.

    Each JSONL line has the structure:
        {"type": "user"|"assistant", "message": {"role": ..., "content": [...]}, ...}

    We extract text content from each entry and return a list of
    {"role": "user"|"assistant", "content": "..."} dicts suitable for mem0/memobase.
    """
    messages: list[dict[str, str]] = []
    with open(session_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg = entry.get("message", {})
            role = msg.get("role", entry.get("type", "user"))
            content_blocks = msg.get("content", [])

            # Extract text from content blocks
            text_parts: list[str] = []
            if isinstance(content_blocks, str):
                text_parts.append(content_blocks)
            elif isinstance(content_blocks, list):
                for block in content_blocks:
                    if isinstance(block, str):
                        text_parts.append(block)
                    elif isinstance(block, dict):
                        btype = block.get("type", "")
                        if btype == "text":
                            text_parts.append(block.get("text", ""))
                        elif btype == "tool_use":
                            # Include tool name and input for context
                            tool_name = block.get("name", "")
                            tool_input = block.get("input", {})
                            # Summarize tool use compactly
                            if isinstance(tool_input, dict):
                                input_str = json.dumps(tool_input, indent=None)
                                # Truncate very long tool inputs
                                if len(input_str) > 500:
                                    input_str = input_str[:500] + "..."
                            else:
                                input_str = str(tool_input)[:500]
                            text_parts.append(f"[Tool: {tool_name}] {input_str}")
                        elif btype == "tool_result":
                            tr = block.get("content", "")
                            if isinstance(tr, str):
                                # Truncate long tool results
                                if len(tr) > 1000:
                                    tr = tr[:1000] + "..."
                                text_parts.append(tr)
                            elif isinstance(tr, list):
                                for sub in tr:
                                    if isinstance(sub, dict) and sub.get("type") == "text":
                                        t = sub.get("text", "")
                                        if len(t) > 1000:
                                            t = t[:1000] + "..."
                                        text_parts.append(t)
                                    elif isinstance(sub, str):
                                        text_parts.append(sub[:1000])

            text = "\n".join(p for p in text_parts if p).strip()
            if not text:
                continue

            # Normalize role to user/assistant
            if role not in ("user", "assistant"):
                role = "user"

            messages.append({"role": role, "content": text})

    return messages


def _chunk_messages(
    messages: list[dict[str, str]],
    max_chars: int = 8000,
) -> list[list[dict[str, str]]]:
    """Split a message list into chunks that fit within a character budget.

    Mem0's LLM-based fact extraction has context limits. We batch messages
    into groups that stay under max_chars total content length.
    """
    chunks: list[list[dict[str, str]]] = []
    current_chunk: list[dict[str, str]] = []
    current_len = 0

    for msg in messages:
        msg_len = len(msg["content"])
        if current_len + msg_len > max_chars and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_len = 0
        current_chunk.append(msg)
        current_len += msg_len

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ---------------------------------------------------------------------------
# Mem0 SystemUnderTest
# ---------------------------------------------------------------------------

class Mem0SUT:
    """SystemUnderTest backed by mem0 open-source Memory class.

    Uses local Qdrant vector store (no cloud), OpenAI for LLM and embeddings.
    Each project gets a fresh Memory instance with a unique collection.
    """

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        embedding_dims: int = 1536,
        infer: bool = True,
    ):
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.embedding_dims = embedding_dims
        self.infer = infer
        self._memory = None
        self._user_id = "codememo_eval"
        self._work_dir: Path | None = None
        self._memory_count = 0

    def ingest(self, session_paths: list[Path]) -> None:
        """Ingest session transcripts into mem0."""
        from mem0 import Memory

        # Clean up previous project's data if any
        if self._work_dir and self._work_dir.exists():
            shutil.rmtree(self._work_dir, ignore_errors=True)
        self._memory = None

        self._work_dir = Path(tempfile.mkdtemp(prefix="codememo_mem0_"))
        qdrant_path = str(self._work_dir / "qdrant_data")

        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": self.llm_model,
                    "temperature": 0.0,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": self.embedding_model,
                    "embedding_dims": self.embedding_dims,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "codememo",
                    "path": qdrant_path,
                    "embedding_model_dims": self.embedding_dims,
                },
            },
            "version": "v1.1",
        }

        self._memory = Memory.from_config(config)
        self._memory_count = 0

        # Ingest each session as a batch of messages
        for i, sp in enumerate(sorted(session_paths)):
            session_id = sp.stem
            print(f"    [{i+1}/{len(session_paths)}] Ingesting {session_id}...")

            messages = _parse_session_to_messages(sp)
            if not messages:
                continue

            # Chunk messages to avoid context length issues
            message_chunks = _chunk_messages(messages, max_chars=8000)

            for chunk in message_chunks:
                try:
                    result = self._memory.add(
                        chunk,
                        user_id=self._user_id,
                        infer=self.infer,
                        metadata={"session_id": session_id},
                    )
                    added = len(result.get("results", []))
                    self._memory_count += added
                except Exception as e:
                    print(f"      Warning: mem0 add failed for chunk: {e}")
                    continue

        print(f"    Total memories stored: {self._memory_count}")
        if self.infer and self._memory_count == 0:
            raise RuntimeError(
                "Mem0 stored 0 memories during ingest. This usually means the "
                f"configured LLM model is unsupported or failing silently "
                f"(model={self.llm_model!r}), or that Mem0's internal extraction "
                "path is not honoring the requested model. Re-run with visible "
                "Mem0 logs, try a known-supported extraction model such as "
                "'gpt-4o-mini', or use --no-infer only if you explicitly want raw "
                "message storage instead of fact extraction."
            )

    def query(self, question: str, max_chunks: int = 20) -> str:
        """Search mem0 memories and format as context text."""
        if self._memory is None:
            raise RuntimeError("Must call ingest() before query()")

        results = self._memory.search(
            query=question,
            user_id=self._user_id,
            limit=max_chunks,
        )

        memories = results.get("results", [])
        if not memories:
            return "(No relevant memories found)"

        # Format memories as context text
        context_parts: list[str] = []
        for i, mem in enumerate(memories):
            memory_text = mem.get("memory", "")
            score = mem.get("score", 0.0)
            metadata = mem.get("metadata", {})
            session_id = metadata.get("session_id", "unknown")

            context_parts.append(
                f"--- [Memory {i+1}, session {session_id}, "
                f"relevance {score:.3f}] ---\n{memory_text}"
            )

        return "\n\n".join(context_parts)

    def stats(self) -> dict:
        """Return memory statistics."""
        return {
            "chunk_count": self._memory_count,
            "knowledge_count": 0,
        }

    def close(self) -> None:
        """Clean up resources."""
        self._memory = None
        if self._work_dir and self._work_dir.exists():
            shutil.rmtree(self._work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Memobase SystemUnderTest
# ---------------------------------------------------------------------------

class MemobaseSUT:
    """SystemUnderTest backed by Memobase.

    Requires a running Memobase server (Docker). The Python client
    communicates over HTTP. Set environment variables:
        MEMOBASE_URL: Server URL (default: http://localhost:8019)
        MEMOBASE_API_KEY: API key / access token for the server
    """

    def __init__(self):
        self._client = None
        self._user = None
        self._user_id = "codememo_eval"

    def ingest(self, session_paths: list[Path]) -> None:
        """Ingest session transcripts into Memobase."""
        from memobase import MemoBaseClient, ChatBlob
        from memobase.core.blob import OpenAICompatibleMessage, BlobType

        # Clean up previous project's data if any
        if self._client and self._user:
            try:
                self._client.delete_user(self._user_id)
            except Exception:
                pass

        api_key = os.environ.get("MEMOBASE_API_KEY", "")
        project_url = os.environ.get("MEMOBASE_URL", "http://localhost:8019")

        if not api_key:
            raise RuntimeError(
                "MEMOBASE_API_KEY environment variable is required. "
                "Set it to your Memobase server's access token."
            )

        self._client = MemoBaseClient(
            api_key=api_key,
            project_url=project_url,
        )

        if not self._client.ping():
            raise RuntimeError(
                f"Cannot reach Memobase server at {project_url}. "
                "Make sure the server is running (docker-compose up)."
            )

        # Create or get user
        self._user = self._client.get_or_create_user(self._user_id)

        # Ingest each session as chat blobs
        for i, sp in enumerate(sorted(session_paths)):
            session_id = sp.stem
            print(f"    [{i+1}/{len(session_paths)}] Ingesting {session_id}...")

            messages = _parse_session_to_messages(sp)
            if not messages:
                continue

            # Chunk messages into reasonable batches
            message_chunks = _chunk_messages(messages, max_chars=8000)

            for chunk in message_chunks:
                # Convert dicts to OpenAICompatibleMessage instances
                oai_messages = [
                    OpenAICompatibleMessage(role=m["role"], content=m["content"])
                    for m in chunk
                ]
                chat_blob = ChatBlob(messages=oai_messages)
                try:
                    self._user.insert(chat_blob, sync=True)
                except Exception as e:
                    print(f"      Warning: memobase insert failed: {e}")
                    continue

        # Flush buffer to ensure all data is processed
        try:
            self._user.flush(blob_type=BlobType.chat, sync=True)
        except Exception as e:
            print(f"      Warning: memobase flush failed: {e}")

    def query(self, question: str, max_chunks: int = 20) -> str:
        """Retrieve context from Memobase using event search."""
        if self._user is None:
            raise RuntimeError("Must call ingest() before query()")

        context_parts: list[str] = []

        # Strategy 1: Search events (episodic memory)
        try:
            events = self._user.search_event(
                query=question,
                topk=max_chunks,
                similarity_threshold=0.1,
            )
            for i, event in enumerate(events):
                content = event.content if hasattr(event, "content") else str(event)
                context_parts.append(
                    f"--- [Event {i+1}] ---\n{content}"
                )
        except Exception as e:
            print(f"      Warning: memobase event search failed: {e}")

        # Strategy 2: Get user context (profile + events combined)
        try:
            context_str = self._user.context(
                max_token_size=2000,
                chats=[{"role": "user", "content": question}],
            )
            if context_str and context_str.strip():
                context_parts.append(
                    f"--- [User Context] ---\n{context_str}"
                )
        except Exception as e:
            print(f"      Warning: memobase context retrieval failed: {e}")

        if not context_parts:
            return "(No relevant memories found)"

        return "\n\n".join(context_parts)

    def stats(self) -> dict:
        """Return memory statistics."""
        return {
            "chunk_count": 0,
            "knowledge_count": 0,
        }

    def close(self) -> None:
        """Clean up resources."""
        # Delete the eval user to clean up server state
        if self._client and self._user:
            try:
                self._client.delete_user(self._user_id)
            except Exception:
                pass
        self._client = None
        self._user = None


# ---------------------------------------------------------------------------
# SUT factory
# ---------------------------------------------------------------------------

SUPPORTED_SYSTEMS = {
    "mem0": "Mem0 open-source (local Qdrant + OpenAI)",
    "memobase": "Memobase (requires running server)",
}


def create_sut(
    system: str,
    llm_model: str = "gpt-4o-mini",
    infer: bool = True,
) -> SystemUnderTest:
    """Create a SystemUnderTest for the given competitor system."""
    if system == "mem0":
        return Mem0SUT(
            llm_model=llm_model,
            infer=infer,
        )
    elif system == "memobase":
        return MemobaseSUT()
    else:
        raise ValueError(
            f"Unknown system: {system}. "
            f"Supported: {', '.join(SUPPORTED_SYSTEMS.keys())}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CodeMemo competitor benchmark evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {k}: {v}" for k, v in SUPPORTED_SYSTEMS.items()
        ),
    )
    parser.add_argument(
        "--system", type=str, required=True,
        choices=list(SUPPORTED_SYSTEMS.keys()),
        help="Competitor memory system to evaluate",
    )
    parser.add_argument(
        "--project", type=str, default=None,
        help="Evaluate a single project (directory name under data/)",
    )
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="Only measure retrieval quality (no answer generation / judging)",
    )
    parser.add_argument(
        "--max-chunks", type=int, default=20,
        help="Number of memories to retrieve per question (default: 20)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory for results (default: results/<system>/)",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        help="OpenAI model for answer generation and judging (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--no-infer", action="store_true",
        help="(mem0 only) Store raw messages without LLM fact extraction",
    )
    args = parser.parse_args()

    # Discover projects
    project_dirs = discover_projects(args.project)
    if not project_dirs:
        print(f"No projects found in {DATA_DIR}")
        if args.project:
            print(f"  (filtered for: {args.project})")
        sys.exit(1)

    print(f"System: {args.system} ({SUPPORTED_SYSTEMS[args.system]})")
    print(f"Found {len(project_dirs)} project(s): "
          f"{', '.join(d.name for d in project_dirs)}")

    # Set up output path
    if args.output is None:
        args.output = Path("evaluation/codememo/results") / args.system

    # Create SUT
    sut = create_sut(
        system=args.system,
        llm_model=args.model,
        infer=not args.no_infer,
    )

    try:
        summary = run_evaluation(
            project_dirs=project_dirs,
            sut=sut,
            retrieval_only=args.retrieval_only,
            max_chunks=args.max_chunks,
            output_path=args.output,
            model=args.model,
        )

        # Add competitor system info to summary
        summary["system"] = args.system
        summary["system_description"] = SUPPORTED_SYSTEMS[args.system]
        if args.system == "mem0":
            summary["mem0_infer"] = not args.no_infer

        # Re-save with system info
        with open(args.output / "codememo_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to {args.output}/")

    finally:
        sut.close()


if __name__ == "__main__":
    main()
