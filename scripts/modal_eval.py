"""Run CodeMemo and LOCOMO ablations on Modal — fire and forget.

Usage:
    # Run all ablations
    modal run scripts/modal_eval.py

    # Run a single config
    modal run scripts/modal_eval.py --config cm_full

    # Run all LOCOMO configs
    modal run scripts/modal_eval.py --config lm_

    # Check results
    modal volume get synapt-eval-vol results/ablation_results.json /tmp/results.json
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
PRIVATE_REPO = REPO_ROOT.parent / "synapt-private"
LOCOMO_DATASET = PRIVATE_REPO / "evaluation" / "dataset" / "locomo10.json"

app = modal.App("synapt-eval")

volume = modal.Volume.from_name("synapt-eval-vol", create_if_missing=True)

_image = (
    modal.Image.debian_slim(python_version="3.13")
    .add_local_dir(
        REPO_ROOT,
        remote_path="/root/synapt",
        copy=True,
        ignore=[".git", "__pycache__", "*.egg-info", ".claude/worktrees"],
    )
    .pip_install(
        "openai>=1.0",
        "huggingface_hub",
    )
    .run_commands("pip install -q /root/synapt")
)

# Bundle LOCOMO dataset if available (lives in private repo)
if LOCOMO_DATASET.exists():
    image = _image.add_local_file(
        LOCOMO_DATASET,
        remote_path="/root/synapt/evaluation/dataset/locomo10.json",
        copy=True,
    )
else:
    image = _image

RESULTS_DIR = "/results"

# ---------------------------------------------------------------------------
# Environment configs (subtractive ablations)
# ---------------------------------------------------------------------------

FULL = {
    "SYNAPT_SUBCHUNK_MIN_TEXT": "1200",
    "SYNAPT_CROSS_LINK_MAX_EXPAND": "3",
    "SYNAPT_MAX_PER_SESSION": "4",
    "SYNAPT_MAX_SUBCHUNKS_PER_SESSION": "2",
}


def minus(feature: str) -> dict:
    env = {**FULL}
    env[f"SYNAPT_DISABLE_{feature.upper()}"] = "1"
    return env


# (description, env_overrides, extra_args, benchmark)
# benchmark: "codememo" or "locomo"
CONFIGS = {
    # CodeMemo ablations
    "cm_full": ("Full (all features ON)", FULL, [], "codememo"),
    "cm_-reranker": ("- reranker", minus("reranker"), [], "codememo"),
    "cm_-intent": ("- intent classification", minus("intent"), [], "codememo"),
    "cm_-dedup": ("- deduplication", minus("dedup"), [], "codememo"),
    "cm_-boosts": ("- working memory boosts", minus("boosts"), [], "codememo"),
    "cm_-clusters": ("- cluster summaries", minus("clusters"), [], "codememo"),
    "cm_-content_profile": ("- content profile", minus("content_profile"), [], "codememo"),
    "cm_-subchunks": ("- sub-chunk splitting", {**FULL, "SYNAPT_SUBCHUNK_MIN_TEXT": "999999999"}, [], "codememo"),
    "cm_-links": ("- cross-session links", {**FULL, "SYNAPT_CROSS_LINK_MAX_EXPAND": "0"}, [], "codememo"),
    "cm_full_pipeline": ("Full pipeline (+ enrich)", FULL, ["--full-pipeline"], "codememo"),
    # LOCOMO ablations
    "lm_full": ("Full (all features ON)", FULL, [], "locomo"),
    "lm_-reranker": ("- reranker", minus("reranker"), [], "locomo"),
    "lm_-intent": ("- intent classification", minus("intent"), [], "locomo"),
    "lm_-dedup": ("- deduplication", minus("dedup"), [], "locomo"),
    "lm_-boosts": ("- working memory boosts", minus("boosts"), [], "locomo"),
    "lm_-clusters": ("- cluster summaries", minus("clusters"), [], "locomo"),
    "lm_-content_profile": ("- content profile", minus("content_profile"), [], "locomo"),
    "lm_-subchunks": ("- sub-chunk splitting", {**FULL, "SYNAPT_SUBCHUNK_MIN_TEXT": "999999999"}, [], "locomo"),
    "lm_-links": ("- cross-session links", {**FULL, "SYNAPT_CROSS_LINK_MAX_EXPAND": "0"}, [], "locomo"),
    "lm_full_pipeline": ("Full pipeline (+ enrich)", FULL, ["--full-pipeline"], "locomo"),
}


# ---------------------------------------------------------------------------
# Modal function — runs a single ablation config
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("openai-secret")],
    volumes={RESULTS_DIR: volume},
    timeout=3600 * 8,  # 8 hours — rate limits make it slow
    cpu=1,
    memory=2048,
)
def run_ablation(
    name: str, desc: str, env_overrides: dict,
    extra_args: list[str] | None = None, benchmark: str = "codememo",
) -> dict:
    """Run a single ablation config (CodeMemo or LOCOMO)."""
    import subprocess
    import sys

    # Apply env overrides
    os.environ.update(env_overrides)

    output_dir = Path(RESULTS_DIR) / name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Benchmark-specific settings
    if benchmark == "locomo":
        module = "evaluation.locomo_eval"
        summary_file = "locomo_summary.json"
        checkpoint_file = "locomo_checkpoint.json"
        score_key = "j_score_overall"
    else:
        module = "evaluation.codememo.eval"
        summary_file = "codememo_summary.json"
        checkpoint_file = "codememo_checkpoint.json"
        score_key = "j_score_overall"

    # Check for existing final results (skip if already complete)
    summary_path = output_dir / summary_file
    if summary_path.exists():
        result = json.loads(summary_path.read_text())
        if result.get("questions_evaluated", 0) > 0:
            print(f"SKIP {name}: already complete, J={result.get(score_key, '?')}%")
            volume.commit()
            return result

    # Reload volume to pick up checkpoints from preempted runs
    volume.reload()

    print(f"=== {name}: {desc} ({benchmark}) ===")
    print(f"Env: {json.dumps(env_overrides, indent=2)}")

    # Run the eval, committing volume after each checkpoint write
    # so preemption doesn't lose progress
    cmd = [sys.executable, "-m", module, "--recalldb", "--output", str(output_dir)]
    if benchmark == "codememo":
        cmd += ["--model", "gpt-4o-mini"]
    cmd += extra_args or []

    checkpoint_path = output_dir / checkpoint_file
    last_size = checkpoint_path.stat().st_size if checkpoint_path.exists() else 0

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, cwd="/root/synapt")
    stdout_lines = []
    for line in proc.stdout:
        stdout_lines.append(line)
        print(line, end="", flush=True)
        # Commit volume whenever the checkpoint file grows (new answer saved)
        if checkpoint_path.exists():
            cur_size = checkpoint_path.stat().st_size
            if cur_size != last_size:
                last_size = cur_size
                volume.commit()
    proc.wait()
    stderr = proc.stderr.read()
    if proc.returncode != 0:
        print(f"STDERR: {stderr[-1000:]}")

    # Read and return results
    volume.commit()
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    return {"error": f"No summary produced, exit={result.returncode}"}


# ---------------------------------------------------------------------------
# Orchestrator — runs all configs serially
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("openai-secret")],
    volumes={RESULTS_DIR: volume},
    timeout=3600 * 24,  # 24 hours
    cpu=1,
    memory=512,
)
def run_all(config_filter: str | None = None):
    """Run all ablation configs serially."""
    t_start = time.time()

    configs_to_run = CONFIGS
    if config_filter:
        # Support comma-separated filters: "subchunks,links,full_pipeline"
        filters = [f.strip() for f in config_filter.split(",")]
        configs_to_run = {
            k: v for k, v in CONFIGS.items()
            if any(f in k for f in filters)
        }

    results = {}
    for name, (desc, env, extra, bench) in configs_to_run.items():
        print(f"\n{'='*60}")
        print(f"Launching: {name} — {desc}")
        print(f"{'='*60}")

        result = run_ablation.remote(name, desc, env, extra, bench)
        results[name] = result

        j = result.get("j_score_overall", "?")
        n = result.get("questions_evaluated", "?")
        print(f">> {name}: J={j}% (n={n})")

    # Save combined results
    elapsed = time.time() - t_start
    combined = {
        "elapsed_min": round(elapsed / 60, 1),
        "configs": {},
    }
    for name, (desc, *_) in configs_to_run.items():
        if name in results and "error" not in results[name]:
            combined["configs"][name] = {
                "description": desc,
                **results[name],
            }

    combined_path = Path(RESULTS_DIR) / "ablation_results.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    volume.commit()

    # Print summary
    print(f"\n\n{'='*70}")
    print(f"ABLATION RESULTS ({elapsed/60:.0f}min)")
    print(f"{'='*70}")
    print(f"{'Config':<25s} {'J-Score':>8s}  Description")
    print("-" * 70)
    for name, (desc, *_) in configs_to_run.items():
        if name in results and "j_score_overall" in results[name]:
            print(f"{name:<25s} {results[name]['j_score_overall']:>8}  {desc}")
        else:
            print(f"{name:<25s} {'FAILED':>8}  {desc}")

    print(f"\nSaved to {combined_path}")
    return combined


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(config: str = ""):
    """Run ablations on Modal.

    Args:
        config: Optional filter — run only configs matching this string.
                Empty = run all.
    """
    if config:
        print(f"Running configs matching: {config}")
        result = run_all.remote(config_filter=config)
    else:
        print("Running all ablation configs")
        result = run_all.remote()

    print(f"\nDone! Results on Modal volume 'synapt-eval-vol'")
    print(f"  modal volume get synapt-eval-vol results/ablation_results.json ./results.json")
