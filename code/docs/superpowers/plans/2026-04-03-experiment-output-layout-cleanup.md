# Experiment Output Layout Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make SAVER training and inference outputs predictable, stage-separated, and lazily created so each experiment is easy to inspect and no unused subtrees are created.

**Architecture:** Keep shared, reusable data products under `code/data_utils/` and move experiment-local outputs into a stricter run layout under `ckpt/<EXP_NAME>/`. Decouple checkpoint storage, evaluation outputs, and logs so each stage writes into one canonical subtree, and remove all eager `mkdir -p` calls that currently create empty directories before work starts.

**Tech Stack:** Bash entry scripts, Python training entrypoints, `pathlib`, existing `saver_agent.experiment_logging`, unittest-based script tests.

---

## Target Layout

### Experiment-local outputs

```text
ckpt/<exp_name>/
  manifest.json
  status.json
  logs/
    pipeline/
      00_full_pipeline.<timestamp>.log
    sft/
      train_saver_sft_run_config.json
      run_weighted_sft_config.json
    rl/
      train_saver_rl_run_config.json
      rl_iteration_metrics.jsonl
    rollout/
      03_batch_rollout_score_summarize.<timestamp>.log
  checkpoints/
    sft/
      checkpoint-*
      epoch_resume/
    rl/
      iter_001/
      iter_002/
  eval/
    sft_epoch_end/
      metrics_history.jsonl
      epoch_001/
        metrics.json
        raw_shards/
        scored_shards/
    batch_rollout/
      <run_name>/
        rollouts.raw.jsonl
        rollouts.scored.jsonl
        summary.json
    rl/
      iter_001/
        raw_shards/
        scored_shards/
  artifacts/
    teacher_judge/
    reports/
```

### Shared preprocessing outputs

```text
data_utils/
  runtime/
    msad_saver_runtime_train.jsonl
    msad_saver_runtime_test.jsonl
  sft/
    msad_saver_sft_train.jsonl
    msad_saver_sft_train.teacher.jsonl
  summaries/
    frame_cache_summary.json
    feature_cache_summary.json
  tensor_cache/
    <prepared_data_stem>/
      <cache_key>/
```

## Non-Negotiable Rules

1. A stage may create a directory only when it is about to write the first file inside that directory.
2. Logs live under `ckpt/<exp_name>/logs/<stage>/`, never inside checkpoint directories.
3. Epoch-end evaluation outputs live under `ckpt/<exp_name>/eval/sft_epoch_end/`, never mixed into SFT checkpoint roots.
4. Batch rollout outputs live under `ckpt/<exp_name>/eval/batch_rollout/<run_name>/`, never in placeholder rollout roots unless files are actually produced.
5. Shared datasets and reusable caches stay under `data_utils/`; experiment directories should reference them in `manifest.json` instead of copying them.
6. Every experiment root must contain a machine-readable `manifest.json` and `status.json` so the user can answer “what was used and what was produced?” without opening scripts.

## Acceptance Criteria

- Running `bash scripts/00_full_pipeline.sh` with a fresh `EXP_NAME=expX` produces no empty `train_artifacts`, empty rollout run directories, or redundant `logs/` subtrees under checkpoint roots.
- `ckpt/<exp_name>/` is readable at a glance: logs, checkpoints, eval, artifacts are separated by responsibility.
- SFT epoch-end evaluation metrics are written once to a canonical location; duplicate `rollout_eval_metrics.jsonl` and duplicate summary logs are removed.
- RL iteration outputs no longer mix checkpoint files with rollout/eval scratch files.
- Existing environment variable overrides still work.
- Existing tests for experiment-name scripts and full-pipeline control flow are updated and green.

### Task 1: Introduce A Canonical Experiment Path Model

**Files:**
- Create: `saver_agent/experiment_paths.py`
- Modify: `saver_agent/experiment_logging.py`
- Modify: `scripts/common_experiment.sh`
- Test: `tests/test_experiment_name_scripts.py`

- [ ] **Step 1: Write the failing tests for new canonical defaults**

Add assertions that an experiment named `exp1` resolves to:

```text
ckpt/exp1/logs/pipeline
ckpt/exp1/checkpoints/sft
ckpt/exp1/checkpoints/rl
ckpt/exp1/eval/batch_rollout/sft_rollout_eval
ckpt/exp1/eval/sft_epoch_end
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m unittest tests.test_experiment_name_scripts -v
```

Expected: failures because current defaults still point to `train_artifacts`, top-level `rollouts`, and checkpoint-local logs.

- [ ] **Step 3: Implement a central path resolver**

Create a small path model in `saver_agent/experiment_paths.py`, for example:

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path
    logs_root: Path
    checkpoints_root: Path
    eval_root: Path
    artifacts_root: Path
    sft_checkpoint_dir: Path
    rl_checkpoint_dir: Path
    sft_eval_dir: Path


def build_experiment_paths(run_root: str | Path) -> ExperimentPaths:
    root = Path(run_root)
    return ExperimentPaths(
        root=root,
        logs_root=root / "logs",
        checkpoints_root=root / "checkpoints",
        eval_root=root / "eval",
        artifacts_root=root / "artifacts",
        sft_checkpoint_dir=root / "checkpoints" / "sft",
        rl_checkpoint_dir=root / "checkpoints" / "rl",
        sft_eval_dir=root / "eval" / "sft_epoch_end",
    )
```

Update `scripts/common_experiment.sh` to mirror the same structure with shell variables.

- [ ] **Step 4: Run the tests again**

Run:

```bash
python -m unittest tests.test_experiment_name_scripts -v
```

Expected: pass with the new layout defaults.

- [ ] **Step 5: Commit**

```bash
git add saver_agent/experiment_paths.py saver_agent/experiment_logging.py scripts/common_experiment.sh tests/test_experiment_name_scripts.py
git commit -m "refactor: centralize experiment output paths"
```

### Task 2: Remove Eager Directory Creation That Produces Empty Trees

**Files:**
- Modify: `scripts/00_full_pipeline.sh`
- Modify: `scripts/02_train_sft_with_rollout_eval.sh`
- Modify: `scripts/03_batch_rollout_score_summarize.sh`
- Modify: `scripts/04_train_rl.sh`
- Test: `tests/test_full_pipeline_script.py`

- [ ] **Step 1: Write failing tests for empty-directory avoidance**

Add tests that a skipped stage does not leave behind:

```text
ckpt/<exp>/artifacts/
ckpt/<exp>/eval/batch_rollout/<run_name>/
ckpt/<exp>/checkpoints/rl/
```

when that stage never wrote a file.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m unittest tests.test_full_pipeline_script -v
```

Expected: failure because `00_full_pipeline.sh` currently runs:

```bash
mkdir -p "${ANNOTATION_DIR}" "${ARTIFACT_DIR}" "${CHECKPOINT_DIR}" "${ROLLOUT_RUN_DIR}"
```

- [ ] **Step 3: Replace eager `mkdir -p` with lazy helpers**

Use shell helpers such as:

```bash
ensure_parent_dir() {
  local target="$1"
  mkdir -p "$(dirname "$target")"
}

ensure_dir_if_stage_runs() {
  local dir="$1"
  [[ -n "$dir" ]] && mkdir -p "$dir"
}
```

Rules:
- only create `ANNOTATION_DIR` when Stage 1 writes data files
- only create checkpoint dirs when SFT or RL actually runs
- only create rollout output dirs when Stage 4 actually writes raw/scored/summary files
- allow `configure_script_logging` to create `logs/pipeline/` because it immediately writes a log file

- [ ] **Step 4: Re-run the script tests**

Run:

```bash
python -m unittest tests.test_full_pipeline_script -v
```

Expected: pass, and tests explicitly verify no unused empty subtree is created.

- [ ] **Step 5: Commit**

```bash
git add scripts/00_full_pipeline.sh scripts/02_train_sft_with_rollout_eval.sh scripts/03_batch_rollout_score_summarize.sh scripts/04_train_rl.sh tests/test_full_pipeline_script.py
git commit -m "refactor: lazily create experiment output directories"
```

### Task 3: Move SFT Logs And Epoch-End Eval Outputs Out Of Checkpoint Roots

**Files:**
- Modify: `train_saver_sft.py`
- Modify: `saver_agent/training.py`
- Modify: `saver_agent/experiment_logging.py`
- Test: `tests/test_train_saver_sft.py`
- Test: `tests/test_saver_agent_training.py`
- Test: `tests/test_full_pipeline_script.py`

- [ ] **Step 1: Write failing tests for new SFT log and eval destinations**

Add tests asserting:
- SFT run config is written under `ckpt/<exp>/logs/sft/`
- epoch-end eval metrics history is written under `ckpt/<exp>/eval/sft_epoch_end/metrics_history.jsonl`
- per-epoch metrics live under `ckpt/<exp>/eval/sft_epoch_end/epoch_001/metrics.json`
- no duplicate `rollout_eval_metrics.jsonl` is written both at checkpoint root and checkpoint-root `logs/`

- [ ] **Step 2: Run the focused tests to verify they fail**

Run:

```bash
python -m unittest tests.test_train_saver_sft tests.test_saver_agent_training tests.test_full_pipeline_script -v
```

Expected: failure because current code writes logs and eval summaries directly into `output_dir` and `output_dir/logs`.

- [ ] **Step 3: Add explicit stage log and eval-output arguments**

Extend `train_saver_sft.py` and `run_weighted_sft(...)` to accept:

```text
--log-dir
--rollout-eval-output-dir
```

Implementation rules:
- default script-level `--log-dir` to `ckpt/<exp>/logs/sft`
- default `--rollout-eval-output-dir` to `ckpt/<exp>/eval/sft_epoch_end`
- keep checkpoints in `ckpt/<exp>/checkpoints/sft`
- keep `epoch_resume/` under the SFT checkpoint dir
- write metrics history only once

- [ ] **Step 4: Remove duplicate writes in `_write_rollout_eval_record`**

Replace:

```python
append_jsonl(Path(output_dir) / "rollout_eval_metrics.jsonl", record)
append_jsonl(Path(output_dir) / "logs" / "rollout_eval_metrics.jsonl", record)
```

with one canonical destination derived from the explicit eval output dir.

- [ ] **Step 5: Re-run the tests**

Run:

```bash
python -m unittest tests.test_train_saver_sft tests.test_saver_agent_training tests.test_full_pipeline_script -v
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add train_saver_sft.py saver_agent/training.py saver_agent/experiment_logging.py tests/test_train_saver_sft.py tests/test_saver_agent_training.py tests/test_full_pipeline_script.py
git commit -m "refactor: separate sft logs and eval outputs from checkpoints"
```

### Task 4: Move RL Logs And Iteration Eval Scratch Out Of RL Checkpoint Roots

**Files:**
- Modify: `train_saver_rl.py`
- Modify: `scripts/00_full_pipeline.sh`
- Modify: `scripts/04_train_rl.sh`
- Test: `tests/test_train_saver_rl.py`
- Test: `tests/test_full_pipeline_script.py`

- [ ] **Step 1: Write failing tests for RL output separation**

Add assertions that:
- RL model checkpoints stay under `ckpt/<exp>/checkpoints/rl`
- RL logs live under `ckpt/<exp>/logs/rl`
- RL rollout shards and scored shards live under `ckpt/<exp>/eval/rl/iter_XXX`
- no rollout scratch directories are created inside the RL checkpoint directory

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m unittest tests.test_train_saver_rl tests.test_full_pipeline_script -v
```

Expected: failure because `train_saver_rl.py` currently mixes iteration directories and rollout scratch under the main output dir.

- [ ] **Step 3: Add explicit RL eval output root**

Introduce a new argument such as:

```text
--eval-output-dir
```

and route iteration scratch to:

```text
ckpt/<exp>/eval/rl/iter_001/
```

while keeping actual RL checkpoints at:

```text
ckpt/<exp>/checkpoints/rl/
```

- [ ] **Step 4: Re-run the tests**

Run:

```bash
python -m unittest tests.test_train_saver_rl tests.test_full_pipeline_script -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add train_saver_rl.py scripts/00_full_pipeline.sh scripts/04_train_rl.sh tests/test_train_saver_rl.py tests/test_full_pipeline_script.py
git commit -m "refactor: separate rl checkpoints from rollout scratch outputs"
```

### Task 5: Reorganize Shared Preprocessing Outputs Inside `data_utils/`

**Files:**
- Modify: `scripts/00_full_pipeline.sh`
- Modify: `scripts/01_build_oracle_and_prepare_sft.sh`
- Modify: `scripts/preproces.sh`
- Modify: `prepare_sft_tensor_cache.py`
- Modify: `annotate_teacher_judge_sft.py`
- Modify: `README.md`
- Test: `tests/test_prepare_sft_tensor_cache.py`
- Test: `tests/test_full_pipeline_script.py`

- [ ] **Step 1: Write failing tests for new shared-data defaults**

Add tests covering these default paths:

```text
data_utils/runtime/msad_saver_runtime_train.jsonl
data_utils/runtime/msad_saver_runtime_test.jsonl
data_utils/sft/msad_saver_sft_train.jsonl
data_utils/sft/msad_saver_sft_train.teacher.jsonl
data_utils/summaries/frame_cache_summary.json
data_utils/summaries/feature_cache_summary.json
data_utils/tensor_cache/<prepared_stem>/<cache_key>/
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m unittest tests.test_prepare_sft_tensor_cache tests.test_full_pipeline_script -v
```

Expected: failure because current defaults still place files flatly under `data_utils/` and `<prepared-data>.tensor_cache`.

- [ ] **Step 3: Implement the new shared-data path policy**

Rules:
- runtime JSONL files go to `data_utils/runtime/`
- SFT JSONL files go to `data_utils/sft/`
- cache summaries go to `data_utils/summaries/`
- tensor cache defaults to `data_utils/tensor_cache/<prepared_stem>/<cache_key>/`
- retain env-var overrides and legacy fallback for one migration window

- [ ] **Step 4: Update README with one canonical tree**

Add a short “Where files go” section showing:
- shared data outputs
- experiment-local outputs
- which paths are reusable versus run-specific

- [ ] **Step 5: Re-run the tests**

Run:

```bash
python -m unittest tests.test_prepare_sft_tensor_cache tests.test_full_pipeline_script -v
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/00_full_pipeline.sh scripts/01_build_oracle_and_prepare_sft.sh scripts/preproces.sh prepare_sft_tensor_cache.py annotate_teacher_judge_sft.py README.md tests/test_prepare_sft_tensor_cache.py tests/test_full_pipeline_script.py
git commit -m "refactor: organize shared preprocessing outputs under data_utils subdirs"
```

### Task 6: Add Experiment Manifest, Status Tracking, And Legacy Cleanup

**Files:**
- Modify: `scripts/00_full_pipeline.sh`
- Modify: `scripts/common_experiment.sh`
- Create: `scripts/clean_legacy_experiment_layout.py`
- Modify: `README.md`
- Test: `tests/test_full_pipeline_script.py`

- [ ] **Step 1: Write failing tests for manifest generation**

Add expectations that each run writes:

```text
ckpt/<exp>/manifest.json
ckpt/<exp>/status.json
```

with fields like:

```json
{
  "experiment_name": "exp1",
  "paths": {
    "sft_checkpoint_dir": "...",
    "rl_checkpoint_dir": "...",
    "sft_eval_dir": "...",
    "batch_rollout_dir": "..."
  },
  "inputs": {
    "runtime_train_jsonl": "...",
    "runtime_test_jsonl": "...",
    "sft_train_jsonl": "..."
  }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m unittest tests.test_full_pipeline_script -v
```

Expected: failure because manifest and status files do not exist yet.

- [ ] **Step 3: Implement manifest + cleanup utility**

Requirements:
- `manifest.json` records resolved paths and main inputs
- `status.json` records per-stage completion and main result files
- `scripts/clean_legacy_experiment_layout.py` optionally deletes empty legacy directories such as:

```text
train_artifacts/
rollouts/<run_name>/
checkpoint-local logs/ duplicates
```

only when they are empty or exact duplicates

- [ ] **Step 4: Re-run tests**

Run:

```bash
python -m unittest tests.test_full_pipeline_script -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/00_full_pipeline.sh scripts/common_experiment.sh scripts/clean_legacy_experiment_layout.py README.md tests/test_full_pipeline_script.py
git commit -m "feat: add experiment manifest and legacy cleanup utility"
```

## Recommended Implementation Order

1. Task 1
2. Task 2
3. Task 3
4. Task 4
5. Task 6
6. Task 5

Reason:
- first establish the path model
- then stop empty-dir pollution
- then separate SFT and RL runtime outputs
- then add manifesting and cleanup
- finally move shared preprocessing defaults, which is the highest-churn compatibility change

## Verification Checklist After The Full Refactor

- [ ] `python -m unittest tests.test_experiment_name_scripts tests.test_full_pipeline_script -v`
- [ ] `python -m unittest tests.test_train_saver_sft tests.test_train_saver_rl tests.test_prepare_sft_tensor_cache -v`
- [ ] Run a smoke pipeline with `EXP_NAME=layout_smoke`
- [ ] Verify `find ckpt/layout_smoke -type d -empty` returns nothing
- [ ] Verify there is no `logs/` directory under `ckpt/layout_smoke/checkpoints/sft`
- [ ] Verify there is no duplicate `rollout_eval_metrics.jsonl`
- [ ] Verify `manifest.json` and `status.json` render the main paths correctly

## Migration Notes

- Keep backward-compatible env vars for one transition cycle.
- If an old explicit path override is set, honor it exactly.
- Do not silently move or delete user-produced files; only write new outputs to the new layout.
- Run the legacy cleanup script only after the new layout is verified on one smoke experiment.
