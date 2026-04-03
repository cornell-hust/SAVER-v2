# Task Plan

## Goal
Audit the `data_utils/` preprocessing outputs and verify they are produced correctly and consumed consistently by SFT training, rollout eval, and RL, with special attention to training/inference observation consistency.

## Phases
- [completed] Phase 1: Inventory `data_utils/` outputs and inspect their actual contents and counts
- [completed] Phase 2: Trace code paths that generate these files
- [completed] Phase 3: Trace code paths that consume these files in SFT, rollout eval, and RL
- [completed] Phase 4: Compare observation/prompt/config pathways for train vs eval and identify mismatches
- [completed] Phase 5: Implement fixes for prepared-data drift detection and cache-summary overwrites, then re-verify

## Notes
- Focus on concrete file paths under `code/data_utils/`
- Avoid guessing; confirm actual filenames, splits, counts, and reader code paths
- Main confirmed bug: existing `msad_saver_sft_train(.teacher).jsonl` is stale relative to current prepared-message construction logic, while runtime train/test caches themselves are complete.
