# Findings

## Data Utils Inventory
- `data_utils/` currently contains runtime train/test JSONL, SFT train JSONL, teacher-annotated SFT JSONL, 4 teacher shards, and cache summaries.

## Verified Facts
- `msad_saver_runtime_train.jsonl` has 480 rows, all `split=train`.
- `msad_saver_runtime_test.jsonl` has 240 rows, all `split=test`.
- `msad_saver_sft_train.jsonl` and `msad_saver_sft_train.teacher.jsonl` both have 4183 rows, all `split=train`, covering the same 480 unique `video_id` values as `runtime_train`.
- Current cache coverage is complete for both runtime datasets: 480/480 train videos and 240/240 test videos have `.frame_cache` and `.feature_cache`.

## Root Cause
- The existing prepared SFT files under `data_utils/` are stale relative to current message-building logic.
- Concrete proof: rebuilding the first 20 runtime-train records with current `build_prepared_sft_examples_from_jsonl(...)` produces 280 prepared examples, and 225/280 differ from the existing `data_utils/msad_saver_sft_train.jsonl`.
- The first observed mismatch is `video_id=Assault_1`, `step_index=3`, where the current code serializes a `seek_evidence` tool observation with 4 evidence frames but the stored prepared JSONL contains only 1 frame.
- This means SFT training has been consuming weaker/staler tool observations than rollout eval / rollout inference, creating a real train-vs-inference observation mismatch.

## Contributing Factors
- `00_full_pipeline.sh` previously skipped Stage 1 whenever the JSONL files already existed, so prepared data could become stale after code changes and still be silently reused.
- `frame_cache_summary.json` and `feature_cache_summary.json` were overwritten by the last cache pass, so the summaries were misleading even though the underlying caches were complete.

## Implemented Fixes
- Added prepared-data metadata sidecars at `<prepared>.meta.json` with schema version plus preview/prompt config snapshots.
- `train_saver_sft.py` now refuses to load prepared data when metadata is missing, unreadable, schema-mismatched, or prompt/preview-incompatible.
- `prepare_sft_tensor_cache.py` and `annotate_teacher_judge_sft.py` now also require valid prepared-data metadata.
- `annotate_teacher_judge_sft.py` now writes metadata for its output JSONL as well.
- `build_saver_data.py` now writes metadata sidecars for both base SFT and teacher-annotated SFT outputs.
- `scripts/00_full_pipeline.sh` now treats prepared metadata sidecars as Stage 1 targets, so missing metadata forces regeneration instead of silently skipping.
- `scripts/00_full_pipeline.sh` now writes split-specific Stage 2 cache summaries (`*.train.json`, `*.test.json`) instead of overwriting a single summary file.
