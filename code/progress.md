# Progress

## 2026-04-03
- Started audit of preprocessing outputs and downstream consumption paths.
- Confirmed current `data_utils/` file inventory before tracing generation and usage code.
- Confirmed runtime train/test JSONL splits and cache coverage are complete: 480/480 train videos and 240/240 test videos have both `.frame_cache` and `.feature_cache`.
- Rebuilt the first 20 train records with current code and compared them against `data_utils/msad_saver_sft_train.jsonl`; 225 of the first 280 prepared examples differ in `messages`, proving prepared SFT drift.
- Implemented prepared-data metadata sidecars plus training/tensor-cache/teacher-judge validation so stale prepared JSONL files can no longer be consumed silently.
- Updated `00_full_pipeline.sh` so Stage 1 rebuilds when prepared metadata sidecars are missing and Stage 2 cache summaries no longer overwrite train/test runs.
