!/bin/bash
顺序执行数据处理流程脚本

步骤2.1: canonical -> agent_train
echo "开始执行步骤2.1: canonical -> agent_train"
python convert_to_saver_agent.py \
  --input ../benchmark_annotations/msad_saver_with_qwen.jsonl \
  --output data_utils/msad_saver_agent_all.jsonl \
  --mode agent_train \
  --adapter msad_saver_qwen \
  --include-splits "train, test"

检查上一步是否成功
if [ ? -ne 0 ]; then
  echo "步骤2.1执行失败，终止流程"
  exit 1
fi

步骤2.2: canonical -> oracle_sft_train
echo "开始执行步骤2.2: canonical -> oracle_sft_train"
python convert_to_saver_agent.py \
  --input ../benchmark_annotations/msad_saver_with_qwen.jsonl \
  --output data_utils/msad_saver_oracle_sft_train.jsonl \
  --mode oracle_sft \
  --adapter msad_saver_qwen \
  --include-splits "train"

检查上一步是否成功
if [ ? -ne 0 ]; then
  echo "步骤2.2执行失败，终止流程"
  exit 1
fi

步骤2.2: canonical -> oracle_sft_test
echo "开始执行步骤2.2: canonical -> oracle_sft_test"
python convert_to_saver_agent.py \
  --input ../benchmark_annotations/msad_saver_with_qwen.jsonl \
  --output data_utils/msad_saver_oracle_sft_test.jsonl \
  --mode oracle_sft \
  --adapter msad_saver_qwen \
  --include-splits "test"

检查上一步是否成功
if [ ? -ne 0 ]; then
  echo "步骤2.2执行失败，终止流程"
  exit 1
fi

步骤2.3: agent_train -> prepared_sft
echo "开始执行步骤2.3: agent_train -> prepared_sft"
python train_saver_sft.py \
  --data data_utils/msad_saver_agent_all.jsonl \
  --data-root /mnt/shared-storage-user/mineru2-shared/zengweijun \
  --include-splits "train" \
  --prepare-output data_utils/msad_saver_agent_train.prepared_sft.jsonl \
  --prepare-only \
  --validate-prepared-data \
  --progress-every 25

检查上一步是否成功
if [ ? -ne 0 ]; then
  echo "步骤2.3执行失败，终止流程"
  exit 1
fi

步骤2.4: prepared_sft -> teacher_judge prepared_sft
echo "开始执行步骤2.4: prepared_sft -> teacher_judge prepared_sft"
torchrun --standalone --nproc_per_node=4 annotate_teacher_judge_sft.py \
  --input data_utils/msad_saver_agent_train.prepared_sft.jsonl \
  --output data_utils/msad_saver_agent_train.prepared_sft.teacher.jsonl \
  --include-splits "train" \
  --model-path /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/Qwen3-VL-32B-Instruct \
  --input-mode auto \
  --torch-dtype auto \
  --device-map auto \
  --attn-implementation flash_attention_3 \
  --max-new-tokens 384 \
  --max-images 8 \
  --topk-frames-per-view 4 \
  --batch-size 2 \
  --progress-every 25

检查上一步是否成功
if [ ? -ne 0 ]; then
  echo "步骤2.4执行失败，终止流程"
  exit 1
fi

步骤2.5: teacher prepared_sft -> tensor cache
echo "开始执行步骤2.5: teacher prepared_sft -> tensor cache"
for i in $(seq 0 5); do
  echo "处理分片 i/3"
  python prepare_sft_tensor_cache.py \
    --prepared-data data_utils/msad_saver_agent_train.prepared_sft.teacher.jsonl \
    --model-path /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/qwen3-vl-8b-Instruct \
    --include-splits train \
    --max-seq-length 4096 \
    --keep-recent-text-messages 12 \
    --max-image-side 0 \
    --max-image-pixels 0 \
    --keep-recent-tool-image-messages 0 \
    --max-total-images 0 \
    --frame-cache-max-cached-videos 128 \
    --num-shards 6 \
    --shard-index i \
    --progress-every 50
done

检查上一步是否成功
if [ $? -ne 0 ]; then
  echo "步骤2.5执行失败，终止流程"
  exit 1
fi

echo "所有步骤执行完成！"