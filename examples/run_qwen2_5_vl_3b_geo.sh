set -x

export HOME=/mnt/2050data/wentao.zhang/MultiModalMath2
export VLLM_ATTENTION_BACKEND=XFORMERS

model_name=$HOME/hub/Qwen2.5-VL-3B-Instruct
n_gpus_per_node=8
tensor_model_parallel_size=2
project_name='verl'
experiment_name='verl_Qwen2.5-VL-3B-Instruct_GRPO'

mm_train_path=$HOME/data/processed/geometry3k/train.parquet
mm_test_path=$HOME/data/processed/geometry3k/test.parquet
gsm8k_train_path=$HOME/data/processed/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/processed/gsm8k/test.parquet

train_files="['$mm_train_path', '$gsm8k_train_path']"
test_files="['$mm_test_path', '$gsm8k_test_path']"

python3 -m verl.trainer.main \
    config=examples/grpo_example.yaml \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    worker.actor.model.model_path=${model_name} \
    worker.rollout.tensor_parallel_size=$tensor_model_parallel_size \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name= $experiment_name \
    trainer.project_name=$project_name \
    trainer.n_gpus_per_node=$n_gpus_per_node
