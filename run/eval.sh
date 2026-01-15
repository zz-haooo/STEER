set -x

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${STEER_ROOT}:$PYTHONPATH"

echo "Current VERL path:"
python3 -c "import verl; print(verl.__file__)"

export PYTHONHASHSEED=42
export PYTORCH_SEED=42
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

project_name='STEER_evaluation'

datasets_dir=${STEER_ROOT}/datasets

# train dataset
train_path=$datasets_dir/DAPO-Math-17k.parquet

#test datasets
aime2024_test_path=$datasets_dir/aime24.parquet
aime2025_test_path=$datasets_dir/aime25.parquet
amc23_test_path=$datasets_dir/amc23.parquet
math500_test_path=$datasets_dir/math500.parquet
olympiad_test_path=$datasets_dir/olympiadbench.parquet
minerva_test_path=$datasets_dir/minerva_math.parquet
gsm8k_test_path=$datasets_dir/gsm8k_test.parquet
omni_test_path=$datasets_dir/omni.parquet

test_files="['$aime2024_test_path', '$math500_test_path', '$olympiad_test_path', '$aime2025_test_path', '$amc23_test_path', '$minerva_test_path', '$gsm8k_test_path', '$omni_test_path']"

# Model path - update to your model location or use environment variable
model_path=${MODEL_PATH:-"zzzzzzzzzzhao/STEER"}


run_name=Multi_datasets_val



export WANDB_INIT_TIMEOUT=300
export WANDB_TIMEOUT=300
export WANDB_RETRY_DELAY=60
export WANDB_MAX_RETRIES=10


save_contents="['hf_model']"

current_datetime=$(date +"%Y%m%d_%H%M%S")
run_name="${run_name}_${current_datetime}"

train_files="['$train_path']"



echo "Start multi-dataset evaluation..."
echo "Test datasets: $test_files"
echo "Model path: $model_path"
echo "Start time: $(date)"

# Extract model name for filename
model_name=$(basename $(dirname $(dirname $(dirname $model_path))))
echo "Model name: $model_name"

# Create output directory
output_dir="${STEER_ROOT}/evaluation_results"
mkdir -p "$output_dir"
echo "Output directory: $output_dir"


# Run evaluation
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='left'  \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.actor.checkpoint.save_contents=${save_contents} \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.rollout_data_dir=${STEER_ROOT}/rollout_data/$project_name/$run_name \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name=$project_name \
    trainer.experiment_name=$run_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=1 \
    trainer.total_epochs=50 \
    trainer.val_only=True \
    trainer.resume_mode=disable
