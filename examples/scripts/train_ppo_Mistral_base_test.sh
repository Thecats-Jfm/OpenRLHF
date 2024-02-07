set -x

rundir="./run/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p $rundir
log_file="$rundir/log.log"

read -r -d '' training_commands <<EOF
examples/train_ppo.py \
    --pretrain /paratera5-data/home/zhangdehao/models/OpenHermes-2.5-Mistral-7B \
    --reward_pretrain /paratera5-data/home/zhangdehao/models/Llama-2-7b-rm-anthropic_hh-lmsys-oasst-webgpt \
    --save_path $rundir \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --micro_train_batch_size 2 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 4 \
    --rollout_batch_size 1024 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data Open-Orca/OpenOrca,Dahoas/full-hh-rlhf,tasksource/oasst1_pairwise_rlhf_reward \
    --prompt_data_probs 0.4,0.5,0.1 \
    --max_samples 8000 \
    --normalize_reward \
    --actor_init_on_gpu \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing \
    --use_wandb *
EOF

# 在这里设置你的环境变量
# 例如: export MY_VARIABLE=value

# 创建日志文件夹和日志文件

if [[ ${1} != "slurm" ]]; then
    deepspeed --master_port 29506 $training_commands 2>&1 | tee $log_file
fi
