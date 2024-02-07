set -x
###
 # @Author: Dylancer1998 bodcoder@gmail.com
 # @Date: 2024-02-07 11:11:31
 # @LastEditors: Dylancer1998 bodcoder@gmail.com
 # @LastEditTime: 2024-02-07 11:11:32
 # @FilePath: /OpenRLHF/examples/scripts/train_ppo_llama_ray_test.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
###

rundir="./run/test_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p $rundir
log_file="$rundir/log.log"
export PATH=$HOME/.local/bin/:$PATH
# export http_proxy=*
# export https_proxy=*
export no_proxy="localhost,127.0.0.1"


ray job submit --address="http://127.0.0.1:8265" \
    -- python3 examples/train_ppo_ray.py \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 1 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 2 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --pretrain ~/models/mistral_continue_dpo_1n_20240110-1428_len_filter_0110_bm25_filtered_bot_number \
    --reward_pretrain ~/models/rm0201HA \
    --save_path $rundir \
    --micro_train_batch_size 8 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 1 \
    --rollout_batch_size 1024 \
    --max_epochs 1 \
    --prompt_max_len 3584 \
    --generate_max_len 512 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data ~/data/2024V1HA_train.json \
    --prompt_data_probs 1 \
    --max_samples 10000 \
    --normalize_reward \
    --actor_init_on_gpu \
    --adam_offload \
    --flash_attn \
    --use_wandb * \
    --gradient_checkpointing 2>&1 | tee $log_file
