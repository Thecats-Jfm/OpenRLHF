import logging
import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import ray
import torch
import torch.nn as nn
from tqdm import tqdm

from openrlhf.models.actor import Actor
from openrlhf.models.utils import compute_reward, masked_mean
from openrlhf.utils.logging import init_logger

logger = init_logger(__name__)


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    values: (B,A)
    returns: (B,A)
    advatanges: (B,A)
    attention_mask: (B, S)
    action_mask: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        self.sequences = self.sequences.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.values = self.values.to(device)
        self.returns = self.returns.to(device)
        self.advantages = self.advantages.to(device)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device)
        if self.action_mask is not None:
            self.action_mask = self.action_mask.to(device)

    def pin_memory(self):
        self.sequences = self.sequences.pin_memory()
        self.action_log_probs = self.action_log_probs.pin_memory()
        self.values = self.values.pin_memory()
        self.returns = self.returns.pin_memory()
        self.advantages = self.advantages.pin_memory()
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.pin_memory()
        if self.action_mask is not None:
            self.action_mask = self.action_mask.pin_memory()
        return self


class NaiveExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn

    # tokenizer
    def tokenize_fn(self, texts, max_length, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], **generate_kwargs) -> Experience:
        self.actor.eval()
        self.critic.eval()
        self.initial_model.eval()
        self.reward_model.eval()

        # generate seq
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
        num_actions = action_mask.size(1)

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask)

        # init log probs
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)

        # values
        value = self.critic(sequences, action_mask, attention_mask)

        # rewards
        r = self.reward_model(sequences, attention_mask)

        reward, kl = compute_reward(
            r,
            self.kl_ctl.value,
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
        )
        advantage, returns = self.get_advantages_and_returns(
            value,
            reward,
            action_mask,
            generate_kwargs["gamma"],
            generate_kwargs["lambd"],
        )

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "return": reward.sum(dim=-1),
            "response_length": action_mask.float().sum(dim=-1),
            "total_length": attention_mask.float().sum(dim=-1),
        }
        # reset model state
        self.actor.train()
        self.critic.train()

        return Experience(
            sequences,
            action_log_probs,
            value,
            returns,
            advantage,
            attention_mask,
            action_mask,
            info,
        )
    @torch.no_grad()  # 停用梯度计算，加速处理并节省内存
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,  # 价值函数的输出，表示每个状态的价值估计
        rewards: torch.Tensor,  # 从环境中获得的奖励
        action_mask: torch.Tensor,  # 一个掩码张量，指示每个时间步的动作是否有效
        gamma: float,  # 折扣因子，用于计算未来奖励的现值
        lambd: float,  # GAE（Generalized Advantage Estimation）的λ参数，用于平衡偏差和方差
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算优势和回报。
        优势函数和回报的计算基于原始的PPO论文。
        注意，奖励可以包含KL散度损失项。
        """
        print(f"[CAT] shape: values:{values.shape},rewards:{rewards.shape},action_mask:{action_mask.shape}")
        print(f"[CAT] GETADVANTAGES: gamma: {gamma}, lambd: {lambd}")
        lastgaelam = 0  # 初始化最后一个时间步的GAE估计为0
        advantages_reversed = []  # 用于存储逆序优势估计的列表
        response_length = rewards.size(1)  # 获取序列长度

        # 使用动作掩码来过滤无效的响应，确保只考虑有效动作的值和奖励
        values = action_mask * values
        rewards = action_mask * rewards

        # 逆序遍历每个时间步，以便于从后向前计算GAE
        for t in reversed(range(response_length)):
            # 如果不是最后一个时间步，则使用下一个时间步的价值；否则假设为0
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            # 计算delta，即TD残差：当前奖励加上折现的下一时刻价值，减去当前价值
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            # 更新GAE估计，加上折现的λ调整后的上一个GAE估计
            lastgaelam = delta + gamma * lambd * lastgaelam
            # 将计算的GAE估计添加到列表中
            advantages_reversed.append(lastgaelam)
        # 将逆序优势列表翻转，以恢复原始顺序，并将列表转换为张量
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        # 计算回报，即优势加上价值
        returns = advantages + values
        # 返回优势和回报，其中优势张量从计算图中分离，以避免影响反向传播
        print(f"[CAT] advantages:{advantages.shape}, returns:{returns.shape}")
        return advantages.detach(), returns



class RemoteExperienceMaker(NaiveExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines

    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], **generate_kwargs) -> Experience:
        self.actor.eval()
        device = torch.cuda.current_device()

        # generate sequence
        start = time.time()
        sequences, attention_mask, action_mask = (
            self._generate_local(prompts, **generate_kwargs)
            if self.vllm_engines is None
            else self._generate_vllm(prompts, **generate_kwargs)
        )
        generate_time = time.time() - start

        num_actions = action_mask.size(1)
        sequences_cpu, attention_mask_cpu, action_mask_cpu = (
            sequences.to("cpu"),
            attention_mask.to("cpu"),
            action_mask.to("cpu"),
        )

        # init log probs
        base_action_log_probs_ref = self.initial_model.forward.remote(sequences_cpu, num_actions, attention_mask_cpu)

        # values
        value_ref = self.critic.forward.remote(sequences_cpu, action_mask_cpu, attention_mask_cpu)

        # rewards
        r_refs = []
        for rm in self.reward_model:
            print("[CAT] shape: sequences:{}, attention_mask:{}".format(sequences_cpu.shape, attention_mask_cpu.shape))
            print(self.tokenizer.decode(sequences_cpu[0]))
            sequences_cpu_path = "run/sequences_cpu.pt"
            torch.save(sequences_cpu, sequences_cpu_path)
            attention_mask_cpu_path = "run/attention_mask_cpu.pt"
            torch.save(attention_mask_cpu,attention_mask_cpu_path)


            tp  = ([    1,  2961,   368,   622,  1156,   272,  3905,   302,  5740,   412,
                    28924, 15423,   460,   741,   302,   516,  1871, 28994,   952, 28747,
                    5740,   412,    13, 28777,  2341, 28747, 18375,   883,    13, 28741,
                    490, 28747, 28705, 28740,    13, 28757, 10565,   441, 17338, 28747,
                    23325,    13, 28765,  9664,   304,  4788, 28747,  5740,   412,   541,
                    28742, 28707,  4085,  2783,   562,   659,   264,  3534,  9410,   395,
                    559,   281, 13774, 28723,    13,  1014,  2296,   460,   272,   865,
                    345, 15342, 28739,   368,   927,   298,  1372,   297,   456,  5313,
                    28747,    13,  5390,   995, 19685,   369,   368,   460,   264,  2229,
                    442,  2007,   304,  1792,  1723,  2663,   693,  2204,   368,   390,
                    8252,   442,  7006,  7839, 28723, 11159, 11194,  1023,   459,  2622,
                    737,  5599,  3842,   562,  3210, 17574,   304,  1933, 28723, 28705,
                    13,  5390,  1791,  1038,   574, 11194,   680,  4229,   304, 10526,
                    28725, 18756,  3479,  7559,   460,  4670, 28723,    13,  5390,   995,
                    1580,  1290,  6212,   272, 10294,  4771,   298,   456,  6073,  1097,
                    28742, 28713,  2809,  3425,   304,  8208,    13,  5390,   995,   460,
                    484, 14758,   395,   528,  5136,  1999,   304,   511,   459,   927,
                    298,  1917,   272,  5088,   356,  2663, 28723,    13,  5390,   995,
                    1580, 16565,   298,   272,  3238,   302,   272,  3905,   368,   460,
                    4543, 28725,   511,   459,   347,  1368,  1997,   982, 28924, 11112,
                    28742, 28707,  1460,   528,   513,   315,   927,  1316,   442,   541,
                    368,  6031,   528, 28723,    13,  1976,  1023,  1038,  5278, 28725,
                    7087, 28725,   442, 14214, 17949, 14915,  2818,   356,   272, 13855,
                    7138, 28725,   459,  1743,  5278, 11194, 28723,    13,  5390,  6746,
                    1864,   574, 11194,   460,   459, 15235,   848,   695, 28725,   304,
                    511,   459, 13750,  3454, 11194, 28723,    13,  5390,   995,  3573,
                    5090,  4389,   272,   345, 15342, 28739,   297,   574,  4802,   497,
                    28723,    13,  5390,  2957,   460,   272,   345, 15342, 28739,   354,
                    456,  7114,  8602,  8048, 12738, 28747,   398,  8459,   412,  4674,
                    438,   368,   304, 24733, 28736,     2,   382,  4171,  1251, 28747,
                    398, 28737,  1388,   805,   586,  8155, 28736,  8602,  8048, 12738,
                    28747,   398,  8459,   412,   480,  1593,   559,  1335,   304, 16217,
                    4911,  1753, 28725,   559, 13790, 16984, 15990,   272,  2003, 28723,
                    985,  2368, 28742, 28707,   737,  8192,  8155,   304,  1743, 14744,
                    298,  4461,   378,  3752,     2,   382,  4171,  1251, 28747,   398,
                    28737,  1388,   805,   559,  8155, 28736,  8602,  8048, 12738, 28747,
                    398,  8459,   412,   480,  1593,   559,  1335,   304, 16217,  4911,
                    1753, 28725,   559, 13790, 16984, 15990,   272,  2003, 28723,   985,
                    2368, 28742, 28707,   737,  8192,  8155,   304,  1743, 14744,   298,
                    4461,   378,  3752,     2,   382,  4171,  1251, 28747,   683,   412,
                    1567,  1236,  8602,  8048, 12738, 28747,   398,  8459,   412, 16217,
                    4911,   852,   298,   368, 28725,   559,  2282, 14123,  1905,   395,
                    290,  3143,  3417,  3752,   345, 28737, 28742, 28719,  1236, 28808,
                    1824,   511,   368,   947,   298,   511,  1110,   398,  6623, 12373,
                    1156,  3071,  3752,     2,   382,  4171,  1251, 28747, 24057,   506,
                    3142,  8602,  8048, 12738, 28747, 28705,   398,  8459,   412,  4674,
                    13803,   304, 23589, 28736,   345, 28749,  1566,   708,   281, 13774,
                    28808,   816,  8869, 28742, 28707,   511,   369,  2781,   398,  4853,
                    12784,  4746,   395,   559,  1628,  3038, 28736,   345,  8779, 28742,
                    28713,  1156,  1545,   746, 28725,  4357,  9525,  2570, 28808,   315,
                    24057,   347,   264, 28313,  2781,     2])

            print(self.tokenizer.decode(tp))
            print('-#'*100)










            log_file_path = "./test2.txt"
            with open(log_file_path, "a") as log_file:  # 使用 "a" 模式以追加的方式写入
                # 写入信息到文件
                log_file.write(f'[CAT] sequences_cpu shape:{sequences_cpu.shape}\n')
                log_file.write(f'[CAT] attention mask shape:{attention_mask_cpu.shape}\n')
                log_file.write(f"[CAT] sequences_cpu:{sequences_cpu[0]}\n")
                log_file.write(f"[CAT] attention_mask:{attention_mask_cpu[0]}\n")
            print(f"sequences_cpu saved to {sequences_cpu_path}")
            time.sleep(10)
            r_refs.append(rm.forward.remote(sequences_cpu, attention_mask_cpu))

        # log probs
        start = time.time()
        action_log_probs = self.actor(sequences, num_actions, attention_mask)
        actor_time = time.time() - start

        # wait initial/critic/reward model done
        start = time.time()
        ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
        wait_time = time.time() - start

        base_action_log_probs, value, rewards = ref_values[0], ref_values[1], ref_values[2:]
        base_action_log_probs, value = base_action_log_probs.to(device), value.to(device)
        rewards = [r.to(device) for r in rewards]
        r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]

        reward, kl = compute_reward(
            r,
            self.kl_ctl.value,
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
        )
        advantage, returns = self.get_advantages_and_returns(
            value,
            reward,
            action_mask,
            generate_kwargs["gamma"],
            generate_kwargs["lambd"],
        )

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "return": reward.sum(dim=-1),
            "response_length": action_mask.float().sum(dim=-1),
            "total_length": attention_mask.float().sum(dim=-1),
        }

        if self.strategy.args.perf:
            batch_size = 1 if isinstance(prompts, str) else len(prompts)
            info["generate_time"] = torch.full((batch_size,), generate_time, device=device)
            info["actor_time"] = torch.full((batch_size,), actor_time, device=device)
            info["wait_time"] = torch.full((batch_size,), wait_time, device=device)

        experience = Experience(
            sequences,
            action_log_probs,
            value,
            returns,
            advantage,
            attention_mask,
            action_mask,
            info,
        )

        # send experience to critic
        experience_cpu = deepcopy(experience)
        experience_cpu.to_device("cpu")
        self._ref = self.critic.append.remote(experience_cpu)

        self.actor.train()  # reset model state
        return experience

    def _generate_local(self, prompts: List[str], **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        return self.actor.generate(**inputs, **kwargs)

    def _generate_vllm(self, prompts: List[str], **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from vllm import SamplingParams

        # round-robin load balance
        rank = torch.distributed.get_rank()
        llm = self.vllm_engines[rank % len(self.vllm_engines)]

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 16),
        )

        # TODO: can't pass `max_length` to vLLM's tokenizer for input truncation, remove this once it is supported.
        input_ids = self.tokenize_fn(prompts, self.prompt_max_len, device="cpu")["input_ids"]
        assert self.tokenizer.padding_side == "left", f"tokenizer padding_size should be left"
        pad_indices = (input_ids != self.tokenizer.pad_token_id).to(dtype=torch.int).argmax(dim=-1)
        prompt_token_ids = []
        for i, pad_index in enumerate(pad_indices.numpy()):
            prompt_token_ids.append(input_ids[i][pad_index:].tolist())
        outputs = ray.get(llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids))

        # NOTE: concat all outputs to following format:
        #
        # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
        # | token token token token token | token token [EOS] [PAD] |
        # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
        # |<---------- prompt ----------->|<-------- answer ------->|
        max_input_len, max_output_len = 0, 0
        for output in outputs:
            # TODO: how to force vLLM generate at least one token?
            output_token_ids = output.outputs[0].token_ids
            if output_token_ids[0] == self.tokenizer.eos_token_id:
                logger.warning(f"Only EOS output for prompt: {output.prompt}")
                output.outputs[0].token_ids = [self.tokenizer.unk_token_id, self.tokenizer.eos_token_id]

            max_input_len = max(max_input_len, len(output.prompt_token_ids))
            max_output_len = max(max_output_len, len(output_token_ids))

        pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
        sequences = []
        for output in outputs:
            # left padding input
            input_len = len(output.prompt_token_ids)
            input_ids = [pad_token_id] * (max_input_len - input_len) + output.prompt_token_ids

            # right padding output
            output_len = len(output.outputs[0].token_ids)
            output_ids = output.outputs[0].token_ids + [pad_token_id] * (max_output_len - output_len)
            if output_ids[output_len - 1] != eos_token_id:
                assert output_len == max_output_len
                output_ids[-1] = eos_token_id

            # concat input and output
            sequences.append(input_ids + output_ids)

        sequences = torch.tensor(sequences)
        sequences, attention_mask, action_mask = self.actor.process_sequences(
            sequences, max_input_len, eos_token_id, pad_token_id
        )
        return sequences.to("cuda"), attention_mask.to("cuda"), action_mask.to("cuda")

    def flush(self):
        "Ensure all experience has been send to critic"
        ray.get(self._ref)
        self._ref = None
