from typing import Optional, Tuple, Union

import bitsandbytes as bnb
import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """

    log_ratio = log_probs - log_probs_base
    return log_ratio * action_mask


def compute_reward(
    r: Union[torch.Tensor, float],
    kl_coef: float,
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    # 打印原始奖励和KL系数
    print(f"[CAT]Original Reward: {r}, KL Coefficient: {kl_coef}")

    # 确保KL系数非负
    if kl_coef <= 0.0:
        kl_coef = 0.0

    # 计算近似KL散度
    kl = compute_approx_kl(log_probs, log_probs_base, action_mask=action_mask)
    # 计算KL散度奖励（惩罚）
    print(f"[CAT]KL Divergence shape: {kl.shape}")
    kl_reward = -kl_coef * kl

    # 将原始奖励限制在 [-10, 10] 的范围内
    r = r.clamp(min=-10, max=10)
    # r = r.clamp(min=0.1, max=1)

    # 计算序列中最后一个有效动作的奖励
    eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
    last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).to(kl.dtype))

    # 计算总奖励
    reward = last_reward + kl_reward
    print(f'[CAT]Last Reward shape: {last_reward.shape},KL reward:{kl_reward.shape}, Total Reward: {reward.shape}')
    print(f"[CAT]Last Reward[0]:{last_reward[0]}, KL reward[0]:{kl_reward[0]}, Total Reward[0]:{reward[0]}")
    # 打印KL散度和最终奖励
    # print(f"[CAT]KL Divergence: {kl}, Final Reward: {reward}")

    return reward, kl



def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = None) -> torch.Tensor:
    if dim is not None:
        return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)
    else:
        return (tensor * mask).sum() / mask.sum()


def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    tensor = tensor * mask
    mean = masked_mean(tensor, mask, dim=dim)
    mean_centered = tensor - mean
    var = masked_mean(mean_centered**2, mask, dim=dim)
    return mean_centered * var.clamp(min=eps).rsqrt()


def find_all_linear_names(model, load_in_4bit=False):
    cls = bnb.nn.Linear4bit if load_in_4bit else nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        # deepseed.zero.init hooks torch.arange to run it on the GPU
        hooked_arange = torch.arange
        torch.arange = deepspeed.runtime.zero.partition_parameters._orig_torch_arange

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

        self.cos_cached = self.cos_cached.to("cuda")
        self.sin_cached = self.sin_cached.to("cuda")
        torch.arange = hooked_arange

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Patch for LLaMA RoPE embedding
# https://github.com/microsoft/DeepSpeed/issues/4932
def replace_rope_embedding():
    from transformers.models.llama import modeling_llama

    modeling_llama.LlamaRotaryEmbedding = LlamaRotaryEmbedding
