from typing import Optional

import deepspeed
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_config, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoConfig, AutoModel, BitsAndBytesConfig
from transformers.deepspeed import HfDeepSpeedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
import time
from openrlhf.utils.logging import init_logger

from .utils import find_all_linear_names, log_probs_from_logits, replace_rope_embedding

# https://github.com/microsoft/DeepSpeed/issues/4932
replace_rope_embedding()

logger = init_logger(__name__)


# Construct transformer with a value head for sequence classification.
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L1310
def get_llm_for_sequence_regression(
    model_name_or_path: str,
    model_type: str,
    *,
    bf16=True,
    load_in_4bit=False,
    lora_rank=0,
    lora_alpha=16,
    target_modules=None,
    normalize_reward=False,
    use_flash_attention_2=False,
    ds_config: dict = None,
    init_score_head: bool = False,
    **kwargs,
) -> nn.Module:
    """Get transformer with a sequence classification head on top (linear layer).

    Args:
        model_name_or_path (str): Path to pretrained model.
        model_type (str): Either "reward" or "critic.
        bf16 (bool, optional): Whether enable bfloat16. Defaults to True.
        normalize_reward (bool, optional): Whether normalize reward. Defaults to False.
        use_flash_attention_2 (bool, optional): Whether use Flash Attention 2.0. Defaults to False.
        ds_config (dict, optional): Deepspeed config, used to automatically splitting the model onto
            multiple gpus during from_pretrained when ZeRO-3 enabled. Defaults to None.

    Returns:
        nn.Module: pretrained transformer model.
    """
    assert (
        model_type == "critic" or model_type == "reward"
    ), f"invalid model_type: {model_type}, should be critic or reward."

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    print(f"[CAT]: origin {config}")
    print(f"[CAT]: model_name_or_path {model_name_or_path}")
    config.normalize_reward = normalize_reward
    config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

    try:
        base_class = AutoModel._model_mapping[type(config)]
        base_pretrained_class = base_class.__base__
        if model_type == "reward":
            cls_class = _get_reward_model(base_pretrained_class, base_class)
        else:
            cls_class = _get_critic_model(base_pretrained_class, base_class)
    except Exception as e:
        print("Failed to load from AutoModel, construct from modelling file.")
        module_file, causal_model_name = config.auto_map["AutoModelForCausalLM"].split(".")

        # special case
        if causal_model_name == "QWenLMHeadModel":
            auto_model_name = "QWenModel"
            pretrained_model_name = "QWenPreTrainedModel"
        elif causal_model_name == "InternLMForCausalLM":
            auto_model_name = "InternLMModel"
            pretrained_model_name = "InternLMPreTrainedModel"
        else:
            if "AutoModel" not in config.auto_map:
                auto_model_name = causal_model_name.split("For")[0] + "Model"
            else:
                auto_model_name = config.auto_map["AutoModel"].split(".")[1]
            pretrained_model_name = causal_model_name.split("For")[0] + "PreTrainedModel"

        logger.info(f"BASE_MODEL_CLASS: {auto_model_name}, PRETRAINED_MODEL_CLASS: {pretrained_model_name}")

        base_pretrained_class = get_class_from_dynamic_module(
            f"{module_file}.{pretrained_model_name}", model_name_or_path
        )
        base_class = get_class_from_dynamic_module(f"{module_file}.{auto_model_name}", model_name_or_path)
        if model_type == "reward":
            cls_class = _get_reward_model(base_pretrained_class, base_class)
        else:
            cls_class = _get_critic_model(base_pretrained_class, base_class)

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    if load_in_4bit:
        assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        nf4_config = None
    print(f'[CAT]:{cls_class}')
    print(f'[CAT]:{config}')

    model = cls_class.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype="auto",
        quantization_config=nf4_config,
        **kwargs,
    )

    # LoRA
    if lora_rank > 0:
        model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules or find_all_linear_names(model, load_in_4bit),
            lora_dropout=0,
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        if load_in_4bit:
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    module = module.to(torch.bfloat16)
                if "norm" in name:
                    module = module.to(torch.float32)
                if "score_head" in name or "embed_tokens" in name:
                    if hasattr(module, "weight"):
                        module = module.to(torch.bfloat16)

    # Mixtral 8x7b - balancing loss
    if "output_router_logits" in model.config.to_dict():
        print("[Mixtral 8x7b] set output_router_logits as True")
        model.config.output_router_logits = True
        deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])

    # NOTE: For reward model training only, intialize score_head manually
    # because deepspeed.zero.Init() will not intialize them.
    # TODO: Find a better way to clarify reward model training.
    if init_score_head:
        if dschf is not None:
            logger.info("initialize score_head for ZeRO-3 reward model training.")
            with deepspeed.zero.GatheredParameters([model.score_head.weight], modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    model.score_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
        else:
            model.score_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))

    return model


def _get_reward_model(base_pretrained_model, base_llm_model):
    class LLMForSequenceRegression(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.score_head = nn.Linear(config.hidden_size, 1, bias=False)

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                # Check if mean is a list and take the first element, otherwise just use the value
                self.mean[0] = config.mean[0] if isinstance(config.mean, list) else config.mean
                self.std[0] = config.std[0] if isinstance(config.std, list) else config.std


        @classmethod
        def _autoset_attn_implementation(cls, config, *args, **kwargs):
            logger.info(
                "Monkey patch for Flash Attention, see https://github.com/huggingface/transformers/issues/28052"
            )
            return config

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
        ) -> torch.Tensor:
            log_file_path = "./test.txt"
            with open(log_file_path, "a") as log_file:  # 使用 "a" 模式以追加的方式写入
                # 写入信息到文件
                log_file.write(f'[CAT] input_ids shape:{input_ids.shape}\n')
                log_file.write(f'[CAT] attention mask shape:{attention_mask.shape}\n')
                log_file.write(f"[CAT] input_ids:{input_ids[0]}\n")
                log_file.write(f"[CAT] attention_mask:{attention_mask[0]}\n")


            print(f'[CAT] reward model forward')
            print(f'[CAT] input_ids shape:{input_ids.shape}')
            print(f'[CAT] attention mask shape:{attention_mask.shape}')
            print(f'[CAT] input_ids:{input_ids[0]}')

            outputs = getattr(self, self.base_model_prefix)(
                input_ids,
                attention_mask=attention_mask,
            )
            last_hidden_states = outputs["last_hidden_state"]
            values = self.score_head(last_hidden_states).squeeze(-1)
            # 使用第二种方法计算end_score替代第一种方法的reward计算
            scores = self.score_head(last_hidden_states)  # size = (B, L, D) 或 (B, L) 取决于score_head的输出
            values = scores.squeeze(-1)
            # left padding in training mode
            if self.training:
                reward = values[:, -1]
            else:
                eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
                reward = values.gather(dim=1, index=eos_indices).squeeze(1)

                print(f"[CAT] eos_indices: {eos_indices}, reward[0]: {reward[0]}")
                with open(log_file_path, "a") as log_file:  # 使用 "a" 模式以追加的方式写入
                    # 写入信息到文件
                    log_file.write(f'[CAT] eos_indices: {eos_indices}, reward[0]: {reward[0]}\n')

                # normalize reward in eval mode
                if self.normalize_reward:
                    print(f'[CAT]:_get_reward_model normalize_reward{reward}')
                    with open(log_file_path, "a") as log_file:
                        log_file.write(f'[CAT]:_get_reward_model normalize_reward{reward}\n')
                    reward = (reward - self.mean) / self.std
                    print(f"[CAT]:after to{reward}")

                    input_ids_path = "run/input_ids.pt"
                    attention_mask_path = "run/attention_mask.pt"

                    # 保存Tensor到文件
                    torch.save(input_ids, input_ids_path)
                    torch.save(attention_mask, attention_mask_path)

                    # 信息提示，表明Tensor已被保存
                    print(f"input_ids saved to {input_ids_path}")
                    print(f"attention_mask saved to {attention_mask_path}")
                    time.sleep(20)
                    exit()

            if return_output:
                return reward, outputs
            else:
                return reward

    return LLMForSequenceRegression


def _get_critic_model(base_pretrained_model, base_llm_model):
    class LLMForSequenceRegression(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.score_head = nn.Linear(config.hidden_size, 1, bias=False)

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                # Check if mean is a list and take the first element, otherwise just use the value
                self.mean[0] = config.mean[0] if isinstance(config.mean, list) else config.mean
                self.std[0] = config.std[0] if isinstance(config.std, list) else config.std

        @classmethod
        def _autoset_attn_implementation(cls, config, *args, **kwargs):
            logger.info(
                "Monkey patch for Flash Attention, see https://github.com/huggingface/transformers/issues/28052"
            )
            return config

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            action_mask: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
        ) -> torch.Tensor:
            outputs = getattr(self, self.base_model_prefix)(
                input_ids,
                attention_mask=attention_mask,
            )
            last_hidden_states = outputs["last_hidden_state"]
            values = self.score_head(last_hidden_states).squeeze(-1)[:, :-1]
            num_actions = action_mask.size(1)

            # normalize reward
            if self.normalize_reward:
                values = (values - self.mean) / self.std
                print(f"[CAT]:after normalize_values{values}")

            if return_output:
                return outputs if num_actions is None else (values[:, -num_actions:], outputs)
            else:
                return values[:, -num_actions:]

    return LLMForSequenceRegression
