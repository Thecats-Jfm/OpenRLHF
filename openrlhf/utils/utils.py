import os
import torch
import deepspeed
import contextlib

from torch import nn
from pathlib import Path

from datasets import Dataset, interleave_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer

from openrlhf.utils import DeepspeedStrategy
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled


DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def get_sp_tokens(args):
    sp_tokens = dict()
    for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
        sp_token = getattr(args, key, None)
        if sp_token is not None:
            sp_tokens[key] = sp_token
    return sp_tokens


def init_new_embeddings(
    embeddings: nn.Embedding | nn.Linear | None,
    new_num_embeddings: int,
    num_new_embeddings: int,
) -> None:
    if embeddings is None:
        return

    params = [embeddings.weight, getattr(embeddings, 'bias', None)]
    context = (
        deepspeed.zero.GatheredParameters(params, modifier_rank=0)
        if is_deepspeed_zero3_enabled()
        else contextlib.nullcontext()
    )
    with context:
        for param in params:
            if param is None:
                continue
            assert param.size(0) == new_num_embeddings
            param_data = param.data
            param_mean = param_data[:-num_new_embeddings].mean(dim=0, keepdim=True)
            param_data[-num_new_embeddings:] = param_mean



def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    # sp_tokens = get_sp_tokens(strategy.args)
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side

    print(f"Tokenizer loaded from `{pretrain}` with the following settings:")
    print(f"- Vocabulary size: {tokenizer.vocab_size}")
    print(f"- Padding side: {tokenizer.padding_side}")
    print(f"- Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"- EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"- BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"- Use fast tokenizer: {use_fast}")

    # Additional special tokens handling and logging
    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict['pad_token'] = '<pad>'
    if tokenizer.eos_token is None:
        special_tokens_dict['eos_token'] = '</s>'
    if tokenizer.bos_token is None:
        special_tokens_dict['bos_token'] = '<s>'

    if special_tokens_dict:
        print(f"Adding new special tokens: {special_tokens_dict}")

    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    new_num_embeddings = len(tokenizer)
    print(f"Number of new tokens added: {num_new_tokens}")
    print(f"New total number of embeddings: {new_num_embeddings}")

    try:
        model.config.pad_token_id = tokenizer.pad_token_id
    except AttributeError:
        model.config['pad_token_id'] = tokenizer.pad_token_id


    if num_new_tokens > 0:
        hf_device_map = getattr(model, 'hf_device_map', {})
        devices = {
            torch.device(device)
            for device in hf_device_map.values()
            if device not in {'cpu', 'disk'}
        }
        is_model_parallel = len(devices) > 1
        print(f"Model parallelism is {'enabled' if is_model_parallel else 'disabled'}.")

        if not is_model_parallel:
            model.resize_token_embeddings(new_num_embeddings)
            print("Resized model token embeddings to match the tokenizer's new vocabulary size.")

    return tokenizer


def get_strategy(args):
    # default args for deepspeed
    if "seed" not in args:
        args.seed = 42
    if "max_norm" not in args:
        args.max_norm = 1.0
    if "micro_train_batch_size" not in args:
        args.micro_train_batch_size = 1
    if "train_batch_size" not in args:
        args.train_batch_size = 8
    if "local_rank" not in args:
        args.local_rank = -1
    if "bf16" not in args:
        args.bf16 = True
    if "adam_offload" not in args:
        args.adam_offload = False
    if "zpg" not in args:
        args.zpg = 1
    if "grad_accum_dtype" not in args:
        args.grad_accum_dtype = "fp32"

    strategy = DeepspeedStrategy(
        seed=args.seed,
        max_norm=args.max_norm,
        micro_train_batch_size=args.micro_train_batch_size,
        train_batch_size=args.train_batch_size,
        zero_stage=args.zero_stage,
        args=args,
    )

    return strategy


def blending_datasets(
    datasets,
    probabilities,
    strategy=None,
    seed=42,
    max_count=2000000,
    return_eval=True,
    stopping_strategy="first_exhausted",
):

    # os.environ["http_proxy"] = "*"
    os.environ["https_proxy"] = "*"
    os.environ["no_proxy"] = "localhost,127.0.0.1"
    https = os.environ.get("https_proxy")
    http = os.environ.get("http_proxy")
    path = os.environ.get("PATH")
    print(f"PATH: {path}")
    print(f"http: {http}")
    print(f"https: {https}")
    datasets = datasets.split(",")
    probabilities = list(map(float, probabilities.split(",")))
    assert len(probabilities) == len(datasets)

    train_data_list = []
    eval_data_list = []
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        dataset_subfold_list = dataset.split("@")
        strategy.print(f"dataset - : {dataset}")
        print(f"[CAT]dataset - : {dataset}")
        # local dir with python script or common local file
        if os.path.exists(dataset):
            print(f'[CAT] load from disk')
            # data = load_from_disk(dataset)
            data = load_dataset('json', data_files=dataset)
        elif dataset.endswith(
            (".json", ".jsonl", ".csv", ".parquet", ".txt")
        ):
            if dataset.endswith((".json", ".jsonl", ".csv", ".parquet", ".txt")):
                files = dataset
                data_type = os.path.splitext(files)[1][1:]
            else:
                path = Path(dataset)
                script = [str(file.resolve()) for file in Path(path).rglob("*.py")]
                extensions = ("*.json", "*.jsonl", "*.csv", "*.parquet", "*.txt")
                files = [str(file) for ext in extensions for file in Path(path).rglob(ext)]
                strategy.print(f"script: {script}")
                strategy.print(f"files: {files}")
                # For dir, follow python script or first file type
                data_type = script[0] if len(script) == 1 else os.path.splitext(files[0])[1][1:]
            # reformat data type
            if data_type in ["json", "jsonl"]:
                data_type = "json"
            elif data_type == "txt":
                data_type = "text"
            elif data_type.endswith(".py"):
                # load local dir with python script
                files = None
            if data_type.endswith(".py"):
                strategy.print(f"load {dataset} with script {data_type}")
            else:
                strategy.print(f"load {files} from {dataset}")
            data = load_dataset(data_type, data_files=files)
        elif len(dataset_subfold_list) == 2:
            dataset = dataset_subfold_list[0]
            subfold = dataset_subfold_list[1]
            data = load_dataset(dataset, data_dir=subfold.strip())
        elif len(dataset_subfold_list) == 1:
            dataset = dataset_subfold_list[0]
            data = load_dataset(dataset)
        else:
            raise Exception(f"Dataset Name {dataset}: Format error")

        if "train" in data:
            if max_count == -1:
                train_data_list.append(data['train'])
            else:
                train_data_list.append(data["train"].select(range(min(max_count, len(data["train"])))))
        else:
            if max_count == -1:
                train_data_list.append(data)
            else:
                train_data_list.append(data.select(range(min(max_count, len(data)))))  # train will contains eval? TODO

        if return_eval:
            if "test" in data:
                if max_count == -1:
                    eval_data = data["test"]
                else:
                    eval_data = data["test"].select(range(min(int(max_count * 0.1), len(data["test"]))))
            elif "validation" in data:
                if max_count == -1:
                    eval_data = data['validation']
                else:
                    eval_data = data["validation"].select(range(min(int(max_count * 0.1), len(data["validation"]))))
            elif "train" in data:
                if max_count == -1:
                    eval_data = data["train"]
                else:
                    eval_data = data["train"].select(range(min(int(max_count * 0.1), int(len(data["train"]) * 0.01))))
            else:
                if max_count == -1:
                    eval_data = data
                else:
                    eval_data = data.select(range(min(int(max_count * 0.1), int(len(data) * 0.001))))
            eval_data_list.append(eval_data)

    # merge datasets
    if strategy.is_rank_0():
        print(train_data_list)

    train_dataset = interleave_datasets(
        train_data_list,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )
    if return_eval:
        eval_dataset = interleave_datasets(
            eval_data_list,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )
        return train_dataset, eval_dataset
    else:
        return train_dataset
