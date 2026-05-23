# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import argparse
import multiprocessing as mp
import os
import time
import warnings
from functools import partial

warnings.filterwarnings("ignore")

import datasets
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, LlamaTokenizerFast, TextStreamer

from moe_infinity import MoE
from moe_infinity.models.modeling_arctic import ArcticTokenizer


class StopWatch(TextStreamer):
    def __init__(self, engine, max_new_tokens=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_prefilling = None
        self.prefilling_time = None
        self.start_decoding = None
        self.decoding_time = None
        self.decoding_iterations = 0
        self.engine = engine
        self.max_new_tokens = max_new_tokens
        self.decode_pbar = None

    @staticmethod
    def _num_tokens(value):
        if torch.is_tensor(value):
            return int(value.numel())
        try:
            return len(value)
        except TypeError:
            return 1

    def _ensure_decode_pbar(self):
        if self.decode_pbar is None:
            self.decode_pbar = tqdm(
                total=self.max_new_tokens,
                desc="Decoding",
                unit="tok",
                leave=False,
            )

    def put(self, value):
        if self.start_prefilling is None:
            self.start_prefilling = time.time()
            return
        elif self.prefilling_time is None:
            self.prefilling_time = time.time() - self.start_prefilling
            self.engine.expert_dispatcher.clear_expert_cache_counts()
            self.start_decoding = time.time()
            #self._ensure_decode_pbar()

        num_tokens = self._num_tokens(value)
        self.decoding_iterations += num_tokens
        if self.decode_pbar is not None:
            self.decode_pbar.update(num_tokens)

        if self.decoding_iterations % 100 == 0:
            current_time = time.time()
            print(f"Prefilling time: {self.prefilling_time} seconds")
            print(f"Decoding time: {self.decoding_time} seconds")
            print(f"Decoding iterations: {self.decoding_iterations}")
            print(
                f"Decoding time per iteration: {(current_time-self.start_decoding) / self.decoding_iterations} seconds"
            )

        return super().put(value)

    def end(self):
        if self.decoding_time is None and self.start_decoding is not None:
            self.decoding_time = time.time() - self.start_decoding

        if self.decode_pbar is not None:
            self.decode_pbar.close()
            self.decode_pbar = None

        return super().end()


parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, required=True)
parser.add_argument("--offload_dir", type=str, required=True)
parser.add_argument("--device_memory_ratio", type=float, default=0.9)
parser.add_argument("--out_len", type=int, default=40)
parser.add_argument("--num_samples", type=int, default=0)
args = parser.parse_args()

model_name = args.model_name_or_path.split("/")[-1]
config = {
    "offload_path": os.path.join(args.offload_dir, model_name),
    "device_memory_ratio": args.device_memory_ratio,
}
model = MoE(args.model_name_or_path, config)

tokenizer = None
if "grok" in model_name:
    tokenizer = LlamaTokenizerFast.from_pretrained(
        "Xenova/grok-1-tokenizer", trust_remote_code=True
    )
elif "arctic" in args.model_name_or_path.lower():
    tokenizer = ArcticTokenizer.from_pretrained(args.model_name_or_path)
else:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, use_fast=False
    )


dataset = datasets.load_dataset("openai/gsm8k", "main", split="test")
all_inputs = dataset["question"]

# dataset_name = "openai/gsm8k"
# names = datasets.get_dataset_config_names(dataset_name)

# pool = mp.Pool(mp.cpu_count())
# all_inputs = [None] * len(names)
# all_inputs = pool.map(partial(datasets.load_dataset, dataset_name), names)

# print(all_inputs)

# text_list = []
# for dataset in all_inputs:
#     if "test" not in dataset:
#         continue
#     for i, text in enumerate(dataset["test"]["question"]):
#         text_list.append(text)

# print(len(text_list))
# all_inputs = text_list

custom_kwargs = {}
if "switch" in args.model_name_or_path.lower():
    custom_kwargs = {"decoder_start_token_id": 0}
elif "nllb" in args.model_name_or_path.lower():
    custom_kwargs = {"forced_bos_token_id": 256057}  # translate to French
elif "mixtral" in args.model_name_or_path.lower():
    custom_kwargs = {"pad_token_id": tokenizer.eos_token_id}
elif "grok" in args.model_name_or_path.lower():
    custom_kwargs = {}
elif "arctic" in args.model_name_or_path.lower():
    custom_kwargs = {"pad_token_id": tokenizer.eos_token_id}
elif (
    "deepseek" in args.model_name_or_path.lower()
    or "qwen3" in args.model_name_or_path.lower()
):
    custom_kwargs = {"pad_token_id": tokenizer.eos_token_id}
else:
    raise ValueError(f"Model {args.model_name_or_path} not supported")

tokenizer.pad_token = tokenizer.eos_token
cnt = 0
max_seq_length = 128
kernel_max_tokens = 128
iter_inputs = all_inputs if args.num_samples <= 0 else all_inputs[: args.num_samples]
for sample_idx, input_text in enumerate(
    tqdm(iter_inputs, desc="Dataset", unit="sample"), start=1
):
    chat_template_kwargs = dict(
        conversation=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": input_text,
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    if "qwen3" in args.model_name_or_path.lower():
        chat_template_kwargs["enable_thinking"] = False

    prompt = tokenizer.apply_chat_template(**chat_template_kwargs)
    print(f"prompt: {prompt}")
    print(f"[sample {sample_idx}] tokenizing input")

    model_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length,
    )
    token_ids = model_inputs["input_ids"].to("cuda:0")
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to("cuda:0")

    input_len = int(token_ids.shape[1])
    available_new_tokens = kernel_max_tokens - input_len
    if available_new_tokens <= 0:
        print(
            f"[sample {sample_idx}] skip: input_len={input_len} reaches kernel limit {kernel_max_tokens}"
        )
        continue
    sample_out_len = min(args.out_len, available_new_tokens)
    if sample_out_len < args.out_len:
        print(
            f"[sample {sample_idx}] clip out_len from {args.out_len} to {sample_out_len} "
            f"to keep input+output <= {kernel_max_tokens}"
        )

    streamer = StopWatch(
        model.engine,
        sample_out_len,
        tokenizer,
        skip_prompt=True,
    )
    with torch.no_grad():
        print(f"[sample {sample_idx}] outputs_text ...")
        #autoregressive
        generate_kwargs = dict(
            input_ids=token_ids,
            streamer=streamer,
            max_new_tokens=sample_out_len,
            min_new_tokens=sample_out_len,
            do_sample=False,
            use_cache=True,
            **custom_kwargs,
        )
        #speculative decoding
        '''generate_kwargs = dict(
            input_ids=token_ids,
            streamer=streamer,
            max_new_tokens=sample_out_len,
            min_new_tokens=sample_out_len,
            gamma=2,
            draft_num_experts=4,
            target_num_experts=8,
            do_sample=False,
            use_cache=True,
            debug=False,
            custom_generate="../",
            **custom_kwargs,
        )'''
        if attention_mask is not None:
            generate_kwargs["attention_mask"] = attention_mask
        outputs = model.generate(**generate_kwargs)

        print(f"Prefilling time: {streamer.prefilling_time} seconds")
        print(f"Decoding time: {streamer.decoding_time} seconds")
        print(f"Decoding iterations: {streamer.decoding_iterations}")
        if streamer.decoding_iterations > 0 and streamer.decoding_time is not None:
            print(
                f"Decoding time per iteration: {streamer.decoding_time / streamer.decoding_iterations} seconds"
            )
        else:
            print("Decoding time per iteration: N/A")
        # print(f"Input tokens: {len(inputs.input_ids[0])}")
