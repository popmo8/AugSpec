import argparse
import os
import sys

from transformers import AutoTokenizer

from moe_infinity import MoE

parser = argparse.ArgumentParser(description="MoE-Infinity Mixtral inference example")
parser.add_argument(
    "--checkpoint",
    default="mistralai/Mixtral-8x7B-v0.1",
    help="HuggingFace model checkpoint (Mixtral-8x7B or Mixtral-8x22B)",
)
parser.add_argument(
    "--offload_dir",
    default="/work/morrisliu07/aug_spec/moe_infinity/offload_output/Mixtral-8x7B-v0.1",
    help="Directory for offloading expert weights",
)
parser.add_argument(
    "--device_memory_ratio",
    type=float,
    default=0.75,
    help="Fraction of GPU memory to use for expert cache (lower on OOM)",
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=64,
)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

config = {
    "offload_path": args.offload_dir,
    "device_memory_ratio": args.device_memory_ratio,
}
model = MoE(args.checkpoint, config)

input_text = "A mixture-of-experts model differs from a dense transformer in that"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda:0")

output_ids = model.generate(
    input_ids,
    max_new_tokens=args.max_new_tokens,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
)

# decode only the newly generated tokens
generated = output_ids[0][input_ids.shape[1]:]
print(tokenizer.decode(generated, skip_special_tokens=True))

# moe-infinity's C++ thread pool hangs on shutdown; force exit
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
