import torch
import os
from transformers import AutoTokenizer
from moe_infinity import MoE

checkpoint = "mistralai/Mixtral-8x7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(
    checkpoint,
    trust_remote_code=True,
)

config = {
    "offload_path": "/work/morrisliu07/moe-infinity-offload/mixtral-8x7b-v0.1",
    "device_memory_ratio": 0.75,
}

model = MoE(checkpoint, config)

input_text = "translate English to German: How old are you?"
encoded = tokenizer(
    input_text,
    return_tensors="pt",
    return_attention_mask=True,
)

input_ids = encoded["input_ids"].to("cuda:0")
attention_mask = encoded["attention_mask"].to("cuda:0")

output_ids = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=64,
    do_sample=False,
)

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)

os._exit(0)