module load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0
export HF_HOME=/work/morrisliu07/.cache/huggingface   # 新模型下到 /work
cd /work/morrisliu07/thesis_experiment/my_moe_infinity-rfang
source .venv/bin/activate

SPDLOG_LEVEL=debug 
CUDA_VISIBLE_DEVICES=0
uv run examples/interface_example.py \
    --model_name_or_path "Qwen/Qwen3-30B-A3B" \
    --offload_dir offload_output \
    --num_samples 5 \
    --device_memory_ratio 0.85 \
    --out_len 20