vllm serve Qwen/Qwen3-VL-8B-Thinking \
    --tensor-parallel-size 1 \
    --allowed-local-media-path / \
    --enforce-eager \
    --dtype half \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.8 \
    --port 8001