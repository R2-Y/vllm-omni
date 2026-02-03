MODEL_PATH="Qwen/Qwen3-Omni-30B-A3B-Instruct"
PORT=8666

# 1. cudagraph + async_chunk
Feature_name="async_chunk_cudagraph"
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port ${PORT} --log-stats --stage-configs-path /rein_workspace/vllm-omni/vllm_omni/model_executor/stage_configs/qwen3_omni_moe_async_chunk.yaml > server_async_chunk_cudagraph.log 2>&1 &
sleep 600;
bash ./benchmark_test.sh ${Feature_name}
pkill -f "VLLM::EngineCore";
ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs kill -9;
sleep 60;


# 2. cudagraph + sequential
Feature_name="sequential_cudagraph"
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port ${PORT} --log-stats --stage-configs-path /rein_workspace/vllm-omni/vllm_omni/model_executor/stage_configs/qwen3_omni_moe.yaml > server_sequential_cudagraph.log 2>&1 &
sleep 600;
bash ./benchmark_test.sh ${Feature_name}
pkill -f "VLLM::EngineCore";
ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs kill -9;
sleep 60;

# 3. eager + async_chunk
Feature_name="async_chunk_eager"
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port ${PORT} --log-stats --stage-configs-path /rein_workspace/vllm-omni/vllm_omni/model_executor/stage_configs/qwen3_omni_moe_async_chunk_eager.yaml > server_async_chunk_eager.log 2>&1 &
sleep 600;
bash ./benchmark_test.sh ${Feature_name}
pkill -f "VLLM::EngineCore";
ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs kill -9;
sleep 60;

# 4. eager + sequential
Feature_name="sequential_eager"
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port ${PORT}  --log-stats --stage-configs-path /rein_workspace/vllm-omni/vllm_omni/model_executor/stage_configs/qwen3_omni_moe_eager.yaml > server_sequential_eager.log 2>&1 &
sleep 600;
bash ./benchmark_test.sh ${Feature_name}
pkill -f "VLLM::EngineCore";
ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs kill -9;
sleep 60;

