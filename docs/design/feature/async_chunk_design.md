# Async Chunk Design

## Table of Contents

1. [Overview](#overview)
2. [Performance](#performance)
3. [Architecture](#architecture)
4. [Configuration](#configuration)
5. [Related Files](#related-files)

## Overview

The `async_chunk` feature enables asynchronous, chunked processing of data across multiple stages in a multi-stage pipeline (e.g., Qwen3-Omni with Thinker → Talker → Code2Wav stages). Instead of waiting for a complete stage output before forwarding to the next stage, this feature allows stages to process and forward data in chunks as it becomes available, significantly reducing latency and improving throughput.

**Chunk Size Definition**

- **Prefill Phase**: `chunk_size = max_batched_num_tokens` for chunked prefill processing
- **Decode Phase**: `chunk_size = 1` for per-token streaming

For qwen3-omni:
- **Thinker → Talker**: Per decode step (typically chunk_size=1)
- **Talker → Code2Wav**: Accumulated to code2wav chunk_size(default=25, current only support default, will support chunk_size soon) before sending
- **Code2Wav**: Streaming decode with code2wav chunk_size

With `async_chunk`:
- Stages can start processing as soon as chunks are available
- Overlapping execution across stages
- Reduced latency and improved throughput
- Better resource utilization
- Async scheduling: Chunk IO (get/put) overlaps with compute via background threads so the scheduler is not blocked waiting for chunks

## Performance
1. **Reduced Latency**: Next stage can start processing immediately
2. **Streaming Support**: Enables streaming for audio generation
3. **IO-Compute Overlap**: Chunk retrieval happens asynchronously while other requests compute
4. **Non-blocking Scheduler**: Requests waiting for chunks don't block the entire scheduler

| Scenario | Input Modality | Output Modality | async_chunk | Input tokens num | Output tokens num | Request num |  TTFT(ms) |  TTFP(ms) | TPOT(ms) |
|--------|----------------------------|------------------------|-------------|-------------|-------------|------------------------|-------------|-------------|-------------|
|single request | text | text + audio | True | 10 | 10 | 1 | 89.90 | 831.18 | 20.28 |
|single request | text | text + audio | False | 10 | 10 | 1 | 98.19 | 3205.68 | 24.57 |
|single request | text | text + audio | True | 2500 | 900 | 1 | 380.03 | 1910.39 | 8.82 |
|single request | text | text + audio | False | 2500 | 900 | 1 | 392.15 | 11696 | 12.01 |

### Performance Data Comparison
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/performance/qwen3-omni_performance.png">
    <img alt="TTFT Performance Data Comparison" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/performance/qwen3-omni_ttft_performance.png" width=100%>
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/performance/qwen3-omni_performance.png">
    <img alt="TPOT Performance Data Comparison" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/performance/qwen3-omni_tpot_performance.png" width=100%>
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/performance/qwen3-omni_performance.png">
    <img alt="TTFP Performance Data Comparison" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/performance/qwen3-omni_ttfp_performance.png" width=100%>
  </picture>
</p>

## Architecture
### Data Flow

#### Sequential Flow
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/qwen3-omni-non-async-chunk.png">
    <img alt="Data Flow between stages" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/qwen3-omni-non-async-chunk.png" width=100%>
  </picture>
</p>

#### Async Chunk Flow

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/qwen3-omni-async-chunk.png">
    <img alt="Data Flow between stages" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/qwen3-omni-async-chunk.png" width=100%>
  </picture>
</p>

### Async Chunk architecture
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/async-chunk-architecture.png">
    <img alt="Async Chunk Architecture" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/async-chunk-architecture.png" width=100%>
  </picture>
</p>


### Key Components

1. **OmniConnector**: Inter-stage data transport only
   - Shared memory or other IPC mechanisms
   - **Transport-only API**: `put(from_stage, to_stage, put_key, data)` and `get(from_stage, to_stage, get_key)` (optionally with timeout)
   - No request-specific state; connector does not track put_requests, get_requests, or payload metadata
   - Chunk keys and request/chunk lifecycle are managed by OmniChunkManager

2. **Stage Input Processors**: Custom functions that process stage outputs into chunks for different models
Here is qwen3-omni showcase for reference:
   - `thinker2talker_async_chunk`: Processes thinker outputs for talker
   - `talker2code2wav_async_chunk`: Processes talker outputs for code2wav

3. **Schedulers**: Modified to handle chunk-based scheduling
   - `OmniARScheduler`: For autoregressive stages
   - `OmniGenerationScheduler`: For generation stages
   - Both schedulers use **OmniChunkManager** and **before/after** hooks around `super().schedule()`: process chunk queues before, restore queues and merge chunk data after

4. **OmniChunkManager**: Owns the full chunk lifecycle when async_chunk is enabled (WIP)
   - **Chunk ID and key construction**: Builds keys like `{req_id}_{stage_id}_{chunk_id}` for put/get
   - **Assembling chunked business data**: Merges thinker embeddings (stage 0), accumulates code_predictor_codes and builds code2wav payloads (stage 1), etc.; uses connector-backed state (e.g. put_requests, get_requests, request_prompt_token_ids, code_prompt_token_ids) where needed
   - **Async get**: `get_chunk(request)` enqueues the request for loading; a background **recv_loop** thread polls the connector and, when data is available, updates the request (e.g. `additional_information`, `prompt_token_ids`) and marks it in `_finished_load_reqs`; scheduler calls `get_finished()` to learn which requests have chunks ready
   - **Async put**: `put_chunk(pooling_output, request, custom_process_input_func)` builds the payload in the main thread (to avoid races on request state), then enqueues a save task; a background **save_loop** thread performs `connector.put()`; payload processing and chunk accumulation (e.g. code2wav chunk_size) remain in the main thread
   - **State**: `_pending_load_reqs`, `_finished_load_reqs`, `_pending_save_reqs`, `_finished_save_reqs`; coordinates with scheduler so requests waiting for chunks are moved to WAITING_FOR_CHUNK and temporarily out of main queues

5. **Model Runners**: Handle chunk processing
   - `OmniGPUModelRunner`: Processes chunks in AR stages
   - `GPUGenerationModelRunner`: Processes chunks in generation stages

6. **Model Implementation**: Model-specific chunk handling
   - `Qwen3OmniMoeForConditionalGeneration`: Main model with async_chunk support
   - `chunked_decode_streaming`: Streaming audio generation

7. **Request status**: （WIP）`RequestStatus.WAITING_FOR_CHUNK` is added via patch (e.g. in `vllm_omni/patch.py`) so requests waiting for a chunk are not scheduled by the base vLLM scheduler until the chunk is ready.

## Configuration

Enable async_chunk in stage configuration YAML:

```yaml
async_chunk: true
stage_args:
  - stage_id: 0
    engine_args:
      custom_process_next_stage_input_func: vllm_omni.model_executor.stage_input_processors.qwen3_omni.thinker2talker_async_chunk
  - stage_id: 1
    engine_args:
      custom_process_next_stage_input_func: vllm_omni.model_executor.stage_input_processors.qwen3_omni.talker2code2wav_async_chunk
```

### Stage Configuration

- `async_chunk: bool`: Enable/disable async chunk mode
- `custom_process_next_stage_input_func: str`: Path to custom chunk processing function,path should under stage input processor. e.g. for qwen3-omni, ```thinker2talker_async_chunk() ``` & ```talker2code2wav_async_chunk()```
- `stage_connector_config: dict`: Connector configuration


### Connector Configuration

```yaml
connectors:
  - from_stage: 0
    to_stage: 1
    spec:
      name: SharedMemoryConnector
      extra:
        stage_id: 0
```


## Related Files

- `vllm_omni/model_executor/stage_input_processors/qwen3_omni.py`: Chunk processing functions
- `vllm_omni/distributed/omni_connectors/omni_chunk_manager.py`: OmniChunkManager (async get/put, recv_loop, save_loop) and connector usage
- `vllm_omni/core/sched/omni_ar_scheduler.py`: AR scheduler with chunk_manager, _process_chunk_queue, restore, _clear_chunk_ready
- `vllm_omni/core/sched/omni_generation_scheduler.py`: Generation scheduler with same async chunk pattern
- `vllm_omni/worker/gpu_model_runner.py`: Model runner with chunk handling
- `vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py`: Model implementation
- `vllm_omni/engine/arg_utils.py`: Configuration definitions
- `vllm_omni/config/model.py`: Model config with async_chunk field
