MODEL_PATH="Qwen/Qwen3-Omni-30B-A3B-Instruct"
PORT=8666

Feature_name=$1

COMMON_ARGS="\
--omni \
--port ${PORT} \
--model ${MODEL_PATH} \
--backend openai-chat-omni \
--endpoint /v1/chat/completions \
--random-input-len 2500 \
--random-output-len 900 \
--ignore-eos \
--percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf"

# 1. max concurrency - text
declare -a CONCURRENCY_LIST=(1 4 8)
declare -a PROMPTS_LIST=(10 40 80)

for i in ${!CONCURRENCY_LIST[@]}; do
	CONC=${CONCURRENCY_LIST[$i]}
	PROMPTS=${PROMPTS_LIST[$i]}
	LOG_FILE="${Feature_name}_bench_conc_c${CONC}_p${PROMPTS}.log"

	echo "Running concurrency mode: c=${CONC}, p=${PROMPTS}"

	vllm bench serve \
	${COMMON_ARGS} \
	--max-concurrency ${CONC} \
	--request-rate inf \
	--num-prompts ${PROMPTS} \
	> ${LOG_FILE} 2>&1

	echo "Finished: ${LOG_FILE}"
	echo
done

# 2. request rate - text 
# declare -a RATE_LIST=(0.1 0.2 0.3 0.4)
# declare -a RATE_PROMPTS_LIST=(10 20 30 40)

# for i in ${!RATE_LIST[@]}; do
# 	RATE=${RATE_LIST[$i]}
# 	PROMPTS=${RATE_PROMPTS_LIST[$i]}
# 	LOG_FILE="${Feature_name}_bench_rate_r${RATE}_p${PROMPTS}.log"

# 	echo "Running rate mode: rate=${RATE}, p=${PROMPTS}"

# 	vllm bench serve \
# 	${COMMON_ARGS} \
# 	--request-rate ${RATE} \
# 	--num-prompts ${PROMPTS} \
# 	> ${LOG_FILE} 2>&1

# 	echo "Finished: ${LOG_FILE}"
# 	echo
# done

# 3. max concurrency - mm
MM_LIMIT='{"image":1,"video":1,"audio":1}'
MM_BUCKET='{"(32, 32, 1)":0.5,"(0, 1, 1)":0.1,"(32, 32, 2)":0.4}'

COMMON_ARGS=(
 --omni
 --dataset-name random-mm
 --port ${PORT}
 --model ${MODEL_PATH}
 --endpoint /v1/chat/completions
 --backend openai-chat-omni
 --random-input-len 2500
 --random-output-len 900
 --random-range-ratio 0.0
 --random-mm-base-items-per-request 2
 --random-mm-num-mm-items-range-ratio 0
 --random-mm-limit-mm-per-prompt "$MM_LIMIT"
 --random-mm-bucket-config "$MM_BUCKET"
 --ignore-eos
 --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf
)

echo "================ Concurrency mode ===================="

CONCURRENCY_LIST=(1 4 8)
PROMPTS_LIST=(10 40 80)

for i in "${!CONCURRENCY_LIST[@]}"; do
   CONC=${CONCURRENCY_LIST[$i]}
   PROMPTS=${PROMPTS_LIST[$i]}
   LOG_FILE="${Feature_name}_mm_bench_conc_c${CONC}_p${PROMPTS}.log"

   echo "🚀 Running concurrency mode: c=${CONC}, prompts=${PROMPTS}"

   vllm bench serve \
     "${COMMON_ARGS[@]}" \
     --max-concurrency ${CONC} \
     --request-rate inf \
     --num-prompts ${PROMPTS} \
     > "${LOG_FILE}" 2>&1

   echo "✅ Finished: ${LOG_FILE}"
   echo
done

# 4. request rate - mm
# echo "================ Request-rate mode ==================="

# RATE_LIST=(0.1 0.2 0.3 0.4)
# RATE_PROMPTS_LIST=(10 20 30 40)

# for i in "${!RATE_LIST[@]}"; do
#    RATE=${RATE_LIST[$i]}
#    PROMPTS=${RATE_PROMPTS_LIST[$i]}
#    LOG_FILE="${Feature_name}_mm_bench_rate_r${RATE}_p${PROMPTS}.log"

#    echo "🚀 Running rate mode: rate=${RATE}, prompts=${PROMPTS}"

#    vllm bench serve \
#      "${COMMON_ARGS[@]}" \
#      --request-rate ${RATE} \
#      --num-prompts ${PROMPTS} \
#      > "${LOG_FILE}" 2>&1

#    echo "✅ Finished: ${LOG_FILE}"
#    echo
# done

# echo "🎯 All benchmark runs completed."
