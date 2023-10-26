
TP=2
PP=1
MT_CHECKPOINT_PATH=/root/models/original_epfLLM_megatron/llama-2-7b-chat-hf-megatron
SHARED_CPT=${MT_CHECKPOINT_PATH}/shard-tp${TP}-pp${PP}

python tools/checkpoint_util.py \
	--target_tensor_parallel_size 1 \
	--target_pipeline_parallel_size 1 \
	--load_dir ${MT_CHECKPOINT_PATH} \
	--save_dir ${SHARED_CPT} \
	--model_type llama2 \
	--true_vocab_size 32000 \
	--bf16
