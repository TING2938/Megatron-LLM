
TP=2
PP=1

TRAINED_PATH=/root/models/original_epfLLM_megatron/llama-2-7b-chat-hf-megatron/shard-tp${TP}-pp${PP}-pretrained
MERGED_PATH=${TRAINED_PATH}-merged

python tools/checkpoint_util.py \
	--target_tensor_parallel_size 1 \
	--target_pipeline_parallel_size 1 \
	--load_dir ${TRAINED_PATH} \
	--save_dir ${MERGED_PATH} \
	--model_type llama2 \
	--true_vocab_size 32000 \
	--bf16
