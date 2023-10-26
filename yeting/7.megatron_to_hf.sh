
TP=2
PP=1

TRAINED_PATH=/root/models/original_epfLLM_megatron/llama-2-7b-chat-hf-megatron/shard-tp${TP}-pp${PP}-pretrained
MERGED_PATH=${TRAINED_PATH}-merged
MERGED_PATH_HF=${MERGED_PATH}-hf

python weights_conversion/megatron_to_hf.py \
    --input_dir ${MERGED_PATH} \
	--output_dir ${MERGED_PATH_HF} \
    --vocab_file /root/models/llama-2-7b-chat-hf-megatron/tokenizer.model
