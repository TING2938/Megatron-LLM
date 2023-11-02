

python3 tools/preprocess_data.py \
	--input=/root/datasets/booksum.jsonl \
	--output_prefix=/root/datasets/booksum_megatron \
	--chunk_size=32 \
	--workers=16 \
	--tokenizer_type=PretrainedFromHF \
    --tokenizer_name_or_path=/root/models/llama-2-7b-chat-hf 

