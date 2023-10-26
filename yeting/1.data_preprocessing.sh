

python3 tools/preprocess_data.py --input=/root/datasets/booksum.jsonl \
	--output_prefix=/root/datasets/booksum_megatron \
	--tokenizer_type=SentencePieceTokenizer \
	--vocab_file=/root/models/llama-2-7b-chat-hf/tokenizer.model \
	--chunk_size=32 \
	--workers=16 \
	--no_new_tokens

