python ../../tools/preprocess_fastchat_instruct_data.py \
	--input=/root/FastChat/data/dummy_conversation.json \
	--tokenizer_path /root/models/llama-2-7b-chat-hf \
	--output_prefix /root/test/test_hf\
	--chunk_size=32 \
	--workers=32
