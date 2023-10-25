
python weights_conversion/hf_to_megatron.py llama2 \
	--size=7 \
	--out=/root/models/llama-2-7b-chat-hf-megatron/ \
	--model-path=/root/models/llama-2-7b-chat-hf/ \
	--cache-dir=/root/models/llama-2-7b-chat-hf_cache

