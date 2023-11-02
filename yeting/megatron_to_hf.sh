
python weights_conversion/megatron_to_hf.py \
    --input_dir /root/models/llama-2-7b-chat-hf-megatron_shard_tp_2_pp_1-pretrained-merged \
	--output_dir /root/models/llama-2-7b-chat-hf-megatron_shard_tp_2_pp_1-pretrained-merged-hf \
    --vocab_file /root/models/llama-2-7b-chat-hf-megatron/tokenizer.model
