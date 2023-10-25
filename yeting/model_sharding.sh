
python tools/checkpoint_util.py \
	--target_tensor_parallel_size 1 \
	--target_pipeline_parallel_size 1 \
	--load_dir /root/models/llama-2-7b-chat-hf-megatron \
	--save_dir /root/models/llama-2-7b-chat-hf-megatron_shard_tp_1_pp_1 \
	--model_type llama2 \
	--true_vocab_size 32000 \
	--bf16
