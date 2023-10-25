LOG_ARGS="--log_interval 1 --save_interval 100 --eval_interval 50"
TRAIN_ARGS="--train_iters 500 --lr_decay_style cosine --lr_warmup_iters 50 --lr 2e-5 --min_lr 2e-6"
DISTRIBUTED_ARGS="--nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8200"

CUDA_VISIBLE_DEVICES=0,1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun $DISTRIBUTED_ARGS finetune.py \
	--tensor_model_parallel_size 2 \
	--pipeline_model_parallel_size 1 \
	--load /root/models/llama-2-7b-chat-hf-megatron \
	--save /root/models/llama-2-7b-chat-hf-megatron_shard_tp_2_pp_1_finetuned \
	--tensorboard_dir /root/models/llama-2-7b-chat-hf-megatron_shard_tp_2_pp_1_tensorboard \
	--data_path /root/datasets/booksum_megatron/booksum_megatron_text_document \
	--model_name llama2 \
	--tokenizer_type SentencePieceTokenizer \
	--vocab_file=/root/models/llama-2-7b-chat-hf-megatron/tokenizer.model \
	--bf16 \
	--use_flash_attn \
	--micro_batch_size 1 \
	--global_batch_size 16 \
	--sequence_parallel \
	--recompute_granularity selective \
	--use_checkpoint_args \
	--no_new_tokens \
	$COMMON_ARGS $LOG_ARGS $TRAIN_ARGS $LLAMA_ARGS
