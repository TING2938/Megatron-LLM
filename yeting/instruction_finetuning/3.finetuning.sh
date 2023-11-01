LOG_ARGS="--log_interval 1 --save_interval 100 --eval_interval 50"
TRAIN_ARGS="--train_iters 6500 --lr_decay_style cosine --lr_warmup_iters 650 --lr 2e-5 --min_lr 2e-6"
DISTRIBUTED_ARGS="--nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8316"

COMMON_ARGS="--use_flash_attn \
	--no_bias_gelu_fusion \
	--seq_length 4096 \
	--max_position_embeddings 4096 \
	--log_interval 1 \
	--save_interval 800 \
	--eval_interval 200 \
	--eval_iters 10 \
	--hidden_dropout 0.0 \
	--position_embedding_type rotary \
	--no_bias_dropout_fusion \
	--use_checkpoint_args \
	--attention_dropout 0.0 \
	--adam_beta1 0.9 \
	--adam_beta2 0.95 \
	--adam_eps 1e-5 \
	--weight_decay 0.1 \
	--sequence_parallel \
	--recompute_granularity selective \
	--log_timers_to_tensorboard \
	--scalar_loss_mask=0.0 \
	--rope_scaling_factor 1.0 \
	--metrics perplexity accuracy count_loss_mask \
	--train_iters 100"

EXTRA_ARGS="--vocab_file=/root/models/original_epfLLM_megatron/llama-2-7b-chat-hf-megatron/tokenizer.model \
	--use_rms_norm \
	--glu_activation swiglu \
	--no_tie_embed_logits \
	--vocab_extra_ids_list <|im_start|>,<|im_end|>,<unk> \
	--layernorm_epsilon 1e-5"

CUDA_VISIBLE_DEVICES="4,5" CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun $DISTRIBUTED_ARGS finetune.py \
	--tensor_model_parallel_size 2 \
	--pipeline_model_parallel_size 1 \
	--load /root/models/original_epfLLM_megatron/llama-2-7b-chat-hf-megatron/shard-tp2-pp1 \
	--save /root/models/original_epfLLM_megatron/llama-2-7b-chat-hf-megatron/shard-tp2-pp1-finetuned \
	--tensorboard_dir /root/models/original_epfLLM_megatron/llama-2-7b-chat-hf-megatron/shard-tp2-pp1-finetuned/logging \
	--data_path /root/datasets/OpenOrca/OpenOrca_megatron \
	--model_name llama2 \
	--tokenizer_type SentencePieceTokenizer \
	--bf16 \
	--use_flash_attn \
	--micro_batch_size 2 \
	--global_batch_size 4 \
	--sequence_parallel \
	--recompute_granularity selective \
	--use_checkpoint_args \
	--no_new_tokens \
	--data_type instruction \
	--variable_seq_lengths \
	--finetune \
	$COMMON_ARGS $LOG_ARGS $EXTRA_ARGS $TRAIN_ARGS $LLAMA_ARGS


COMMON_ARGS="--use_flash_attn \
	--no_bias_gelu_fusion \
	--seq_length 4096 \
	--max_position_embeddings 4096 \
	--log_interval 1 \
	--save_interval 800 \
	--eval_interval 200 \
	--eval_iters 10 \
	--hidden_dropout 0.0 \
	--position_embedding_type rotary \
	--no_bias_dropout_fusion \
	--use_checkpoint_args \
	--attention_dropout 0.0 \
	--adam_beta1 0.9 \
	--adam_beta2 0.95 \
	--adam_eps 1e-5 \
	--weight_decay 0.1 \
	--sequence_parallel \
	--recompute_granularity selective \
	--log_timers_to_tensorboard \
	--scalar_loss_mask=0.0 \
	--rope_scaling_factor 1.0 \
	--metrics perplexity accuracy count_loss_mask \
	--train_iters 100"