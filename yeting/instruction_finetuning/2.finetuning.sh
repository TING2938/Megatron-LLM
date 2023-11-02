
TP=2
PP=1
MT_CHECKPOINT_PATH=/root/models/original_epfLLM_megatron/llama-2-7b-chat-hf-megatron
SHARED_CPT=${MT_CHECKPOINT_PATH}/shard-tp${TP}-pp${PP}

CHECKPOINT_PATH=${SHARED_CPT}
DATA_ARGS="--data_path /root/test/test_hf"
TRAINED_PATH=${SHARED_CPT}-finetuned
TENSORBOARD_PATH=$TRAINED_PATH/logging
MODEL=llama2
MICRO_BATCH=1
GLOBAL_BATCH=2
TOKENIZER=PretrainedFromHF
TOKENIZER_PATH=/root/models/llama-2-7b-chat-hf 

DISTRIBUTED_ARGS="--nproc_per_node 2 \
	--nnodes 1 \
	--node_rank 0 \
	--master_addr localhost \
	--master_port 6100"

EXTRA_ARGS="--use_rms_norm \
	--glu_activation swiglu \
	--no_tie_embed_logits \
	--layernorm_epsilon 1e-5"

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
	--lr_decay_style cosine \
	--lr_warmup_fraction 0.1 \
	--lr 2e-5 \
	--min_lr 2e-6 \
	--finetune \
    --variable_seq_lengths \
    --data_type=fastchat_instruction \
	--weight_decay 0.1 \
	--sequence_parallel \
	--recompute_granularity selective \
	--log_timers_to_tensorboard \
	--scalar_loss_mask=0.0 \
	--rope_scaling_factor 1.0 \
	--metrics perplexity accuracy count_loss_mask \
	--train_iters 10"


CUDA_VISIBLE_DEVICES="2,3" CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=16 \
torchrun $DISTRIBUTED_ARGS ../../finetune.py \
       --tensor_model_parallel_size $TP \
       --pipeline_model_parallel_size $PP  \
       --load $CHECKPOINT_PATH \
       --save $TRAINED_PATH \
       --tensorboard_dir $TENSORBOARD_PATH \
       $DATA_ARGS \
       --model_name $MODEL \
       --tokenizer_type $TOKENIZER \
	   --tokenizer_name_or_path ${TOKENIZER_PATH} \
       --bf16 \
       --global_batch_size $GLOBAL_BATCH \
       --micro_batch_size $MICRO_BATCH \
       --num_workers=2 \
       $EXTRA_ARGS \
       $COMMON_ARGS


