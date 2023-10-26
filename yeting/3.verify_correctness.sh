# arguments required by `torchrun`
DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8100"
LLAMA_ARGS="--use_rms_norm --glu_activation swiglu --no_tie_embed_logits --no_new_tokens --layernorm_epsilon 1e-5"
COMMON_ARGS="--hidden_dropout 0.0 --attention_dropout 0.0 --no_bias_gelu_fusion"

MT_CHECKPOINT_PATH=/root/models/original_epfLLM_megatron/llama-2-7b-chat-hf-megatron/
HF_CHECKPOINT_PATH=/root/models/llama-2-7b-chat-hf
DATA_PATH=/root/datasets/booksum_megatron/booksum_megatron_text_document

torchrun $DISTRIBUTED_ARGS verify_correctness.py \
	--model_name=llama \
	--model_size=7 \
	--load ${MT_CHECKPOINT_PATH} \
	--data_path ${DATA_PATH} \
	--tokenizer_type=SentencePieceTokenizer \
	--vocab_file ${MT_CHECKPOINT_PATH}/tokenizer.model \
	--huggingface_cache ${HF_CHECKPOINT_PATH} \
	--huggingface_device=cuda:5 \
	$COMMON_ARGS $LLAMA_ARGS  # dont include LLAMA_ARGS if using Falcon
