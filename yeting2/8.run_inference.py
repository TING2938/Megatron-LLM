import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer

tokenizer=LlamaTokenizer.from_pretrained("/root/models/llama-2-7b-chat-hf-megatron_shard_tp_2_pp_1-pretrained-merged-hf")
print(tokenizer)
pipeline = transformers.pipeline(
    "text-generation",
    model=LlamaForCausalLM.from_pretrained("/root/models/llama-2-7b-chat-hf-megatron_shard_tp_2_pp_1-pretrained-merged-hf"),
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device="cuda:4"
)
prompt = """#= a function that returns the fibonacci number of its argument =#
function fibonacci(n::Int)::Int
"""
sequences = pipeline(prompt, max_new_tokens=100, do_sample=True, top_k=20,
                     num_return_sequences=1)
for sequence in sequences:
    print(sequence["generated_text"])
