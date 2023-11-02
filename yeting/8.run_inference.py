import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer

TP = 2
PP = 1

TRAINED_PATH = f"/root/models/original_epfLLM_megatron/llama-2-7b-chat-hf-megatron/shard-tp{TP}-pp{PP}-pretrained"
MERGED_PATH = f"{TRAINED_PATH}-merged"
MERGED_PATH_HF = f"{MERGED_PATH}-hf"
# MERGED_PATH_HF = f"/root/models/llama-2-7b-chat-hf"

tokenizer = LlamaTokenizer.from_pretrained(MERGED_PATH_HF)
model = LlamaForCausalLM.from_pretrained(MERGED_PATH_HF)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device="cuda:2"
)
prompt = """#= a function that returns the fibonacci number of its argument =#
function fibonacci(n::Int)::Int
"""
sequences = pipeline(prompt, max_new_tokens=100, do_sample=True, top_k=20,
                     num_return_sequences=1)
for sequence in sequences:
    print(sequence["generated_text"])
