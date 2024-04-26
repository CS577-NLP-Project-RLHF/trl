import torch
import transformers

from transformers import LlamaForCausalLM, LlamaTokenizer

#model_dir = "meta-llama/Llama-2-7b-hf"
model_dir = "results/checkpoint_36"
model = LlamaForCausalLM.from_pretrained(model_dir)
tokenizer = LlamaTokenizer.from_pretrained(model_dir)

pipeline = transformers.pipeline(
    "text-generation",

    model=model,

    tokenizer=tokenizer,

    torch_dtype=torch.float16,

    device_map="auto",

)

sequences = pipeline(
    'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',

    do_sample=True,

    top_k=10,

    num_return_sequences=1,

    eos_token_id=tokenizer.eos_token_id,

    max_length=1000,

)

for seq in sequences:

    print(f"{seq['generated_text']}")

