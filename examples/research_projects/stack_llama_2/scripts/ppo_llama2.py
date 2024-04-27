# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, pipeline, LlamaForCausalLM, LlamaTokenizer

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import LengthSampler
from trl.import_utils import is_npu_available, is_xpu_available

import os
from rouge import Rouge 

input_min_text_length = 6
input_max_text_length = 500


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf", metadata={"help": "the model name"}) # "huggyllama/llama-7b" "/root/llama/llama-2-7b-hf"
    dataset_name: Optional[str] = field(default="lvwerra/stack-exchange-paired", metadata={"help": "the dataset name"}) # "Anthropic/hh-rlhf"
    log_with: Optional[str] = field(default='wandb', metadata={"help": "use 'wandb' to log with wandb"})
    use_safetensors: Optional[bool] = field(default=False, metadata={"help": "Use safetensors"})
    seed: Optional[int] = field(default=0, metadata={"help": "the random seed"})
    use_score_scaling: Optional[bool] = field(default=False, metadata={"help": "Use score scaling"})
    use_score_norm: Optional[bool] = field(
        default=False, metadata={"help": "Use score normalization. Only applicable if use_score_scaling is True"}
    )
    score_clip: Optional[float] = field(default=None, metadata={"help": "Score clipping"})
    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


def create_and_prepare_dataset(tokenizer):
    dataset = load_dataset(script_args.dataset_name, split="train[:10000]")

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(example):
        text_size = input_size()
        prompt = "Question: " + example["question"] + "\n\nAnswer: "
        example["input_ids"] = tokenizer.encode(prompt)[:text_size]
        example["query"] = tokenizer.decode(example["input_ids"])
        return example

    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format("torch")
    return dataset


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    script_args.model_name,
    #device_map={"": "xpu:0"} if is_xpu_available() else {"": "npu:0"} if is_npu_available else {"": 0},
    peft_config=lora_config,
    quantization_config=nf4_config,
    #reward_adapter=script_args.rm_adapter,
    use_safetensors=script_args.use_safetensors,
)
model = model.bfloat16().cuda()

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, model_max_length=1024, max_length=1024, truncation=True, padding=True)
tokenizer.pad_token = "[PAD]"
tokenizer.padding_side = "left"

dataset = create_and_prepare_dataset(tokenizer)


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


config = PPOConfig(
    model_name=script_args.model_name,
    log_with=script_args.log_with,
    learning_rate=1e-5,
    batch_size=4,
    mini_batch_size=1,
    gradient_accumulation_steps=1,
    optimize_cuda_cache=True,
    seed=script_args.seed,
    use_score_scaling=script_args.use_score_scaling,
    use_score_norm=script_args.use_score_norm,
    score_clip=script_args.score_clip,
)

ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
)

generation_kwargs = {
    "top_k": 10,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    #"max_new_tokens": 500,
    #'temperature': 0.7,
    #'repetition_penalty': 10.0,
    #'num_return_sequences': 1,
    #'eos_token_id': tokenizer.eos_token_id,
    'max_length': 1000,
}
"""
generation_kwargs = {
    "top_k": 0.0,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "max_new_tokens": 800,
}
"""

eval_q = 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n'
eval_tensor = [torch.Tensor(tokenizer.encode(eval_q)).int()]
eval_ref = """
Given your interest in "Breaking Bad" and "Band of Brothers," you seem to appreciate intense dramas with deep character development and gripping storylines. Here are a few recommendations that might resonate with you:

The Wire – This series offers a gritty, realistic exploration of life and drug trafficking in Baltimore through the eyes of both law enforcers and drug dealers. Its complex narratives and deep dive into societal issues might appeal to a fan of "Breaking Bad."
The Sopranos – Often regarded as one of the greatest TV shows of all time, this series delves into the life of Tony Soprano, a mob boss balancing his criminal organization with his family life. It shares the moral complexity and deep character studies that "Breaking Bad" excels in.
Mad Men – While this series is less about crime and more about the advertising industry of the 1960s, its focus on character development, moral dilemmas, and personal transformation might strike a chord similar to "Breaking Bad."
Fargo – Inspired by the original Coen Brothers film, this anthology series captures the dark comedy and moral ambiguity of its source material while delivering thrilling crime stories that might appeal to a fan of intricate narratives and strong character arcs.
Band of Gold – If you liked "Band of Brothers," you might also enjoy "Band of Gold," a British television drama series about the lives of a group of women who turn to prostitution to survive.
Generation Kill – From the creators of "The Wire," this miniseries follows the early phase of the Iraq War, depicting the lives of Marines in the 1st Reconnaissance Battalion. Like "Band of Brothers," it provides a gritty, realistic portrayal of soldiers in conflict.
These series offer a mix of critical acclaim and strong narrative depth, likely to satisfy your tastes based on your previous favorites.

"""
rouge = Rouge()

reward_model_name = '/root/trl/examples/research_projects/stack_llama/scripts/gpt2_peft_stack-exchange-paired_rmts__100000_2e-05/checkpoint-32000'
reward_model = pipeline("text-classification", model=reward_model_name)
reward_kwargs = {"return_all_scores": True, "function_to_apply": "none"}


for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    question_tensors = batch["input_ids"]
    #print(len(question_tensors), question_tensors[0].size())

    response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt=False,
        **generation_kwargs,
    )
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    #batch["response"] = [r[:500] for r in batch["response"]]
    #response_tensors = [torch.Tensor(tokenizer.encode(r)).int() for r in batch["response"]]
    

    # Compute reward score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    #texts = [tokenizer.decode(tokenizer.encode(t)[:1024]) for t in texts]
    """
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(ppo_trainer.accelerator.device)
    raw_rewards = ppo_trainer.accelerator.unwrap_model(ppo_trainer.model).compute_reward_score(**inputs)
    rewards = [raw_rewards[i, -1, 1] for i in range(len(raw_rewards))]  # take last token
    """
    #print(texts)
    pipe_outputs = reward_model(texts, truncation=True, **reward_kwargs)
    #print(pipe_outputs)
    rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

    # Run PPO step
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)
    
    print()
    print('Epoch', epoch)
    print('ppo/loss/value', stats['ppo/loss/value'])
    
    #print(eval_tensor[0].size())
    eval_response_tensors = ppo_trainer.generate(
        eval_tensor,
        return_prompt=False,
        **generation_kwargs,
    )
    eval_res = tokenizer.batch_decode(eval_response_tensors, skip_special_tokens=True)
    print(eval_res)
    scores = rouge.get_scores(eval_res[0], eval_ref)
    print(scores)
    
    
    output_dir = 'results/checkpoint_'+str(epoch)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    ppo_trainer.model.save_pretrained(output_dir)
    
