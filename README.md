## DPO training

Source: https://huggingface.co/docs/trl/main/en/dpo_trainer

I tried running examples/scripts/dpo.py with:

```
pip install -r requirements.txt

python examples/scripts/dpo.py \
    --dataset_name=trl-internal-testing/hh-rlhf-trl-style \
    --model_name_or_path=gpt2 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="dpo_anthropic_hh" \
    --optim rmsprop \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=16
```

It was able to run successfully. As next steps, we can update the code in dpo.py or the arguments to use a different dataset and model for our project.

## StackLLaMA

Source: https://huggingface.co/blog/stackllama, https://huggingface.co/blog/dpo-trl

I found an implementation of StackLLaMA at: trl/examples/research_projects/stack_llama_2/scripts
/dpo_llama2.py

When I tried running dpo_llama2.py with the default configuration, I got error:

```
OSError: You are trying to access a gated repo.
Make sure to have access to it at https://huggingface.co/meta-llama/Llama-2-7b-hf.
```

Instead of using meta-llama/Llama-2-7b-hf, I found a smaller llama2 model: https://huggingface.co/ahxt/llama2_xs_460M_experimental, along with a llama2 model that doesn't need access request: https://huggingface.co/openlm-research/open_llama_3b_v2

I was able to run dpo_llama2.py with:

```
cd examples/research_projects/stack_llama_2/scripts
pip install -r requirements.txt

python dpo_llama2.py --model_name_or_path ahxt/llama2_xs_460M_experimental
```

With 50GB of space on the device, I got an error "No space left on device".



# TRL - Transformer Reinforcement Learning
> Full stack library to fine-tune and align large language models.

<p align="center">
    <a href="https://github.com/huggingface/trl/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/huggingface/trl.svg?color=blue">
    </a>
    <a href="https://huggingface.co/docs/trl/index">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/trl/index.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/huggingface/trl/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/trl.svg">
    </a>
</p>


## What is it?

The `trl` library is a full stack tool to fine-tune and align transformer language and diffusion models using methods such as Supervised Fine-tuning step (SFT), Reward Modeling (RM) and the Proximal Policy Optimization (PPO) as well as Direct Preference Optimization (DPO). 

The library is built on top of the [`transformers`](https://github.com/huggingface/transformers) library and thus allows to use any model architecture available there.


## Highlights

- **`Efficient and scalable`**: 
    - [`accelerate`](https://github.com/huggingface/accelerate) is the backbone of `trl` which allows to scale model training from a single GPU to a large scale multi-node cluster with methods such as DDP and DeepSpeed.
    - [`PEFT`](https://github.com/huggingface/peft) is fully integrated and allows to train even the largest models on modest hardware with quantisation and methods such as LoRA or QLoRA.
    - [`unsloth`](https://github.com/unslothai/unsloth) is also integrated and allows to significantly speed up training with dedicated kernels.
- **`CLI`**: With the [CLI](https://huggingface.co/docs/trl/clis) you can fine-tune and chat with LLMs without writing any code using a single command and a flexible config system.
- **`Trainers`**: The Trainer classes are an abstraction to apply many fine-tuning methods with ease such as the [`SFTTrainer`](https://huggingface.co/docs/trl/sft_trainer), [`DPOTrainer`](https://huggingface.co/docs/trl/trainer#trl.DPOTrainer), [`RewardTrainer`](https://huggingface.co/docs/trl/reward_trainer), [`PPOTrainer`](https://huggingface.co/docs/trl/trainer#trl.PPOTrainer), [`CPOTrainer`](https://huggingface.co/docs/trl/trainer#trl.CPOTrainer), and [`ORPOTrainer`](https://huggingface.co/docs/trl/trainer#trl.ORPOTrainer).
- **`AutoModels`**: The [`AutoModelForCausalLMWithValueHead`](https://huggingface.co/docs/trl/models#trl.AutoModelForCausalLMWithValueHead) & [`AutoModelForSeq2SeqLMWithValueHead`](https://huggingface.co/docs/trl/models#trl.AutoModelForSeq2SeqLMWithValueHead) classes add an additional value head to the model which allows to train them with RL algorithms such as PPO.
- **`Examples`**: Train GPT2 to generate positive movie reviews with a BERT sentiment classifier, full RLHF using adapters only, train GPT-j to be less toxic, [StackLlama example](https://huggingface.co/blog/stackllama), etc. following the [examples](https://github.com/huggingface/trl/tree/main/examples).

## Installation

### Python package
Install the library with `pip`:
```bash
pip install trl
```

### From source
If you want to use the latest features before an official release you can install from source:
```bash
pip install git+https://github.com/huggingface/trl.git
```

### Repository
If you want to use the examples you can clone the repository with the following command:
```bash
git clone https://github.com/huggingface/trl.git
```

## Command Line Interface (CLI)

You can use TRL Command Line Interface (CLI) to quickly get started with Supervised Fine-tuning (SFT), Direct Preference Optimization (DPO) and test your aligned model with the chat CLI: 

**SFT:**

```bash
trl sft --model_name_or_path facebook/opt-125m --dataset_name imdb --output_dir opt-sft-imdb
```

**DPO:**

```bash
trl dpo --model_name_or_path facebook/opt-125m --dataset_name trl-internal-testing/hh-rlhf-trl-style --output_dir opt-sft-hh-rlhf 
```

**Chat:**

```bash
trl chat --model_name_or_path Qwen/Qwen1.5-0.5B-Chat
```

Read more about CLI in the [relevant documentation section](https://huggingface.co/docs/trl/main/en/clis) or use `--help` for more details.

## How to use

For more flexibility and control over the training, you can use the dedicated trainer classes to fine-tune the model in Python.

### `SFTTrainer`

This is a basic example of how to use the `SFTTrainer` from the library. The `SFTTrainer` is a light wrapper around the `transformers` Trainer to easily fine-tune language models or adapters on a custom dataset.

```python
# imports
from datasets import load_dataset
from trl import SFTTrainer

# get dataset
dataset = load_dataset("imdb", split="train")

# get trainer
trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
)

# train
trainer.train()
```

### `RewardTrainer`

This is a basic example of how to use the `RewardTrainer` from the library. The `RewardTrainer` is a wrapper around the `transformers` Trainer to easily fine-tune reward models or adapters on a custom preference dataset.

```python
# imports
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer

# load model and dataset - dataset needs to be in a specific format
model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=1)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

...

# load trainer
trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
)

# train
trainer.train()
```

### `PPOTrainer`

This is a basic example of how to use the `PPOTrainer` from the library. Based on a query the language model creates a response which is then evaluated. The evaluation could be a human in the loop or another model's output.

```python
# imports
import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import respond_to_batch

# get models
model = AutoModelForCausalLMWithValueHead.from_pretrained('gpt2')
model_ref = create_reference_model(model)

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# initialize trainer
ppo_config = PPOConfig(batch_size=1, mini_batch_size=1)

# encode a query
query_txt = "This morning I went to the "
query_tensor = tokenizer.encode(query_txt, return_tensors="pt")

# get model response
response_tensor  = respond_to_batch(model, query_tensor)

# create a ppo trainer
ppo_trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer)

# define a reward for response
# (this could be any reward such as human feedback or output from another model)
reward = [torch.tensor(1.0)]

# train model for one step with ppo
train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)
```

### `DPOTrainer`

`DPOTrainer` is a trainer that uses [Direct Preference Optimization algorithm](https://arxiv.org/abs/2305.18290). This is a basic example of how to use the `DPOTrainer` from the library. The `DPOTrainer` is a wrapper around the `transformers` Trainer to easily fine-tune reward models or adapters on a custom preference dataset.

```python
# imports
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer

# load model and dataset - dataset needs to be in a specific format
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

...

# load trainer
trainer = DPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
)

# train
trainer.train()
```

## Development

If you want to contribute to `trl` or customizing it to your needs make sure to read the [contribution guide](https://github.com/huggingface/trl/blob/main/CONTRIBUTING.md) and make sure you make a dev install:

```bash
git clone https://github.com/huggingface/trl.git
cd trl/
make dev
```

## References

### Proximal Policy Optimisation
The PPO implementation largely follows the structure introduced in the paper **"Fine-Tuning Language Models from Human Preferences"** by D. Ziegler et al. \[[paper](https://arxiv.org/pdf/1909.08593.pdf), [code](https://github.com/openai/lm-human-preferences)].

### Direct Preference Optimization
DPO is based on the original implementation of **"Direct Preference Optimization: Your Language Model is Secretly a Reward Model"** by E. Mitchell et al. \[[paper](https://arxiv.org/pdf/2305.18290.pdf), [code](https://github.com/eric-mitchell/direct-preference-optimization)]


## Citation

```bibtex
@misc{vonwerra2022trl,
  author = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang},
  title = {TRL: Transformer Reinforcement Learning},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/trl}}
}
```
