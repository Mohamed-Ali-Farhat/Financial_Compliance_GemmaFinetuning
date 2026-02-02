# Finetuning Google Gemma on Tunisian Banking Compliance

This notebook outlines the process for fine-tuning the **Gemma-2b** model using Low-Rank Adaptation (LoRA) on a specialized Tunisian banking compliance dataset.

---

## 1. Environment Setup
Install the necessary libraries for PEFT (Parameter-Efficient Fine-Tuning), quantization, and dataset handling.

```python
!pip uninstall -y bitsandbytes peft trl accelerate transformers datasets
!pip install -U transformers accelerate peft trl bitsandbytes datasets
```

---

## 2. Imports and Configuration

Configure the environment and Hugging Face authentication.

```python
import os
import transformers
import torch
from google.colab import userdata
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GemmaTokenizer

# Fetch Hugging Face token from Colab Secret Keys
os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')
```

---

## 3. Model Loading and Quantization

Load the model in 4-bit quantization to fit within standard GPU memory constraints.

```python
model_id = "google/gemma-2b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"":0},
    token=os.environ['HF_TOKEN']
)
```

---

## 4. Zero-Shot Testing

Test the base model's performance before any fine-tuning.

```python
text = "WHY THE SKY IS BLUE?"
device = "cuda:0"
inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 5. Weights & Biases Integration

Setup experiment tracking to monitor the training loss and performance.

```python
import wandb
os.environ["WANDB_DISABLED"] = "false"
wandb.login()
os.environ["WANDB_PROJECT"] = "Tunisia-Banking-Compliance-Gemma"
```

---

## 6. LoRA Configuration

Define the Low-Rank Adaptation parameters focusing on the `q_proj` and `v_proj` modules.

```python
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
```

---

## 7. Dataset Loading and Formatting

Load the specialized dataset and define the prompt template used for training.

```python
data = load_dataset("MedAliFarhat/Tunisia-Banking-Compliance-qa")

def formatting_func(example):
    text = f"### Instruction: Étant donné la législation bancaire tunisienne, répondez à la question.\n"
    text += f"### Question: {example['question']}\n"
    text += f"### Réponse (Article {example['article_number']}): {example['answer']}"
    return [text]
```

---

## 8. Training Execution

Using the `SFTTrainer` (Supervised Fine-tuning Trainer) to execute the training process.

```python
trainer = SFTTrainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=50,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs_compliance",
        optim="paged_adamw_8bit"
    ),
    peft_config=lora_config,
    formatting_func=formatting_func,
)

trainer.train()
```

---

## 9. Model Persistence

Save the fine-tuned adapter and tokenizer, then back them up to Google Drive.

```python
output_dir = "outputs_compliance"
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Mount Drive and copy weights
from google.colab import drive
drive.mount('/content/drive')
!cp -r outputs_compliance /content/drive/MyDrive/
```
