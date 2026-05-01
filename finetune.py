"""
Unsloth Fine-Tuning Script for German Financial Document Parser (GDPR SLM)
Model: Qwen3.5-2B (bf16 LoRA — QLoRA NOT recommended for Qwen3.5)
GPU: RTX 4060 Ti 8GB
Dataset: ~1000 German financial document -> JSON extraction examples
"""

import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# ============================================================================
# 1. MODEL CONFIGURATION
# ============================================================================
MODEL_NAME = "Qwen/Qwen3.5-2B"   # Dense 2B model (bf16 LoRA = ~5GB VRAM)
MAX_SEQ_LENGTH = 2048
DTYPE = None                       # Auto-detect
LOAD_IN_4BIT = False               # QLoRA NOT recommended for Qwen3.5!

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
    load_in_16bit=True,             # bf16 LoRA (recommended for Qwen3.5)
    full_finetuning=False,
)

# ============================================================================
# 2. LoRA ADAPTER CONFIGURATION
# ============================================================================
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                           # Rank 16: good balance for structured extraction
    lora_alpha=16,                  # Equal to rank (Qwen3.5 recommended)
    lora_dropout=0,                 # 0 = Unsloth optimized
    target_modules=[                # Target ALL linear layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    use_gradient_checkpointing="unsloth",  # 30% less VRAM
    random_state=3407,
    max_seq_length=MAX_SEQ_LENGTH,
)

# ============================================================================
# 3. DATASET LOADING & CHAT TEMPLATE
# ============================================================================
DATASET_PATH = os.path.join(os.path.dirname(__file__), "train_cleaned.jsonl")

dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# Split into train and eval (90/10)
dataset = dataset.train_test_split(test_size=0.1, seed=3407)
train_dataset = dataset["train"]
eval_dataset  = dataset["test"]

print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

# Apply chat template
def formatting_func(examples):
    """Format each example using the tokenizer's chat template."""
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return {"text": texts}

train_dataset = train_dataset.map(formatting_func, batched=True)
eval_dataset  = eval_dataset.map(formatting_func, batched=True)

# ============================================================================
# 4. TRAINING CONFIGURATION (Optimized for RTX 4060 Ti 8GB)
# ============================================================================
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=SFTConfig(
        # --- Core Training ---
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        num_train_epochs=3,
        per_device_train_batch_size=1,        # Minimum for 8GB VRAM
        gradient_accumulation_steps=8,        # Effective batch size = 8

        # --- Learning Rate ---
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,

        # --- Regularization ---
        weight_decay=0.01,
        max_grad_norm=1.0,

        # --- Precision ---
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),

        # --- Logging & Evaluation ---
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,

        # --- Output ---
        output_dir=OUTPUT_DIR,
        report_to="none",

        # --- Optimizer & Misc ---
        optim="adamw_8bit",
        seed=3407,
        dataset_num_proc=2,
        packing=False,
    ),
)

# ============================================================================
# 5. SHOW MEMORY STATS & TRAIN
# ============================================================================
print("\n" + "="*60)
print("GPU Memory Stats BEFORE Training:")
gpu_stats = torch.cuda.get_device_properties(0)
reserved_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"  GPU: {gpu_stats.name}")
print(f"  Total VRAM: {max_memory} GB")
print(f"  Reserved: {reserved_memory} GB")
print(f"  Available for training: {max_memory - reserved_memory:.3f} GB")
print("="*60 + "\n")

# Train!
trainer_stats = trainer.train()

# ============================================================================
# 6. POST-TRAINING STATS
# ============================================================================
print("\n" + "="*60)
print("Training Complete!")
print(f"  Total training time: {trainer_stats.metrics['train_runtime']:.1f}s")
print(f"  Final training loss: {trainer_stats.metrics['train_loss']:.4f}")
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f"  Peak VRAM used: {used_memory} GB / {max_memory} GB")
print("="*60 + "\n")

# ============================================================================
# 7. SAVE THE MODEL
# ============================================================================
SAVE_DIR = os.path.join(os.path.dirname(__file__), "finetuned_adapter_qwen3.5")

print("Saving LoRA adapter...")
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"LoRA adapter saved to: {SAVE_DIR}")

print("\nDone! Run 'python inference.py' to test your fine-tuned model.")
