"""
Convert Fine-Tuned Model (LoRA adapter + base) to GGUF format.
Exports for use with Ollama, llama.cpp, LM Studio, etc.
Usage:
    python convert_to_gguf.py              # Default: Q8_0 quantization
    python convert_to_gguf.py q4_k_m       # Smallest, fastest
    python convert_to_gguf.py q5_k_m       # Good balance
    python convert_to_gguf.py q8_0         # Best quality
    python convert_to_gguf.py f16          # Full 16-bit (largest)
"""

import os
import sys
from unsloth import FastLanguageModel

# ============================================================================
# CONFIGURATION
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_DIR = os.path.join(SCRIPT_DIR, "finetuned_adapter_qwen3.5")
MAX_SEQ_LENGTH = 2048

# Choose quantization method from CLI arg or default to q8_0
QUANT_METHOD = sys.argv[1] if len(sys.argv) > 1 else "q8_0"
OUTPUT_DIR = os.path.join(SCRIPT_DIR, f"gguf_{QUANT_METHOD}")

print(f"Quantization method: {QUANT_METHOD}")
print(f"Output directory: {OUTPUT_DIR}")

# ============================================================================
# LOAD MODEL + LoRA ADAPTER
# ============================================================================
print("\nLoading base model + LoRA adapter...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_DIR,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=False,   # Must match training — Qwen3.5-2B was trained in bf16
    load_in_16bit=True,
)

# ============================================================================
# EXPORT TO GGUF
# ============================================================================
print(f"\nExporting to GGUF ({QUANT_METHOD})... This may take a few minutes.")
model.save_pretrained_gguf(
    OUTPUT_DIR,
    tokenizer,
    quantization_method=QUANT_METHOD,
)

print(f"\n✅ GGUF model saved to: {OUTPUT_DIR}")
print(f"\nTo use with Ollama, create a Modelfile:")
print(f"  echo 'FROM ./{os.path.basename(OUTPUT_DIR)}/unsloth.{QUANT_METHOD.upper()}.gguf' > Modelfile")
print(f"  ollama create gdpr-slm -f Modelfile")
print(f"  ollama run gdpr-slm")
