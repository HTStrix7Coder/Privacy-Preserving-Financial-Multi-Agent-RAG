"""
Inference Script for Fine-Tuned German Financial Document Parser (GDPR SLM)
Model: Qwen3.5-2B with LoRA adapter
"""

import os
import json
import sys
from unsloth import FastLanguageModel

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
ADAPTER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetuned_adapter_qwen3.5")
MAX_SEQ_LENGTH = 2048
DTYPE = None
LOAD_IN_4BIT = False        # Must match training: bf16 LoRA for Qwen3.5

SYSTEM_PROMPT = "You are a specialized German financial parser. Extract entities into valid JSON."

# ============================================================================
# 2. LOAD MODEL + LoRA ADAPTER
# ============================================================================
print("Loading base model + LoRA adapter...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_DIR,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
    load_in_16bit=True,
)

FastLanguageModel.for_inference(model)
print("Model loaded and ready for inference!\n")

# ============================================================================
# 3. EXTRACTION FUNCTION
# ============================================================================
def extract_entities(text: str) -> dict | None:
    """
    Extract financial entities from German text and return as dict.
    Returns None if extraction fails.
    """
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": f"Extract data from this text:\n\n{text}"}]},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
        top_p=0.95,
    )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    # Parse JSON from response
    try:
        clean = response.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        print(f"Warning: Could not parse JSON from response:\n{response}")
        return None

# ============================================================================
# 4. RUN INFERENCE
# ============================================================================
if __name__ == "__main__":

    # --- Get text from: CLI arg > built-in sample ---
    filepath = sys.argv[1] if len(sys.argv) > 1 else None

    if filepath and os.path.exists(filepath):
        print(f"\nReading document: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()[:5000]
    else:
        print("\nUsing built-in sample document...")
        text = """Konditionenblatt
Deutsche Bank AG

15.03.2023
Fixed Rate Bond Series 500
(die "Schuldverschreibungen")

Währung: Euro ("EUR")
Gesamtnennbetrag: EUR 50.000.000,-
Festgelegte Stückelung: EUR 1.000,-
Begebungstag: 01.04.2023
Fälligkeitstag: 01.04.2028
Zinssatz: 3,50% p.a.
ISIN: DE000DB7XYZ1
"""

    print("=" * 60)
    print("Extracting entities...")
    print("=" * 60)

    result = extract_entities(text)

    if result:
        print("\nExtracted JSON:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("\nExtraction failed.")
