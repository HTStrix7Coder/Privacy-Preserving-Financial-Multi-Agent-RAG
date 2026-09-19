"""
Ollama Inference Script for GDPR SLM (Qwen3.5-2B GGUF)
Uses the Ollama API to run inference on German financial documents.

Prerequisites:
    1. Ollama running: ollama serve &
    2. Model registered: ollama create gdpr-slm-qwen3.5 -f GDPR_SLM/gguf_q8_0_gguf_qwen3.5/Modelfile

Usage:
    python ollama_inference.py                          # Interactive prompt
    python ollama_inference.py path/to/document.txt     # From file
    python ollama_inference.py --batch-dir path/to/dir  # Batch process
"""

import os
import sys
import json
import argparse
import requests

# ============================================================================
# CONFIGURATION
# ============================================================================
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("EXTRACTOR_MODEL", "gdpr-slm-qwen3.5")
SYSTEM_PROMPT = "You are a specialized German financial document parser. Extract key entities from financial documents into valid JSON only. Output ONLY raw JSON. No explanations, no markdown, no extra text."


# ============================================================================
# EXTRACTION FUNCTION
# ============================================================================
def extract_entities(text: str) -> dict | None:
    """Extract financial entities from German text using Ollama API."""
    
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract data from this text:\n\n{text}"},
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.95,
                "num_predict": 512,
            },
        },
    )

    if response.status_code != 200:
        print(f"Error: Ollama returned status {response.status_code}")
        print(response.text)
        return None

    result = response.json()
    content = result["message"]["content"]

    # Parse JSON from response — strip Qwen3.5 <think>...</think> block first
    try:
        import re
        clean = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        clean = clean.removeprefix("```json").removesuffix("```").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        print(f"Warning: Could not parse JSON:\n{content}")
        return None


# ============================================================================
# BATCH PROCESSING
# ============================================================================
def batch_process(directory: str):
    """Process all .txt files in a directory."""
    txt_files = sorted([f for f in os.listdir(directory) if f.endswith(".txt")])
    print(f"Found {len(txt_files)} .txt files in {directory}\n")

    results = []
    for i, filename in enumerate(txt_files, 1):
        filepath = os.path.join(directory, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()[:5000]

        print(f"[{i}/{len(txt_files)}] Processing {filename}...", end=" ")
        result = extract_entities(text)

        if result:
            results.append({"file": filename, "data": result})
            print(f"✅ ISIN: {result.get('isin', 'N/A')}")
        else:
            results.append({"file": filename, "data": None})
            print("❌ Failed")

    # Save all results
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    success = sum(1 for r in results if r["data"])
    print(f"\nDone! {success}/{len(results)} successful. Saved to: {output_file}")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    # Check if Ollama is running
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
    except requests.ConnectionError:
        print("❌ Ollama is not running! Start it with:")
        print("   ollama serve")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Ollama inference for GDPR SLM")
    parser.add_argument("file", nargs="?", help="Path to a .txt document")
    parser.add_argument("--batch-dir", type=str, help="Process all .txt files in a directory")
    args = parser.parse_args()

    # Batch mode
    if args.batch_dir:
        batch_process(args.batch_dir)
        sys.exit(0)

    # Single file or interactive
    if args.file:
        filepath = args.file
    else:
        filepath = input("Enter path to a .txt document (or press Enter for sample): ").strip()

    if filepath:
        print(f"\nReading document: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()[:5000]
    else:
        text = ("Konditionenblatt\nDeutsche Bank AG\n15.03.2023\n"
                "Fixed Rate Bond Series 500\n(die \"Schuldverschreibungen\")\n"
                "Währung: Euro (\"EUR\")\nGesamtnennbetrag: EUR 50.000.000,-\n"
                "Festgelegte Stückelung: EUR 1.000,-\nBegebungstag: 01.04.2023\n"
                "Fälligkeitstag: 01.04.2028\nZinssatz: 3,50% p.a.\n"
                "ISIN: DE000DB7XYZ1")

    print("=" * 60)
    print("Extracting entities via Ollama...")
    print("=" * 60)

    result = extract_entities(text)

    if result:
        print("\nExtracted JSON:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("\nExtraction failed.")
