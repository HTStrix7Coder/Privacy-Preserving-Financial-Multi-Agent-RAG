import sys
import json
import time
import logging
import requests
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "gdpr-slm-qwen3.5"

SYSTEM_PROMPT = """You are an automated German financial document parser. Extract key entities from financial documents into valid JSON only. Output ONLY raw JSON. No explanations, no markdown, no extra text."""

# Synthetic Golden Dataset to establish baseline performance
TEST_CASES = [
    {
        "text": """ENDGÜLTIGE BEDINGUNGEN vom 15. Oktober 2019.
Emittentin: Deutsche Pfandbriefbank AG, Unterschleißheim.
Wertpapierart: Fix-to-Float Schuldverschreibungen.
ISIN: DE000A2YN3J7. WKN: A2YN3J.```
Gesamtnennbetrag: bis zu EUR 150.000.000.
Festgelegte Stückelung: EUR 1.000.
Ausgabetag: 22.10.2019. Rückzahlungstag: 22.10.2029.
Zinsen: Vom 22.10.2019 bis 21.10.2024 mit 0,875% p.a. (Festzinsperiode).
Danach variabel basierend auf 3-Monats-EURIBOR zzgl. 95 Basispunkte p.a.
Rang: nicht nachrangig.""",
        "ground_truth": {
            "issuer": "Deutsche Pfandbriefbank AG",
            "isin": "DE000A2YN3J7",
            "currency": "EUR"
        }
    },
    {
        "text": """ENDGÜLTIGE BEDINGUNGEN vom 12. Mai 2024.
Emittentin: Commerzbank AG, Frankfurt am Main.
Wertpapierart: Variabel verzinsliche Schuldverschreibungen (Floating Rate Notes).
ISIN: DE000CBK1234. WKN: CBK123.
Gesamtnennbetrag: EUR 500.000.000.
Festgelegte Stückelung: EUR 100.000.
Ausgabetag: 15.05.2024. Rückzahlungstag: 15.05.2028.
Zinsen: Die Schuldverschreibungen werden variabel verzinst. Der Basiszinssatz entspricht dem 3-Monats-EURIBOR, zuzüglich einer festgelegten Marge von 1,25 % p.a. 
Rang: Nicht nachrangig (Senior Preferred).""",
        "ground_truth": {
            "issuer": "Commerzbank AG",
            "isin": "DE000CBK1234",
            "interest_rate_type": "floating"
        }
    }
]
#define the chat template
def extract_entities(text: str):
    response = requests.post(f"{OLLAMA_URL}/api/chat", json={
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract data from this text:\n\n{text}"}
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 512}
    })
    response.raise_for_status()
    content = response.json()["message"]["content"]
    clean = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    clean = clean.removeprefix("```json").removesuffix("```").strip()
    
    try:
        raw = json.loads(clean)
        return {str(k).lower().strip(): v for k, v in raw.items() if k}
    except Exception:
        return {}

def calculate_metrics(pred: dict, truth: dict):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Check predictions against truth
    for key, pred_val in pred.items():
        if key in truth:
            # Type cast to string for safety during comparison
            if str(pred_val).lower().strip() == str(truth[key]).lower().strip():
                true_positives += 1
            else:
                false_positives += 1
        else:
            # Model hallucinated extra keys not in our ground truth scope
            pass # We ignore extra benign keys for precision calculations in basic setups
            
    # Check truth against predictions to find false negatives
    for key, truth_val in truth.items():
        if key not in pred or str(pred.get(key, "")).lower().strip() != str(truth_val).lower().strip():
            false_negatives += 1
            
    return true_positives, false_positives, false_negatives

def run_evaluation():
    logging.info(f"Starting formal evaluation suite for Fine-Tuned Model: {MODEL_NAME}")
    
    total_tp, total_fp, total_fn = 0, 0, 0
    latencies = []
    
    for idx, case in enumerate(TEST_CASES):
        logging.info(f"Evaluating Document Sample {idx + 1}/{len(TEST_CASES)}...")
        
        start_time = time.time()
        prediction = extract_entities(case["text"])
        latency = time.time() - start_time
        latencies.append(latency)
        
        tp, fp, fn = calculate_metrics(prediction, case["ground_truth"])
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
    # Calculate Macro Metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    
    print("\n========================================================")
    print("FINANCIAL EXTRACTION SLM EVALUATION REPORT")
    print("========================================================")
    print(f"Model Under Test:       {MODEL_NAME}")
    print(f"Total Test Samples:     {len(TEST_CASES)}")
    print("--------------------------------------------------------")
    print(f"Precision:              {precision:.4f}  ({(precision*100):.1f}%)")
    print(f"Recall:                 {recall:.4f}  ({(recall*100):.1f}%)")
    print(f"F1-Score:               {f1_score:.4f}  ({(f1_score*100):.1f}%)")
    print(f"Average GPU Latency:    {avg_latency:.2f} seconds / document")
    print("========================================================\n")
    
if __name__ == "__main__":
    run_evaluation()
