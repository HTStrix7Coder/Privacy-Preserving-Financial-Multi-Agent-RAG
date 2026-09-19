"""
Comprehensive Evaluation Suite for GDPR-Sovereign Financial Compliance System

Evaluates:
1. Fine-Tuned SLM Extractor (Field-level Precision, Recall, F1 on holdout test split)
2. Compliance Reasoning Agent (Decision accuracy on regulatory edge cases)

Usage:
    python evaluate_model.py --samples 25                    # Evaluate on 25 holdout documents
    python evaluate_model.py --mode compliance              # Evaluate compliance reasoner
    python evaluate_model.py --mode all --samples 50        # Full comprehensive benchmark
"""

import os
import re
import sys
import json
import time
import random
import logging
import argparse
import requests
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("EvalSuite")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_EXTRACTOR = os.getenv("EXTRACTOR_MODEL", "gdpr-slm-qwen3.5")
DEFAULT_REASONER = os.getenv("COMPLIANCE_MODEL", "qwen2.5:7b")

DATASET_PATH = Path(__file__).parent / "train_cleaned.jsonl"
OUTPUT_REPORT = Path(__file__).parent / "eval_results.json"

EXTRACTOR_SYSTEM = """You are an automated German financial document parser. Extract key entities from financial documents into valid JSON only. Output ONLY raw JSON. No explanations, no markdown, no extra text."""

CORE_FIELDS = [
    "isin",
    "issuer",
    "currency",
    "total_volume",
    "denomination",
    "issue_date",
    "maturity_date",
    "interest_rate_type",
    "document_type",
    "instrument_name",
]

# ============================================================================
# CURATED COMPLIANCE REASONING BENCHMARK
# ============================================================================
COMPLIANCE_BENCHMARK_CASES = [
    {
        "name": "Pfandbriefbank AG Fix-to-Float",
        "description": "Standard non-subordinated bond with Euribor floating tranche and clean supervisory record.",
        "extracted_data": {
            "document_type": "Final Terms",
            "issuer": "Deutsche Pfandbriefbank AG",
            "instrument_name": "Fix-to-Float Schuldverschreibungen",
            "isin": "DE000A2YN3J7",
            "currency": "EUR",
            "total_volume": 150000000.0,
            "denomination": 1000.0,
            "issue_date": "2019-10-22",
            "maturity_date": "2029-10-22",
            "interest_rate_type": "mixed",
        },
        "bafin_status": "Clean: No active BaFin warnings, sanctions, or investigations found.",
        "expected_status": "COMPLIANT",
    },
    {
        "name": "Wirecard AG Fraud Anomaly",
        "description": "Issuer subject to critical BaFin criminal fraud alerts and active trading suspension.",
        "extracted_data": {
            "document_type": "Final Terms",
            "issuer": "Wirecard AG",
            "instrument_name": "Senior Unsecured Floating Rate Notes",
            "isin": "DE000WND1234",
            "currency": "EUR",
            "total_volume": 500000000.0,
            "denomination": 100000.0,
            "issue_date": "2020-01-15",
            "maturity_date": "2024-01-15",
            "interest_rate_type": "floating",
        },
        "bafin_status": "CRITICAL RISK: Multiple fraud investigations (§ 44 KWG). Trading suspended. Insolvency active.",
        "expected_status": "NON-COMPLIANT",
    },
    {
        "name": "Deutsche Bank Tier 2 Subordinated Bail-In",
        "description": "Subordinated instrument subject to statutory bail-in rules under § 10 KWG.",
        "extracted_data": {
            "document_type": "Final Terms",
            "issuer": "Deutsche Bank AG",
            "instrument_name": "Tier 2 Subordinated Notes",
            "isin": "DE000DB9XYZ4",
            "currency": "EUR",
            "total_volume": 750000000.0,
            "denomination": 100000.0,
            "issue_date": "2022-09-10",
            "maturity_date": "2032-09-10",
            "interest_rate_type": "fixed",
        },
        "bafin_status": "NOTICE: Ongoing BaFin monitor for risk management systems. No active bond issuance restrictions.",
        "expected_status": "REVIEW REQUIRED",
    },
]

# ============================================================================
# NORMALIZATION & MATCHING UTILITIES
# ============================================================================
def normalize_val(val: Any) -> str:
    """Normalize values for robust equality comparison."""
    if val is None:
        return ""
    # Convert floats/ints like 150000000.0 -> "150000000"
    if isinstance(val, (int, float)):
        try:
            return str(int(val)) if float(val).is_integer() else str(val)
        except (ValueError, OverflowError):
            return str(val)
    
    s = str(val).strip().lower()
    # Normalize common date formats YYYY-MM-DD
    s = re.sub(r"\s+", " ", s)
    return s

def values_match(pred: Any, truth: Any) -> bool:
    """Check if predicted and ground truth values match semantically."""
    norm_p = normalize_val(pred)
    norm_t = normalize_val(truth)
    if norm_p == norm_t:
        return True
    
    # Substring match for lengthy institution names (e.g. "Commerzbank" in "Commerzbank Aktiengesellschaft")
    if len(norm_t) > 5 and (norm_t in norm_p or norm_p in norm_t):
        return True
        
    return False

# ============================================================================
# DATASET LOADING (10% TEST SPLIT REPRODUCTION)
# ============================================================================
def load_holdout_test_split(dataset_path: Path, seed: int = 3407, test_ratio: float = 0.1) -> List[Dict[str, Any]]:
    """Reproduces the exact holdout test set from finetune.py."""
    if not dataset_path.exists():
        logger.error(f"Dataset file not found at {dataset_path}")
        return []

    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    random.seed(seed)
    # Shuffle with exact same seed
    indices = list(range(len(records)))
    random.shuffle(indices)

    test_size = max(1, int(len(records) * test_ratio))
    test_indices = indices[:test_size]

    test_set = []
    for idx in test_indices:
        item = records[idx]
        user_msg = next((m["content"] for m in item["messages"] if m["role"] == "user"), "")
        asst_msg = next((m["content"] for m in item["messages"] if m["role"] == "assistant"), "{}")

        # Strip prompt prefix from user text
        clean_text = user_msg.replace("Extract data from this text:\n\n", "").strip()

        try:
            truth_dict = json.loads(asst_msg)
            truth_normalized = {str(k).lower().strip(): v for k, v in truth_dict.items()}
            test_set.append({"text": clean_text, "ground_truth": truth_normalized})
        except json.JSONDecodeError:
            continue

    logger.info(f"Loaded {len(test_set)} holdout test examples (10% split of {len(records)} total records).")
    return test_set

# ============================================================================
# EXTRACTION INFERENCE
# ============================================================================
def call_extraction_model(text: str, model_name: str) -> Tuple[Dict[str, Any], float]:
    """Call Ollama extraction model and return parsed JSON + latency."""
    start_time = time.time()
    try:
        res = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": EXTRACTOR_SYSTEM},
                    {"role": "user", "content": f"Extract data from this text:\n\n{text[:5000]}"},
                ],
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 512},
            },
            timeout=60,
        )
        latency = time.time() - start_time
        res.raise_for_status()

        content = res.json().get("message", {}).get("content", "")
        clean = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        clean = clean.removeprefix("```json").removesuffix("```").strip()

        parsed = json.loads(clean)
        return {str(k).lower().strip(): v for k, v in parsed.items() if k}, latency
    except Exception as e:
        latency = time.time() - start_time
        return {}, latency

# ============================================================================
# COMPLIANCE REASONER INFERENCE
# ============================================================================
def call_compliance_reasoner(case: Dict[str, Any], model_name: str) -> Tuple[str, float]:
    """Evaluates compliance reasoning agent verdict."""
    start_time = time.time()
    prompt = f"""
Analyze this bond for German compliance:
Issuer: {case['extracted_data'].get('issuer')}
ISIN: {case['extracted_data'].get('isin')}
Instrument: {case['extracted_data'].get('instrument_name')}
Interest Type: {case['extracted_data'].get('interest_rate_type')}
BaFin Status: {case['bafin_status']}

Determine status: COMPLIANT | REVIEW REQUIRED | NON-COMPLIANT
Respond with: COMPLIANCE STATUS: [STATUS]
"""
    try:
        res = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a German financial compliance officer."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 256},
            },
            timeout=60,
        )
        latency = time.time() - start_time
        res.raise_for_status()
        content = res.json().get("message", {}).get("content", "")
        
        status = "UNKNOWN"
        for line in content.splitlines():
            if "COMPLIANCE STATUS:" in line.upper():
                if "NON-COMPLIANT" in line.upper():
                    status = "NON-COMPLIANT"
                elif "REVIEW REQUIRED" in line.upper():
                    status = "REVIEW REQUIRED"
                elif "COMPLIANT" in line.upper():
                    status = "COMPLIANT"
                break
        return status, latency
    except Exception:
        return "ERROR", time.time() - start_time

# ============================================================================
# BENCHMARK RUNNERS
# ============================================================================
def run_extraction_benchmark(model_name: str, sample_limit: int) -> Dict[str, Any]:
    logger.info(f"=== Starting Field-Level Extraction Benchmark on {model_name} ===")
    test_set = load_holdout_test_split(DATASET_PATH)
    if not test_set:
        logger.error("No test set available to evaluate.")
        return {}

    eval_samples = test_set[:sample_limit]
    logger.info(f"Evaluating {len(eval_samples)} holdout documents...")

    field_tp = defaultdict(int)
    field_fp = defaultdict(int)
    field_fn = defaultdict(int)
    latencies = []

    for idx, sample in enumerate(eval_samples, 1):
        pred, latency = call_extraction_model(sample["text"], model_name)
        latencies.append(latency)
        truth = sample["ground_truth"]

        # Track evaluated fields
        all_keys = set(CORE_FIELDS).union(truth.keys())

        for key in all_keys:
            has_truth = key in truth and truth[key] is not None
            has_pred = key in pred and pred[key] is not None

            if has_truth and has_pred:
                if values_match(pred[key], truth[key]):
                    field_tp[key] += 1
                else:
                    field_fp[key] += 1
                    field_fn[key] += 1
            elif has_truth and not has_pred:
                field_fn[key] += 1
            elif not has_truth and has_pred:
                field_fp[key] += 1

        print(f"\rProgress: [{idx}/{len(eval_samples)}] Docs Processed | Avg Latency: {sum(latencies)/len(latencies):.2f}s", end="", flush=True)

    print("\n")

    # Calculate per-field metrics
    field_metrics = {}
    total_tp, total_fp, total_fn = 0, 0, 0

    for key in CORE_FIELDS:
        tp = field_tp[key]
        fp = field_fp[key]
        fn = field_fn[key]

        total_tp += tp
        total_fp += fp
        total_fn += fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        field_metrics[key] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    macro_prec = sum(m["precision"] for m in field_metrics.values()) / len(field_metrics)
    macro_rec = sum(m["recall"] for m in field_metrics.values()) / len(field_metrics)
    macro_f1 = sum(m["f1"] for m in field_metrics.values()) / len(field_metrics)

    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * (micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    return {
        "model": model_name,
        "sample_count": len(eval_samples),
        "average_latency_sec": round(sum(latencies) / len(latencies), 2),
        "macro_metrics": {
            "precision": round(macro_prec, 4),
            "recall": round(macro_rec, 4),
            "f1": round(macro_f1, 4),
        },
        "micro_metrics": {
            "precision": round(micro_prec, 4),
            "recall": round(micro_rec, 4),
            "f1": round(micro_f1, 4),
        },
        "field_breakdown": field_metrics,
    }

def run_compliance_benchmark(model_name: str) -> Dict[str, Any]:
    logger.info(f"=== Starting Compliance Reasoner Benchmark on {model_name} ===")
    results = []
    correct = 0

    for case in COMPLIANCE_BENCHMARK_CASES:
        verdict, lat = call_compliance_reasoner(case, model_name)
        is_match = verdict == case["expected_status"]
        if is_match:
            correct += 1

        results.append({
            "case_name": case["name"],
            "expected": case["expected_status"],
            "predicted": verdict,
            "correct": is_match,
            "latency_sec": round(lat, 2),
        })

    accuracy = correct / len(COMPLIANCE_BENCHMARK_CASES)
    return {
        "model": model_name,
        "total_cases": len(COMPLIANCE_BENCHMARK_CASES),
        "accuracy": round(accuracy, 4),
        "cases": results,
    }

# ============================================================================
# REPORT PRINTER
# ============================================================================
def print_formal_report(ext_res: Dict[str, Any], comp_res: Dict[str, Any]):
    print("\n" + "=" * 78)
    print(" 🏛️  GDPR-SOVEREIGN SYSTEM: FORMAL EMPIRICAL BENCHMARK REPORT")
    print("=" * 78)

    if ext_res:
        print(f"\n[PART I: EXTRACTOR SLM ({ext_res.get('model')})]")
        print(f"Holdout Samples Tested: {ext_res.get('sample_count')} documents (10% unseen split)")
        print(f"Average Latency:        {ext_res.get('average_latency_sec')} s / document")
        print("-" * 78)
        print(f"{'FIELD NAME':<22} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10} | {'SUPPORT'}")
        print("-" * 78)

        for field, m in ext_res.get("field_breakdown", {}).items():
            print(f"{field:<22} | {m['precision']*100:>8.1f}% | {m['recall']*100:>8.1f}% | {m['f1']:>10.4f} | {m['support']}")

        print("-" * 78)
        macro = ext_res["macro_metrics"]
        micro = ext_res["micro_metrics"]
        print(f"{'OVERALL (Macro-Avg)':<22} | {macro['precision']*100:>8.1f}% | {macro['recall']*100:>8.1f}% | {macro['f1']:>10.4f} |")
        print(f"{'OVERALL (Micro-Avg)':<22} | {micro['precision']*100:>8.1f}% | {micro['recall']*100:>8.1f}% | {micro['f1']:>10.4f} |")

    if comp_res:
        print("\n" + "-" * 78)
        print(f"[PART II: COMPLIANCE REASONER ({comp_res.get('model')})]")
        print(f"Decision Accuracy: {comp_res.get('accuracy')*100:.1f}% ({sum(1 for c in comp_res['cases'] if c['correct'])}/{comp_res['total_cases']} cases)")
        print("-" * 78)
        for c in comp_res.get("cases", []):
            mark = "✅" if c["correct"] else "❌"
            print(f"{mark} {c['case_name']:<42} -> Expected: {c['expected']:<16} Got: {c['predicted']}")

    print("=" * 78 + "\n")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Empirical Evaluation Benchmark")
    parser.add_argument("--mode", choices=["all", "extraction", "compliance"], default="extraction")
    parser.add_argument("--samples", type=int, default=25, help="Number of holdout test samples")
    parser.add_argument("--extractor-model", default=DEFAULT_EXTRACTOR)
    parser.add_argument("--reasoner-model", default=DEFAULT_REASONER)
    args = parser.parse_args()

    # Check Ollama connectivity
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
    except requests.ConnectionError:
        logger.warning(f"Ollama is offline at {OLLAMA_URL}. Ensure 'ollama serve' is running to perform live inference.")
        print(f"\n⚠️  Ollama offline. To run evaluation, start Ollama first:\n   ollama serve &\n")
        sys.exit(1)

    ext_results = {}
    comp_results = {}

    if args.mode in ["all", "extraction"]:
        ext_results = run_extraction_benchmark(args.extractor_model, args.samples)

    if args.mode in ["all", "compliance"]:
        comp_results = run_compliance_benchmark(args.reasoner_model)

    print_formal_report(ext_results, comp_results)

    # Save to disk
    full_output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "extractor_benchmark": ext_results,
        "compliance_benchmark": comp_results,
    }
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    logger.info(f"Evaluation report saved to {OUTPUT_REPORT}")
