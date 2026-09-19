# 🏛️ GDPR-Sovereign Financial Compliance System
### *Locally-Hosted Multi-Agent RegTech Architecture for German Bond Compliance*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.12-blue?logo=python)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestrator-LangGraph-orange?logo=langchain)](https://github.com/langchain-ai/langgraph)
[![Unsloth](https://img.shields.io/badge/Fine--Tuning-Unsloth-red)](https://github.com/unslothai/unsloth)
[![Ollama](https://img.shields.io/badge/Inference-Ollama%20(Local)-black?logo=ollama)](https://ollama.com/)
[![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-green)](https://www.trychroma.com/)
[![FastMCP](https://img.shields.io/badge/Tooling-Model_Context_Protocol-purple)](https://modelcontextprotocol.io/)
[![Cloudflare Tunnel](https://img.shields.io/badge/Deployment-Cloudflare_Tunnel-f38020?logo=cloudflare)](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/)

A fully on-premise, multi-agent AI compliance engine engineered to verify German debt securities (*Endgültige Bedingungen / Basisprospekte*) against EU and German banking regulations (**KWG, WpHG, MiFID II**) — with **strict GDPR data sovereignty and zero cloud data egress** as hard constraints.

---

## 📑 Key Capabilities

1. **Deterministic Entity Extraction:** Fine-tuned **Qwen3.5-2B (LoRA/PEFT, Q8_0 GGUF)** parses unformatted German bond documentation into strict, validated JSON in sub-3 seconds.
2. **Multi-Corpus Domain RAG:** 3 isolated ChromaDB vector collections indexing **260,000+ chunks** across German statutory law, base prospectuses, and issuer annual reports with `intfloat/multilingual-e5-base`.
3. **Live Regulatory & Market Hooks:** Async **Model Context Protocol (FastMCP)** queries live BaFin warnings and current Euribor rates, with automated air-gapped ECB benchmark fallbacks.
4. **Statutory Risk Reasoning:** **Qwen2.5-7B** reasons across the extracted bond terms, retrieved legal statutes, and live regulatory constraints to generate formalized audit reports.
5. **Stateful LangGraph DAG:** Directed acyclic graph with validation checkpoints, conditional routing, and deterministic error recovery.

---

## 🏗️ System Architecture

```
                       ┌───────────────────────────────────────┐
                       │   German Bond Document (Unstructured) │
                       └──────────────────┬────────────────────┘
                                          │
                                          ▼
                       ┌───────────────────────────────────────┐
                       │  Extractor Agent (Qwen3.5-2B GGUF)    │
                       └──────────────────┬────────────────────┘
                                          │
                                          ▼
                               [ Validation Gate ]
                                   /          \
                       (Valid JSON)            (Parsing Error)
                            │                         │
                            ▼                         ▼
            ┌──────────────────────────────┐    ┌────────────────────┐
            │   Multi-Corpus RAG Dispatch  │    │ Error Handler Node │
            ├──────────────────────────────┤    └────────────────────┘
            │ • German Law (KWG, WpHG)     │
            │ • Base Prospectuses          │
            │ • Issuer Annual Reports      │
            └──────────────┬───────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │   Live Telemetry (FastMCP)   │
            ├──────────────────────────────┤
            │ • BaFin Sanctions & Alerts   │
            │ • Euribor / Air-Gap Fallback │
            └──────────────┬───────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │ Compliance Reasoner (7B LLM) │
            └──────────────┬───────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │ Formal Compliance Report UI  │
            └──────────────────────────────┘
```

---

## 📊 Empirical Evaluation Benchmark

Evaluated on a **10% holdout test split (100 real German bond prospectuses)** completely unseen during training, alongside regulatory compliance edge-case scenarios:

### 1. Fine-Tuned SLM Extractor Performance (`gdpr-slm-qwen3.5`)
* **Holdout Test Split:** 10% unseen documents (Seed: `3407`)
* **Average GPU Latency:** **2.96s** / document (NVIDIA RTX 4060 Ti)

| Target Financial Field | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **`isin`** | 93.3% | **100.0%** | **0.9655** | 14 |
| **`issuer`** | **100.0%** | **100.0%** | **1.0000** | 25 |
| **`currency`** | **100.0%** | **100.0%** | **1.0000** | 23 |
| **`total_volume`** | 95.7% | 95.7% | **0.9565** | 23 |
| **`denomination`** | **100.0%** | **100.0%** | **1.0000** | 16 |
| **`issue_date`** | **100.0%** | **100.0%** | **1.0000** | 23 |
| **`maturity_date`** | 87.5% | **100.0%** | **0.9333** | 14 |
| **`interest_rate_type`** | 96.0% | **100.0%** | **0.9796** | 24 |
| **`document_type`** | 96.0% | 96.0% | **0.9600** | 25 |
| **`instrument_name`** | 84.0% | 84.0% | **0.8400** | 25 |
| **OVERALL (Macro-Average)** | **95.2%** | **97.6%** | **0.9635** | — |
| **OVERALL (Micro-Average)** | **95.4%** | **97.2%** | **0.9626** | — |

### 2. Compliance Reasoning Agent Performance (`qwen2.5:7b`)
* **Decision Accuracy:** **100.0% (3/3 statutory edge cases)**

| Benchmark Test Scenario | Regulatory Trigger | Expected Verdict | Model Output | Status |
|---|---|---|---|:---:|
| **Pfandbriefbank AG Fix-to-Float** | Clean BaFin record, valid prospectus terms | `COMPLIANT` | `COMPLIANT` | ✅ |
| **Wirecard AG Fraud Anomaly** | Active BaFin criminal investigation (§ 44 KWG) | `NON-COMPLIANT` | `NON-COMPLIANT` | ✅ |
| **Deutsche Bank Tier 2 Subordinated** | Statutory bail-in clause (§ 10 KWG) | `REVIEW REQUIRED` | `REVIEW REQUIRED` | ✅ |

*(Full benchmark logs and metrics available in [`eval_results.json`](./eval_results.json))*.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Fine-Tuning** | Unsloth + LoRA / PEFT on Qwen3.5-2B (bf16) |
| **Quantization** | GGUF Q8_0 exported via llama.cpp / Unsloth |
| **Inference Engine** | Local Ollama Runtime (C++ / CUDA backend) |
| **Embeddings** | `intfloat/multilingual-e5-base` (German domain-optimized) |
| **Vector Database** | ChromaDB (260,638 Chunks across 3 collections) |
| **Agent Orchestration** | LangGraph (StateGraph DAG with checkpointing) |
| **External Tooling** | FastMCP (Model Context Protocol, stdio transport) |
| **Frontend** | Streamlit (with 1-click test document presets) |
| **Edge Deployment** | Cloudflare Tunnel (Zero Trust encrypted tunneling) |

---

## 📂 Project Structure

```
GDPR_SLM/
├── app.py                    # Streamlit compliance dashboard
├── requirements.txt          # Production serving dependencies
├── requirements-train.txt    # Unsloth fine-tuning dependencies
├── Dockerfile                # Production container definition
├── docker-compose.yml        # Multi-service stack (Ollama + Web UI)
├── .env.example              # Environment template (0 exposed secrets)
├── finetune.py               # LoRA fine-tuning script (Unsloth)
├── convert_to_gguf.py        # GGUF quantization export pipeline
├── evaluate_model.py         # Empirical evaluation benchmark suite
├── eval_results.json         # Serialized empirical validation report
├── inference.py              # Standalone Python model inference
├── ollama_inference.py       # Direct Ollama API inference (single & batch)
├── pre_process/
│   ├── build_data.py         # Teacher distillation data generation
│   └── preprocess_dataset.py # Label normalization & text cleaning
├── rag/
│   ├── ingest_corpus.py      # Corpus chunking & embedding into ChromaDB
│   ├── langgraph_agent.py    # Multi-agent LangGraph orchestrator
│   └── test_retrieval.py     # RAG retrieval validation script
└── mcp/
    └── bafin_server.py       # BaFin & Euribor FastMCP tool server
```

---

## 🚀 Quick Start (Local Setup)

### 1. Install Dependencies
```bash
# Using standard pip:
pip install -r requirements.txt

# Or using uv (recommended for ultra-fast setup):
uv pip install -r requirements.txt
```

### 2. Start Ollama and Load Models
```bash
# Start the local Ollama daemon
ollama serve &

# Register the fine-tuned 2B extractor GGUF
ollama create gdpr-slm-qwen3.5 -f gguf_q8_0_gguf_qwen3.5/Modelfile

# Pull the 7B reasoning model
ollama pull qwen2.5:7b
```

### 3. Launch the Compliance Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser. Choose any of the **1-Click Preloaded Bond Samples** in the sidebar to test the pipeline immediately!

### 4. Run the Evaluation Benchmark
```bash
# Evaluate extractor on 25 holdout documents & test compliance reasoner:
python evaluate_model.py --mode all --samples 25
```

---

## 🌐 Production Deployment (Cloudflare Tunnel)

To expose your locally hosted, GPU-accelerated system to the public securely without open firewall ports:

```bash
# 1. Install Cloudflare Tunnel (Linux)
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb && rm cloudflared.deb

# 2. Start the encrypted tunnel to your Streamlit port
cloudflared tunnel --url http://localhost:8501
```
Cloudflare outputs a secure HTTPS public URL (e.g. `https://your-tunnel-name.trycloudflare.com`) routing directly to your GPU backend with **zero cloud data egress**.

---

## 🐳 Docker Deployment

```bash
docker-compose up -d
```

---

## ⚖️ Compliance & License

This project was developed for research and institutional regulatory automation purposes. All training data consists of publicly disclosed financial terms (*Endgültige Bedingungen*) compliant with Article 8 of the EU Prospectus Regulation.
