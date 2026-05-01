# 🏛️ GDPR-Sovereign Financial Compliance System

A fully on-device, multi-agent AI system for automated regulatory compliance checking of German bond documents — built with GDPR data sovereignty as a hard constraint.

## What It Does

Paste a raw German financial document (e.g. *Endgültige Bedingungen*), and the system will:

1. **Extract** structured entities (issuer, ISIN, rates, dates) using a fine-tuned 2B-parameter LLM
2. **Retrieve** relevant context from 260K+ chunks across German financial law (KWG, WpHG, MiFID II), base prospectuses, and issuer annual reports
3. **Query** live regulatory data (BaFin warnings, Euribor rates) via MCP tool calls
4. **Analyze** compliance against all retrieved context using a reasoning LLM
5. **Report** a structured compliance verdict with legal references and recommendations

**Zero cloud dependency. All inference runs locally via Ollama.**

## Architecture

```
Document → Fine-Tuned Extractor (Qwen3.5-2B, LoRA) → Validation Gate
    → Multi-Corpus RAG (ChromaDB, 3 collections) → Live Data (MCP/BaFin/Euribor)
    → Compliance Reasoner (Qwen2.5-7B) → Streamlit Dashboard
```

Orchestrated as a **LangGraph state machine** with conditional routing and error recovery.

## Tech Stack

| Layer | Technology |
|---|---|
| Fine-Tuning | Unsloth + LoRA/PEFT on Qwen3.5-2B |
| Training Data | Knowledge distillation from Gemini 2.5 Flash (~1K examples) |
| Quantization | GGUF Q8_0 → Ollama |
| Embeddings | `intfloat/multilingual-e5-base` (German-optimized) |
| Vector DB | ChromaDB (260K+ chunks, 3 collections) |
| Orchestration | LangGraph (StateGraph DAG) |
| Live Data | FastMCP (Model Context Protocol) |
| Frontend | Streamlit |

## Project Structure

```
GDPR_SLM/
├── app.py                    # Streamlit dashboard
├── finetune.py               # LoRA fine-tuning (Unsloth)
├── inference.py              # Direct model inference
├── ollama_inference.py       # Ollama API inference (single + batch)
├── evaluate_model.py         # F1/Precision/Recall evaluation suite
├── convert_to_gguf.py        # GGUF quantization export
├── pre_process/
│   ├── build_data.py         # Synthetic training data generation (Gemini teacher)
│   └── preprocess_dataset.py # Label normalization & text cleaning
├── rag/
│   ├── ingest_corpus.py      # Corpus chunking & embedding into ChromaDB
│   ├── langgraph_agent.py    # Multi-agent LangGraph orchestrator
│   └── test_retrieval.py     # RAG retrieval validation
└── mcp/
    └── bafin_server.py       # BaFin & Euribor MCP tool server
```

## Quick Start

```bash
# 1. Start Ollama with the fine-tuned model
ollama serve &
ollama create gdpr-slm-qwen3.5 -f gguf_q8_0_gguf_qwen3.5/Modelfile

# 2. Ingest the RAG corpus (first time only)
python rag/ingest_corpus.py --all

# 3. Launch the dashboard
streamlit run app.py
```

## License

This project was built for educational and research purposes.
