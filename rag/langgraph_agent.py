"""
Enterprise-Grade German Financial Compliance System
Orchestrates multiple specialized agents using LangGraph for document extraction,
context retrieval, and regulatory compliance checking.

Usage:
    python GDPR_SLM/rag/langgraph_agent.py --demo
    python GDPR_SLM/rag/langgraph_agent.py path/to/document.txt --model qwen2.5:7b
"""

import os
import re
import sys
import json
import logging
import asyncio
import argparse
import operator
import requests
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import TypedDict, Annotated, Dict, Any, List, Tuple

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import chromadb
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END

# ============================================================================
# CONFIGURATION & LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)

SCRIPT_DIR = Path(__file__).parent
CHROMA_DB_DIR = str(SCRIPT_DIR / "chroma_db")
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
OLLAMA_URL = "http://localhost:11434"

EXTRACTOR_MODEL = "gdpr-slm-qwen3.5"
COMPLIANCE_MODEL = "qwen2.5:7b"
N_RESULTS = 3

# ============================================================================
# SYSTEM PROMPTS
# ============================================================================
EXTRACTOR_SYSTEM = """You are an automated German financial document parser. Extract key entities from financial documents into valid JSON only. Output ONLY raw JSON. No explanations, no markdown, no extra text."""

COMPLIANCE_SYSTEM = """You are a formalized German financial compliance and risk analyst agent. Analyze a SPECIFIC bond using the provided excerpts from Law, the issuer's Base Prospectus, and their Annual Report.

You MUST respond using EXACTLY this structure — no deviations:

COMPLIANCE STATUS: COMPLIANT | REVIEW REQUIRED | NON-COMPLIANT

KEY FINDINGS:
- [Finding about the bond's terms vs. German Law]
- [Finding about if the bond's features match the Base Prospectus]
- [Risk finding based on the issuer's financial health in the Annual Report]

LEGAL & PROSPECTUS REFERENCES:
- [Law/Prospectus name]: [What it requires/allows and whether this bond meets it]

LIVE DATA OVERSIGHT:
- [Reference the BaFin warnings and current Euribor rates explicitly]

RECOMMENDATION:
[One paragraph with concrete next steps or final judgment for this specific bond]

Rules:
- NEVER say "hypothetical" — analyze the actual bond data provided.
- Cite specific article numbers or document sections when possible.
- Reference the actual issuer, ISIN, and amounts from the extracted data.
- If the Base Prospectus or Law is missing relevant details, state that explicitly."""


# ============================================================================
# STATE DEFINITION
# ============================================================================
class ComplianceState(TypedDict):
    document_text: str
    compliance_model: str
    
    extracted_json: Dict[str, Any] | None
    extraction_error: str | None
    
    law_context: str | None
    prospectus_context: str | None
    annual_context: str | None
    
    bafin_warnings: str | None
    euribor_rates: str | None
    
    compliance_report: str | None
    compliance_status: str | None
    steps_completed: Annotated[List[str], operator.add]


# ============================================================================
# AGENT ABSTRACTIONS
# ============================================================================
class ResourceProvider:
    """Singleton provider for shared RAG embedding models and vector collections."""
    _model: SentenceTransformer | None = None
    _law: chromadb.Collection | None = None
    _prospectus: chromadb.Collection | None = None
    _annual: chromadb.Collection | None = None
    _logger = logging.getLogger("ResourceProvider")

    @classmethod
    def get(cls):
        if cls._model is None:
            cls._logger.info("Initializing multilingual embedding model...")
            cls._model = SentenceTransformer(EMBEDDING_MODEL)
            client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
            cls._law = client.get_collection("law_corpus")
            cls._prospectus = client.get_collection("base_prospectuses")
            cls._annual = client.get_collection("annual_reports")
            cls._logger.info(f"RAG initialized: Law({cls._law.count()}) | Prosp({cls._prospectus.count()}) | Annual({cls._annual.count()})")
        return cls._model, cls._law, cls._prospectus, cls._annual


class InformationExtractionAgent:
    """Agent responsible for parsing raw document unstructured text into structured JSON."""
    def __init__(self):
        self.logger = logging.getLogger("ExtractorAgent")

    def invoke(self, text: str) -> Dict[str, Any]:
        self.logger.info("Executing structured data extraction protocol...")
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": EXTRACTOR_MODEL,
                "messages": [
                    {"role": "system", "content": EXTRACTOR_SYSTEM},
                    {"role": "user", "content": f"Extract data from this text:\n\n{text[:5000]}"},
                ],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 512},
            },
            timeout=60,
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "")
        clean = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        clean = clean.removeprefix("```json").removesuffix("```").strip()
        
        raw_extracted = json.loads(clean)
        # Normalize all dictionary keys to lowercase to prevent capitalization mismatches
        extracted = {str(k).lower().strip(): v for k, v in raw_extracted.items() if k}
        
        self.logger.info(f"Extraction successful. Recovered {len(extracted)} entity fields.")
        return extracted


class ComplianceAnalyzerAgent:
    """Agent responsible for reasoning over RAG context and Extracted JSON to produce compliance reports."""
    def __init__(self):
        self.logger = logging.getLogger("ComplianceAgent")

    def invoke(self, state: ComplianceState, llm_model: str) -> Tuple[str, str]:
        self.logger.info("Executing comprehensive law and risk compliance analysis...")
        
        user_prompt = f"""RELEVANT LAW EXCERPTS:
{state['law_context']}

{'='*50}
RELEVANT BASE PROSPECTUS EXCERPTS:
{state['prospectus_context']}

{'='*50}
RELEVANT ANNUAL REPORT EXCERPTS (RISK):
{state['annual_context']}

{'='*50}
LIVE MARKET & REGULATORY DATA (VIA MCP SERVER):
BaFin Status: {state.get('bafin_warnings', 'Not checked')}
Current Market Rates: {state.get('euribor_rates', 'Not checked')}

{'='*50}
BOND DATA TO ANALYZE (use this specific data — do NOT say you need more information):
{json.dumps(state['extracted_json'], indent=2, ensure_ascii=False)}

Analyze the bond above against ALL the excerpts and write your compliance report now."""

        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": llm_model,
                "messages": [
                    {"role": "system", "content": COMPLIANCE_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 1024, "num_ctx": 16384},
            },
            timeout=120,
        )
        response.raise_for_status()
        content = response.json()["message"]["content"]
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        status = "REVIEW REQUIRED"
        for line in content.splitlines():
            if "COMPLIANCE STATUS:" in line:
                if "NON-COMPLIANT" in line:
                    status = "NON-COMPLIANT"
                elif "COMPLIANT" in line and "NON" not in line:
                    status = "COMPLIANT"
                break

        self.logger.info(f"Compliance analysis concluded. Status: {status}")
        return content, status


# ============================================================================
# LANGGRAPH NODE FUNCTIONS
# ============================================================================
def extract_data_node(state: ComplianceState) -> dict:
    agent = InformationExtractionAgent()
    try:
        extracted = agent.invoke(state["document_text"])
        return {"extracted_json": extracted, "extraction_error": None, "steps_completed": ["extract_data"]}
    except Exception as e:
        agent.logger.error(f"Extraction execution failed: {e}")
        return {"extracted_json": None, "extraction_error": str(e), "steps_completed": ["extract_data"]}


def validate_extraction_node(state: ComplianceState) -> dict:
    logger = logging.getLogger("ValidationNode")
    if state.get("extracted_json") is not None and isinstance(state["extracted_json"], dict):
        logger.info("Extraction validated successfully. Valid JSON dictionary recovered.")
        return {"steps_completed": ["validate_extraction"]}
    else:
        logger.warning("Extraction validation failed. No valid JSON recovered.")
        return {"extraction_error": "Validation Failed: The SLM failed to produce a valid JSON dictionary. Cannot proceed.", "steps_completed": ["validate_extraction"]}


def route_after_validation(state: ComplianceState) -> str:
    if state.get("extracted_json") is not None and isinstance(state["extracted_json"], dict):
        return "retrieve_all_context"
    return "handle_extraction_error"


def handle_extraction_error_node(state: ComplianceState) -> dict:
    logger = logging.getLogger("ErrorNode")
    error = state.get("extraction_error") or "Unknown analytical validation error."
    logger.error(f"Terminating pipeline due to prior error: {error}")
    return {
        "compliance_report": f"COMPLIANCE STATUS: ERROR\n\nPipeline terminated: {error}",
        "compliance_status": "ERROR",
        "steps_completed": ["handle_extraction_error"],
    }


def _query_collection_helper(model, collection, queries: List[str], n: int = N_RESULTS) -> str:
    all_chunks = []
    for query in queries:
        emb = model.encode(f"query: {query}", normalize_embeddings=True).tolist()
        results = collection.query(query_embeddings=[emb], n_results=n, include=["documents", "metadatas"])
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            source = meta.get("source", "unknown").replace(".pdf.txt", "")
            all_chunks.append(f"[{source}]\n{doc}")
    return "\n\n---\n\n".join(set(all_chunks))


def retrieve_all_context_node(state: ComplianceState) -> dict:
    logger = logging.getLogger("ContextRetrievalNode")
    logger.info("Dispatching parallel RAG inquiries over Law, Prospectuses, and Annual Reports...")

    bond = state["extracted_json"]
    model, law_col, prosp_col, ann_col = ResourceProvider.get()

    rate_type = bond.get("interest_rate_type", "")
    issuer = bond.get("issuer", "")
    
    law_queries = [
        f"Prospektpflicht {bond.get('document_type', '')} Endgültige Bedingungen Wertpapierprospektgesetz",
        f"{'Festverzinsliche' if rate_type == 'fixed' else 'Variabel verzinsliche' if rate_type == 'floating' else 'Stufenzins Fix-to-Float'} Schuldverschreibungen Regulierung",
        "Nachrangige Verbindlichkeiten Insolvenz Senior Haftungsreihenfolge",
    ]
    
    prosp_queries = [
        f"{issuer} Basisprospekt Schuldverschreibungen Emissionsbedingungen",
        f"{issuer} {'Festverzinsliche' if rate_type == 'fixed' else 'Stufenzins'} Zinszahlungbedingungen",
    ]
    
    ann_queries = [
        f"{issuer} Jahresabschluss Eigenkapitalquote Risikobericht",
        f"{issuer} Liquidität Rating Solvenz",
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        f_law = executor.submit(_query_collection_helper, model, law_col, law_queries)
        f_prosp = executor.submit(_query_collection_helper, model, prosp_col, prosp_queries)
        f_ann = executor.submit(_query_collection_helper, model, ann_col, ann_queries)
        
        law_ctx, prosp_ctx, ann_ctx = f_law.result(), f_prosp.result(), f_ann.result()

    logger.info(f"RAG retrieval complete. Recovered {len(law_ctx)} bytes (Law), {len(prosp_ctx)} bytes (Prosp), {len(ann_ctx)} bytes (Annual).")
    return {
        "law_context": law_ctx,
        "prospectus_context": prosp_ctx,
        "annual_context": ann_ctx,
        "steps_completed": ["retrieve_all_context"]
    }


async def _mcp_dispatch(issuer: str, rate_type: str) -> Tuple[str, str]:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SCRIPT_DIR.parent / "mcp" / "bafin_server.py")],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            try:
                bafin_res = await session.call_tool("get_bafin_warnings", arguments={"issuer_name": issuer})
                bafin_text = bafin_res.content[0].text
            except Exception as e:
                bafin_text = f"Warning: MCP BaFin service unreachable ({e})"
            
            euribor_text = "N/A (Fixed rate bond)"
            if rate_type in ["mixed", "floating"]:
                try:
                    eur_res = await session.call_tool("get_live_euribor_rate", arguments={"maturity": "3M"})
                    euribor_text = eur_res.content[0].text
                except Exception as e:
                    euribor_text = f"Warning: MCP Market Data service unreachable ({e})"
                
            return bafin_text, euribor_text


def fetch_live_data_node(state: ComplianceState) -> dict:
    logger = logging.getLogger("MCPNode")
    logger.info("Initializing Model Context Protocol (MCP) clients for live telemetry...")
    
    bond = state["extracted_json"]
    issuer = bond.get("issuer", "")
    rate_type = bond.get("interest_rate_type", "")
    
    bafin_text, euribor_text = asyncio.run(_mcp_dispatch(issuer, rate_type))
    
    logger.info("Live telemetry acquired successfully via MCP services.")
    return {
        "bafin_warnings": bafin_text,
        "euribor_rates": euribor_text,
        "steps_completed": ["fetch_live_data"]
    }


def run_compliance_check_node(state: ComplianceState) -> dict:
    agent = ComplianceAnalyzerAgent()
    try:
        content, status = agent.invoke(state, state.get("compliance_model", COMPLIANCE_MODEL))
        return {
            "compliance_report": content,
            "compliance_status": status,
            "steps_completed": ["run_compliance_check"],
        }
    except Exception as e:
        agent.logger.error(f"Compliance analysis execution failed: {e}")
        return {
            "compliance_report": f"COMPLIANCE STATUS: ERROR\n\nAgent error: {e}",
            "compliance_status": "ERROR",
            "steps_completed": ["run_compliance_check"],
        }


def format_report_node(state: ComplianceState) -> dict:
    logger = logging.getLogger("ReportingNode")
    logger.info("Generating finalized compliance report assembly...")

    status = state.get("compliance_status", "UNKNOWN")
    extraction = state.get("extracted_json", {})
    
    output = f"""
================================================================================
GERMAN FINANCIAL COMPLIANCE REPORT
================================================================================
Timestamp:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
Architecture:  LangGraph Orchestration
Trace:         {' -> '.join(state.get('steps_completed', []))}
================================================================================
OVERALL SYSTEM STATUS: [{status}]
================================================================================
"""
    if extraction:
        output += f"""
EXTRACTED FINANCIAL ENTITIES:
  Document Type:      {extraction.get('document_type', 'N/A')}
  Issuer:             {extraction.get('issuer', 'N/A')}
  ISIN:               {extraction.get('isin', 'N/A')}
  Instrument:         {extraction.get('instrument_name', 'N/A')}
  Currency:           {extraction.get('currency', 'N/A')}
  Total Volume:       {f"{extraction.get('total_volume'):,}" if isinstance(extraction.get('total_volume'), (int, float)) else extraction.get('total_volume', 'N/A')} {extraction.get('currency', '')} 
  Denomination:       {f"{extraction.get('denomination'):,}" if isinstance(extraction.get('denomination'), (int, float)) else extraction.get('denomination', 'N/A')}
  Issue Date:         {extraction.get('issue_date', 'N/A')}
  Maturity Date:      {extraction.get('maturity_date', 'N/A')}
  Interest Rate Type: {extraction.get('interest_rate_type', 'N/A')}

================================================================================
COMPLIANCE AGENT ANALYSIS:
{state.get('compliance_report', 'No report generated.')}
================================================================================
"""
    print("\n" + output)
    return {"steps_completed": ["format_report"]}


# ============================================================================
# ORCHESTRATOR GRAPH CONSTRUCTION
# ============================================================================
def compile_orchestrator() -> StateGraph:
    logger = logging.getLogger("Orchestrator")
    logger.info("Compiling StateGraph topology...")
    
    graph = StateGraph(ComplianceState)

    graph.add_node("extract_data", extract_data_node)
    graph.add_node("validate_extraction", validate_extraction_node)
    graph.add_node("handle_extraction_error", handle_extraction_error_node)
    graph.add_node("retrieve_all_context", retrieve_all_context_node)
    graph.add_node("fetch_live_data", fetch_live_data_node)
    graph.add_node("run_compliance_check", run_compliance_check_node)
    graph.add_node("format_report", format_report_node)

    graph.set_entry_point("extract_data")
    graph.add_edge("extract_data", "validate_extraction")
    graph.add_conditional_edges(
        "validate_extraction",
        route_after_validation,
        {
            "retrieve_all_context": "retrieve_all_context",
            "handle_extraction_error": "handle_extraction_error",
        },
    )
    graph.add_edge("handle_extraction_error", END)
    graph.add_edge("retrieve_all_context", "fetch_live_data")
    graph.add_edge("fetch_live_data", "run_compliance_check")
    graph.add_edge("run_compliance_check", "format_report")
    graph.add_edge("format_report", END)

    logger.info("StateGraph compiled successfully.")
    return graph.compile()


# ============================================================================
# EXECUTION ENTRY POINT
# ============================================================================
DEMO_TEXT = """
ENDGÜLTIGE BEDINGUNGEN vom 15. Oktober 2019.
Emittentin: Deutsche Pfandbriefbank AG, Unterschleißheim.
Wertpapierart: Fix-to-Float Schuldverschreibungen.
ISIN: DE000A2YN3J7. WKN: A2YN3J.
Gesamtnennbetrag: bis zu EUR 150.000.000.
Festgelegte Stückelung: EUR 1.000.
Ausgabetag: 22.10.2019. Rückzahlungstag: 22.10.2029.
Zinsen: Vom 22.10.2019 bis 21.10.2024 mit 0,875% p.a. (Festzinsperiode).
Danach variabel basierend auf 3-Monats-EURIBOR zzgl. 95 Basispunkte p.a.
Rang: nicht nachrangig.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangGraph Compliance Agent")
    parser.add_argument("file", nargs="?", help="Path to .txt document")
    parser.add_argument("--demo", action="store_true", help="Execute built-in demonstration dataset")
    parser.add_argument("--model", default=COMPLIANCE_MODEL, help="Target Compliance LLM")
    args = parser.parse_args()

    main_logger = logging.getLogger("Main")
    main_logger.info("Initializing Agent System Operations...")

    if args.demo:
        text = DEMO_TEXT
        main_logger.info("Loaded built-in demonstration dataset.")
    elif args.file:
        text = Path(args.file).read_text(encoding="utf-8", errors="ignore")
        main_logger.info(f"Loaded target document from path: {args.file}")
    else:
        main_logger.error("Invalid execution arguments. Use <file.txt> or --demo.")
        sys.exit(1)

    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        main_logger.info("Ollama inference engine status: ONLINE.")
    except requests.ConnectionError:
        main_logger.error("Ollama inference engine is offline. Start the service with 'ollama serve'.")
        sys.exit(1)

    app = compile_orchestrator()
    initial_state: ComplianceState = {
        "document_text": text,
        "compliance_model": args.model,
        "extracted_json": None,
        "extraction_error": None,
        "law_context": None,
        "prospectus_context": None,
        "annual_context": None,
        "bafin_warnings": None,
        "euribor_rates": None,
        "compliance_report": None,
        "compliance_status": None,
        "steps_completed": [],
    }

    main_logger.info("Triggering LangGraph execution pipeline...")
    final_state = app.invoke(initial_state)
    main_logger.info(f"Pipeline execution finalized. Status: {final_state.get('compliance_status', 'UNKNOWN')}")
