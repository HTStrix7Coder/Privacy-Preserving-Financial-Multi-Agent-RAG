import sys
import json
import time
import pandas as pd
from pathlib import Path
import streamlit as st

# Ensure the rag module can be imported
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR))

from rag.langgraph_agent import compile_orchestrator, ComplianceState, COMPLIANCE_MODEL

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="GDPR_Sovereign_System | Core",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Enterprise Fintech Styling & German Accent
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Subtle German Flag Top Accent Bar */
    .stApp > header {
        background-color: transparent;
        border-top: 5px solid;
        border-image: linear-gradient(90deg, #111111 33%, #FF0000 33%, #FF0000 66%, #FFCC00 66%) 1;
    }
    
    .status-compliant {
        color: #065F46; background-color: #D1FAE5; border-left: 6px solid #059669; padding: 15px; font-size: 1.1rem; font-weight: 600; border-radius: 4px;
    }
    .status-review {
        color: #92400E; background-color: #FEF3C7; border-left: 6px solid #D97706; padding: 15px; font-size: 1.1rem; font-weight: 600; border-radius: 4px;
    }
    .status-non-compliant {
        color: #991B1B; background-color: #FEE2E2; border-left: 6px solid #DC2626; padding: 15px; font-size: 1.1rem; font-weight: 600; border-radius: 4px;
    }
    
    .metric-container {
        padding: 10px;
        background-color: rgba(128,128,128,0.05);
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 6px;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-label { font-size: 0.85rem; opacity: 0.7; font-weight: 500; text-transform: uppercase;}
    .metric-value { font-size: 1.4rem; font-weight: 700; margin-top: 5px;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# UI HEADER & METRICS
# ============================================================================
st.markdown("<h1>GDPR_Sovereign_System <span style='font-size: 0.5em; opacity: 0.5;'>v2.0</span></h1>", unsafe_allow_html=True)
st.markdown("##### Souveräne Verifizierung via Fine-Tuning und Multi-Agenten RAG (Sovereign Verification via Fine-Tuned Extraction and Multi-Agent RAG)")
st.markdown("---")

# Top Level Stats row
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown('<div class="metric-container"><div class="metric-label">Extraktions-Engines (Extractor)</div><div class="metric-value">Qwen3.5-2B (Fine-Tuned)</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric-container"><div class="metric-label">Vektor-Datenbank (Vector DB)</div><div class="metric-value">260,638 Chunks</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-container"><div class="metric-label">Orchestrierung (Orchestration)</div><div class="metric-value">LangGraph (Lokal)</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown('<div class="metric-container"><div class="metric-label">Live-Telemetrie (Live Data)</div><div class="metric-value">Aktiv (MCP)</div></div>', unsafe_allow_html=True)


# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
with st.sidebar:
    flag_path = str(SCRIPT_DIR / "Flag_of_Germany.svg.png")
    st.image(flag_path, width=70)
    st.subheader("⚙️ Systemkonfiguration")
    target_model = st.selectbox(
        "Compliance Reasoner Model",
        [COMPLIANCE_MODEL, "llama3:8b", "mistral"],
        index=0,
        help="Das sekundäre LLM für rechtliche Referenzierung (Secondary LLM for legal cross-referencing)."
    )
    
    st.markdown("---")
    st.markdown("### 📄 Dokumenteneingabe")
    input_method = st.radio(
        "Eingabequelle (Input Source)",
        ["Beispieldokument (1-Click Sample)", "Text einfügen (Paste)", "Datei hochladen (.txt)"]
    )

    SAMPLE_DOCUMENTS = {
        "1. Pfandbriefbank AG (Fix-to-Float — Triggers Euribor MCP)": (
            "ENDGÜLTIGE BEDINGUNGEN vom 15. Oktober 2019.\n"
            "Emittentin: Deutsche Pfandbriefbank AG, Unterschleißheim.\n"
            "Wertpapierart: Fix-to-Float Schuldverschreibungen.\n"
            "ISIN: DE000A2YN3J7. WKN: A2YN3J.\n"
            "Gesamtnennbetrag: bis zu EUR 150.000.000.\n"
            "Festgelegte Stückelung: EUR 1.000.\n"
            "Ausgabetag: 22.10.2019. Rückzahlungstag: 22.10.2029.\n"
            "Zinsen: Vom 22.10.2019 bis 21.10.2024 mit 0,875% p.a. (Festzinsperiode).\n"
            "Danach variabel basierend auf 3-Monats-EURIBOR zzgl. 95 Basispunkte p.a.\n"
            "Rang: nicht nachrangig."
        ),
        "2. Commerzbank AG (Senior Preferred — Fixed 3.50%)": (
            "ENDGÜLTIGE BEDINGUNGEN vom 12. Mai 2024.\n"
            "Emittentin: Commerzbank AG, Frankfurt am Main.\n"
            "Wertpapierart: Festverzinsliche Schuldverschreibungen (Fixed Rate Notes).\n"
            "ISIN: DE000CBK1234. WKN: CBK123.\n"
            "Gesamtnennbetrag: EUR 500.000.000.\n"
            "Festgelegte Stückelung: EUR 100.000.\n"
            "Ausgabetag: 15.05.2024. Rückzahlungstag: 15.05.2029.\n"
            "Zinssatz: 3,50% p.a. zahlbar jährlich nachträglich.\n"
            "Rang: Nicht nachrangig (Senior Preferred)."
        ),
        "3. Deutsche Bank AG (Subordinated / Tier 2 Bail-in Risk)": (
            "ENDGÜLTIGE BEDINGUNGEN vom 04. September 2022.\n"
            "Emittentin: Deutsche Bank AG, Frankfurt am Main.\n"
            "Wertpapierart: Nachrangige Schuldverschreibungen (Tier 2 Subordinated Notes).\n"
            "ISIN: DE000DB9XYZ4. WKN: DB9XYZ.\n"
            "Gesamtnennbetrag: EUR 750.000.000.\n"
            "Festgelegte Stückelung: EUR 100.000.\n"
            "Ausgabetag: 10.09.2022. Rückzahlungstag: 10.09.2032.\n"
            "Zinssatz: 5,25% p.a. bis zum ersten Zinsanpassungstag.\n"
            "Rang: Nachrangig gemäß § 10 KWG (Bail-in fähig)."
        )
    }

    st.markdown("---")
    with st.expander("Technische Methodik (Specs)"):
        st.caption("Core Extractor: Qwen3.5-2B (PEFT/LoRA Finetuned)")
        st.caption("Schema: Strict Financial Entity JSON Output")
        st.caption("Inference Engine: Ollama (Local Streaming API)")
        st.caption("Quantization: 8-bit Symmetric (Q8_0 GGUF)")
        st.caption("Reasoning Engine: Qwen2.5-7B-Instruct")
        st.caption("Data Layer: Multi-Corpus RAG (260,638 Chunks)")
        st.caption("Logic Control: LangGraph Directed Acyclic Graph (DAG)")
        st.caption("Real-time Hooks: Model Context Protocol (FastMCP)")
        st.caption("Target Policy: 100% Local / GDPR / Air-Gapped")

# ============================================================================
# MAIN APPLICATION BODY
# ============================================================================
doc_text = ""
if "Beispieldokument" in input_method:
    selected_sample = st.sidebar.selectbox(
        "Vordefiniertes Testdokument wählen:",
        list(SAMPLE_DOCUMENTS.keys())
    )
    doc_text = st.sidebar.text_area(
        "Dokumentenvorschau (Editierbar):",
        value=SAMPLE_DOCUMENTS[selected_sample],
        height=280
    )
elif "Datei hochladen (.txt)" in input_method:
    uploaded_file = st.sidebar.file_uploader("Upload 'Endgültige Bedingungen' (Final Terms)", type=["txt"])
    if uploaded_file is not None:
        doc_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
else:
    doc_text = st.sidebar.text_area("Unstrukturierten Dokumententext einfügen (Paste text)...", height=300, placeholder="ENDGÜLTIGE BEDINGUNGEN vom 15. Oktober 2019...")

if not doc_text.strip():
    st.info("👈 Bitte legen Sie in der Seitenleiste ein Finanzdokument an, um die Compliance-Prüfung zu starten.")
    st.stop()

if st.button("▶ Multi-Agenten-Verifizierungsprozess Starten (Execute Pipeline)", type="primary"):
    
    # Progress UI
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner("LangGraph-Orchestrierungsstatus wird kompiliert..."):
        app = compile_orchestrator()
        
        initial_state: ComplianceState = {
            "document_text": doc_text,
            "compliance_model": target_model,
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

    # Stream execution
    nodes = ["extract_data", "validate_extraction", "retrieve_all_context", "fetch_live_data", "run_compliance_check", "format_report"]
    current_state = initial_state.copy()
    
    for output in app.stream(initial_state):
        for key, value in output.items():
            current_state.update(value)
            
            # Update Progress Bar elegantly
            if key in nodes:
                idx = nodes.index(key) + 1
                progress = int((idx / len(nodes)) * 100)
                progress_bar.progress(progress)
                status_text.caption(f"System-Checkpoint Erreicht: **{key.upper()}**")
                
    time.sleep(0.5) # UX Pause
    status_text.empty()
    progress_bar.empty()

    final_state = current_state

    # ============================================================================
    # RESULTS DASHBOARD
    # ============================================================================
    
    st.markdown("### Verifizierungsergebnisse (Verification Results Dashboard)")
    st.markdown("---")
    
    # Top Status Banner
    status = final_state.get('compliance_status', 'UNKNOWN')
    if status == "COMPLIANT":
        st.markdown(f'<div class="status-compliant">SYSTEMURTEIL: <strong>{status} (Zugelassen)</strong></div>', unsafe_allow_html=True)
    elif status == "REVIEW REQUIRED":
        st.markdown(f'<div class="status-review">SYSTEMURTEIL: <strong>{status} (Prüfung Erforderlich)</strong> — Manuelle Aufsicht angeraten</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-non-compliant">SYSTEMURTEIL: <strong>{status} (Nicht Konform)</strong> — Regulatorischer Verstoß erkannt</div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2-Column Split for Data vs Report
    left_col, right_col = st.columns([1, 1.4], gap="large")
    
    with left_col:
        st.subheader("I. Extrahierte Entitäten")
        
        # Always attempt to render the data payload so the user can debug what the LLM structured, even on error.
        if final_state.get("extracted_json"):
            extracted = final_state["extracted_json"]
            safe_items = [(str(k), str(v) if v is not None else "N/A") for k, v in extracted.items()]
            df = pd.DataFrame(safe_items, columns=["Entity Key", "Identifizierter Wert (Value)"])
            st.dataframe(df, hide_index=True, use_container_width=True)
            
        if final_state.get("extraction_error"):
            st.error(f"Fehler (Pipeline Alert): {final_state['extraction_error']}")
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("II. Live-Telemetrie (MCP Constraints)")
        
        # Show MCP results dynamically
        bafin = final_state.get("bafin_warnings") or ""
        if "Clean" in bafin:
            st.success(f"**BaFin API Check:** {bafin}")
        elif bafin:
            st.warning(f"**BaFin API Check:** {bafin}")
            
        mkt = final_state.get("euribor_rates") or ""
        if mkt:
            st.info(f"**Markt API Check:** {mkt}")
        
        with st.expander("Originales Quelldokument anzeigen (View Source)"):
            st.text(doc_text)

    with right_col:
        st.subheader("III. Regulatorische Analyse (Multi-Corpus)")
        report = final_state.get("compliance_report", "")
        
        if report:
            # Clean up raw artifacts
            clean_report = report.replace(f"COMPLIANCE STATUS: {status}", "").strip()
            
            # Make the subtitles pop
            clean_report = clean_report.replace("KEY FINDINGS:", "### Zentrale Erkenntnisse (Key Findings)")
            clean_report = clean_report.replace("LEGAL & PROSPECTUS REFERENCES:", "### Gesetzliche & Prospekt-Validierung (Statutory References)")
            clean_report = clean_report.replace("LIVE DATA OVERSIGHT:", "### Live-Marktprüfung (Market Auditing)")
            clean_report = clean_report.replace("RECOMMENDATION:", "### Strategische Empfehlung (Strategic Recommendation)")

            st.markdown(clean_report)
        else:
            st.error("Es wurde kein Compliance-Bericht generiert (No compliance report generated).")
