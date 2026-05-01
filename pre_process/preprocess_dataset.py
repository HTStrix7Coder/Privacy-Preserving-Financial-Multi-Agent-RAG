import json
import re
import os

def clean_text(text):
    """Removes page markers and normalizes whitespace."""
    # Remove common German/English page markers: "- 2 -", "Page 1", "Seite 1"
    text = re.sub(r'\n-?\s*\d+\s*-?\n', '\n', text)
    text = re.sub(r'\n(Page|Seite)\s*\d+\s*\n', '\n', text, flags=re.IGNORECASE)
    
    # Collapse multiple blank lines into one
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Collapse multiple horizontal spaces but preserve single newlines
    text = re.sub(r'[^\S\n]{2,}', ' ', text)
    
    return text.strip()

def normalize_labels(content):
    """Standardizes JSON keys and values in the assistant's extraction matching core categories."""
    try:
        data = json.loads(content)
        
        # 1. Normalize interest_rate_type
        raw_rate = str(data.get("interest_rate_type", "")).lower().strip()
        
        if any(kw in raw_rate for kw in ["mixed", "kombination", "fixed to floating", "fixed to reset"]):
            data["interest_rate_type"] = "mixed"
        elif any(kw in raw_rate for kw in ["stepped", "step-up", "step up", "stepup", "stufenzins", "stufenverzinsung"]):
            data["interest_rate_type"] = "stepped"
        elif any(kw in raw_rate for kw in ["capped", "floored", "collared", "corridor", "cap", "floor", "range"]):
            data["interest_rate_type"] = "capped_floored_floating"
        elif any(kw in raw_rate for kw in ["floating", "variable", "variabel", "euribor", "adjustable", "floater", "cms"]):
            data["interest_rate_type"] = "floating"
        elif any(kw in raw_rate for kw in ["fixed", "festzins", "fixzins", "feste", "festverzinslich", "festverzinsliche", "festverzinsung", "festzins-anleihe"]):
            data["interest_rate_type"] = "fixed"
        elif any(kw in raw_rate for kw in ["zero", "nullkupon", "discount"]):
            data["interest_rate_type"] = "zero"
        elif any(kw in raw_rate for kw in ["inflation"]):
            data["interest_rate_type"] = "inflation_linked"
        elif "none" in raw_rate or not raw_rate:
            data["interest_rate_type"] = None
        else:
            # Fallback to a clean lowercase version if no keyword matches
            data["interest_rate_type"] = raw_rate if raw_rate else None

        # 2. Normalize document_type
        raw_doc = str(data.get("document_type", "")).lower().strip()
        
        if any(kw in raw_doc for kw in ["final terms", "endgültige bedingungen", "endgültige angebotsbedingungen", "endgültige anleihebedingungen", "issue terms", "pricing supplement", "angebot", "term sheet", "konditionenblatt", "terms of issue", "issuance terms", "final bond terms"]):
            data["document_type"] = "Final Terms"
        elif any(kw in raw_doc for kw in ["prospectus", "wertpapierprospekt", "offering circular", "listing prospectus", "prospekt"]):
            data["document_type"] = "Prospectus"
        elif any(kw in raw_doc for kw in ["emissionsbedingungen", "terms and conditions", "anleihebedingungen", "bond conditions", "security data sheet", "conditions", "satzung", "bedingungen", "bearer bond terms", "bond terms", "bond"]):
            data["document_type"] = "Terms and Conditions"
        elif any(kw in raw_doc for kw in ["globalurkunde", "sammelurkunde", "global note", "global certificate", "global bearer note", "inhaberglobalurkunde"]):
            data["document_type"] = "Global Note"
        elif any(kw in raw_doc for kw in ["pfandbrief", "hypothekenpfandbrief", "covered bond"]):
             data["document_type"] = "Pfandbrief"
        elif any(kw in raw_doc for kw in ["inhaberschuldverschreibung", "schuldverschreibung", "anleihe", "notes", "landesschatzanweisung"]):
             data["document_type"] = "Bond"
        else:
            # Fallback to "Other" or title case for unrecognized types
            data["document_type"] = raw_doc.title() if raw_doc else "Other"
            
        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return content

def preprocess(input_file, output_file):
    print(f"Cleaning and normalizing {input_file} -> {output_file}...")
    processed_count = 0
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            row = json.loads(line)
            
            for msg in row["messages"]:
                if msg["role"] == "user":
                    msg["content"] = clean_text(msg["content"])
                elif msg["role"] == "assistant":
                    msg["content"] = normalize_labels(msg["content"])
            
            f_out.write(json.dumps(row, ensure_ascii=False) + '\n')
            processed_count += 1
            
    print(f"Done! Processed {processed_count} rows.")

if __name__ == "__main__":
    preprocess("train.jsonl", "train_cleaned.jsonl")
