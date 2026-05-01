import os
import json
import time
import google.generativeai as genai

# Setup your Teacher Model
genai.configure(api_key="AIzaSyDkHSPI1e6f23oNFijzctE9U3q9-YQQ9MU") #type: ignore
model = genai.GenerativeModel('gemini-2.5-flash') #type: ignore

SYSTEM_PROMPT = "You are a specialized German financial parser. Extract entities into valid JSON."

def get_json_from_gemini(text_block):
    prompt = f"""
    Analyze the following German financial text. Extract the key data into a strict JSON object.
    
    Rules:
    1. Convert all dates to YYYY-MM-DD.
    2. Convert amounts to floats.
    3. If a field is not found in the text, use null.
    4. Output ONLY raw JSON.
    
    Keys required: document_type, issuer, instrument_name, isin, currency, total_volume, denomination, issue_date, maturity_date, interest_rate_type.
    
    TEXT:
    {text_block}
    """
    try:
        response = model.generate_content(prompt)
        clean_json_str = response.text.strip().removeprefix("```json").removesuffix("```").strip()
        json_data = json.loads(clean_json_str)
        
        # If it found absolutely nothing, skip it to keep training data clean
        if all(value is None for value in json_data.values()):
            return None
            
        return clean_json_str
    except Exception as e:
        print(f"Skipping due to error: {e}")
        return None

def process_corpus_head(input_folder, output_jsonl_file, max_files=5):
    """Reads the FIRST part of TXT files and distills them into the training file."""
    processed_count = 0
    
    with open(output_jsonl_file, "a", encoding="utf-8") as outfile:
        for filename in os.listdir(input_folder):
            if not filename.endswith(".txt"):
                continue
                
            if processed_count >= max_files:
                break
                
            filepath = os.path.join(input_folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                full_text = f.read()
            
            truncated_text = full_text[:5000]
            
            print(f"Extracting data from {filename}...")
            assistant_json = get_json_from_gemini(truncated_text)
            
            if assistant_json:
                training_row = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Extract data from this text:\n\n{truncated_text}"},
                        {"role": "assistant", "content": assistant_json}
                    ]
                }
                outfile.write(json.dumps(training_row, ensure_ascii=False) + "\n")
                
            time.sleep(1) 
            processed_count += 1
            print(f"Finished file: {filename} ({processed_count}/{max_files})")

# --- EXECUTION ---
INPUT_DIR = r"corpus_safe\safe_corpus\txt\Final_terms" 
OUTPUT_FILE = "train.jsonl"

print("Starting Document-Level Data Generation...")
process_corpus_head(INPUT_DIR, OUTPUT_FILE, max_files=1000)
print("Done!")
