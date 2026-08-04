import json
import re
from datetime import datetime
from huggingface_hub import HfApi

# 1. Configuration
CATALOG_FILE = "models_catalog.json"
TRUSTED_AUTHORS = ["bartowski", "city96", "Qwen", "meta-llama"]
# Regex to find parameter counts like "70B", "8x7b", "3.8B", "0.5B"
PARAM_REGEX = re.compile(r'(\d+(?:\.\d+)?(?:x\d+)?)[Bb]')

def load_catalog():
    with open(CATALOG_FILE, 'r') as f:
        return json.load(f)

def save_catalog(data):
    with open(CATALOG_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def extract_params(model_name):
    match = PARAM_REGEX.search(model_name)
    if match:
        raw_val = match.group(1)
        # Handle MoE models like 8x7B (simplified by just taking the base param size)
        if 'x' in raw_val.lower():
            parts = raw_val.lower().split('x')
            return float(parts[0]) * float(parts[1])
        return float(raw_val)
    return None

def main():
    api = HfApi()
    catalog = load_catalog()
    
    # 2. Fetch top trending GGUF models
    print("Fetching trending GGUF models from Hugging Face...")
    models = api.list_models(
        tags="gguf", 
        sort="trending_score", 
        direction=-1, 
        limit=50
    )
    
    # Filter by trusted authors
    trusted_models = [m for m in models if m.author in TRUSTED_AUTHORS]
    
    updated = False
    
    # 3. Match models to tiers
    for tier in catalog["model_tiers"]:
        min_params = tier["min_params_billion"]
        
        # Find the highest trending model that fits this tier
        for model in trusted_models:
            params = extract_params(model.id)
            if params is None:
                continue
                
            # Logic: If it meets the min_params for this tier, but isn't big enough for the next tier up
            # (Assuming tiers are sorted highest to lowest in your JSON)
            if params >= min_params:
                new_example = f"{model.id.split('/')[-1]} ({params}B)"
                
                # If we found a trending model different from the current example, stage it
                if new_example not in tier["example"]:
                    print(f"Update found for {min_params}B+ tier: {new_example}")
                    tier["example"] = new_example
                    updated = True
                break # Move to the next tier once we fill this one

    # 4. Save if changes were made
    if updated:
        catalog["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        save_catalog(catalog)
        print("Catalog updated successfully.")
    else:
        print("No updates needed. Catalog is current.")

if __name__ == "__main__":
    main()
