import json
import re
import os
from datetime import datetime
from huggingface_hub import HfApi

CATALOG_FILE = "models_catalog.json"
CHANGELOG_FILE = "changelog.md"

TRUSTED_ORGS = [
    "meta-llama", "Qwen", "deepseek-ai", "mistralai", 
    "google", "microsoft", "Alibaba-NLP"
]
TRUSTED_CONVERTERS = [
    "bartowski", "city96", "unsloth", "lmstudio-community", "NousResearch"
]

# Strict ranges prevent a 72B model from overwriting the 8B tier
TIERS = [
    {"name": "70B+", "min": 70, "max": 1000},
    {"name": "32B-69B", "min": 32, "max": 69.9},
    {"name": "12B-31B", "min": 12, "max": 31.9},
    {"name": "8B-11B", "min": 8, "max": 11.9},
    {"name": "3B-7B", "min": 3, "max": 7.9},
    {"name": "Sub-2B", "min": 0, "max": 2.9}
]

def parse_params(model_id):
    # Handles MoE (e.g., 8x7B -> 56B total weight for hardware calculations)
    moe_match = re.search(r'(\d+)x(\d+(?:\.\d+)?)b', model_id, re.IGNORECASE)
    if moe_match:
        return float(moe_match.group(1)) * float(moe_match.group(2))

    # Handles standard notation (e.g., 14B, 0.5B)
    std_match = re.search(r'(\d+(?:\.\d+)?)[Bb]', model_id)
    if std_match:
        return float(std_match.group(1))
    
    return None

def main():
    api = HfApi()
    
    # 1. Search text-generation, not just GGUF. Sort by downloads as a quality proxy.
    print("Querying Hugging Face for top text-generation models...")
    models = api.list_models(
        pipeline_tag="text-generation",
        sort="downloads",
        
        limit=300
    )
    
    tier_winners = {tier["name"]: None for tier in TIERS}
    
    # 2. Filter & Rank
    for model in models:
        author = model.id.split('/')[0]
        if author not in TRUSTED_ORGS and author not in TRUSTED_CONVERTERS:
            continue
            
        params = parse_params(model.id)
        if params is None:
            continue
            
        # 3. Range-based assignment
        for tier in TIERS:
            if tier["min"] <= params <= tier["max"]:
                # Because we sorted by downloads, the first valid model we hit is the "best"
                # (In v2, this is where you merge LMSYS Elo scores instead of relying purely on downloads)
                if tier_winners[tier["name"]] is None:
                    
                    tier_winners[tier["name"]] = {
                        "family": author,
                        "model": model.id.split('/')[-1],
                        "params": params,
                        "context": 131072 if "128k" in model.id.lower() else 32768, # Placeholder logic
                        "hf": model.id,
                        "downloads": getattr(model, 'downloads', 0),
                        "release": getattr(model, 'lastModified', datetime.now()).strftime("%Y-%m-%d")
                    }
                break
                
    # 4. Compare with existing catalog & generate Changelog
    with open(CATALOG_FILE, 'r') as f:
        catalog = json.load(f)
        
    changelog_lines = ["### 🤖 Model Updates\n"]
    updated = False
    
    for i, tier in enumerate(catalog.get("model_tiers", [])):
        tier_name = TIERS[i]["name"]
        new_winner = tier_winners.get(tier_name)
        
        if new_winner and new_winner["hf"] != tier.get("hf"):
            old_model = tier.get("model", "None")
            changelog_lines.append(f"- **{tier_name} tier**: `{old_model}` ↓ `{new_winner['model']}` (+{new_winner['downloads']} downloads)")
            # Update the JSON structure with the rich metadata
            catalog["model_tiers"][i] = new_winner
            updated = True
            
    if updated:
        catalog["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        with open(CATALOG_FILE, 'w') as f:
            json.dump(catalog, f, indent=2)
            
        with open(CHANGELOG_FILE, 'w') as f:
            f.write("\n".join(changelog_lines))
        print("Catalog updated. Changelog generated.")
    else:
        print("No updates required.")

if __name__ == "__main__":
    main()
