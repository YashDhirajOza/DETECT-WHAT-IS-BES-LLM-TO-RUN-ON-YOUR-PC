import json
import logging
import os
import re
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional
from huggingface_hub import HfApi

# -----------------------------------------------------------------------------
# Configuration & Logging Setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

CATALOG_FILE = "models_catalog.json"
CHANGELOG_FILE = "changelog.md"

# Case-insensitive trusted author sets
TRUSTED_ORGS = {
    "meta-llama",
    "qwen",
    "deepseek-ai",
    "mistralai",
    "google",
    "microsoft",
    "alibaba-nlp",
}

TRUSTED_CONVERTERS = {
    "bartowski",
    "city96",
    "unsloth",
    "lmstudio-community",
    "nousresearch",
}

# Non-overlapping parameter ranges (in Billions)
TIERS = [
    {"name": "70B+", "min": 70.0, "max": 1000.0},
    {"name": "32B-69B", "min": 32.0, "max": 69.9},
    {"name": "12B-31B", "min": 12.0, "max": 31.9},
    {"name": "8B-11B", "min": 8.0, "max": 11.9},
    {"name": "3B-7B", "min": 3.0, "max": 7.9},
    {"name": "Sub-2B", "min": 0.0, "max": 2.9},
]

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def parse_params(model_id: str) -> Optional[float]:
    """Parses parameter counts in billions from model ID (handles MoE and standard notation)."""
    # MoE notation like 8x7B -> 56B total params
    moe_match = re.search(r'(\d+)x(\d+(?:\.\d+)?)b', model_id, re.IGNORECASE)
    if moe_match:
        return float(moe_match.group(1)) * float(moe_match.group(2))

    # Standard notation like 70B, 14B, 0.5B
    std_match = re.search(r'(\d+(?:\.\d+)?)[Bb]', model_id)
    if std_match:
        return float(std_match.group(1))

    return None


def extract_release_date(last_modified: Any) -> str:
    """Safely extracts YYYY-MM-DD from datetime object, ISO string, or NoneType."""
    if isinstance(last_modified, datetime):
        return last_modified.strftime("%Y-%m-%d")
    elif isinstance(last_modified, str):
        return last_modified.split("T")[0]
    return datetime.now().strftime("%Y-%m-%d")


def atomic_write_json(file_path: str, data: Dict[str, Any]) -> None:
    """Writes JSON atomically using a temporary file to avoid corrupting data on interrupt."""
    dir_name = os.path.dirname(os.path.abspath(file_path)) or "."
    with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tf:
        json.dump(data, tf, indent=2)
        temp_name = tf.name
    os.replace(temp_name, file_path)


# -----------------------------------------------------------------------------
# Core Pipeline Functions
# -----------------------------------------------------------------------------
def fetch_trending_models(api: HfApi, limit: int = 300) -> List[Any]:
    """Queries Hugging Face API for top text-generation models sorted by downloads."""
    logging.info(f"Querying Hugging Face for top {limit} text-generation models...")
    try:
        return list(
            api.list_models(
                pipeline_tag="text-generation",
                sort="downloads",
                limit=limit
            )
        )
    except Exception as e:
        logging.error(f"Failed to query Hugging Face API: {e}")
        return []


def get_model_metadata(api: HfApi, model_obj: Any, params: float) -> Optional[Dict[str, Any]]:
    """Fetches rich model information with fallback error handling."""
    model_id = model_obj.id
    try:
        info = api.model_info(model_id)
        
        # Safe attribute extraction
        last_mod = getattr(info, "lastModified", None) or getattr(model_obj, "lastModified", None)
        release_date = extract_release_date(last_mod)
        downloads = getattr(info, "downloads", None) or getattr(model_obj, "downloads", 0) or 0
        
        # Context window estimate
        model_id_lower = model_id.lower()
        if "128k" in model_id_lower or "131k" in model_id_lower:
            context = 131072
        elif "32k" in model_id_lower:
            context = 32768
        elif "16k" in model_id_lower:
            context = 16384
        else:
            context = 32768

        author = model_id.split("/")[0]

        return {
            "family": author,
            "model": model_id.split("/")[-1],
            "params": params,
            "context": context,
            "hf": model_id,
            "downloads": downloads,
            "release": release_date
        }
    except Exception as e:
        logging.warning(f"Failed to fetch detailed metadata for {model_id}: {e}")
        return None


def select_best_models(api: HfApi, models: List[Any]) -> Dict[str, Optional[Dict[str, Any]]]:
    """Filters models by trusted author/converters and assigns them to parameter tiers."""
    tier_winners: Dict[str, Optional[Dict[str, Any]]] = {tier["name"]: None for tier in TIERS}

    for model in models:
        author = model.id.split("/")[0].lower()
        
        # Case-insensitive trust verification
        if author not in TRUSTED_ORGS and author not in TRUSTED_CONVERTERS:
            continue

        params = parse_params(model.id)
        if params is None:
            continue

        # Range-based tier matching
        for tier in TIERS:
            if tier["min"] <= params <= tier["max"]:
                if tier_winners[tier["name"]] is None:
                    metadata = get_model_metadata(api, model, params)
                    if metadata:
                        tier_winners[tier["name"]] = metadata
                        logging.info(f"Selected winner for {tier['name']} tier: {metadata['hf']}")
                break

    return tier_winners


def update_catalog_and_changelog(tier_winners: Dict[str, Optional[Dict[str, Any]]]) -> bool:
    """Updates the catalog JSON and generates changelog.md if changes occurred."""
    if not os.path.exists(CATALOG_FILE):
        logging.error(f"Catalog file '{CATALOG_FILE}' not found.")
        return False

    with open(CATALOG_FILE, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    changelog_lines = ["### 🤖 Model Catalog Updates\n"]
    updated = False

    for i, tier in enumerate(catalog.get("model_tiers", [])):
        tier_name = TIERS[i]["name"]
        new_winner = tier_winners.get(tier_name)

        if new_winner and new_winner["hf"] != tier.get("hf"):
            old_model = tier.get("model", "None")
            changelog_lines.append(
                f"- **{tier_name} tier**: `{old_model}` ↓ `{new_winner['model']}` "
                f"(+{new_winner['downloads']:,} downloads)"
            )
            catalog["model_tiers"][i] = new_winner
            updated = True

    if updated:
        catalog["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        
        # Atomic write for JSON
        atomic_write_json(CATALOG_FILE, catalog)
        
        # Write changelog for GitHub Action PR Body
        with open(CHANGELOG_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(changelog_lines))
            
        logging.info("Catalog updated successfully. Changelog generated.")
        return True
    
    logging.info("No updates required. Catalog is up to date.")
    return False


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
def main() -> None:
    api = HfApi()
    models = fetch_trending_models(api, limit=300)
    if not models:
        logging.error("No models retrieved from Hugging Face. Aborting.")
        return

    tier_winners = select_best_models(api, models)
    update_catalog_and_changelog(tier_winners)


if __name__ == "__main__":
    main()
