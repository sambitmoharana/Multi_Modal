import os
from pathlib import Path
from typing import Dict, List, Set

# Local .env parser fallback
def _load_dotenv():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        env_path = Path(".env").resolve()
    if env_path.exists():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ[k.strip()] = v.strip().strip("'\"")
        except Exception:
            pass

_load_dotenv()

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = PROJECT_ROOT.parent / "dataset"

# Allowed Vocabularies
ALLOWED_CLAIM_STATUSES: Set[str] = {"supported", "contradicted", "not_enough_information"}

ALLOWED_ISSUE_TYPES: Set[str] = {
    "dent", "scratch", "crack", "glass_shatter", "broken_part",
    "missing_part", "torn_packaging", "crushed_packaging",
    "water_damage", "stain", "none", "unknown"
}

ALLOWED_OBJECT_PARTS: Dict[str, Set[str]] = {
    "car": {
        "front_bumper", "rear_bumper", "door", "hood", "windshield",
        "side_mirror", "headlight", "taillight", "fender",
        "quarter_panel", "body", "unknown"
    },
    "laptop": {
        "screen", "keyboard", "trackpad", "hinge", "lid",
        "corner", "port", "base", "body", "unknown"
    },
    "package": {
        "box", "package_corner", "package_side", "seal",
        "label", "contents", "item", "unknown"
    }
}

ALLOWED_RISK_FLAGS: Set[str] = {
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required"
}

ALLOWED_SEVERITIES: Set[str] = {"none", "low", "medium", "high", "unknown"}

# Image Processing Quality Heuristics
# Blurriness: variance of Laplacian below threshold
BLUR_THRESHOLD = 50.0

# Brightness: average brightness below LOW_LIGHT or above GLARE
LOW_LIGHT_THRESHOLD = 40
GLARE_THRESHOLD = 240

# Setup logging configuration format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
