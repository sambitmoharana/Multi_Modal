import os
import logging
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import cv2
import numpy as np
from PIL import Image

from src import config

logger = logging.getLogger("damage_claim.utils")

def resolve_path(path_str: str, base_dir: Path) -> Path:
    """Resolves a file path against a base directory or absolute workspace root."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    
    # Try resolving relative to the provided base_dir
    resolved = (base_dir / path).resolve()
    if resolved.exists():
        return resolved
        
    # Try resolving relative to the workspace root directory (two levels up from src)
    resolved_root = (config.PROJECT_ROOT.parent / path).resolve()
    return resolved_root

def parse_image_paths(paths_str: str) -> List[str]:
    """Parses semicolon-separated image paths from CSV cell."""
    if not paths_str or pd.isna(paths_str):
        return []
    return [p.strip() for p in str(paths_str).split(";") if p.strip()]

def load_claims_data(file_path: str, base_dir: Path) -> pd.DataFrame:
    """Loads claims CSV file and validates required columns."""
    resolved = resolve_path(file_path, base_dir)
    logger.info(f"Loading claims from {resolved}")
    if not resolved.exists():
        raise FileNotFoundError(f"Claims file not found: {resolved}")
    return pd.read_csv(resolved)

def load_user_history(file_path: str, base_dir: Path) -> pd.DataFrame:
    """Loads user history CSV file."""
    resolved = resolve_path(file_path, base_dir)
    logger.info(f"Loading user history from {resolved}")
    if not resolved.exists():
        raise FileNotFoundError(f"User history file not found: {resolved}")
    return pd.read_csv(resolved)

def load_evidence_requirements(file_path: str, base_dir: Path) -> pd.DataFrame:
    """Loads evidence requirements CSV file."""
    resolved = resolve_path(file_path, base_dir)
    logger.info(f"Loading evidence requirements from {resolved}")
    if not resolved.exists():
        raise FileNotFoundError(f"Evidence requirements file not found: {resolved}")
    return pd.read_csv(resolved)

def run_image_heuristics(image_path: str) -> List[str]:
    """
    Perform local image analysis using OpenCV and Pillow.
    Detects blurriness via Laplacian variance and lighting issues.
    """
    flags = []
    if not os.path.exists(image_path):
        return ["damage_not_visible"]

    try:
        # Load using OpenCV
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"OpenCV failed to read image at {image_path}. Flagging blurry/unusable.")
            return ["blurry_image"]

        # Dimensions / obstruction check
        h, w, c = img.shape
        if h < 200 or w < 200:
            flags.append("cropped_or_obstructed")

        # Convert to grayscale for statistical metrics
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Blurriness detector (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < config.BLUR_THRESHOLD:
            logger.info(f"Image {image_path} Laplacian variance: {laplacian_var:.2f} < {config.BLUR_THRESHOLD}. Flagging blurry.")
            flags.append("blurry_image")

        # 2. Brightness checks (low light or glare)
        mean_brightness = np.mean(gray)
        if mean_brightness < config.LOW_LIGHT_THRESHOLD:
            logger.info(f"Image {image_path} mean brightness: {mean_brightness:.2f} < {config.LOW_LIGHT_THRESHOLD}. Flagging low light.")
            flags.append("low_light_or_glare")
        elif mean_brightness > config.GLARE_THRESHOLD:
            logger.info(f"Image {image_path} mean brightness: {mean_brightness:.2f} > {config.GLARE_THRESHOLD}. Flagging glare.")
            flags.append("low_light_or_glare")

    except Exception as e:
        logger.error(f"Error running image heuristics on {image_path}: {e}")
        flags.append("blurry_image")

    return flags
