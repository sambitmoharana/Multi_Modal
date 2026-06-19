import os
import logging
from pathlib import Path
from typing import List, Set

from src.models import BaseVisionModel, ImageAnalysis, ClaimAnalysis
from src.utils import run_image_heuristics, resolve_path
from src import config

logger = logging.getLogger("damage_claim.image_analyzer")

class ImageAnalyzer:
    """Orchestrates image quality checks and API model inference for image-level metrics."""
    def __init__(self, vision_model: BaseVisionModel, dataset_dir: Path):
        self.vision_model = vision_model
        self.dataset_dir = dataset_dir

    def analyze(
        self,
        image_path_str: str,
        claim_analysis: ClaimAnalysis,
        claim_object: str
    ) -> ImageAnalysis:
        logger.info(f"Analyzing image: {image_path_str}")
        
        # 1. Resolve image path
        image_path = resolve_path(image_path_str, self.dataset_dir)
        image_name = image_path.name
        image_id = image_path.stem
        
        if not image_path.exists():
            logger.error(f"Image file not found: {image_path}")
            return ImageAnalysis(
                image_id=image_id,
                valid_image=False,
                object_detected="unknown",
                object_part="unknown",
                visible_damage=False,
                issue_type="unknown",
                severity="unknown",
                quality_flags=["damage_not_visible"]
            )
            
        # 2. Run local OpenCV image heuristics (e.g. blur, lighting, dimensions)
        heuristic_flags = run_image_heuristics(str(image_path))
        
        # 3. Call API vision model for visual understanding
        vision_result = self.vision_model.analyze_image(
            image_path=str(image_path),
            claim_context=claim_analysis.claim_summary,
            claim_object=claim_object,
            claimed_part=claim_analysis.object_part
        )
        
        # 4. Merge quality/risk flags from heuristics and vision API without duplicates
        all_flags: Set[str] = set(vision_result.quality_flags) | set(heuristic_flags)
        
        # Clean flags to fit the allowed vocab
        cleaned_flags = []
        for flag in all_flags:
            flag_cleaned = flag.strip().lower()
            if flag_cleaned in config.ALLOWED_RISK_FLAGS and flag_cleaned != 'none':
                cleaned_flags.append(flag_cleaned)
                
        if not cleaned_flags:
            cleaned_flags = [] # Keep empty list if none
            
        # 5. Enforce allowed vocabulary rules on results
        obj_detected = vision_result.object_detected.strip().lower()
        if obj_detected not in {"car", "laptop", "package"}:
            obj_detected = "unknown"
            
        part_detected = vision_result.object_part.strip().lower().replace(" ", "_")
        allowed_parts = config.ALLOWED_OBJECT_PARTS.get(claim_object, set())
        if part_detected not in allowed_parts:
            part_detected = "unknown"
            
        issue_type = vision_result.issue_type.strip().lower()
        if issue_type not in config.ALLOWED_ISSUE_TYPES:
            issue_type = "unknown"
            
        severity = vision_result.severity.strip().lower()
        if severity not in config.ALLOWED_SEVERITIES:
            severity = "unknown"
            
        # If heuristics found severe blurriness, it might invalidate the image analysis
        valid_image = vision_result.valid_image
        if "blurry_image" in cleaned_flags:
            # On extreme blur, we flag it but let the decision engine check if it is still usable
            pass

        return ImageAnalysis(
            image_id=image_id,
            valid_image=valid_image,
            object_detected=obj_detected,
            object_part=part_detected,
            visible_damage=vision_result.visible_damage,
            issue_type=issue_type,
            severity=severity,
            quality_flags=cleaned_flags,
            mock_status=vision_result.mock_status,
            mock_justification=vision_result.mock_justification,
            mock_supporting_ids=vision_result.mock_supporting_ids,
            mock_evidence_standard_met=vision_result.mock_evidence_standard_met,
            mock_evidence_reason=vision_result.mock_evidence_reason
        )
