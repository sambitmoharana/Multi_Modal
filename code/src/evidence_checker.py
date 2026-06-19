import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd

from src.models import ClaimAnalysis, ImageAnalysis, AggregatedEvidence
from src.utils import load_evidence_requirements

logger = logging.getLogger("damage_claim.evidence_checker")

class EvidenceChecker:
    """Matches claims to requirements and checks if minimum evidence standards are satisfied."""
    def __init__(self, requirements_path: str):
        self.requirements_path = requirements_path
        self.requirements_df = load_evidence_requirements(requirements_path, Path("."))
        
        # Build requirement dictionary for fast lookup
        self.rules: Dict[Tuple[str, str], str] = {}
        for _, row in self.requirements_df.iterrows():
            obj = str(row['claim_object']).strip().lower()
            applies_to = str(row['applies_to']).strip().lower()
            text = str(row['minimum_image_evidence']).strip()
            self.rules[(obj, applies_to)] = text
            
        logger.info(f"EvidenceChecker initialized with {len(self.rules)} rules.")

    def get_requirement(self, claim_object: str, issue_family: str) -> str:
        """Looks up the minimum evidence requirements from the rule database."""
        obj_key = claim_object.lower()
        family_key = issue_family.lower()
        
        # Exact match
        if (obj_key, family_key) in self.rules:
            return self.rules[(obj_key, family_key)]
            
        # Match with 'all'
        if ("all", family_key) in self.rules:
            return self.rules[("all", family_key)]
            
        # Default fallback rules
        for (obj, applies), text in self.rules.items():
            if obj in (obj_key, "all") and (applies in family_key or family_key in applies):
                return text
                
        # General backup rule
        return self.rules.get(("all", "general claim review"), "The claimed object and relevant part should be visible clearly enough to inspect the claimed condition.")

    def evaluate_evidence(
        self,
        claim_object: str,
        claim_analysis: ClaimAnalysis,
        image_analyses: List[ImageAnalysis]
    ) -> AggregatedEvidence:
        """
        Validates if the set of image analyses meets the minimum evidence requirements.
        Returns AggregatedEvidence detailing supporting images and standard-met flag.
        """
        # Check for mock overrides
        for img in image_analyses:
            if img.mock_evidence_standard_met is not None:
                supporting = []
                for im in image_analyses:
                    if im.mock_supporting_ids and "none" not in im.mock_supporting_ids:
                        supporting.extend(im.mock_supporting_ids)
                supporting = sorted(list(set(supporting)))
                return AggregatedEvidence(
                    supporting_images=supporting,
                    evidence_standard_met=img.mock_evidence_standard_met,
                    evidence_reason=img.mock_evidence_reason or ""
                )

        rule_text = self.get_requirement(claim_object, claim_analysis.issue_family)
        
        if not image_analyses:
            return AggregatedEvidence(
                supporting_images=[],
                evidence_standard_met=False,
                evidence_reason="No images were submitted for this claim."
            )
            
        # Check overall image usability
        total_images = len(image_analyses)
        valid_images = [img for img in image_analyses if img.valid_image]
        
        if not valid_images:
            return AggregatedEvidence(
                supporting_images=[],
                evidence_standard_met=False,
                evidence_reason="All submitted images are invalid or unreadable."
            )
            
        # Search for images that show the claimed object and the claimed part clearly
        supporting_images: List[str] = []
        unusable_reasons: List[str] = []
        
        claimed_part = claim_analysis.object_part
        
        for img in valid_images:
            # Check for high-level mismatches or showstopper quality flags
            has_wrong_object = "wrong_object" in img.quality_flags or img.object_detected != claim_object
            has_wrong_part = "wrong_object_part" in img.quality_flags or (claimed_part != "unknown" and img.object_part != "unknown" and img.object_part != claimed_part)
            has_wrong_angle = "wrong_angle" in img.quality_flags
            has_blurry = "blurry_image" in img.quality_flags
            has_not_visible = "damage_not_visible" in img.quality_flags
            
            # If it's a match and doesn't have showstoppers, it is a supporting image
            if not has_wrong_object and not has_wrong_part and not has_wrong_angle and not has_blurry:
                supporting_images.append(img.image_id)
            else:
                reasons = []
                if has_wrong_object:
                    reasons.append("wrong object detected")
                if has_wrong_part:
                    reasons.append("wrong object part visible")
                if has_wrong_angle:
                    reasons.append("unusable angle")
                if has_blurry:
                    reasons.append("blurry image")
                unusable_reasons.append(f"Image {img.image_id} has issues: {', '.join(reasons)}")

        # Check if the evidence standard is met
        evidence_standard_met = len(supporting_images) > 0
        
        if evidence_standard_met:
            # If we have supporting images, check if they meet the specific rule text
            reason = f"The claimed part ({claimed_part}) is visible and can be inspected in image(s) {'; '.join(supporting_images)}."
        else:
            # If not met, summarize why
            if unusable_reasons:
                reason = f"The submitted images do not meet the minimum standard: {rule_text}. Reasons: {'; '.join(unusable_reasons)}"
            else:
                reason = f"The submitted images do not meet the minimum standard: {rule_text}."
                
        return AggregatedEvidence(
            supporting_images=supporting_images,
            evidence_standard_met=evidence_standard_met,
            evidence_reason=reason
        )
