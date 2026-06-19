import logging
from src.models import BaseVisionModel, ClaimAnalysis
from src import config

logger = logging.getLogger("damage_claim.claim_parser")

class ClaimParser:
    """Extracts and normalizes claims data from conversation transcript."""
    def __init__(self, vision_model: BaseVisionModel):
        self.vision_model = vision_model

    def parse(self, user_claim: str, claim_object: str) -> ClaimAnalysis:
        logger.info(f"Parsing claim conversation for object '{claim_object}'")
        
        # Extract parsing details via model adapter
        parsed = self.vision_model.parse_claim(user_claim, claim_object)
        
        # 1. Enforce normalized vocabulary for issue_type
        issue_type = parsed.issue_type.strip().lower()
        if issue_type not in config.ALLOWED_ISSUE_TYPES:
            logger.warning(f"Extracted issue type '{issue_type}' not allowed. Defaulting to 'unknown'.")
            issue_type = "unknown"
            
        # 2. Enforce normalized vocabulary for object_part based on object type
        object_part = parsed.object_part.strip().lower().replace(" ", "_")
        allowed_parts = config.ALLOWED_OBJECT_PARTS.get(claim_object, set())
        if object_part not in allowed_parts:
            logger.warning(f"Extracted object part '{object_part}' not allowed for {claim_object}. Defaulting to 'unknown'.")
            object_part = "unknown"
            
        # 3. Maintain structured issue family matching
        issue_family = parsed.issue_family.strip().lower()
        if not issue_family:
            issue_family = "general claim review"
            
        return ClaimAnalysis(
            claim_summary=parsed.claim_summary,
            issue_type=issue_type,
            object_part=object_part,
            issue_family=issue_family
        )
