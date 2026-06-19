import logging
from typing import List, Dict, Any
import pandas as pd

from src.models import ClaimDecision, RiskAssessment, AggregatedEvidence, ClaimAnalysis
from src import config

logger = logging.getLogger("damage_claim.output_generator")

class OutputGenerator:
    """Formats and writes system predictions to output.csv."""
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.columns = [
            "user_id", "image_paths", "user_claim", "claim_object",
            "evidence_standard_met", "evidence_standard_met_reason",
            "risk_flags", "issue_type", "object_part", "claim_status",
            "claim_status_justification", "supporting_image_ids",
            "valid_image", "severity"
        ]

    def generate_csv(self, records: List[Dict[str, Any]]) -> None:
        logger.info(f"Generating output CSV with {len(records)} rows...")
        
        formatted_rows = []
        for rec in records:
            # Extract models
            claim_info: ClaimAnalysis = rec["claim_info"]
            evidence: AggregatedEvidence = rec["evidence"]
            risk: RiskAssessment = rec["risk"]
            decision: ClaimDecision = rec["decision"]
            
            # Format risk flags
            # Combine image risk flags and history risk flags
            all_risk_flags = set(risk.history_flags)
            for img in rec["image_analyses"]:
                all_risk_flags.update(img.quality_flags)
                
            risk_flags_str = ";".join(sorted(all_risk_flags)) if all_risk_flags else "none"
            
            # Format supporting image IDs
            supporting_ids_str = ";".join(decision.supporting_image_ids) if decision.supporting_image_ids else "none"
            if supporting_ids_str == "":
                supporting_ids_str = "none"

            row = {
                "user_id": rec["user_id"],
                "image_paths": rec["image_paths"],
                "user_claim": rec["user_claim"],
                "claim_object": rec["claim_object"],
                "evidence_standard_met": str(evidence.evidence_standard_met).lower(),
                "evidence_standard_met_reason": evidence.evidence_reason,
                "risk_flags": risk_flags_str,
                "issue_type": claim_info.issue_type,
                "object_part": claim_info.object_part,
                "claim_status": decision.claim_status,
                "claim_status_justification": decision.claim_status_justification,
                "supporting_image_ids": supporting_ids_str,
                "valid_image": str(decision.valid_image).lower(),
                "severity": decision.severity
            }
            formatted_rows.append(row)
            
        df = pd.DataFrame(formatted_rows)
        
        # Ensure exact column selection and order
        df = df[self.columns]
        
        # Save to output file
        df.to_csv(self.output_path, index=False)
        logger.info(f"Successfully saved output file to: {self.output_path}")
