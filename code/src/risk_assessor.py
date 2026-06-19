import os
import logging
from pathlib import Path
from typing import List

import pandas as pd
from src.models import RiskAssessment
from src.utils import load_user_history

logger = logging.getLogger("damage_claim.risk_assessor")

class RiskAssessor:
    """Evaluates user profile metadata and history risk logs."""
    def __init__(self, history_csv_path: str):
        self.history_csv_path = history_csv_path
        self.history_df = load_user_history(history_csv_path, Path("."))
        
        # Build index for fast user lookup
        self.user_records = {}
        for _, row in self.history_df.iterrows():
            user_id = str(row['user_id']).strip()
            self.user_records[user_id] = row
            
        logger.info(f"RiskAssessor initialized with {len(self.user_records)} user records.")

    def assess_user_risk(self, user_id: str) -> RiskAssessment:
        logger.info(f"Assessing risk for user: {user_id}")
        
        if user_id not in self.user_records:
            logger.info(f"User {user_id} not found in history. Defaulting to low risk profile.")
            return RiskAssessment(
                user_history_risk=False,
                manual_review_required=False,
                history_flags=[],
                history_summary="New user with no prior claim history"
            )
            
        record = self.user_records[user_id]
        
        # Read profile columns
        past_claim_count = int(record.get('past_claim_count', 0))
        accept_claim = int(record.get('accept_claim', 0))
        manual_review_claim = int(record.get('manual_review_claim', 0))
        rejected_claim = int(record.get('rejected_claim', 0))
        last_90_days_claim_count = int(record.get('last_90_days_claim_count', 0))
        
        # Parse history flags column
        history_flags_str = str(record.get('history_flags', 'none'))
        raw_flags = [f.strip().lower() for f in history_flags_str.split(';')]
        history_flags = [f for f in raw_flags if f not in ('none', '')]
        
        history_summary = str(record.get('history_summary', 'No notable history.'))

        # Risk heuristics
        user_history_risk = "user_history_risk" in history_flags
        manual_review_required = "manual_review_required" in history_flags
        
        # Structural overrides based on raw counts
        if rejected_claim >= 3:
            user_history_risk = True
            
        if manual_review_claim >= 2 or last_90_days_claim_count >= 5:
            manual_review_required = True
            
        # Re-compile clean list of flags
        final_flags = []
        if user_history_risk:
            final_flags.append("user_history_risk")
        if manual_review_required:
            final_flags.append("manual_review_required")

        return RiskAssessment(
            user_history_risk=user_history_risk,
            manual_review_required=manual_review_required,
            history_flags=final_flags,
            history_summary=history_summary
        )
