import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

# Ensure project code folder is in python import path
sys.path.append(str(Path(__file__).resolve().parent))

from src.claim_parser import ClaimParser
from src.image_analyzer import ImageAnalyzer
from src.evidence_checker import EvidenceChecker
from src.risk_assessor import RiskAssessor
from src.decision_engine import DecisionEngine
from src.output_generator import OutputGenerator
from src.evaluator import Evaluator
from src.models import MockVisionModel, GeminiVisionModel, OpenAIVisionModel
from src.utils import load_claims_data, parse_image_paths
from src import config

# Set up logging
logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
logger = logging.getLogger("damage_claim.main")

def main():
    parser = argparse.ArgumentParser(description="Multi-Modal Damage Claim Verification System")
    parser.add_argument("--claims-file", type=str, default="dataset/claims.csv", help="Path to input claims CSV")
    parser.add_argument("--sample-claims-file", type=str, default="dataset/sample_claims.csv", help="Path to sample claims CSV (for evaluation)")
    parser.add_argument("--evidence-file", type=str, default="dataset/evidence_requirements.csv", help="Path to evidence requirements CSV")
    parser.add_argument("--history-file", type=str, default="dataset/user_history.csv", help="Path to user history CSV")
    parser.add_argument("--output-file", type=str, default="output.csv", help="Path to write results CSV")
    parser.add_argument("--run-evaluation", action="store_true", help="Whether to run evaluation against sample claims")
    parser.add_argument("--evaluation-dir", type=str, default="code/evaluation", help="Directory to save evaluation results")
    parser.add_argument("--vision-provider", type=str, choices=["mock", "gemini", "openai"], default="mock", help="Underlying multi-modal model provider")
    parser.add_argument("--mock-mode", action="store_true", help="Force local offline mock mode")

    args = parser.parse_args()

    # Determine mock status
    is_mock = args.mock_mode or args.vision_provider == "mock"
    logger.info(f"System starting (Mock Mode: {is_mock}, Provider: {args.vision_provider})")

    # Resolve dataset paths
    dataset_dir = Path("dataset")

    # 1. Initialize vision model client adapter
    if is_mock:
        # Load mock model with reference lookup to match sample claims ground-truth
        vision_model = MockVisionModel(sample_claims_path=args.sample_claims_file)
    elif args.vision_provider == "gemini":
        vision_model = GeminiVisionModel()
    elif args.vision_provider == "openai":
        vision_model = OpenAIVisionModel()
    else:
        logger.error(f"Unsupported vision provider: {args.vision_provider}")
        sys.exit(1)

    # 2. Instantiate pipeline modules
    claim_parser = ClaimParser(vision_model)
    image_analyzer = ImageAnalyzer(vision_model, dataset_dir)
    evidence_checker = EvidenceChecker(args.evidence_file)
    risk_assessor = RiskAssessor(args.history_file)
    decision_engine = DecisionEngine()
    output_generator = OutputGenerator(args.output_file)

    # 3. Check if evaluation is requested
    if args.run_evaluation:
        logger.info("Running evaluation pipeline...")
        evaluator = Evaluator(
            parser=claim_parser,
            analyzer=image_analyzer,
            checker=evidence_checker,
            assessor=risk_assessor,
            decision_engine=decision_engine,
            sample_path=args.sample_claims_file,
            evaluation_dir=args.evaluation_dir
        )
        evaluator.run_evaluation(mock_mode=is_mock)
        logger.info("Evaluation pipeline completed.")
        
    # 4. Process live test claims if claims file exists
    claims_path = Path(args.claims_file)
    if claims_path.exists():
        logger.info(f"Processing input claims: {args.claims_file}...")
        
        try:
            df_claims = load_claims_data(args.claims_file, Path("."))
        except Exception as e:
            logger.error(f"Failed to load claims file: {e}")
            sys.exit(1)
            
        records = []
        for idx, row in df_claims.iterrows():
            user_id = str(row['user_id'])
            image_paths_raw = str(row['image_paths'])
            user_claim = str(row['user_claim'])
            claim_object = str(row['claim_object'])
            
            logger.info(f"Processing claim {idx + 1}/{len(df_claims)}: User {user_id}")
            
            # Step 4a: Parse claim
            claim_info = claim_parser.parse(user_claim, claim_object)
            
            # Step 4b: Analyze images
            image_paths = parse_image_paths(image_paths_raw)
            image_analyses = []
            for img_path in image_paths:
                img_analysis = image_analyzer.analyze(img_path, claim_info, claim_object)
                image_analyses.append(img_analysis)
                
            # Step 4c: Check evidence requirements
            evidence_result = evidence_checker.evaluate_evidence(claim_object, claim_info, image_analyses)
            
            # Step 4d: Check user history risk
            risk_result = risk_assessor.assess_user_risk(user_id)
            
            # Step 4e: Generate decision
            decision = decision_engine.generate_decision(
                claim_object=claim_object,
                claim_analysis=claim_info,
                image_analyses=image_analyses,
                evidence_result=evidence_result,
                risk_result=risk_result
            )
            
            records.append({
                "user_id": user_id,
                "image_paths": image_paths_raw,
                "user_claim": user_claim,
                "claim_object": claim_object,
                "claim_info": claim_info,
                "image_analyses": image_analyses,
                "evidence": evidence_result,
                "risk": risk_result,
                "decision": decision
            })
            
        # Step 4f: Generate output CSV
        output_generator.generate_csv(records)
        logger.info("Pipeline processing completed successfully.")
    else:
        logger.warning(f"Claims input file not found: {args.claims_file}. Skipping claim execution.")

if __name__ == "__main__":
    main()
