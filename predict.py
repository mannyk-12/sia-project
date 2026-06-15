import os
import sys
import json
import ast
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# 1. IMPORT VERIFIED PROJECT MODULES (Single Source of Truth)
# ---------------------------------------------------------------------------
try:
    from dossier_generator import generate_dossier
except ImportError:
    print("❌ ERROR: dossier_generator.py not found in the current directory.")
    sys.exit(1)

try:
    from signal_fusion import (
        preprocess_dataframe,
        compute_signal1,
        compute_signal2,
        fuse_signals,
        build_classifier_input
    )
except ImportError:
    print("❌ ERROR: signal_fusion.py not found in the current directory.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 2. CONFIGURATION & CONSTANTS
# ---------------------------------------------------------------------------
REPO_ID = "Mr-Manny12/sia-deberta"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REQUIRED_COLUMNS = [
    'Ticket_ID', 'Customer_Name', 'Customer_Email', 'Ticket_Subject', 
    'Ticket_Description', 'Issue_Category', 'Priority_Level', 
    'Ticket_Channel', 'Submission_Date', 'Resolution_Time_Hours', 
    'Assigned_Agent', 'Satisfaction_Score'
]

# ---------------------------------------------------------------------------
# 3. INFERENCE ENGINE
# ---------------------------------------------------------------------------
class SIAInferenceEngine:
    def __init__(self):
        print(f"🚀 Booting SIA Inference Engine from Hub: {REPO_ID}")
        print(f"⚙️ Execution Device: {DEVICE}")
        
        # Load correct JSON keys from Hub (Fixes Bugs 5, 6, 7)
        self._load_hub_artifacts()
        
        print("📦 Loading DeBERTa-v3-base and Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(REPO_ID).to(DEVICE)
        self.model.eval()
        
        print("🧠 Loading Semantic Embedder (all-mpnet-base-v2)...")
        self.embedder = SentenceTransformer('all-mpnet-base-v2')

    def _load_hub_artifacts(self):
        """Fetches and parses custom JSON configurations with correct Phase 1 keys."""
        try:
            # Bug 5 Fix: Key is 'threshold', not 'optimal_threshold'
            thresh_path = hf_hub_download(repo_id=REPO_ID, filename="classification_threshold.json")
            with open(thresh_path, 'r') as f:
                self.threshold = json.load(f).get("threshold", 0.495)
                
            # Bug 6 Fix: Keys are 'rule_weight' and 'embed_weight'
            fusion_path = hf_hub_download(repo_id=REPO_ID, filename="fusion_weights.json")
            with open(fusion_path, 'r') as f:
                fusion_data = json.load(f)
                self.w_rule = fusion_data.get("rule_weight", 0.4)
                self.w_embed = fusion_data.get("embed_weight", 0.6)
                
            # Bug 7 Fix: Keys are 't_low', 't_medium', 't_high'
            quant_path = hf_hub_download(repo_id=REPO_ID, filename="severity_quantile_thresholds.json")
            with open(quant_path, 'r') as f:
                quantiles = json.load(f)
                self.t_low = quantiles.get("t_low", 1.8)
                self.t_medium = quantiles.get("t_medium", 2.4)
                self.t_high = quantiles.get("t_high", 3.0)
                
            print(f"✅ Configurations loaded. Locked Inference Threshold: {self.threshold}")
            print(f"✅ Fusion Weights: Rule={self.w_rule}, Embed={self.w_embed}")
        except Exception as e:
            print(f"❌ Failed to download required configuration artifacts from Hub: {e}")
            sys.exit(1)

    def process_and_infer(self, df):
        """Orchestrates Phase 1 logic and Phase 2 classification."""
        
        # 1. Preprocessing (Fixes Bug 2, Bug 8)
        print("🔍 Preprocessing Text and Deriving Metadata...")
        df = preprocess_dataframe(df)
        
        # 2. Compute Signal 1 (Fixes Bug 1)
        print("📐 Computing Signal 1 (Rule-Based Matrix & Negation Logic)...")
        s1_results = df.apply(compute_signal1, axis=1)
        df['rule_score'] = [res[0] for res in s1_results]
        df['matched_keywords'] = [res[1] for res in s1_results]
        
        # 3. Compute Signal 2 (Fixes Bug 4)
        print("🌌 Computing Signal 2 (Semantic Embeddings & Micro-Clusters)...")
        embeddings = self.embedder.encode(df['ticket_text'].tolist(), show_progress_bar=True)
        df = compute_signal2(df, embeddings)
        
        # 4. Fusion (Fixes Bug 3 - No Normalization Applied)
        print("🔗 Fusing Signals (Raw Score Projection)...")
        df = fuse_signals(
            df, 
            rule_weight=self.w_rule, 
            embed_weight=self.w_embed, 
            t_low=self.t_low, 
            t_medium=self.t_medium, 
            t_high=self.t_high
        )
        
        # 5. Build Final Classifier Strings (Fixes Bug 9)
        texts = df.apply(build_classifier_input, axis=1).tolist()
        
        # 6. DeBERTa Inference (Fixes Bug 10 - max_length=512)
        print(f"🤖 Running DeBERTa-v3-base inference (Threshold: {self.threshold})...")
        predictions = []
        confidences = []
        batch_size = 32
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch_texts, 
                truncation=True, 
                padding='max_length', 
                max_length=512, 
                return_tensors="pt"
            ).to(DEVICE)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                mismatch_probs = probs[:, 1].cpu().numpy()
                
            for prob in mismatch_probs:
                confidences.append(prob)
                predictions.append(1 if prob >= self.threshold else 0)
                
        df['confidence'] = confidences
        df['predicted_mismatch'] = predictions
        
        # 7. Final Typology Determination
        def get_mismatch_type(row):
            if row['predicted_mismatch'] == 0: 
                return "Consistent"
            return "Hidden Crisis" if row['severity_delta'] > 0 else "False Alarm"
            
        df['mismatch_type'] = df.apply(get_mismatch_type, axis=1)
        
        return df

# ---------------------------------------------------------------------------
# 4. MAIN EXECUTION ROUTINE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIA Phase 3 Inference Engine")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory path")
    args = parser.parse_args()

    # Directory Setup
    os.makedirs(args.output, exist_ok=True)
    dossier_dir = os.path.join(args.output, "dossiers")
    os.makedirs(dossier_dir, exist_ok=True)

    # Load Data & Validate
    print(f"📥 Loading dataset from {args.input}...")
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"❌ ERROR: Input file '{args.input}' not found.")
        sys.exit(1)
        
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"❌ ERROR: Input CSV is missing required dataset columns: {missing_cols}")
        sys.exit(1)
        
    # Handle pre-existing parsed list columns if running multiple passes
    if 'matched_keywords' in df.columns and df['matched_keywords'].dtype == object:
        df['matched_keywords'] = df['matched_keywords'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Execute Pipeline
    engine = SIAInferenceEngine()
    final_df = engine.process_and_infer(df)
    
    # Save Predictions CSV
    output_csv = os.path.join(args.output, "predictions.csv")
    columns_to_save = ['Ticket_ID', 'Priority_Level', 'inferred_severity', 'predicted_mismatch', 'mismatch_type', 'confidence']
    final_df[columns_to_save].to_csv(output_csv, index=False)
    print(f"\n💾 Predictions successfully saved to {output_csv}")
    
    # Generate Dossiers for Flagged Mismatches
    print(f"📋 Assembling Evidence Dossiers for Flagged Mismatches...")
    mismatches = final_df[final_df['predicted_mismatch'] == 1]
    
    # Helper to map numerical semantic score back to string for the dossier layout
    def map_semantic_to_string(score):
        if score >= engine.t_high: return "Critical"
        elif score >= engine.t_medium: return "High"
        elif score >= engine.t_low: return "Medium"
        return "Low"

    for _, row in tqdm(mismatches.iterrows(), total=len(mismatches)):
        embed_str = map_semantic_to_string(row['embed_severity_score'])
        
        dossier = generate_dossier(
            ticket=row.to_dict(),
            classifier_confidence=row['confidence'],
            inferred_severity=row['inferred_severity'],
            matched_keywords=row['matched_keywords'],
            rule_score=row['rule_score'],
            embed_severity=embed_str 
        )
        
        # Write Dossier JSON
        dossier_path = os.path.join(dossier_dir, f"dossier_{row['Ticket_ID']}.json")
        with open(dossier_path, 'w') as f:
            json.dump(dossier, f, indent=4)
            
    print(f"✨ SUCCESS: {len(mismatches)} Dossiers generated in {dossier_dir}/")
    print("Pipeline Execution Complete. Standing by.")