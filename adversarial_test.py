import os
import sys
import json
import torch
import pandas as pd
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# IMPORT SINGLE SOURCE OF TRUTH MODULES
# ---------------------------------------------------------------------------
try:
    from signal_fusion import (
        preprocess_dataframe,
        compute_signal1,
        compute_signal2,
        fuse_signals,
        build_classifier_input
    )
    from dossier_generator import generate_dossier
except ImportError:
    print("❌ ERROR: signal_fusion.py or dossier_generator.py not found in the current directory.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# ADVERSARIAL TICKETS DATASET
# ---------------------------------------------------------------------------
ADVERSARIAL_TICKETS = [
    {
        'Ticket_ID': 'ADV_001', 'Ticket_Subject': 'Quick question about our account',
        'Ticket_Description': 'Good morning. Our entire production database cluster has been inaccessible since 3am and approximately 50,000 active users cannot log in or complete transactions.',
        'Ticket_Channel': 'Email', 'Priority_Level': 'Low', 'Resolution_Time_Hours': 96,
        'expected_mismatch': True, 'trap': 'Ultra-polite tone masking complete outage'
    },
    {
        'Ticket_ID': 'ADV_002', 'Ticket_Subject': 'URGENT DISASTER TERRIBLE EMERGENCY',
        'Ticket_Description': 'This is absolutely horrible. The button in the settings page changed from blue to grey. I am furious. Needs immediate escalation to the CEO.',
        'Ticket_Channel': 'Email', 'Priority_Level': 'Critical', 'Resolution_Time_Hours': 0.5,
        'expected_mismatch': True, 'trap': 'All escalation keywords in subject, trivial cosmetic issue in description'
    },
    {
        'Ticket_ID': 'ADV_003', 'Ticket_Subject': 'API latency inquiry',
        'Ticket_Description': 'We are seeing p99 latency spike to 45 seconds on our payment gateway endpoint causing cascade failures across our checkout pipeline. Transaction success rate dropped from 99.2 percent to 12 percent. SLA breach window opens in 2 hours.',
        'Ticket_Channel': 'Email', 'Priority_Level': 'Low', 'Resolution_Time_Hours': 120,
        'expected_mismatch': True, 'trap': 'Technical jargon describes catastrophic impact without common crisis keywords'
    },
    {
        'Ticket_ID': 'ADV_004', 'Ticket_Subject': 'Not urgent but database seems down',
        'Ticket_Description': 'I know this probably is not urgent and I do not want to cause fuss, but our database might not be working. No data loads anywhere in the app. Maybe it is just me? Sorry to bother you.',
        'Ticket_Channel': 'Chat', 'Priority_Level': 'Low', 'Resolution_Time_Hours': 84,
        'expected_mismatch': True, 'trap': 'Negated urgency language hiding total data failure'
    },
    {
        'Ticket_ID': 'ADV_005', 'Ticket_Subject': 'Minor concern',
        'Ticket_Description': 'Oh wonderful, just woke up to find that 15,000 customer records have been permanently deleted from production. Totally fine. No rush at all. We love losing years of customer data first thing in the morning.',
        'Ticket_Channel': 'Email', 'Priority_Level': 'Low', 'Resolution_Time_Hours': 200,
        'expected_mismatch': True, 'trap': 'Heavy sarcasm, severity must be inferred from facts not tone'
    },
    {
        'Ticket_ID': 'ADV_006', 'Ticket_Subject': 'CRITICAL URGENT EMERGENCY HELP ASAP',
        'Ticket_Description': 'Hi team! The dashboard loads in 3.1 seconds instead of 2.8 seconds. I know it is not a big deal but I thought I should mention it. Thank you so much for all your hard work!',
        'Ticket_Channel': 'Email', 'Priority_Level': 'Critical', 'Resolution_Time_Hours': 1.0,
        'expected_mismatch': True, 'trap': 'All urgency keywords in subject, genuinely trivial issue in description'
    },
    {
        'Ticket_ID': 'ADV_007', 'Ticket_Subject': 'Users calling us',
        'Ticket_Description': 'Customers have been contacting us non-stop since midnight. They cannot complete purchases. Our support queue has 847 open tickets. We are losing approximately 12000 dollars in revenue every 10 minutes.',
        'Ticket_Channel': 'Chat', 'Priority_Level': 'Medium', 'Resolution_Time_Hours': 180,
        'expected_mismatch': True, 'trap': 'Describes business consequences not technical keywords'
    },
    {
        'Ticket_ID': 'ADV_008', 'Ticket_Subject': 'Question about export feature',
        'Ticket_Description': 'I cannot not get this resolved immediately. It is not unimportant that we not lose this data. Our audit logs have been corrupted and we face regulatory penalties by end of business today.',
        'Ticket_Channel': 'Email', 'Priority_Level': 'Low', 'Resolution_Time_Hours': 110,
        'expected_mismatch': True, 'trap': 'Double negations designed to confuse negation detection'
    },
    {
        'Ticket_ID': 'ADV_009', 'Ticket_Subject': 'Account issue',
        'Ticket_Description': 'I wanted to share some general thoughts. The weather has been nice lately. I have been using your service for 3 years and it has been great overall. Anyway our SSO integration has been completely non-functional for 6 days and 200 employees cannot access any company data. Let me know when you get a chance.',
        'Ticket_Channel': 'Email', 'Priority_Level': 'Low', 'Resolution_Time_Hours': 144,
        'expected_mismatch': True, 'trap': 'Critical info buried after extensive irrelevant padding'
    },
    {
        'Ticket_ID': 'ADV_010', 'Ticket_Subject': 'Password reset not working',
        'Ticket_Description': 'I tried to reset my password but did not receive the email. I checked spam. I can still log in with my old password so it is not blocking me. Just wanted to report it. No rush.',
        'Ticket_Channel': 'Email', 'Priority_Level': 'Medium', 'Resolution_Time_Hours': 6.0,
        'expected_mismatch': False, 'trap': 'Legitimately medium priority, system must NOT flag this'
    }
]

# Standardize Dataset Defaults
for ticket in ADVERSARIAL_TICKETS:
    ticket.update({
        'Customer_Email': 'test@company.com',
        'Issue_Category': 'Technical',
        'Customer_Name': 'Test User',
        'Submission_Date': '2024-01-01',
        'Assigned_Agent': 'Agent001',
        'Satisfaction_Score': 3
    })

df_adv = pd.DataFrame(ADVERSARIAL_TICKETS)

# ---------------------------------------------------------------------------
# INFERENCE ENGINE INITIALIZATION
# ---------------------------------------------------------------------------
REPO_ID = "Mr-Manny12/sia-deberta"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n🚀 Booting SIA Adversarial Evaluation Engine from Hub: {REPO_ID}")
print(f"⚙️ Execution Device: {DEVICE}")

try:
    # 1. Fetch Configuration Artifacts (Exact JSON Keys)
    thresh_path = hf_hub_download(repo_id=REPO_ID, filename="classification_threshold.json")
    with open(thresh_path, 'r') as f:
        threshold = json.load(f).get("threshold", 0.495)
        
    fusion_path = hf_hub_download(repo_id=REPO_ID, filename="fusion_weights.json")
    with open(fusion_path, 'r') as f:
        fw = json.load(f)
        w_rule = fw.get("rule_weight", 0.4)
        w_embed = fw.get("embed_weight", 0.6)
        
    quant_path = hf_hub_download(repo_id=REPO_ID, filename="severity_quantile_thresholds.json")
    with open(quant_path, 'r') as f:
        sq = json.load(f)
        t_low = sq.get("t_low", 1.8)
        t_med = sq.get("t_medium", 2.4)
        t_high = sq.get("t_high", 3.0)

    # 2. Load Models
    print("📦 Loading DeBERTa-v3-base and Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    model = AutoModelForSequenceClassification.from_pretrained(REPO_ID).to(DEVICE)
    model.eval()
    
    print("🧠 Loading Semantic Embedder (all-mpnet-base-v2)...")
    embedder = SentenceTransformer('all-mpnet-base-v2')

except Exception as e:
    print(f"❌ Failed to load artifacts from HF Hub: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# RUN STRICT INFERENCE PIPELINE
# ---------------------------------------------------------------------------
print("\n🔍 Executing SIA Core Pipeline on Adversarial Data...")

# 1. Preprocess
df = preprocess_dataframe(df_adv)

# 2. Signals
s1_results = df.apply(compute_signal1, axis=1)
df['rule_score'] = [res[0] for res in s1_results]
df['matched_keywords'] = [res[1] for res in s1_results]

embeddings = embedder.encode(df['ticket_text'].tolist(), show_progress_bar=False)
df = compute_signal2(df, embeddings)

# 3. Fuse Signals
df = fuse_signals(df, rule_weight=w_rule, embed_weight=w_embed, t_low=t_low, t_medium=t_med, t_high=t_high)

# 4. Input Prefixing
texts = df.apply(build_classifier_input, axis=1).tolist()

# 5. DeBERTa Batch Inference
predictions = []
confidences = []

print(f"🤖 Running Transformer Classification (Threshold: {threshold})...")
inputs = tokenizer(texts, truncation=True, padding='max_length', max_length=512, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
    
for prob in probs:
    confidences.append(float(prob))
    # Threshold boolean evaluation
    predictions.append(bool(prob >= threshold))
    
df['predicted_mismatch'] = predictions
df['confidence'] = confidences

# ---------------------------------------------------------------------------
# EVALUATE RESULTS & SCORE
# ---------------------------------------------------------------------------
print("\n" + "="*115)
print(f"{'TICKET':<10} | {'TRAP DESCRIPTION':<70} | {'EXPECTED':<10} | {'ACTUAL':<10} | {'STATUS'}")
print("="*115)

passed_count = 0
results_payload = []

for _, row in df.iterrows():
    expected = row['expected_mismatch']
    actual = row['predicted_mismatch']
    trap = row['trap']
    t_id = row['Ticket_ID']
    
    passed = expected == actual
    if passed:
        passed_count += 1
        status_tag = "✅ PASS"
    else:
        status_tag = "❌ FAIL"
        
    print(f"{t_id:<10} | {trap[:68]:<70} | {str(expected):<10} | {str(actual):<10} | {status_tag}")
    
    # Save to JSON payload
    results_payload.append({
        "ticket_id": t_id,
        "trap": trap,
        "confidence": row['confidence'],
        "expected_mismatch": expected,
        "predicted_mismatch": actual,
        "passed": passed
    })

print("="*115)

# ---------------------------------------------------------------------------
# COMPETITION BONUS SCORING
# ---------------------------------------------------------------------------
print(f"\n🎯 FINAL ADVERSARIAL SCORE: {passed_count} / 10")

if passed_count >= 7:
    print("🏆 BONUS UNLOCKED: Model successfully defeated >= 70% of adversarial traps! (10% Competition Bonus Awarded)")
else:
    print("⚠️ NO BONUS: Model failed to clear the 7/10 adversarial trap threshold.")

# ---------------------------------------------------------------------------
# SAVE PAYLOAD
# ---------------------------------------------------------------------------
os.makedirs("outputs", exist_ok=True)
output_file = "outputs/adversarial_results.json"

with open(output_file, 'w') as f:
    json.dump({
        "total_score": passed_count,
        "bonus_awarded": bool(passed_count >= 7),
        "results": results_payload
    }, f, indent=4)

print(f"\n💾 Detailed results saved to: {output_file}")