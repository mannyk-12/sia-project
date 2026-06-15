"""
SIA PROJECT: SIGNAL FUSION MODULE (PHASE 1)
Single Source of Truth for Preprocessing, Signal Generation, and Cross-Modal Prefixing.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import spacy

# ---------------------------------------------------------
# SPACY NLP INITIALIZATION
# ---------------------------------------------------------
import spacy
import en_core_web_sm

# Load the model directly from the installed wheel package
nlp = en_core_web_sm.load()

# ---------------------------------------------------------
# CONSTANTS & CONFIGURATION
# ---------------------------------------------------------
FREE_EMAIL_DOMAINS = [
    'gmail.com', 'yahoo.com', 'hotmail.com', 
    'outlook.com', 'aol.com', 'icloud.com'
]

NEGATION_WORDS = {"not", "n't", "never", "no"}

PRIORITY_TO_NUM = {
    "Low": 1, 
    "Medium": 2, 
    "High": 3, 
    "Critical": 4
}

SIGNAL_1_TIERS = {
    'fraud_critical': {
        'weight': 2.0, 
        'words': ['fraud', 'unauthorized', 'stolen', 'hacked', 'compromised', 'scam', 'breach', 'illegal charge']
    },
    'tech_critical': {
        'weight': 1.5, 
        'words': ['crash', 'outage', 'data loss', 'completely down', 'system down', 'deleted']
    },
    'billing_high': {
        'weight': 1.0, 
        'words': ['charged twice', 'overcharged', 'refund', 'wrong amount', 'cancel subscription']
    },
    'account_med': {
        'weight': 0.6, 
        'words': ['login failed', 'locked out', 'not syncing', 'sync error', 'spinning wheel', 'keeps failing']
    },
    'inquiry_low': {
        'weight': -0.8, 
        'words': ['how to', 'how do i', 'hours of operation', 'wondering', 'headquarters', 'can you']
    },
    'escalation': {
        'weight': 1.0, 
        'words': ['legal action', 'manager', 'unacceptable', 'losing revenue', 'asap']
    }
}

# ---------------------------------------------------------
# EXPOSED FUNCTIONS
# ---------------------------------------------------------

def get_customer_tier(email):
    """Maps email domains to exact CONSUMER or ENTERPRISE labels."""
    if pd.isna(email):
        return "ENTERPRISE"
    email_lower = str(email).lower()
    for domain in FREE_EMAIL_DOMAINS:
        if domain in email_lower:
            return "CONSUMER"
    return "ENTERPRISE"

def get_resolution_bucket(hours):
    """Maps numerical resolution hours to exact categorical bucket strings."""
    try:
        h = float(hours)
        if h <= 4: return "VERY_FAST"
        elif h <= 24: return "FAST"
        elif h <= 72: return "MODERATE"
        elif h <= 168: return "SLOW"
        else: return "VERY_SLOW"
    except (ValueError, TypeError):
        return "MODERATE"

def preprocess_dataframe(df):
    """Adds all engineered columns needed for Phase 1 scoring and inference."""
    df = df.copy()
    
    # Text concatenation
    subj = df['Ticket_Subject'].fillna('').astype(str)
    desc = df['Ticket_Description'].fillna('').astype(str)
    df['ticket_text'] = subj + '. ' + desc
    
    # Metadata derivations
    df['customer_tier'] = df['Customer_Email'].apply(get_customer_tier)
    df['resolution_bucket'] = df['Resolution_Time_Hours'].apply(get_resolution_bucket)
    
    return df

def compute_signal1(row):
    """Computes Rule-Based NLP score with Negation Lookback and Channel Multiplier."""
    text = str(row.get('ticket_text', '')).lower()
    channel = str(row.get('Ticket_Channel', '')).lower()
    
    # Base score strictly starts at 2.0
    score = 2.0
    matched_keywords = []
    
    # Tokenize for accurate negation window checking
    doc = nlp(text)
    tokens = [t.text for t in doc]
    
    for tier, data in SIGNAL_1_TIERS.items():
        weight = data['weight']
        for word in data['words']:
            if word in text:
                # spaCy Negation Check (3-token lookback)
                word_base = word.split()[0]
                try:
                    idx = tokens.index(word_base)
                    start_idx = max(0, idx - 3)
                    context_window = set(tokens[start_idx:idx])
                    
                    if context_window.intersection(NEGATION_WORDS):
                        score -= weight  # Invert effect if negated
                    else:
                        score += weight
                    matched_keywords.append(word)
                except ValueError:
                    # Fallback if substring match is offset from spacy tokenization
                    score += weight
                    matched_keywords.append(word)
                    
    # Channel Multiplier (applied AFTER keyword scoring)
    multiplier = 1.15 if 'chat' in channel else 1.0
    score *= multiplier
    
    # Hard clipping to [1.0, 4.0] boundary
    score = max(1.0, min(4.0, score))
    
    # Return unique matched keywords
    return score, list(set(matched_keywords))

def compute_signal2(df, embeddings):
    """Computes MPNet Semantic Cluster score without normalization."""
    df = df.copy()
    
    n_clusters = min(100, len(df))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster_id'] = kmeans.fit_predict(embeddings)
    
    # Signal 2 is the mean of Signal 1 (rule_score) within that semantic micro-cluster
    cluster_means = df.groupby('cluster_id')['rule_score'].mean().to_dict()
    df['embed_severity_score'] = df['cluster_id'].map(cluster_means)
    
    return df

def fuse_signals(df, rule_weight=0.4, embed_weight=0.6, t_low=1.8, t_medium=2.4, t_high=3.0):
    """Fuses rules and embeddings, assigns discrete severity, and computes mismatch."""
    df = df.copy()
    
    # Raw Score Fusion
    df['fused_severity_score'] = (rule_weight * df['rule_score']) + (embed_weight * df['embed_severity_score'])
    
    # Fixed Severity Thresholds Mapping
    def map_severity(score):
        if score >= t_high: return "Critical"
        elif score >= t_medium: return "High"
        elif score >= t_low: return "Medium"
        return "Low"
        
    df['inferred_severity'] = df['fused_severity_score'].apply(map_severity)
    
    # Mismatch Labeling Delta >= 2
    df['agent_priority_val'] = df['Priority_Level'].map(PRIORITY_TO_NUM).fillna(2)
    df['inferred_severity_val'] = df['inferred_severity'].map(PRIORITY_TO_NUM)
    df['severity_delta'] = df['inferred_severity_val'] - df['agent_priority_val']
    
    df['is_mismatch'] = (df['severity_delta'].abs() >= 2).astype(int)
    
    return df

def build_classifier_input(row):
    """Builds the exact token stream prefix array matching Phase 2 Training."""
    channel = str(row.get('Ticket_Channel', 'UNKNOWN')).upper().replace(" ", "_")
    tier = str(row.get('customer_tier', 'CONSUMER')).upper()
    restime = str(row.get('resolution_bucket', 'MODERATE')).upper()
    category = str(row.get('Issue_Category', 'UNKNOWN')).upper().replace(" ", "_")
    signal_sev = str(row.get('inferred_severity', 'UNKNOWN')).upper()
    
    text = str(row.get('ticket_text', ''))
    
    prefix = f"[CHANNEL:{channel}] [TIER:{tier}] [RESTIME:{restime}] [CATEGORY:{category}] [SIGNAL_SEVERITY:{signal_sev}] "
    return prefix + text