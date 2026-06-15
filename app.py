import streamlit as st
import pandas as pd
import torch
import json
import plotly.express as px
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# SINGLE SOURCE OF TRUTH IMPORTS
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
    st.error("❌ ERROR: Required modules (signal_fusion.py, dossier_generator.py) not found in the directory.")
    st.stop()

# ---------------------------------------------------------------------------
# PAGE CONFIGURATION & SIDEBAR
# ---------------------------------------------------------------------------
st.set_page_config(page_title="SIA — Support Integrity Auditor", page_icon="🛡️", layout="wide")

st.sidebar.title("🛡️ SIA Project")
st.sidebar.markdown("""
**Support Integrity Auditor** Automated Semantic Priority Triage & Mismatch Detection.

**Model Engine:**
- Architecture: `microsoft/deberta-v3-base`
- Task: Binary Classification
- Environment: HF Hub (`Mr-Manny12/sia-deberta`)
""")

REQUIRED_COLUMNS = [
    'Ticket_ID', 'Customer_Email', 'Ticket_Subject', 'Ticket_Description', 
    'Issue_Category', 'Priority_Level', 'Ticket_Channel', 'Resolution_Time_Hours'
]

# ---------------------------------------------------------------------------
# SYSTEM INITIALIZATION (Cached Once)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_sia_pipeline():
    repo_id = "Mr-Manny12/sia-deberta"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # 1. Fetch Configuration Artifacts (Using precise keys specified)
        thresh_path = hf_hub_download(repo_id=repo_id, filename="classification_threshold.json")
        with open(thresh_path, 'r') as f:
            threshold = json.load(f).get('threshold', 0.495)
            
        fusion_path = hf_hub_download(repo_id=repo_id, filename="fusion_weights.json")
        with open(fusion_path, 'r') as f:
            fw = json.load(f)
            w_rule = fw.get('rule_weight', 0.4)
            w_embed = fw.get('embed_weight', 0.6)
            
        quant_path = hf_hub_download(repo_id=repo_id, filename="severity_quantile_thresholds.json")
        with open(quant_path, 'r') as f:
            sq = json.load(f)
            t_low = sq.get('t_low', 1.8)
            t_med = sq.get('t_medium', 2.4)
            t_high = sq.get('t_high', 3.0)

        # 2. Load Models
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForSequenceClassification.from_pretrained(repo_id).to(device)
        model.eval()
        
        embedder = SentenceTransformer('all-mpnet-base-v2')
        
        return {
            'tokenizer': tokenizer, 'model': model, 'embedder': embedder, 'device': device,
            'threshold': threshold, 'w_rule': w_rule, 'w_embed': w_embed,
            't_low': t_low, 't_medium': t_med, 't_high': t_high
        }
    except Exception as e:
        return {"error": str(e)}

with st.spinner("Booting SIA Engine from Hugging Face Hub..."):
    pipeline = load_sia_pipeline()
    if "error" in pipeline:
        st.error(f"❌ Failed to load model from Hub: {pipeline['error']}")
        st.stop()

# ---------------------------------------------------------------------------
# INFERENCE PIPELINE EXECUTOR
# ---------------------------------------------------------------------------
def run_inference_pipeline(df):
    """Executes inference sequence exactly as prescribed."""
    # 1. Preprocess
    df = preprocess_dataframe(df)
    
    # 2. Compute Signals
    s1_results = df.apply(compute_signal1, axis=1)
    df['rule_score'] = [res[0] for res in s1_results]
    df['matched_keywords'] = [res[1] for res in s1_results]
    
    embeddings = pipeline['embedder'].encode(df['ticket_text'].tolist(), show_progress_bar=False)
    df = compute_signal2(df, embeddings)
    
    # 3. Fuse Signals
    df = fuse_signals(
        df, 
        rule_weight=pipeline['w_rule'], 
        embed_weight=pipeline['w_embed'], 
        t_low=pipeline['t_low'], 
        t_medium=pipeline['t_medium'], 
        t_high=pipeline['t_high']
    )
    
    # 4. Build Input Strings
    texts = df.apply(build_classifier_input, axis=1).tolist()
    
    # 5. Tokenize and Infer (DeBERTa batch size 16)
    predictions = []
    confidences = []
    batch_size = 16
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = pipeline['tokenizer'](
            batch, truncation=True, padding='max_length', max_length=512, return_tensors="pt"
        ).to(pipeline['device'])
        
        with torch.no_grad():
            outputs = pipeline['model'](**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            
        for prob in probs:
            confidences.append(float(prob))
            predictions.append(1 if prob >= pipeline['threshold'] else 0)
            
    df['confidence'] = confidences
    df['is_mismatch'] = predictions
    
    # 6. Typology Mapping with Strict "Trust but Verify" Guardrails
    def get_prediction_label(row):
        delta = row['severity_delta']
        ai_mismatch = row['is_mismatch']
        
        # GUARDRAIL 1: Mathematical Override (Catches AI False Negatives / Traps)
        # If the objective math proves a severe mismatch (Delta of 2 or more), it overrides the AI.
        if delta >= 2:
            return "Hidden Crisis"
        elif delta <= -2:
            return "False Alarm"
            
        # GUARDRAIL 2: AI False Positive Protection
        # If the AI hallucinates a mismatch but the delta is 0, force it to Consistent.
        if ai_mismatch == 1 and delta == 0:
            return "Consistent"
            
        # 3. Standard AI Nuance 
        # If the delta is borderline (1 or -1), trust DeBERTa's semantic judgment.
        if ai_mismatch == 1:
            return "Hidden Crisis" if delta > 0 else "False Alarm"
            
        return "Consistent"
        
    df['Prediction'] = df.apply(get_prediction_label, axis=1)
    
    # 7. Dossier Generation (For Mismatches only)
    dossiers = []
    for _, row in df.iterrows():
        if row['Prediction'] in ["Hidden Crisis", "False Alarm"]:
            es = row['embed_severity_score']
            if es >= pipeline['t_high']: e_str = "Critical"
            elif es >= pipeline['t_medium']: e_str = "High"
            elif es >= pipeline['t_low']: e_str = "Medium"
            else: e_str = "Low"
            
            dossier = generate_dossier(
                ticket=row.to_dict(), 
                classifier_confidence=row['confidence'],
                inferred_severity=row['inferred_severity'], 
                matched_keywords=row['matched_keywords'],
                rule_score=row['rule_score'], 
                embed_severity=e_str
            )
            dossiers.append(dossier)
        else:
            dossiers.append(None)
    
    df['dossier'] = dossiers
    return df

# ---------------------------------------------------------------------------
# UI TABS
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🎯 Single Ticket Analyzer", "📂 Batch CSV Upload", "📊 Priority Mismatch Dashboard"])

# ==============================================================================
# TAB 1: SINGLE TICKET ANALYZER
# ==============================================================================
with tab1:
    st.subheader("Live Semantic Triage Simulator")
    
    with st.form("single_ticket_form"):
        col1, col2 = st.columns([2, 1])
        with col1:
            t_id = st.text_input("Ticket ID", value="TKT-SIM-001")
            t_subj = st.text_input("Ticket Subject")
            t_desc = st.text_area("Ticket Description", height=150)
            c_email = st.text_input("Customer Email (Optional)", value="customer@example.com")
        with col2:
            t_cat = st.selectbox("Issue Category", ["Technical", "Billing", "Account", "General Inquiry", "Fraud"])
            p_level = st.selectbox("Assigned Priority", ["Low", "Medium", "High", "Critical"])
            t_chan = st.selectbox("Ticket Channel", ["Chat", "Email", "Web Form"])
            r_time = st.number_input("Resolution Time (Hours)", min_value=0.0, value=24.0, step=1.0)
            
        submitted = st.form_submit_button("Run Integrity Audit", type="primary")

    if submitted:
        if not t_subj.strip() and not t_desc.strip():
            st.warning("Please provide a Ticket Subject or Description.")
        else:
            single_df = pd.DataFrame([{
                'Ticket_ID': t_id, 'Customer_Email': c_email, 'Ticket_Subject': t_subj,
                'Ticket_Description': t_desc, 'Issue_Category': t_cat, 'Priority_Level': p_level,
                'Ticket_Channel': t_chan, 'Resolution_Time_Hours': r_time
            }])
            
            with st.spinner("Analyzing semantic vectors and compliance rules..."):
                res_df = run_inference_pipeline(single_df)
                
            row = res_df.iloc[0]
            st.markdown("---")
            st.markdown("### 📋 Audit Verdict")
            
            # Confidence Progress Bar
            st.progress(row['confidence'], text=f"Mismatch Confidence Score: {row['confidence']:.2%}")
            
            # Verdict Cards
            if row['Prediction'] == "Consistent":
                st.success(f"**🟢 CONSISTENT:** The human-assigned '{p_level}' priority correctly aligns with the semantic severity.")
            elif row['Prediction'] == "Hidden Crisis":
                st.error(f"**🔴 HIDDEN CRISIS DETECTED:** Ticket severely under-prioritized. Inferred Objective Severity is **{row['inferred_severity']}**.")
            else:
                st.warning(f"**🟠 FALSE ALARM DETECTED:** Ticket over-prioritized. Inferred Objective Severity is **{row['inferred_severity']}**.")
                
            # Dossier JSON Display
            if row['dossier'] is not None:
                with st.expander("🔍 View Zero-Hallucination Evidence Dossier", expanded=True):
                    st.json(row['dossier'])

# ==============================================================================
# TAB 2: BATCH CSV UPLOAD
# ==============================================================================
with tab2:
    st.subheader("Bulk Auditing & CSV Processor")
    uploaded_file = st.file_uploader("Upload CRM Export Dataset (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        
        st.markdown("##### Upload Preview (First 5 Rows)")
        st.dataframe(batch_df.head(5))
        
        # Column Validation
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in batch_df.columns]
        if missing_cols:
            st.error(f"❌ **Invalid Dataset Format.** The uploaded CSV is missing the following required columns:\n\n`{', '.join(missing_cols)}`")
        else:
            if st.button("Run Global Batch Audit", type="primary"):
                # Memory-safe batch chunking: 50 rows per batch
                processed_chunks = []
                chunk_size = 50
                total_chunks = (len(batch_df) + chunk_size - 1) // chunk_size
                
                progress_bar = st.progress(0, text="Initializing batch processor...")
                
                for i in range(total_chunks):
                    chunk = batch_df.iloc[i * chunk_size : (i + 1) * chunk_size]
                    with st.spinner(f"Running inference on batch {i+1} of {total_chunks}..."):
                        processed_chunks.append(run_inference_pipeline(chunk))
                    
                    progress_pct = min((i + 1) * chunk_size, len(batch_df)) / len(batch_df)
                    progress_bar.progress(progress_pct, text=f"Processed {min((i+1)*chunk_size, len(batch_df))} / {len(batch_df)} tickets")
                
                final_batch_df = pd.concat(processed_chunks, ignore_index=True)
                st.session_state['dashboard_data'] = final_batch_df  # Cache for Tab 3
                
                # Render Stylized Dataframe
                display_cols = ['Ticket_ID', 'Priority_Level', 'inferred_severity', 'Prediction', 'confidence']
                display_df = final_batch_df[display_cols].copy()
                display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.2%}")
                
                def color_prediction(val):
                    if val == 'Consistent': return 'color: #2ca02c; font-weight: bold'
                    if val == 'Hidden Crisis': return 'color: #d62728; font-weight: bold'
                    if val == 'False Alarm': return 'color: #ff7f0e; font-weight: bold'
                    return ''
                
                st.markdown("### 🗄️ Inference Results")
                st.dataframe(display_df.style.map(color_prediction, subset=['Prediction']), use_container_width=True)
                
                # CSV Download Extractor
                export_df = final_batch_df.drop(columns=['dossier']).to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Predictions CSV", 
                    data=export_df, 
                    file_name='sia_audit_predictions.csv', 
                    mime='text/csv'
                )
                
                # Expandable Dossier Views
                st.markdown("### 🔎 Generated Mismatch Dossiers")
                
                # UPDATE: Filter by the final Guardrail Prediction, not the raw AI boolean
                flagged_rows = final_batch_df[final_batch_df['Prediction'].isin(["Hidden Crisis", "False Alarm"])]
                
                if flagged_rows.empty:
                    st.success("No priority mismatches were flagged in this dataset.")
                else:
                    for _, f_row in flagged_rows.iterrows():
                        label = f"{f_row['Ticket_ID']} | {f_row['Prediction']} (Assigned: {f_row['Priority_Level']} ➡️ Inferred: {f_row['inferred_severity']})"
                        with st.expander(label):
                            st.json(f_row['dossier'])

# ==============================================================================
# TAB 3: PRIORITY MISMATCH DASHBOARD
# ==============================================================================
with tab3:
    st.subheader("System Analytics & Triage Integrity Metrics")
    
    if 'dashboard_data' not in st.session_state:
        st.info("⚠️ Dashboard empty. Please upload and process a CSV dataset in the **Batch CSV Upload** tab to populate analytics.")
    else:
        results = st.session_state['dashboard_data']
        
        # Calculate Summary Metrics
        total_tickets = len(results)
        hidden_crisis_ct = len(results[results['Prediction'] == 'Hidden Crisis'])
        false_alarm_ct = len(results[results['Prediction'] == 'False Alarm'])
        total_flagged = hidden_crisis_ct + false_alarm_ct
        mismatch_rate = (total_flagged / total_tickets) if total_tickets > 0 else 0
        
        # Metrics Header
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Tickets Processed", f"{total_tickets:,}")
        c2.metric("Total Flagged Mismatches", f"{total_flagged:,}")
        c3.metric("🔴 Hidden Crises", f"{hidden_crisis_ct:,}")
        c4.metric("🟠 False Alarms", f"{false_alarm_ct:,}")
        c5.metric("Overall Mismatch Rate", f"{mismatch_rate:.1%}")
        
        st.markdown("---")
        
        # Color Map
        color_map = {'Consistent': '#2ca02c', 'Hidden Crisis': '#d62728', 'False Alarm': '#ff7f0e'}
        
        row1_col1, row1_col2 = st.columns(2)
        
        # Chart 1: Mismatch Type Distribution
        with row1_col1:
            fig1 = px.pie(
                results, 
                names='Prediction', 
                title='Mismatch Type Distribution',
                color='Prediction', 
                color_discrete_map=color_map,
                hole=0.4
            )
            st.plotly_chart(fig1, use_container_width=True)
            
        # Chart 2: Category Bar
        with row1_col2:
            cat_df = results.groupby(['Issue_Category', 'Prediction']).size().reset_index(name='Ticket Count')
            fig2 = px.bar(
                cat_df, 
                x='Issue_Category', y='Ticket Count', color='Prediction', 
                title='Mismatch Rate by Issue Category',
                barmode='group', 
                color_discrete_map=color_map
            )
            st.plotly_chart(fig2, use_container_width=True)

        row2_col1, row2_col2 = st.columns(2)
        
        # Chart 3: Severity Delta Heatmap (Competition Requirement)
        with row2_col1:
            cat_order = ['Low', 'Medium', 'High', 'Critical']
            heatmap_data = pd.crosstab(results['Priority_Level'], results['inferred_severity'])
            
            # Reindex to ensure strict Low -> Critical ordering
            heatmap_data = heatmap_data.reindex(index=cat_order, columns=cat_order, fill_value=0)
            
            fig3 = px.imshow(
                heatmap_data, 
                text_auto=True, 
                color_continuous_scale='Blues', 
                labels=dict(x="Objective Inferred Severity (System)", y="Assigned Priority (Human)", color="Ticket Count"),
                title="Severity Delta Heatmap (Agent vs. System)"
            )
            fig3.update_xaxes(side="bottom")
            st.plotly_chart(fig3, use_container_width=True)
            
        # Chart 4: Channel Bar
        with row2_col2:
            chan_df = results.groupby(['Ticket_Channel', 'Prediction']).size().reset_index(name='Ticket Count')
            fig4 = px.bar(
                chan_df, 
                x='Ticket_Channel', y='Ticket Count', color='Prediction', 
                title='Mismatch Rate by Intake Channel',
                barmode='group', 
                color_discrete_map=color_map
            )
            st.plotly_chart(fig4, use_container_width=True)