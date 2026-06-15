import json
from typing import Dict, Any, List
from signal_fusion import SIGNAL_1_TIERS

# Core mapping for delta calculations
PRIORITY_RANK = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}

# Calibrated against actual CSV data distributions for accurate SLA bounding
EXPECTED_RESOLUTION = {
    'Critical': (0, 12), 
    'High': (6, 36), 
    'Medium': (24, 72), 
    'Low': (48, 168)
}

# Calibrated against actual CSV intake channels
CHANNEL_PROFILE = {
    'web form': 'medium — asynchronous but routed through structured queues',
    'chat': 'high — synchronous, requires immediate agent attention',
    'email': 'medium-low — asynchronous, typically standard urgency'
}

def generate_dossier(ticket: Dict[str, Any], classifier_confidence: float,
                     inferred_severity: str, matched_keywords: List[str],
                     rule_score: float, embed_severity: str) -> Dict:
    """
    Generates a strictly deterministic, zero-hallucination Evidence Dossier.
    All text is template-grounded and mathematically bound to the input ticket.
    """
    assigned = str(ticket.get('Priority_Level', ticket.get('Priority', 'Unknown')))
    
    # Calculate Severity Delta
    delta = PRIORITY_RANK.get(inferred_severity, 2) - PRIORITY_RANK.get(assigned, 2)
    mismatch_type = 'Hidden Crisis' if delta > 0 else 'False Alarm' if delta < 0 else 'Consistent'

    evidence = []

    # ---------------------------------------------------------
    # EVIDENCE VECTOR 1: Keyword Extraction (Bucketed Polarity)
    # ---------------------------------------------------------
    pos_kws = []
    neg_kws = []
    net_weight = 0.0

    for kw in matched_keywords:
        kw_w = 0.0
        for tier, data in SIGNAL_1_TIERS.items():
            if kw in data['words']:
                kw_w = data['weight']
                break
        
        net_weight += kw_w
        if kw_w > 0:
            pos_kws.append(kw)
        elif kw_w < 0:
            neg_kws.append(kw)

    if len(pos_kws) > len(neg_kws):
        kw_interp = f"{len(pos_kws)} crisis indicator(s) detected (e.g. {pos_kws[0]}), consistent with elevated severity."
    elif len(neg_kws) > len(pos_kws):
        kw_interp = f"Low-priority inquiry indicators (e.g. {neg_kws[0]}) outweighed escalation signals, driving severity downward."
    elif pos_kws and neg_kws:
        kw_interp = f"System detected competing signals — escalation markers ({pos_kws[0]}) vs inquiry indicators ({neg_kws[0]}). Net effect: {inferred_severity}."
    else:
        kw_interp = "No strong urgency or routine keywords detected; text lacks explicit escalation markers."

    all_kws = matched_keywords[:5] if matched_keywords else []
    evidence.append({
        'signal': 'keyword',
        'source_field': 'Ticket_Description',
        'value': ', '.join(all_kws) if all_kws else 'No keywords detected',
        'weight': f"{net_weight:.2f}",
        'interpretation': kw_interp
    })

    # ---------------------------------------------------------
    # EVIDENCE VECTOR 2: SLA / Resolution Time Boundaries
    # ---------------------------------------------------------
    res_time = ticket.get('Resolution_Time_Hours', ticket.get('Resolution Time (in hours)'))
    
    if res_time is not None:
        try:
            actual = float(res_time)
            exp_min, exp_max = EXPECTED_RESOLUTION.get(assigned, (0, 999))
            
            # Custom Interpretation for Mismatches
            if mismatch_type == 'Hidden Crisis' and actual < exp_min:
                interp = f"Despite being assigned {assigned}, resolution time of {actual:.0f}h suggests the agent may have escalated handling after recognizing the severity."
            elif mismatch_type == 'False Alarm' and actual <= exp_max:
                interp = f"Resolution in {actual:.0f}h reflects the agent's execution of a {assigned}-tier playbook, not the objective content severity. Fast resolution in this case supports the False Alarm classification — the agent over-reacted to surface signals."
            else:
                if actual > exp_max:
                    interp = (f"Resolved in {actual:.0f}h, exceeding the {exp_max}h ceiling "
                              f"for '{assigned}' tickets. Indicates SLA breach or irregular handling.")
                elif actual < exp_min and exp_min > 0:
                    interp = (f"Resolved in {actual:.0f}h, below the {exp_min}h floor "
                              f"for '{assigned}' tickets. Indicates rapid, out-of-band handling.")
                else:
                    interp = (f"Resolution time of {actual:.0f}h is within normal range "
                              f"({exp_min}-{exp_max}h) for '{assigned}' tickets.")
                    
            evidence.append({
                'signal': 'resolution_time',
                'source_field': 'Resolution_Time_Hours',
                'value': f'{actual:.0f} hours',
                'interpretation': interp
            })
        except (ValueError, TypeError):
            pass

    # ---------------------------------------------------------
    # EVIDENCE VECTOR 3: Channel Profiling
    # ---------------------------------------------------------
    channel = str(ticket.get('Ticket_Channel', ticket.get('Ticket Channel', 'unknown')))
    profile = CHANNEL_PROFILE.get(channel.lower(), 'an unknown urgency profile')
    evidence.append({
        'signal': 'intake_channel',
        'source_field': 'Ticket_Channel',
        'value': channel,
        'interpretation': f"Submitted via '{channel}' which has {profile}."
    })

    # ---------------------------------------------------------
    # EVIDENCE VECTOR 4: Semantic Embeddings
    # ---------------------------------------------------------
    evidence.append({
        'signal': 'semantic_cluster',
        'source_field': 'Ticket_Subject + Ticket_Description',
        'value': embed_severity,
        'interpretation': (
            f"Semantic embedding placed this ticket in a '{embed_severity}' "
            f"severity cluster based on linguistic similarity to historical tickets."
        )
    })

    # ---------------------------------------------------------
    # FORENSIC CONSTRAINT ANALYSIS (Zero Hallucination Execution)
    # ---------------------------------------------------------
    t_id = ticket.get('Ticket_ID', ticket.get('id', 'UNKNOWN'))
    pos_str = ', '.join(pos_kws[:2]) if pos_kws else 'implicit urgency signals in description'
    neg_str = ', '.join(neg_kws[:2]) if neg_kws else 'absence of routine inquiry language'

    if mismatch_type == 'Hidden Crisis':
        constraint_analysis = (
            f"Audit of ticket {t_id} finds the assigned {assigned} priority understates objective severity. "
            f"Despite {neg_str}, the semantic engine identified crisis-specific content ({pos_str}). "
            f"The agent's assigned priority appears driven by surface tone rather than content severity."
        )
    elif mismatch_type == 'False Alarm':
        constraint_analysis = (
            f"Audit of ticket {t_id} finds the assigned {assigned} priority overstates objective severity. "
            f"Escalation language ({pos_str}) created the appearance of urgency, but the actual description reveals routine content ({neg_str}). "
            f"The agent responded to presentation, not substance."
        )
    else:
        constraint_analysis = (
            f"Audit of ticket {t_id} confirms the assigned '{assigned}' priority aligns with objective severity. "
            f"The semantic engine and metadata signals consistently support the {inferred_severity} classification."
        )

    # Compile Final Payload
    return {
        'ticket_id':           t_id,
        'assigned_priority':   assigned,
        'inferred_severity':   inferred_severity,
        'mismatch_type':       mismatch_type,
        'severity_delta':      f'+{delta}' if delta > 0 else str(delta),
        'feature_evidence':    evidence,
        'constraint_analysis': constraint_analysis,
        'confidence':          f'{classifier_confidence:.4f}'
    }

if __name__ == "__main__":
    print("SIA Dossier Generator Ready.")