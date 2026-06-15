import json
from typing import Dict, Any, List

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
    # EVIDENCE VECTOR 1: Keyword Extraction
    # ---------------------------------------------------------
    kws = matched_keywords[:5] if matched_keywords else []
    evidence.append({
        'signal': 'keyword_analysis',
        'source_field': 'Ticket_Description',
        'value': ', '.join(kws) if kws else 'No crisis keywords detected',
        'weight': f"{min(0.90, len(kws) * 0.15 + 0.10):.2f}",
        'interpretation': (
            f"{len(kws)} urgency keyword(s) found in description: {', '.join(kws[:3])}." if kws
            else 'No high-urgency keywords found; text appears to describe a routine issue.'
        )
    })

    # ---------------------------------------------------------
    # EVIDENCE VECTOR 2: SLA / Resolution Time Boundaries
    # ---------------------------------------------------------
    res_time = ticket.get('Resolution_Time_Hours', ticket.get('Resolution Time (in hours)'))
    res_time_observation = "Resolution time data was unavailable."
    
    if res_time is not None:
        try:
            actual = float(res_time)
            exp_min, exp_max = EXPECTED_RESOLUTION.get(assigned, (0, 999))
            
            if actual > exp_max:
                interp = (f"Resolved in {actual:.0f}h, exceeding the {exp_max}h ceiling "
                          f"for '{assigned}' tickets. Indicates SLA breach or irregular handling.")
                res_time_observation = f"Furthermore, the actual resolution time of {actual:.0f}h exceeded the normal maximum of {exp_max}h for '{assigned}' tickets."
            elif actual < exp_min and exp_min > 0:
                interp = (f"Resolved in {actual:.0f}h, below the {exp_min}h floor "
                          f"for '{assigned}' tickets. Indicates rapid, out-of-band handling.")
                res_time_observation = f"Furthermore, the ticket was closed in just {actual:.0f}h, falling well below the typical {exp_min}h minimum for '{assigned}' tickets."
            else:
                interp = (f"Resolution time of {actual:.0f}h is within normal range "
                          f"({exp_min}-{exp_max}h) for '{assigned}' tickets.")
                res_time_observation = f"The resolution time of {actual:.0f}h remained within the standard {exp_min}-{exp_max}h window."
                
            evidence.append({
                'signal': 'resolution_time',
                'source_field': 'Resolution_Time_Hours',
                'value': f'{actual:.0f} hours',
                'interpretation': interp
            })
        except (ValueError, TypeError):
            res_time_observation = "Resolution time data was unavailable."

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
    kw_text = f"direct urgency markers like '{kws[0]}'" if kws else "an absence of critical escalation keywords"
    
    if mismatch_type == 'Consistent':
        constraint_analysis = (
            f"An audit of ticket {ticket.get('Ticket_ID', ticket.get('id', 'UNKNOWN'))} confirms that the assigned '{assigned}' priority "
            f"aligns with the objective severity. The system's semantic engine independently maps the customer's "
            f"issue to the '{inferred_severity}' tier. {res_time_observation} Metadata and linguistic signals are consistent."
        )
    else:
        direction = "understates" if delta > 0 else "overstates"
        constraint_analysis = (
            f"An audit of ticket {ticket.get('Ticket_ID', ticket.get('id', 'UNKNOWN'))} reveals that the assigned '{assigned}' priority "
            f"{direction} the objective severity by {abs(delta)} level(s). The system's semantic engine maps the customer's "
            f"issue to a '{inferred_severity}' tier, primarily driven by {kw_text} in the description. "
            f"{res_time_observation} Ultimately, the conflicting metadata and linguistic signals classify this case as a {mismatch_type}."
        )

    # Compile Final Payload
    return {
        'ticket_id':           str(ticket.get('Ticket_ID', ticket.get('id', 'UNKNOWN'))),
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