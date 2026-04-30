"""SimEvent type registry — engagement-agnostic event vocabulary."""

from confabra.schemas import SimEventType

# Registry: event_type → required payload fields
SIM_EVENT_REGISTRY: dict[str, list[str]] = {
    SimEventType.ACCOUNT_CREATED: ["plan_tier", "industry"],
    SimEventType.CONVERSATION_STARTED: ["surface_channel", "agent_id", "customer_name", "domain"],
    SimEventType.CONVERSATION_ENDED: ["duration_minutes", "turn_count", "resolution_status"],
    SimEventType.PAYMENT_RECEIVED: ["amount_cents", "plan_tier"],
    SimEventType.PAYMENT_FAILED: ["amount_cents", "failure_reason"],
    SimEventType.FEATURE_USAGE: ["feature_name", "usage_count"],
    SimEventType.SUPPORT_TICKET_OPENED: ["ticket_id", "category", "priority"],
    SimEventType.SUPPORT_TICKET_RESOLVED: ["ticket_id", "resolution_minutes"],
    SimEventType.CHURN_SIGNAL_DETECTED: ["signal_type", "strength"],  # strength: 0.0-1.0
    SimEventType.ESCALATION_DETECTED: ["conversation_id", "escalation_reason"],
    SimEventType.RENEWAL_APPROACHING: ["days_remaining", "plan_tier"],
    SimEventType.RENEWAL_COMPLETED: ["plan_tier", "renewed_months"],
    SimEventType.RENEWAL_LAPSED: ["plan_tier", "lapse_reason"],
    SimEventType.COACHING_NOTE_ISSUED: ["agent_id", "criterion", "coaching_id"],
    SimEventType.SCORE_CORRECTION: ["agent_id", "criterion", "original_score", "corrected_score"],
}


def validate_event_payload(event_type: SimEventType, payload: dict) -> list[str]:
    """
    Validate that a payload contains all required fields for the given event type.

    Returns a list of missing required field names; an empty list means the
    payload is valid. Callers can use this to assert correctness during testing
    or to raise structured errors before emitting events into the stream.
    """
    required = SIM_EVENT_REGISTRY.get(event_type, [])
    return [f for f in required if f not in payload]
