"""Curated lexicons for the SaaS conversation profile."""
import re

# ============================================================
# EMPATHY LEXICON
# ============================================================

# Phrases indicating the agent acknowledges the customer's situation
ACKNOWLEDGMENT_PHRASES = [
    "i understand",
    "i can see",
    "i can imagine",
    "that makes sense",
    "i hear you",
    "i appreciate",
    "thank you for letting us know",
    "thank you for reaching out",
    "sorry you're experiencing",
    "sorry you're hitting",
    "sorry you're running into",
    "sorry you're seeing",
    "sorry to hear",
    "i'm sorry that",
    "i know this is",
    "i realize this is",
    "that must be",
    "understandably",
    "completely understand",
    "totally understand",
    "i see what you mean",
    "i get that",
    "i know how",
    "i'd be happy",
    "happy to help",
    "no problem",
    "got it",
    "good question",
    "of course",
    "absolutely",
    "sure thing",
]

# Terms indicating emotional awareness
EMOTION_LEXICON = [
    "frustrated",
    "frustration",
    "inconvenient",
    "inconvenience",
    "confusing",
    "confusion",
    "stressful",
    "stress",
    "upset",
    "disappointing",
    "disappointed",
    "annoyed",
    "concerned",
    "worry",
    "worried",
    "anxious",
    "difficult",
    "overwhelming",
    "trouble",
    "struggle",
    "struggling",
    "glad",
    "happy",
    "pleased",
    "delighted",
    "wonderful",
    "great to hear",
    "good luck",
]

# Apology terms
APOLOGY_LEXICON = [
    "apologize",
    "apologies",
    "my apologies",
    "i'm sorry",
    "so sorry",
    "sorry about",
    "sorry for",
    "sincere apologies",
    "we're sorry",
    "deeply sorry",
]

# Action verbs for follow-through detection (must follow acknowledgment phrase within 20 tokens)
ACTION_VERB_LEXICON = [
    "fix",
    "resolve",
    "send",
    "escalate",
    "refund",
    "replace",
    "schedule",
    "confirm",
    "process",
    "update",
    "check",
    "look into",
    "investigate",
    "handle",
    "arrange",
    "coordinate",
    "expedite",
    "prioritize",
    "follow up",
    "create",
    "submit",
    "open",
    "close",
    "transfer",
    "connect",
    "reach out",
    "get back",
    "reach back",
    "walk",
    "guide",
    "show",
    "help",
    "assist",
    "explain",
    "clarify",
    "take you through",
]

# ============================================================
# RESOLUTION LEXICON
# ============================================================

# Verb stems that, when preceded by \bi'?ve\s+, signal a completed action.
# Used by _already_resolved() in the resolution extractor, AND by
# COMPLETION_RESOLUTION_PATTERNS so that solution_provided fires for all
# completion verbs — not just "fixed".
COMPLETION_VERB_PATTERNS = [
    "fixed", "resolved", "updated", "applied", "corrected",
    "pushed", "deployed", "patched", "removed", "added",
    "reset", "changed", "adjusted", "rebuilt", "cleared",
    "sent", "issued", "processed", "created",
]

# Dynamic alternation built from COMPLETION_VERB_PATTERNS so that adding a
# verb to the list above automatically extends both _already_resolved detection
# and solution_provided triggering.
_COMPLETION_VERB_ALT = "|".join(re.escape(v) for v in COMPLETION_VERB_PATTERNS)

# Completion-oriented subset of RESOLUTION_PATTERNS.
# Only patterns in this list can trigger the pronoun-reference fallback
# in _classify_solution_type() — investigative I'll-verbs are excluded.
COMPLETION_RESOLUTION_PATTERNS = [
    r"\byou can\b",
    r"\bwe'?ve\b",
    rf"\bi'?ve\s+(?:{_COMPLETION_VERB_ALT})\b",  # covers all completion verbs
    r"\bhere'?s how\b",
    r"\bto resolve\b",
    r"\bthe solution\b",
    r"\bplease try\b",
    r"\bsteps to\b",
    r"\bfollow these\b",
    r"\bthe fix\b",
    r"\bwe have\b",
    r"\bi have\b",
    r"\bthis will\b",
    r"\byou should be able\b",
    r"\bthat should\b",
    r"\bthis should\b",
    r"\byou'?ll want to\b",
    r"\byou'?d need to\b",
    r"\byou need to\b",
    r"\bgo to\b",
    r"\bclick (?:on |the )\b",
    r"\bcopy (?:the |that )\b",
    r"\bpaste (?:it|that|the)\b",
    r"\bwhat you need to do\b",
    # Escalation / routing — completion actions (agent routes to resolution)
    r"\bi'?m (?:escalating|routing|transferring)\b",
    r"\bi'?ll (?:fix|update|resolve|escalate|loop in|bring in)\b",
    r"\bescalating this\b",
    r"\bour (?:engineering|billing|technical|infrastructure) team\b.{0,40}\bwill\b",
]

# Full resolution pattern set — includes COMPLETION_RESOLUTION_PATTERNS plus
# investigative I'll-verbs that set solution_provided=True but are NOT
# completion-oriented (excluded from pronoun-reference fallback).
# NOTE: \byou should\b removed (RFORGE-43 — matched vague promises)
# NOTE: dig in included alongside dig into (must match "I'll dig in tomorrow")
RESOLUTION_PATTERNS = COMPLETION_RESOLUTION_PATTERNS + [
    r"\bi'?ll (?:look into|dig into|dig in|investigate|check)\b",
]

# Patterns indicating deflection without help
DEFLECTION_PATTERNS = [
    r"\bplease contact\b",
    r"\bcheck (?:the )?documentation\b",
    r"\bcheck (?:the )?docs\b",
    r"\bhave you (?:checked|tried|looked at)\b",
    r"\brefer to\b",
    r"\bour support team\b",
    r"\bsubmit a ticket\b",
    r"\bopen a ticket\b",
    r"\bfeel free to contact\b",
    r"\byou can find\b.{0,30}\bhelp center\b",
    r"\boutside (?:my|our) (?:scope|expertise)\b",
]

# Future-tense / next-steps language
NEXT_STEPS_PATTERNS = [
    r"\bwill\b",
    r"\bgoing to\b",
    r"\bnext step\b",
    r"\bwe'?ll\b",
    r"\bi'?ll\b",
    r"\bby [a-z]+ (?:morning|afternoon|evening|day|week|friday|monday)\b",
    r"\bwithin \d+",
    r"\bby end of\b",
    r"\btoday\b",
    r"\btomorrow\b",
]

# Temporal anchors (subset of next_steps)
TEMPORAL_ANCHOR_PATTERNS = [
    r"\bby [a-z]+ (?:morning|afternoon|evening|day|week|friday|monday)\b",
    r"\bwithin \d+[-–]\d+ (?:hours?|minutes?|days?|business days?)\b",
    r"\bwithin \d+ (?:hours?|minutes?|days?|business days?)\b",
    r"\bby end of (?:day|week|business day)\b",
    r"\bby tomorrow\b",
    r"\bby today\b",
    r"\bin \d+ (?:hours?|minutes?|days?)\b",
    r"\btoday\b",
]

# Specific actor patterns (make next_steps actionable)
SPECIFIC_ACTOR_PATTERNS = [
    r"\bi'?ll\b",
    r"\bour team will\b",
    r"\bour engineering team\b",
    r"\bour billing team\b",
    r"\bi'?m going to\b",
    r"\bi will\b",
    r"\bwe will\b",
    r"\bwe'?ll\b",
]

# Ownership language
OWNERSHIP_PATTERNS = [
    r"\bi will\b",
    r"\bi'?ve\b",
    r"\blet me\b",
    r"\bi can handle\b",
    r"\bi'?ll take\b",
    r"\bi'?m handling\b",
    r"\bi'?m going to\b",
    r"\bi'?m taking\b",
    r"\bmy responsibility\b",
    r"\bi'?m on it\b",
    # RFORGE-6: explicit I'll-action and escalation patterns
    r"\bi'?ll (?:fix|update|resolve|look into|dig into|dig in|investigate|check)\b",
    r"\bi'?m (?:escalating|routing|transferring)\b",
    r"\bi'?ve (?:already|just|now)\b",
]

# Issue keyword overlap — terms that appear in customer descriptions of their issue.
# Used to verify solution_type=="complete" (solution references the customer's issue).
ISSUE_KEYWORDS = [
    "billing",
    "charge",
    "refund",
    "payment",
    "subscription",
    "account",
    "access",
    "login",
    "password",
    "feature",
    "integration",
    "api",
    "data",
    "export",
    "import",
    "sync",
    "error",
    "bug",
    "broken",
    "not working",
    "failed",
    "crash",
    "slow",
    "performance",
    "timeout",
    "webhook",
    "notification",
]

# ============================================================
# BRAND VOICE LEXICONS
# ============================================================

# Warm-exploratory brand voice characteristic terms
WARM_TERMS = [
    "happy to",
    "love to",
    "excited",
    "wonderful",
    "great",
    "amazing",
    "fantastic",
    "absolutely",
    "definitely",
    "certainly",
    "of course",
    "let's",
    "together",
    "journey",
    "explore",
    "discover",
    "wondering",
    "curious",
    "interesting",
    "love",
    "enjoy",
    "appreciate",
    "delight",
    "glad",
    "thrilled",
    "warmly",
]

# Direct-clinical brand voice characteristic terms
CLINICAL_TERMS = [
    "confirm",
    "verify",
    "ensure",
    "proceed",
    "execute",
    "implement",
    "utilize",
    "leverage",
    "optimize",
    "configure",
    "initiate",
    "terminate",
    "escalate",
    "document",
    "remediate",
    "resolve",
    "validate",
    "authorize",
    "authenticate",
    "provision",
]

# Hedging terms (associated with warm-exploratory / less directive voice)
HEDGING_LEXICON = [
    "might",
    "could",
    "perhaps",
    "let's",
    "it seems",
    "maybe",
    "possibly",
    "it appears",
    "it looks like",
    "wondering if",
    "i believe",
    "i think",
    "seems like",
    "if that's",
    "if you'd",
    "you might want",
]

# Directive terms (associated with direct-clinical voice)
DIRECTIVE_LEXICON = [
    "do",
    "click",
    "follow",
    "use",
    "check",
    "select",
    "enter",
    "navigate",
    "go to",
    "open",
    "close",
    "click on",
    "tap",
    "scroll",
    "type",
    "copy",
    "paste",
    "download",
    "upload",
    "enable",
    "disable",
    "toggle",
]

# Contraction patterns — high contraction count = informal (lower formality score)
CONTRACTION_PATTERNS = [
    r"\bdon'?t\b",
    r"\bwon'?t\b",
    r"\bcan'?t\b",
    r"\bi'?m\b",
    r"\bi'?ll\b",
    r"\bi'?ve\b",
    r"\bwe'?re\b",
    r"\bwe'?ll\b",
    r"\bwe'?ve\b",
    r"\byou'?re\b",
    r"\byou'?ll\b",
    r"\byou'?ve\b",
    r"\bit'?s\b",
    r"\bthat'?s\b",
    r"\bthere'?s\b",
    r"\bthey'?re\b",
    r"\bisn'?t\b",
    r"\baren'?t\b",
    r"\bhasn'?t\b",
    r"\bhaven'?t\b",
    r"\bwasn'?t\b",
    r"\bweren'?t\b",
    r"\bdidn'?t\b",
    r"\bshouldn'?t\b",
    r"\bwouldn'?t\b",
    r"\bcouldn'?t\b",
]

# ============================================================
# SYNONYM MAP (for KB text normalization in accuracy extractor)
# ============================================================

SYNONYM_MAP: dict[str, str] = {
    "reimbursement": "refund",
    "reimburse": "refund",
    "terminate": "cancel",
    "cancellation": "cancel",
    "cancelled": "cancel",
    "subscription end": "cancel",
    "billing": "payment",
    "charge": "payment",
    "invoice": "payment",
    "invoiced": "payment",
    "login": "access",
    "sign in": "access",
    "authenticate": "access",
    "authorization": "access",
    "reach out": "contact",
    "get in touch": "contact",
    "drop us a line": "contact",
    "per": "according to",
    "per our": "according to our",
    "as per": "according to",
}

# ============================================================
# BRAND VOICE VARIANT FEATURE PROFILES
# ============================================================
# Calibrated ranges per brand voice variant.
# These are approximate ranges derived from typical exemplar responses.
# Structure: {field: (min, max)}

BRAND_VOICE_FEATURE_PROFILES: dict[str, dict] = {
    "bv_warm_exploratory": {
        "sentence_length_range": (8, 25),      # words per sentence
        "question_count_range": (1, 5),         # questions per response
        "hedging_range": (2, 10),               # hedging terms
        "directive_range": (0, 3),              # directive terms
        "aligned_term_field": "warm_terms_count",
        "aligned_terms_min": 2,
        "formality_range": (0.2, 0.6),          # lower = less formal
    },
    "bv_direct_clinical": {
        "sentence_length_range": (5, 15),
        "question_count_range": (0, 2),
        "hedging_range": (0, 2),
        "directive_range": (3, 15),
        "aligned_term_field": "clinical_terms_count",
        "aligned_terms_min": 2,
        "formality_range": (0.5, 1.0),
    },
    "bv_baseline": {
        "sentence_length_range": (6, 20),
        "question_count_range": (0, 4),
        "hedging_range": (0, 6),
        "directive_range": (0, 8),
        "aligned_term_field": "warm_terms_count",
        "aligned_terms_min": 0,  # no minimum for baseline
        "formality_range": (0.3, 0.8),
    },
}
