"""Curated lexicons for the Professional Services conversation profile.

PS conversations are more formal than SaaS by default. Vocabulary centers on
client engagements, deliverables, project management, and SOW language.
"""

# ============================================================
# EMPATHY LEXICON
# ============================================================

# Phrases indicating the consultant acknowledges the client's situation
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
    "i appreciate your patience with",
    "i understand the impact this has",
    "that's a valid concern regarding the project",
    "i recognize the urgency here",
    "i appreciate you flagging this",
    "i understand how critical this is",
    "that's an important point",
    "we take that seriously",
    "i can see why that's a concern",
    "we appreciate your candor",
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
    "concerned",
    "worry",
    "worried",
    "anxious",
    "difficult",
    "overwhelming",
    "pressure",
    "urgent",
    "critical",
    "blocked",
    "at risk",
    "jeopardized",
    "impacted",
    "setback",
]

# Apology terms
APOLOGY_LEXICON = [
    "apologize",
    "apologies",
    "my apologies",
    "our apologies",
    "i'm sorry",
    "so sorry",
    "sorry about",
    "sorry for",
    "sincere apologies",
    "we're sorry",
    "deeply sorry",
    "we regret",
    "i regret",
    "we understand this is not ideal",
]

# Action verbs for follow-through detection (must follow acknowledgment phrase within 20 tokens)
ACTION_VERB_LEXICON = [
    "resolve",
    "escalate",
    "schedule",
    "confirm",
    "update",
    "investigate",
    "coordinate",
    "prioritize",
    "follow up",
    "review",
    "assess",
    "align",
    "facilitate",
    "engage",
    "assign",
    "allocate",
    "mobilize",
    "remediate",
    "mitigate",
    "address",
    "action",
    "deliver",
    "communicate",
    "brief",
    "consult",
    "advise",
    "document",
    "prepare",
    "present",
    "clarify",
]

# ============================================================
# RESOLUTION LEXICON
# ============================================================

# Patterns indicating a solution or path forward has been provided (regex patterns)
RESOLUTION_PATTERNS = [
    r"\byou can\b",
    r"\bwe'?ve\b",
    r"\bi'?ve addressed\b",
    r"\bhere'?s how\b",
    r"\bto resolve\b",
    r"\bthe solution\b",
    r"\bplease review\b",
    r"\bsteps to\b",
    r"\bthe approach\b",
    r"\bour recommendation\b",
    r"\bwe have\b",
    r"\bi have\b",
    r"\bthis will\b",
    r"\bthis approach will\b",
    r"\bthat should\b",
    r"\bthis should\b",
    r"\bper the sow\b",
    r"\bas outlined in\b",
    r"\bour team will\b",
    r"\bper our agreement\b",
    r"\bas agreed\b",
    r"\bper the engagement\b",
    r"\bin accordance with\b",
]

# Patterns indicating deflection without help
DEFLECTION_PATTERNS = [
    r"\bplease contact\b",
    r"\brefer to\b",
    r"\byour project manager\b",
    r"\boutside (?:the )?scope\b",
    r"\bnot covered (?:in|by|under)\b",
    r"\boutside (?:my|our) (?:scope|expertise|remit)\b",
    r"\byou'?ll need to\b.{0,40}\bwith (?:your|the) (?:account|project|engagement)\b",
    r"\braise a change request\b",
    r"\bsubmit a change order\b",
    r"\bthis would require a new sow\b",
    r"\byour account team\b",
    r"\byour engagement manager\b",
]

# Future-tense / next-steps language
NEXT_STEPS_PATTERNS = [
    r"\bwill\b",
    r"\bgoing to\b",
    r"\bnext step\b",
    r"\baction item\b",
    r"\bwe'?ll\b",
    r"\bi'?ll\b",
    r"\bby [a-z]+ (?:morning|afternoon|evening|day|week|friday|monday)\b",
    r"\bwithin \d+",
    r"\bby end of\b",
    r"\btoday\b",
    r"\btomorrow\b",
    r"\bby (?:the )?next (?:call|meeting|check.?in|session)\b",
    r"\bby (?:the )?end of (?:sprint|phase|milestone)\b",
    r"\bprior to (?:the )?next\b",
    r"\bin (?:the )?next (?:business day|working day)\b",
]

# Temporal anchors (subset of next_steps)
TEMPORAL_ANCHOR_PATTERNS = [
    r"\bby [a-z]+ (?:morning|afternoon|evening|day|week|friday|monday)\b",
    r"\bwithin \d+ (?:hour|minute|day|business day|working day)\b",
    r"\bby end of (?:day|week|business day|sprint|phase)\b",
    r"\bby tomorrow\b",
    r"\bby today\b",
    r"\bin \d+ (?:hour|minute|day|business day)\b",
    r"\bby (?:the )?next (?:call|meeting|check.?in)\b",
    r"\bbefore (?:the )?next (?:milestone|phase|sprint)\b",
]

# Specific actor patterns (make next_steps actionable)
SPECIFIC_ACTOR_PATTERNS = [
    r"\bi'?ll\b",
    r"\bour team will\b",
    r"\bour delivery team\b",
    r"\bour project team\b",
    r"\bthe engagement team\b",
    r"\bour technical lead\b",
    r"\bi'?m going to\b",
    r"\bi will\b",
    r"\bwe will\b",
    r"\bwe'?ll\b",
    r"\bour solutions architect\b",
    r"\bthe workstream lead\b",
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
    r"\bi'?ll own\b",
    r"\bwe'?ll own\b",
    r"\bwe'?re accountable\b",
    r"\bwe'?re responsible for\b",
]

# Issue keyword overlap — terms that appear in client descriptions of their issue.
# Used to verify solution_type=="complete" (solution references the client's issue).
ISSUE_KEYWORDS = [
    "deliverable",
    "milestone",
    "scope",
    "timeline",
    "sow",
    "requirement",
    "stakeholder",
    "workstream",
    "dependency",
    "blocker",
    "risk",
    "escalation",
    "delay",
    "overdue",
    "sign-off",
    "approval",
    "feedback",
    "review",
    "integration",
    "data",
    "migration",
    "configuration",
    "training",
    "documentation",
    "handover",
    "go-live",
    "cutover",
    "acceptance",
    "uat",
    "testing",
    "deployment",
    "rollout",
    "budget",
    "resource",
    "capacity",
    "availability",
    "access",
    "environment",
    "change request",
    "change order",
]

# ============================================================
# BRAND VOICE LEXICONS
# ============================================================

# Warm-relational brand voice characteristic terms (PS equivalent of warm_exploratory)
WARM_TERMS = [
    "happy to",
    "glad to",
    "pleased to",
    "delighted to",
    "appreciate",
    "value",
    "partnership",
    "collaborate",
    "together",
    "jointly",
    "shared",
    "our shared",
    "committed",
    "dedicated",
    "invested",
    "supportive",
    "proactive",
    "thoughtful",
    "transparent",
    "open",
    "candid",
    "trust",
    "relationship",
]

# Formal-professional brand voice characteristic terms
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
    "document",
    "remediate",
    "resolve",
    "validate",
    "authorize",
    "provision",
    "facilitate",
    "coordinate",
    "mobilize",
    "align",
    "assess",
    "evaluate",
    "mitigate",
    "escalate",
    "remediate",
    "deliverable",
    "milestone",
    "workstream",
    "engagement",
    "stakeholder",
]

# Hedging terms (less common in PS but present in exploratory / discovery phases)
HEDGING_LEXICON = [
    "might",
    "could",
    "perhaps",
    "it seems",
    "maybe",
    "possibly",
    "it appears",
    "it looks like",
    "we believe",
    "we think",
    "seems like",
    "subject to",
    "pending",
    "contingent on",
    "depending on",
    "assuming",
    "provided that",
    "if confirmed",
]

# Directive terms (high in PS given instructional / advisory nature)
DIRECTIVE_LEXICON = [
    "please",
    "review",
    "confirm",
    "approve",
    "sign off",
    "complete",
    "submit",
    "provide",
    "send",
    "share",
    "upload",
    "download",
    "follow",
    "use",
    "check",
    "select",
    "navigate",
    "go to",
    "open",
    "close",
    "enable",
    "disable",
    "configure",
    "implement",
    "execute",
    "run",
    "test",
    "validate",
    "document",
]

# Contraction patterns — high contraction count = informal (lower formality score).
# PS voice generally avoids contractions in written comms.
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
    # PS-specific
    "change request": "scope change",
    "change order": "scope change",
    "cr": "scope change",
    "deliverable": "output",
    "artefact": "output",
    "artifact": "output",
    "workplan": "project plan",
    "work plan": "project plan",
    "statement of work": "sow",
    "master services agreement": "msa",
    "nda": "non-disclosure agreement",
    "poc": "proof of concept",
    "pilot": "proof of concept",
    "go-live": "cutover",
    "go live": "cutover",
    "launch": "cutover",
    "user acceptance testing": "uat",
    "signoff": "sign-off",
    "sign off": "sign-off",
    "check-in": "status call",
    "check in": "status call",
    "standup": "status call",
    "stand-up": "status call",
    "retrospective": "retro",
    "requirements gathering": "discovery",
    "scoping": "discovery",
    # Shared with SaaS
    "reimbursement": "refund",
    "reimburse": "refund",
    "terminate": "cancel",
    "cancellation": "cancel",
    "cancelled": "cancel",
    "billing": "payment",
    "invoice": "payment",
    "invoiced": "payment",
    "login": "access",
    "sign in": "access",
    "authenticate": "access",
    "authorization": "access",
    "reach out": "contact",
    "get in touch": "contact",
    "per our": "according to our",
    "as per": "according to",
    "per the": "according to the",
    "per": "according to",
}

# ============================================================
# BRAND VOICE VARIANT FEATURE PROFILES
# ============================================================
# Calibrated ranges per brand voice variant.
# PS is generally more formal than SaaS — formality_range is shifted upward.
# Structure: {field: (min, max)}

BRAND_VOICE_FEATURE_PROFILES: dict[str, dict] = {
    "bv_warm_relational": {
        "sentence_length_range": (10, 28),      # words per sentence — slightly longer than SaaS warm
        "question_count_range": (1, 4),          # questions per response
        "hedging_range": (1, 6),                 # hedging terms
        "directive_range": (1, 5),               # directive terms
        "aligned_term_field": "warm_terms_count",
        "aligned_terms_min": 2,
        "formality_range": (0.5, 0.75),          # more formal than SaaS warm exploratory
    },
    "bv_formal_advisory": {
        "sentence_length_range": (8, 22),
        "question_count_range": (0, 2),
        "hedging_range": (0, 3),
        "directive_range": (4, 18),
        "aligned_term_field": "clinical_terms_count",
        "aligned_terms_min": 3,
        "formality_range": (0.7, 1.0),           # high formality
    },
    "bv_baseline": {
        "sentence_length_range": (8, 24),
        "question_count_range": (0, 3),
        "hedging_range": (0, 5),
        "directive_range": (2, 10),
        "aligned_term_field": "warm_terms_count",
        "aligned_terms_min": 0,  # no minimum for baseline
        "formality_range": (0.5, 0.85),          # PS baseline is more formal than SaaS baseline
    },
}
