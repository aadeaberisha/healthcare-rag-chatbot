INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore the system message",
    "disregard previous instructions",
    "you are now",
    "act as",
    "system:",
    "assistant:",
]

def is_prompt_injection(text: str) -> bool:
    """
    Detects simple prompt-injection attempts in user input.
    This is a lightweight guardrail for demo purposes.
    """
    t = text.lower()
    return any(p in t for p in INJECTION_PATTERNS)
