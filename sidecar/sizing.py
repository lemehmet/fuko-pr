"""Estimate the context window a review job needs, for context-fit routing.

A deliberately rough ``chars / 4`` token estimate over the PR diff plus the
injected knowledge, with a fixed reserve for the system prompt and the model's
own output. PR-Agent compresses and clips the diff internally, so this is only
accurate enough to skip a provider whose window clearly cannot hold the job; the
reserve is the safety margin.
"""

_CHARS_PER_TOKEN = 4
_RESERVE_TOKENS = 8000


def estimate_tokens(char_count: int) -> int:
    """Approximate a token count from a character count."""
    return char_count // _CHARS_PER_TOKEN


def required_context(
    diff_chars: int, knowledge_chars: int, reserve_tokens: int = _RESERVE_TOKENS
) -> int:
    """Estimate the context window a review of this size needs (input + reserve)."""
    return estimate_tokens(diff_chars + knowledge_chars) + reserve_tokens
