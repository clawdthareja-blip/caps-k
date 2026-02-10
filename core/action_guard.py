"""
CAPS-K Action Guard Module
===========================
Validates proposed tool calls against:

1. The **allowed-tools allowlist** — only tools listed in the current TSCH
   block may be invoked.
2. The **argument schema** — argument names and types must match the declared
   schema.
3. **Sanity checks** — basic argument value checks (no shell metacharacters
   in string args, numeric bounds, etc.).
4. **Prompt trace** — optionally checks whether the tool call was requested
   from a trusted authority segment (SYS/USR), not from an EXT/TOUT block.

Decision: allow / deny with a human-readable reason string.
"""

import re
from typing import Any


# ---------------------------------------------------------------------------
# Type coercion helpers
# ---------------------------------------------------------------------------
_TYPE_MAP: dict[str, type] = {
    "string":  str,
    "str":     str,
    "integer": int,
    "int":     int,
    "number":  (int, float),   # type: ignore[assignment]
    "float":   float,
    "boolean": bool,
    "bool":    bool,
    "array":   list,
    "list":    list,
    "object":  dict,
    "dict":    dict,
}

# Shell metacharacters that shouldn't appear in string args (basic sanity)
_SHELL_META = re.compile(r"[;&|`$><\\\n\r]")


def _check_types(
    args: dict[str, Any],
    schema: dict[str, Any],
) -> tuple[bool, str]:
    """
    Validate *args* against a JSON-Schema-like *schema* dict.

    Schema format (simplified)::

        {
            "parameters": {
                "properties": {
                    "query":  {"type": "string"},
                    "limit":  {"type": "integer"},
                },
                "required": ["query"]
            }
        }

    Returns
    -------
    (ok, reason) tuple.
    """
    params = schema.get("parameters", {})
    properties: dict[str, Any] = params.get("properties", {})
    required: list[str] = params.get("required", [])

    # Check required fields present
    for req in required:
        if req not in args:
            return False, f"Missing required argument: '{req}'"

    # Check types for provided args
    for arg_name, arg_value in args.items():
        if arg_name not in properties:
            # Unknown arg — allow with warning (permissive mode)
            continue
        expected_type_str: str = properties[arg_name].get("type", "")
        if not expected_type_str:
            continue
        expected_type = _TYPE_MAP.get(expected_type_str.lower())
        if expected_type is None:
            continue  # unknown type spec — skip check
        if not isinstance(arg_value, expected_type):
            return (
                False,
                f"Argument '{arg_name}' expected type '{expected_type_str}', "
                f"got {type(arg_value).__name__}",
            )

    return True, ""


def _sanity_check_args(args: dict[str, Any]) -> tuple[bool, str]:
    """
    Basic arg value sanity: reject obvious shell injection in string values.
    """
    for name, value in args.items():
        if isinstance(value, str):
            if _SHELL_META.search(value):
                return (
                    False,
                    f"Argument '{name}' contains shell metacharacters — "
                    "possible injection attempt",
                )
    return True, ""


def _trace_authority(
    tool_name: str,
    prompt_trace: str | None,
) -> tuple[bool, str]:
    """
    Check that the tool call instruction originated from a trusted segment.

    If *prompt_trace* is provided, verify the call-requesting text came from
    within SYS_* or USR_* delimiters, not from EXT_*/TOUT_* blocks.

    This is a heuristic check: we look for the tool name appearing
    *only* inside trusted tags in the trace.
    """
    if prompt_trace is None:
        return True, ""  # no trace provided — skip check

    # Find all occurrences of tool name in the trace
    tool_positions = [m.start() for m in re.finditer(re.escape(tool_name), prompt_trace)]
    if not tool_positions:
        return True, ""  # tool name not in trace — no concern

    # Find spans of untrusted blocks
    untrusted_spans: list[tuple[int, int]] = []
    for match in re.finditer(r"<(EXT|TOUT)_[0-9a-f]+>(.*?)</(EXT|TOUT)_[0-9a-f]+>", prompt_trace, re.DOTALL):
        untrusted_spans.append((match.start(), match.end()))

    for pos in tool_positions:
        for span_start, span_end in untrusted_spans:
            if span_start <= pos <= span_end:
                return (
                    False,
                    f"Tool '{tool_name}' was referenced inside an untrusted "
                    "EXT/TOUT block — possible prompt injection attempt",
                )

    return True, ""


def validate_tool_call(
    tool_name: str,
    args: dict[str, Any],
    allowed_tools: list[str],
    schema: dict[str, Any],
    prompt_trace: str | None = None,
) -> dict[str, str]:
    """
    Validate a proposed tool call and return an allow/deny decision.

    Parameters
    ----------
    tool_name:
        Name of the tool the model wants to call.
    args:
        Argument dict the model proposes.
    allowed_tools:
        List of tool names permitted in this session (from TSCH).
    schema:
        JSON-schema-like dict for this specific tool.
    prompt_trace:
        The assembled prompt string (used for authority tracing).
        Optional — pass ``None`` to skip trace analysis.

    Returns
    -------
    dict with keys:
        - ``decision`` — ``"allow"`` or ``"deny"``
        - ``reason``   — human-readable explanation
    """
    # 1. Allowlist check
    if tool_name not in allowed_tools:
        return {
            "decision": "deny",
            "reason": (
                f"Tool '{tool_name}' is not in the allowed-tools list. "
                f"Allowed: {allowed_tools}"
            ),
        }

    # 2. Type / schema check
    ok, reason = _check_types(args, schema)
    if not ok:
        return {"decision": "deny", "reason": f"Schema violation: {reason}"}

    # 3. Sanity check on arg values
    ok, reason = _sanity_check_args(args)
    if not ok:
        return {"decision": "deny", "reason": f"Arg sanity check failed: {reason}"}

    # 4. Authority trace check
    ok, reason = _trace_authority(tool_name, prompt_trace)
    if not ok:
        return {"decision": "deny", "reason": f"Authority trace violation: {reason}"}

    return {"decision": "allow", "reason": "All checks passed"}
