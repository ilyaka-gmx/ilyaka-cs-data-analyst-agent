"""
TOON formatter for compact multi-record tool outputs.

TOON format: field names declared once as header, pipe-delimited rows.
Reduces token usage by 30-60% vs JSON for tabular data sent to LLMs.

Reference: https://toonformat.dev
"""


def to_toon(records: list[dict], name: str, fields: list[str]) -> str:
    """Convert a list of dicts to TOON format.

    Args:
        records: List of dictionaries to format.
        name: Label for the record set (e.g., "examples", "results").
        fields: Ordered list of field names to include.

    Returns:
        TOON-formatted string.

    Example:
        >>> to_toon([{"a": 1, "b": "hello"}], "data", ["a", "b"])
        'data[1]{a | b}:\\n1 | hello'
    """
    if not records:
        return f"{name}[0]: (empty)"
    header = f"{name}[{len(records)}]{{{' | '.join(fields)}}}:"
    rows = []
    for r in records:
        values = []
        for f in fields:
            val = str(r.get(f, ""))
            val = val.replace("|", "\\|")
            values.append(val)
        rows.append(" | ".join(values))
    return header + "\n" + "\n".join(rows)
