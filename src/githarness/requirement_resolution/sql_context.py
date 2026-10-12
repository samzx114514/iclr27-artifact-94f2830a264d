"""Remove fixed SQL transport assets from evolving requirement feedback."""

from __future__ import annotations


SQL_REQUEST_PREAMBLE = (
    "Write a SQL query to answer the following question. Return only the SQL "
    "query in a ```sql code block. Select only the column(s) the question asks "
    "for — do not include extra columns such as IDs, counts, or aggregates "
    "unless explicitly requested."
)
_SCHEMA_MARKER = "Database schema:\n"
_EVIDENCE_MARKER = "\n\nEvidence:\n"


def compact_sql_feedback(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("SQL feedback must be a string")
    compact = value.strip()
    if compact.startswith(SQL_REQUEST_PREAMBLE):
        compact = compact[len(SQL_REQUEST_PREAMBLE):].lstrip()
    if compact.startswith(_SCHEMA_MARKER):
        remainder = compact[len(_SCHEMA_MARKER):]
        if _EVIDENCE_MARKER in remainder:
            _, tail = remainder.split(_EVIDENCE_MARKER, 1)
            compact = "Evidence:\n" + tail
        elif "\n\n" in remainder:
            _, compact = remainder.rsplit("\n\n", 1)
    return compact.strip() or value.strip()
