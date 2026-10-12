import hashlib
import json
from typing import Any, Mapping

from .base import UpdateDomainContract

CONTRACT = UpdateDomainContract("code", "patch.diff", ("/testbed",), ("testbed",))


def record_testbed_result(
    command: str,
    value: Any,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("/testbed execute returned an invalid result")
    returncode = value.get("returncode")
    output = value.get("output", "")
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        raise ValueError("/testbed result requires an integer returncode")
    if not isinstance(output, str):
        raise ValueError("/testbed result output must be text")
    runs.append({
        "command": command,
        "returncode": returncode,
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "output_bytes": len(output.encode("utf-8")),
    })
    return {"returncode": returncode, "output": output}


def test_receipt(runs: list[dict[str, Any]]) -> str:
    return json.dumps(
        {"schema_version": 1, "workspace": "/testbed", "test_runs": runs},
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
