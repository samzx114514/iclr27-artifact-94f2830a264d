"""Shared domain capability contract."""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class UpdateDomainContract:
    domain: str
    artifact_path: str
    public_capabilities: tuple[str, ...]
    runtime_asset_keys: tuple[str, ...]

    def validate_runtime_assets(self, assets: Mapping[str, Any]) -> None:
        allowed = {"capabilities", *self.runtime_asset_keys}
        unknown = set(assets) - allowed
        if unknown:
            raise ValueError(
                f"unsupported {self.domain} runtime assets: {sorted(unknown)}"
            )
        capabilities = assets.get("capabilities", self.public_capabilities)
        if not isinstance(capabilities, (list, tuple)):
            raise ValueError("public runtime capabilities must be a list")
        if tuple(capabilities) != self.public_capabilities:
            raise ValueError("public runtime capabilities do not match domain contract")
