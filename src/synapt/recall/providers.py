"""Provider protocols for the OSS/premium seam.

OSS defines the protocols; premium implements them via entry points.
This module contains:
- OrgProvider / OrgInfo — org identity resolution (premium#553)
- EntitlementProvider / FreeEntitlementProvider — capability gating (premium#558)

Premium registers implementations under the 'synapt.providers' entry point group.
OSS falls back to free/default providers when premium is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Org seam (premium#553)
# ---------------------------------------------------------------------------


@dataclass
class OrgInfo:
    """Resolved org identity."""

    org_id: str
    source: str  # "manifest" | "license" | "account" | "env"
    name: str | None = None
    metadata: dict | None = None


@runtime_checkable
class OrgProvider(Protocol):
    """Protocol for org identity resolution.

    OSS default: resolves org_id from gripspace manifest URL.
    Premium: resolves from license/account with membership + settings.
    """

    def resolve_org(self) -> OrgInfo | None:
        """Resolve the current org. Returns None if no org context."""
        ...

    def org_metadata(self, org_id: str) -> dict:
        """Return org-level settings/metadata."""
        ...

    def validate_org(self, org_id: str) -> bool:
        """Check whether the given org_id is valid for this session."""
        ...


# ---------------------------------------------------------------------------
# Entitlements seam (premium#558)
# ---------------------------------------------------------------------------


@runtime_checkable
class EntitlementProvider(Protocol):
    """Protocol for capability/entitlement queries.

    OSS default: free tier (all features disabled).
    Premium: resolves from license.key / account.json / env.
    """

    def has_feature(self, feature: str) -> bool:
        """Return whether a named premium capability is enabled."""
        ...

    def tier(self) -> str:
        """Return the current entitlement tier (free, team, enterprise)."""
        ...

    def is_degraded(self) -> bool:
        """Return True if entitlements have expired or are degraded."""
        ...

    def summary(self) -> dict[str, bool]:
        """Return a dict of feature name -> enabled status."""
        ...


class FreeEntitlementProvider:
    """Default provider when premium is not installed. Everything is free tier."""

    def has_feature(self, feature: str) -> bool:
        return False

    def tier(self) -> str:
        return "free"

    def is_degraded(self) -> bool:
        return False

    def summary(self) -> dict[str, bool]:
        return {}
