"""Pydantic schema for the final caller-info record.

The reconciler emits a free-form dict; this schema is the gate that
turns it into a validated, normalized output. Failures are collected
per-field rather than raised - Layer 5 already chose the best candidate
for each key, and we'd rather emit an unvalidated fallback string than
drop the whole record.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

from src.validators import (
    is_valid_email_shape,
    is_valid_phone,
    normalize_email,
    normalize_name,
    normalize_phone,
)


class CallerInfo(BaseModel):
    first_name: str = ""
    last_name: str = ""
    email: str = ""
    phone_number: str = ""

    @field_validator("first_name", "last_name", mode="before")
    @classmethod
    def _names(cls, v):
        return normalize_name(v or "")

    @field_validator("email", mode="before")
    @classmethod
    def _email(cls, v):
        return normalize_email(v or "")

    @field_validator("phone_number", mode="before")
    @classmethod
    def _phone(cls, v):
        return normalize_phone(v or "")


class CallerInfoReport(BaseModel):
    """Validated caller info + a list of per-field validation notes."""

    info: CallerInfo
    warnings: List[str] = Field(default_factory=list)


def validate_caller_info(raw: dict) -> CallerInfoReport:
    """Always returns a report; never raises. Warnings list flags fields that
    look structurally bad so observability layers can flag them downstream."""
    safe = {
        "first_name": raw.get("first_name", "") or "",
        "last_name": raw.get("last_name", "") or "",
        "email": raw.get("email", "") or "",
        "phone_number": raw.get("phone_number", "") or "",
    }
    try:
        info = CallerInfo(**safe)
    except ValidationError:
        # Pydantic shouldn't raise for these (validators always return strings),
        # but defensively fall through to manual normalization just in case.
        info = CallerInfo(
            first_name=normalize_name(safe["first_name"]),
            last_name=normalize_name(safe["last_name"]),
            email=normalize_email(safe["email"]),
            phone_number=normalize_phone(safe["phone_number"]),
        )

    warnings: list[str] = []
    if info.email and not is_valid_email_shape(info.email):
        warnings.append(f"email shape invalid: {info.email!r}")
    if info.phone_number and not is_valid_phone(info.phone_number):
        warnings.append(f"phone not parseable: {info.phone_number!r}")

    return CallerInfoReport(info=info, warnings=warnings)
