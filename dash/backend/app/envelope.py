"""
Shared agent response envelope.

Every agent in the project (soft power, policy stance, trade, events) returns
this same shape so the orchestrator can fan out and the insight-fusion agent
can rank / corroborate without per-agent parsing. Field names follow the
Trade agent's contract:

    { "agent", "metadata": { "query_type", "year", "sector", "data_quality", ... },
      "insights": [ { "entity_iso3", "claim", "score", "confidence", "reason", "evidence" } ] }

Keep this file dependency-free apart from pydantic so other agents can copy it
verbatim.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Insight(BaseModel):
    entity_iso3: str = Field(min_length=3, max_length=3, description="Canonical ISO3 the claim is about")
    claim: str = Field(min_length=1, description="One human-readable sentence")
    score: float = Field(ge=0.0, le=1.0, description="Magnitude of the claim, normalised to 0-1")
    confidence: float = Field(ge=0.0, le=1.0, description="How much fusion should trust the claim")
    reason: str = Field(min_length=1, description="Why this confidence -- names the signal and its caveats")
    evidence: dict[str, Any] = Field(default_factory=dict, description="Raw values behind the claim")

    @field_validator("entity_iso3")
    @classmethod
    def upper_iso3(cls, value: str) -> str:
        return value.upper()


class Metadata(BaseModel):
    query_type: str
    year: Optional[int] = None
    sector: Optional[str] = None
    entities: list[str] = Field(default_factory=list)
    data_quality: dict[str, Any] = Field(default_factory=dict)
    model_version: Optional[str] = None
    data_source: Optional[str] = None


class Envelope(BaseModel):
    agent: str
    metadata: Metadata
    insights: list[Insight] = Field(default_factory=list)


class SoftPowerQuery(BaseModel):
    """Request body for POST /agent/query."""

    query_type: Literal["profile", "peers"]
    iso3: list[str] = Field(min_length=1, max_length=25, description="Canonical ISO3 codes, resolved by the orchestrator")
    year: Optional[int] = Field(default=None, description="Profile year; defaults to each country's latest year")
    limit: int = Field(default=5, ge=1, le=25, description="Peers per country")

    @field_validator("iso3")
    @classmethod
    def upper_codes(cls, values: list[str]) -> list[str]:
        return [v.strip().upper() for v in values]
