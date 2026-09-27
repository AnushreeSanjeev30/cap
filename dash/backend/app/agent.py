"""
Soft Power agent: turns the dashboard's data store into the shared envelope.

Confidence is derived from data quality rather than hard-coded, so the fusion
agent can rank soft power claims against other agents' claims:

    confidence = BASE[query_type] x coverage_factor x precision_factor

    coverage_factor  = 1 - 0.5 x kpi_missing_share
        share of the 22 raw KPIs that were missing before imputation. Halved
        because imputed values still carry information (they're interpolated
        from the country's own history), they just deserve less trust.
    precision_factor = 1 - min(0.8, ci_width / 20)
        Kalman 95% CI width in score points. A 20-point band means "we barely
        know the level"; floor at 0.2 so a wide CI down-weights rather than zeroes.

BASE values come from trust_report.md: rankings agree with 4 external indices
(Spearman 0.62-0.87) so profiles start high; peers start lower because 31/54
embedding inputs have VIF > 10, so similarity can be driven by a cluster of
near-duplicate indicators.
"""

from .envelope import Envelope, Insight, Metadata, SoftPowerQuery
from .file_data import get_file_store

AGENT_NAME = "soft_power"
MODEL_VERSION = "kalman-xgb-1.1"
SCORE_SCALE = 100.0

BASE_CONFIDENCE = {"profile": 0.9, "peers": 0.75}


def coverage_factor(missing_share):
    return 1 - 0.5 * (missing_share or 0.0)


def precision_factor(ci_width):
    if ci_width is None:
        return 0.5
    return 1 - min(0.8, ci_width / 20)


def clamp01(value):
    return max(0.0, min(1.0, value))


def run_query(query: SoftPowerQuery, store=None) -> Envelope:
    store = store or get_file_store()
    handler = {"profile": _profile, "peers": _peers}[query.query_type]
    insights, year, quality = handler(query, store)
    return Envelope(
        agent=AGENT_NAME,
        metadata=Metadata(
            query_type=query.query_type,
            year=year,
            entities=query.iso3,
            data_quality=quality,
            model_version=MODEL_VERSION,
            data_source="files",
        ),
        insights=insights,
    )


def _profile(query, store):
    insights = []
    unknown, missing_year = [], []
    years = set()
    for iso3 in query.iso3:
        points = [p for p in store.timeseries.get(iso3, []) if p.get("score") is not None]
        if not points:
            unknown.append(iso3)
            continue
        if query.year is None:
            point = max(points, key=lambda p: p["year"])
        else:
            point = next((p for p in points if p["year"] == query.year), None)
            if point is None:
                missing_year.append(iso3)
                continue
        year = point["year"]
        years.add(year)
        name = store.country_names.get(iso3, iso3)
        q = store.quality_by_key.get((iso3, year), {})
        missing_share = q.get("kpi_missing_share")
        ci_width = q.get("ci_width")
        n_ranked = sum(
            1 for pts in store.timeseries.values()
            if any(p["year"] == year and p.get("score") is not None for p in pts)
        )

        cov, prec = coverage_factor(missing_share), precision_factor(ci_width)
        confidence = BASE_CONFIDENCE["profile"] * cov * prec

        claim = f"{name} ranks #{point['rank']} of {n_ranked} in soft power in {year} (score {point['score']:.1f}/100)"
        change = _rank_change(store, iso3, year, 5)
        if change:
            direction = "up" if change > 0 else "down"
            places = "place" if abs(change) == 1 else "places"
            claim += f", {direction} {abs(change)} {places} since {year - 5}"
        elif change == 0:
            claim += f", rank unchanged since {year - 5}"

        reason_parts = ["Kalman-smoothed composite of 23 KPIs across 5 dimensions"]
        if ci_width is not None:
            reason_parts.append(f"95% CI width {ci_width:.1f} pts")
        else:
            reason_parts.append("no Kalman CI for this year")
        if missing_share is not None:
            reason_parts.append(f"{missing_share:.0%} of raw KPIs imputed")

        latest = store.latest_by_iso3.get(iso3, {})
        insights.append(
            Insight(
                entity_iso3=iso3,
                claim=claim,
                score=round(clamp01(point["score"] / SCORE_SCALE), 4),
                confidence=round(clamp01(confidence), 4),
                reason="; ".join(reason_parts),
                evidence={
                    "raw_score": round(point["score"], 2),
                    "rank": point["rank"],
                    "n_ranked": n_ranked,
                    "rank_change_5y": change,
                    "ci_95": [q.get("ci_lower"), q.get("ci_upper")],
                    "dimensions": {k: point.get(k) for k in ("D1", "D2", "D3", "D4", "D5")},
                    "stability_class": latest.get("stability_class"),
                    "trend_slope": latest.get("trend_slope"),
                    "confidence_factors": {
                        "base": BASE_CONFIDENCE["profile"],
                        "coverage": round(cov, 4),
                        "precision": round(prec, 4),
                    },
                },
            )
        )

    quality = {"year_coverage": _year_coverage(store)}
    if unknown:
        quality["unknown_iso3"] = unknown
    if missing_year:
        quality["no_data_for_year"] = missing_year
    year = query.year if query.year is not None else (max(years) if years else None)
    return insights, year, quality


def _peers(query, store):
    insights = []
    unknown = []
    for iso3 in query.iso3:
        rows = store.get_peers(iso3, query.limit)
        if not rows:
            unknown.append(iso3)
            continue
        name = store.country_names.get(iso3, iso3)
        own_missing = _latest_missing_share(store, iso3)
        for row in rows:
            peer = row["iso3"]
            explanation = row.get("explanation") or {}
            dims = [d["name"] for d in explanation.get("dimensions", [])]
            indicators = [i["label"] for i in explanation.get("indicators", [])]

            pair_missing = (own_missing + _latest_missing_share(store, peer)) / 2
            cov = coverage_factor(pair_missing)
            confidence = BASE_CONFIDENCE["peers"] * cov

            claim = f"{row['name']} is a soft power peer of {name} (similarity {row['similarity']:.2f})"
            if dims:
                reason = f"Closest on {', '.join(dims)}"
                if indicators:
                    reason += f"; top shared indicators: {', '.join(indicators)}"
            else:
                reason = "Cosine similarity of 2000-2024 KPI profiles; no per-dimension breakdown available"
            reason += "; embedding inputs are highly collinear, read as indicator clusters"

            insights.append(
                Insight(
                    entity_iso3=iso3,
                    claim=claim,
                    score=round(clamp01(row["similarity"]), 4),
                    confidence=round(clamp01(confidence), 4),
                    reason=reason,
                    evidence={
                        "peer_iso3": peer,
                        "similarity": row["similarity"],
                        "peer_score": row.get("score"),
                        "peer_rank": row.get("rank"),
                        "shared_dimensions": explanation.get("dimensions", []),
                        "shared_indicators": explanation.get("indicators", []),
                        "confidence_factors": {"base": BASE_CONFIDENCE["peers"], "coverage": round(cov, 4)},
                    },
                )
            )

    coverage = _year_coverage(store)
    quality = {
        "year_coverage": coverage,
        "note": "peer embeddings pool each country's KPI mean/std over all years, not a single year",
    }
    if unknown:
        quality["unknown_iso3"] = unknown
    return insights, coverage[1] if coverage else None, quality


def _rank_change(store, iso3, year, lookback):
    """Positive = climbed. None if either year is missing."""
    by_year = {p["year"]: p for p in store.timeseries.get(iso3, [])}
    now, then = by_year.get(year), by_year.get(year - lookback)
    if not now or not then or now.get("rank") is None or then.get("rank") is None:
        return None
    return then["rank"] - now["rank"]


def _latest_missing_share(store, iso3):
    points = store.timeseries.get(iso3, [])
    if not points:
        return 0.0
    year = max(p["year"] for p in points)
    return store.quality_by_key.get((iso3, year), {}).get("kpi_missing_share") or 0.0


def _year_coverage(store):
    years = [p["year"] for pts in store.timeseries.values() for p in pts]
    return [min(years), max(years)] if years else []
