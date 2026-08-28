# External Benchmark Comparisons

`MODEL_TRUST_GUIDE.md` (and the checks it documents in `model_trust_suite.py`)
answer one question: is this pipeline *internally* well-behaved — does it
overfit, leak the target, miscalibrate its intervals, fall apart under
re-weighting? All of those checks can pass and the score can still fail to
mean anything outside this codebase. This document is about the other
question: **does this model's country ranking agree with anyone else who has
tried to measure something like "soft power," using completely different
data and methodology?**

Section 9 of `model_trust_suite.py` (`check_external_benchmark`) now runs
that comparison against **four** independent, published indices instead of
one. This file is the detail behind that section: what each index actually
measures, how the reference data was sourced, what the correlation came out
to, and — more informative than the correlation number itself — *where* each
index disagrees with this model and why that disagreement is worth reading
rather than explaining away.

Run `.venv/Scripts/python.exe model_trust_suite.py` to regenerate the numbers
below from whatever is currently in `output/soft_power_predictions.csv`.

---

## The four benchmarks, at a glance

| Index | What it actually measures | Built from | Countries in reference file | Spearman ρ vs. this model |
|---|---|---|---|---|
| Brand Finance **Global Soft Power Index** (GSPI) 2024 | Global *perception* of a nation's influence, across business, culture, governance, media, diplomacy | Survey of 170,000+ respondents, 100+ countries, rating all 193 UN member states | 20 (hand-selected to span rank 1–193) | **0.806** |
| Anholt-Ipsos **Nation Brands Index** (NBI) 2023 | Perception of "nation brand" — exports, governance, culture, people, tourism, immigration, investment | ~60,000 interviews across 20 panel countries, rating 60 nations | 57 (full published table minus 3 non-sovereign entries) | **0.874** |
| **Good Country Index** (GCI) 2023, v1.6 | What a country contributes to the *common good of humanity and the planet* — a deliberately different construct from reputation or influence | 35 objective datasets across 7 categories (science, culture, peace, world order, planet, prosperity/equality, health) | 50 (full published top-50 table) | **0.617** |
| Portland/USC **Soft Power 30** 2019 (final edition) | The index this project's own six-pillar framing (enterprise, culture, digital, government, engagement, education) most directly echoes | Objective data + polling across 25 countries, ranking 30 nations | 11 of 30 (partial — see caveats) | **0.782** |

All four are **PASS** under the suite's threshold (ρ ≥ 0.5). That's the headline result, but the useful part of this exercise is the disagreements, not the average.

---

## 1. Brand Finance Global Soft Power Index 2024 — ρ = 0.806, n = 20

Reference file: `external_benchmark_gspi2024.csv`.

**Provenance:** the full 193-country table sits behind Brand Finance's PDF
report, which isn't machine-readable through the tools available here. The
20 countries in the reference file were corroborated across multiple
independent news sources reporting the same index (Brand Finance's own press
releases, Modern Diplomacy, Gulf News, a UPSC current-affairs summary) and
were chosen to span the *entire* rank range (1 to 193), not clustered at the
top — that matters for Spearman, since a sample bunched at one end can look
correlated by coincidence.

**Biggest disagreements** (model rank − GSPI rank):

| Country | Model rank | GSPI rank | Gap |
|---|---|---|---|
| Vanuatu | 120 | 191 | model 71 places *more favorable* |
| Kiribati | 126 | 193 | model 67 places more favorable |
| Nauru | 137 | 192 | model 55 places more favorable |
| Saudi Arabia | 70 | 18 | model 52 places *less favorable* |
| Turkiye | 76 | 25 | model 51 places less favorable |
| UAE | 54 | 10 | model 44 places less favorable |
| Qatar | 64 | 21 | model 43 places less favorable |
| China | 28 | 3 | model 25 places less favorable |

Two distinct patterns, both informative:

- **Gulf states + Türkiye + China rank far better on GSPI than in this
  model.** These are exactly the countries known for heavy, deliberate
  investment in soft-power projection (state media, cultural diplomacy,
  sports sponsorship, expos) that shows up in a *perception* survey faster
  than it shows up in this pipeline's underlying KPIs (freedom-house scores,
  institutional quality, R&D). This isn't a bug in either index — it's the
  gap between "perceived influence right now" and "measured
  developmental/institutional capacity," and it's worth naming explicitly in
  the write-up rather than treating as noise.
- **Pacific micro-states rank far better in this model than on GSPI.** GSPI
  effectively can't distinguish among countries nobody has an opinion about —
  survey respondents likely have no view of Vanuatu or Kiribati at all, so
  they default to the bottom of a "perception" ranking regardless of the
  country's actual KPI profile. This model, by contrast, scores them on
  their own indicator values, which can genuinely put a small, stable,
  reasonably-governed micro-state above a country most respondents have
  heard of but view unfavorably. This is a real, structural difference in
  what a perception survey vs. a KPI composite can measure at the low-
  visibility end of the distribution — flag it if any Pacific-nation ranking
  claim comes up in the defense.

## 2. Anholt-Ipsos Nation Brands Index 2023 — ρ = 0.874, n = 57

Reference file: `external_benchmark_nbi2023.csv`. Highest correlation and
largest reference sample of the four — this is the strongest single piece of
external-validity evidence the project has.

**Provenance:** Ipsos publishes a full 60-nation table with both rank and
score in its own press-release PDF, which *was* machine-readable. All 60
rows were transcribed; 3 (Scotland, Wales, Northern Ireland) were dropped
because they're not sovereign states with an iso3 code to join on, leaving
57.

**Biggest disagreements** (model rank − NBI rank, all in the direction of
NBI ranking the country noticeably better than this model does):

| Country | Model rank | NBI rank | Gap |
|---|---|---|---|
| Tanzania | 143 | 58 | 85 |
| Dominican Republic | 117 | 49 | 68 |
| Kenya | 122 | 56 | 66 |
| Egypt | 100 | 36 | 64 |
| Vietnam | 98 | 47 | 51 |

Every large disagreement here runs the same direction: middle-income and
developing countries with an active tourism/export brand story (Egypt,
Vietnam, Dominican Republic, Kenya, Tanzania) score meaningfully better on a
consumer-perception survey about "would I visit / buy from / invest in this
country" than they do on this pipeline's institutional-and-development-
weighted KPI set. NBI is explicitly a *reputation-for-engagement* metric
(tourism, exports, people); this model leans harder on governance and
knowledge-economy indicators where these same countries score lower. Same
lesson as GSPI's Gulf-state gap, from the other income tier.

## 3. Good Country Index 2023 (v1.6) — ρ = 0.617, n = 50

Reference file: `external_benchmark_gci2023.csv`. Lowest of the four
correlations — expected, and worth stating plainly in the write-up: **GCI is
not a soft-power index.** It measures net contribution to humanity and the
planet (peacekeeping, foreign aid, low emissions, open science, press
freedom, refugee hosting) — a normative "good citizen of the world" score,
not perceived influence or capacity. Including it was a deliberate
robustness check: if this model's ranking correlated with GCI *as strongly*
as with the reputation-focused indices, that would be a warning sign that
the model is really just measuring "rich, orderly countries" and mislabeling
it "soft power." A moderate 0.617 — positive but visibly weaker than the
three reputation-based benchmarks (0.78–0.87) — is the *expected* and
reassuring outcome: overlapping (rich, stable countries tend to score well
on both), but clearly not the same construct.

**Biggest disagreements:**

| Country | Model rank | GCI rank | Gap |
|---|---|---|---|
| Bosnia and Herzegovina | 104 | 42 | 62 |
| Moldova | 89 | 37 | 52 |
| Albania | 95 | 49 | 46 |
| United States | 8 | 50 (last in this table) | −42 |
| Montenegro | 83 | 44 | 39 |

The United States entry is the one worth quoting directly in a defense:
**this model ranks the US 8th; GCI ranks it last (50th) among the countries
compared.** That is not a contradiction — it's the clearest possible
illustration of the difference between "has soft power / developmental
capacity" (this model, and every reputation-based index above) and
"contributes net-positive to the rest of the world relative to its size and
wealth" (GCI's specific normative claim, driven by things like military
spending, carbon emissions per capita, and foreign aid as a share of GDP).
Citing this pairing is a good way to preempt the objection "isn't your score
just GDP in a trench coat" — the GCI comparison shows the model is not
simply reproducing wealth/power rankings under a different name, because a
wealth/power ranking and a common-good ranking diverge sharply exactly where
they should.

## 4. Portland/USC Soft Power 30, 2019 (final edition) — ρ = 0.782, n = 11

Reference file: `external_benchmark_sp30_2019.csv`. Weakest-evidence
benchmark of the four, included anyway because it's the direct intellectual
ancestor of this project's own six-pillar structure (enterprise, culture,
digital, government, engagement, education) and a discontinued index is
still worth checking against if it's the one you're implicitly building on.

**Caveats specific to this one, more serious than the others:**
- The index was discontinued after the 2019 edition — Portland/USC stopped
  publishing it, so there is no more recent year to compare against, and the
  comparison year (2019) is older than any of the other three benchmarks.
- Only **11 of the 30** ranked countries could be reliably sourced: the full
  report is a 7.4MB image-only PDF (`The-Soft-Power-30-Report-2019-1.pdf`)
  that this suite's tooling can't OCR, so the 11 reference rows come from
  cross-corroborated press coverage (USC Center on Public Diplomacy, a
  Japanese foreign-press-center summary, British Council) rather than the
  primary source table directly.
- With n=11, a single disagreement swings the correlation more than it would
  with n=50+ — treat 0.782 here as suggestive, not as strong a result as the
  NBI or GSPI numbers.

**Disagreements** (all modest, consistent with the small, mostly-plausible
sample):

| Country | Model rank | SP30 rank | Gap |
|---|---|---|---|
| Sweden | 13 | 4 | 9 |
| Switzerland | 14 | 6 | 8 |
| South Korea | 11 | 19 | −8 (model *more* favorable) |
| UK | 6 | 2 | 4 |
| Singapore | 25 | 21 | 4 |

Nothing here rises to the level of the GSPI/NBI/GCI stories above — mostly
small reshuffling among countries both indices already agree belong near the
top. If a fuller reference table is ever transcribed by hand from the PDF,
re-run the suite; this entry is the one most likely to change with more
data.

---

## What to actually claim in the capstone write-up

- **Don't claim** "our model is validated against the Global Soft Power
  Index" as a single blanket statement — say which index, what it measures,
  and the correlation, because the four numbers (0.62–0.87) tell a more
  precise story together than any one of them alone.
- **Do claim**: three independently-built reputation/influence indices
  (GSPI, NBI, SP30) each show strong positive rank agreement (ρ = 0.78–0.87)
  with this model's KPI-composite ranking, despite sharing no input data or
  methodology with it or each other — that's real evidence the ranking
  reflects something outside this codebase, not an artifact of feature
  choice.
- **Do claim**, as a robustness argument: agreement is visibly *weaker*
  (ρ = 0.617) against the Good Country Index, which measures a genuinely
  different construct (contribution to the common good vs. reputation/
  capacity) — the model isn't simply a proxy for wealth or power, since a
  wealth/power ranking and a common-good ranking should and do diverge.
- **Name the specific disagreements** (Gulf states + Türkiye + China on
  GSPI, developing-tourism-economy countries on NBI, the US on GCI) as
  findings, not embarrassments — each one cleanly identifies what this
  model's KPI set does and doesn't capture relative to a perception survey
  or a normative "good citizen" index.
- **Flag the evidence quality honestly**: GSPI and SP30 comparisons rest on
  hand-picked/partial reference subsets (20 and 11 countries respectively),
  not full published tables — NBI (n=57, full table) and GCI (n=50, full
  top-50 table) are the stronger pieces of evidence of the four.

## Extending this later

Each reference file is a plain CSV (`iso3, country, <index>_rank_<year>`,
optionally a score column) read by `check_external_benchmark` in
`model_trust_suite.py` via the `EXTERNAL_BENCHMARKS` list at the top of that
section. To add a fifth benchmark or a fuller version of an existing one:

1. Add/extend the CSV with `iso3` codes that exist in
   `output/soft_power_predictions.csv` (check with a quick merge first —
   micro-states and non-UN entities are the usual mismatch).
2. Add one entry to `EXTERNAL_BENCHMARKS` with the file name, its rank
   column, a label, and a `note` describing what the index measures and how
   the reference data was sourced (this note is what makes a low or high
   correlation *interpretable* rather than just a number).
3. Re-run `model_trust_suite.py` — it picks up new entries automatically and
   regenerates `trust_report.md`.

The Soft Power 30 entry is the best candidate to strengthen first if anyone
has access to a text-extractable copy of the 2019 report (or wants to
manually transcribe the remaining 19 countries from the image-only PDF) —
n=11 is the thinnest sample of the four.
