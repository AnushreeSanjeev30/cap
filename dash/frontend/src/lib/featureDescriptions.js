const DESCRIPTIONS = {
  score_roll3_mean: "Three-year rolling average of the composite soft power score. It captures recent level and momentum while smoothing one-year noise.",
  property_rights: "Heritage Foundation measure of legal protection for private property and contract enforcement.",
  govt_integrity: "Governance indicator for corruption control, public trust, and integrity of government institutions.",
  judicial_effectiveness: "Measure of how independent, efficient, and reliable the judicial system is.",
  business_freedom: "Regulatory environment for starting, operating, and closing businesses.",
  investment_freedom: "Openness to investment flows and limits on capital movement.",
  fh_combined_score: "Freedom House combined political rights and civil liberties score.",
  fh_total_score: "Freedom House aggregate freedom score.",
  fh_status_num: "Numeric coding of Freedom House status.",
  tourist_arrivals: "International inbound tourist arrivals, used as a proxy for cultural reach and attractiveness.",
  unesco_total_sites: "Total UNESCO World Heritage sites, a proxy for recognized cultural and natural assets.",
  unesco_cultural_sites: "UNESCO cultural heritage sites.",
  internet_pct_sp: "Share of the population using the internet.",
  rnd_pct_gdp_sp: "Research and development spending as a share of GDP.",
  sci_journal_articles: "Scientific and technical journal publication output.",
  ai_publications: "AI-related publication output.",
  hightech_exports_pct: "High-technology exports as a share of manufactured exports.",
  ict_patents: "ICT patent activity.",
  rd_researchers_per_mil: "Researchers in R&D per million people.",
  trade_pct_gdp: "Trade openness measured as trade share of GDP.",
  life_expectancy: "Life expectancy at birth.",
  infant_mortality: "Infant mortality rate. Lower values generally indicate stronger human development.",
  physicians_per_1k: "Physicians per 1,000 people.",
  tertiary_enroll_pct: "Tertiary education enrollment rate.",
  score_slope: "Estimated trend slope of the composite score over time.",
  score_r2: "Trend fit strength for the composite score.",
  score_volatility: "Recent variability in the composite score.",
  influence_growth: "Estimated growth signal in influence-related indicators.",
  momentum_5y: "Five-year momentum in the composite soft power score.",
  year_norm: "Normalized year feature used by the predictive model.",
};

export function describeFeature(raw, label) {
  const key = String(raw || "")
    .replace(/_lag[12]$/, "")
    .replace(/_slope$/, "")
    .replace(/_r2$/, "");
  if (DESCRIPTIONS[key]) return DESCRIPTIONS[key];
  if (String(raw || "").endsWith("_lag1")) return `${label} from the previous year.`;
  if (String(raw || "").endsWith("_lag2")) return `${label} from two years earlier.`;
  if (String(raw || "").endsWith("_slope")) return `Estimated time trend for ${label}.`;
  return "Model feature used in the SHAP attribution. Positive SHAP values push the prediction up; negative values pull it down.";
}
