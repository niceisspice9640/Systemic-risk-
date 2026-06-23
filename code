"""
EuRepoC Global Dataset of Cyber Incidents (v1.3.2) - Cleaning Pipeline
Research project: Cyber and Operations Risk composite index (2015-2025)

Decisions implemented (see Cleaning_Log sheet in output for full rationale):
1. Dates parsed from EU format (dd.mm.yyyy)
2. Rows with no usable date (start_date AND end_date both "Not available") dropped
3. Window filtered to 2015-2024 (2025 handled separately via TableView provisional data)
4. Flag added: ci_only_post2023_rule = incidents that qualify ONLY via the
   critical-infrastructure inclusion criterion (in effect since Feb 2023),
   used for sensitivity-testing the apparent 2022-2024 incident surge
5. Multi-value fields (incident_type, receiver_category) exploded into long-format
   tables for category-level time series, while the main cleaned file preserves
   the original semicolon-separated strings for reference
"""
import pandas as pd

RAW_PATH = "eurepoc_global_dataset_1_3.csv"
df = pd.read_csv(RAW_PATH, low_memory=False)
n_raw = len(df)

# --- 1. Parse dates ---
df["start_date_parsed"] = pd.to_datetime(df["start_date"], format="%d.%m.%Y", errors="coerce")
df["end_date_parsed"] = pd.to_datetime(df["end_date"], format="%d.%m.%Y", errors="coerce")

# --- 2. Drop rows with no usable date ---
no_date_mask = df["start_date_parsed"].isna() & df["end_date_parsed"].isna()
n_dropped_nodate = no_date_mask.sum()
df = df[~no_date_mask].copy()

# Fill the rare case where start is missing but end exists
df["start_date_parsed"] = df["start_date_parsed"].fillna(df["end_date_parsed"])
df["year"] = df["start_date_parsed"].dt.year

# --- 3. Filter to study window ---
n_before_window = len(df)
df = df[(df["year"] >= 2015) & (df["year"] <= 2024)].copy()
n_after_window = len(df)

# --- 4. CI-only post-2023-rule flag ---
df["ci_only_post2023_rule"] = (
    df["inclusion_criterion"].str.strip() == "Attack on critical infrastructure target(s)"
)

# --- 5. Explode multi-value fields ---
def explode_field(frame, col):
    out = frame[["incident_id", "year", col]].copy()
    out[col] = out[col].astype(str).str.split(";")
    out = out.explode(col)
    out[col] = out[col].str.strip()
    return out

incident_type_long = explode_field(df, "incident_type")
receiver_category_long = explode_field(df, "receiver_category")

# --- Annual summary table ---
total_by_year = df.groupby("year").size().rename("total_incidents")
ci_only_by_year = df[df["ci_only_post2023_rule"]].groupby("year").size().rename("ci_only_incidents")
annual_summary = pd.concat([total_by_year, ci_only_by_year], axis=1).fillna(0)
annual_summary["ci_only_incidents"] = annual_summary["ci_only_incidents"].astype(int)
annual_summary["incidents_excl_ci_only"] = annual_summary["total_incidents"] - annual_summary["ci_only_incidents"]
annual_summary["pct_ci_only"] = (annual_summary["ci_only_incidents"] / annual_summary["total_incidents"] * 100).round(1)
annual_summary = annual_summary.reset_index()

# --- Incident type and receiver category pivot tables ---
incident_type_by_year = (
    incident_type_long.groupby(["year", "incident_type"]).size().unstack(fill_value=0)
)
receiver_category_by_year = (
    receiver_category_long.groupby(["year", "receiver_category"]).size().unstack(fill_value=0)
)

# --- Cleaning log ---
cleaning_log = pd.DataFrame({
    "Step": [
        "Raw incidents loaded",
        "Dropped (no usable start_date or end_date)",
        "Remaining after date cleaning",
        "Filtered to 2015-2024 window",
        "Final cleaned incident count (2015-2024)",
        "Incidents flagged ci_only_post2023_rule",
    ],
    "Count / Note": [
        n_raw,
        n_dropped_nodate,
        n_before_window,
        f"Window applied: 2015-2024 (incidents outside window: {n_before_window - n_after_window})",
        n_after_window,
        int(df["ci_only_post2023_rule"].sum()),
    ]
})

print("Raw:", n_raw)
print("Dropped (no date):", n_dropped_nodate)
print("After window filter (2015-2024):", n_after_window)
print("CI-only flagged:", df["ci_only_post2023_rule"].sum())
print()
print(annual_summary)

# Save outputs
df.to_csv("eurepoc_cleaned_2015_2024.csv", index=False)
incident_type_long.to_csv("eurepoc_incident_type_long.csv", index=False)
receiver_category_long.to_csv("eurepoc_receiver_category_long.csv", index=False)
annual_summary.to_csv("eurepoc_annual_summary.csv", index=False)
incident_type_by_year.to_csv("eurepoc_incident_type_by_year.csv")
receiver_category_by_year.to_csv("eurepoc_receiver_category_by_year.csv")
cleaning_log.to_csv("eurepoc_cleaning_log.csv", index=False)
