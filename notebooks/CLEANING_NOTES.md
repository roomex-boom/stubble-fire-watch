# Data Cleaning Handoff Notes

## Project
**Machine Learning-Based Prediction of Agricultural Residue Burning Activity Across Districts of Northwestern India**

## 1. Cleaning Status

All selected datasets have been cleaned and exported.

- Expected files: 15
- Files successfully verified: 15
- Missing files: 0
- Raw datasets: Preserved unchanged
- Cleaning notebook: `02_Data_Cleaning.ipynb`
- Audit notebook: `Data_Auditing.ipynb`

## 2. Cleaned Dataset Inventory

All paths below are relative to `data/processed/cleaned/`.

### CREAMS

| Cleaned file | Description |
|---|---|
| `creams_daily_clean.csv` | District-level daily burning records |
| `creams_bulletins_clean.csv` | Bulletin metadata |
| `creams_validation_clean.json` | Original validation report |
| `creams_state_daily_clean.csv` | State-level daily records |
| `creams_year_compare_clean.csv` | Historical seasonal comparisons |

### FIRMS

| Cleaned file | Description |
|---|---|
| `fires/fires_district_daily_clean.csv.gz` | District-level daily satellite detections |
| `fires/fires_district_season_clean.csv` | District-level seasonal detections |
| `fires/fires_season_totals_clean.csv` | State-level seasonal totals |

### Paddy

| Cleaned file | Description |
|---|---|
| `paddy/paddy_rice_area_district_clean.csv` | District-level rice area and production |
| `paddy/paddy_rice_area_state_clean.csv` | State-level rice area and production |

### Boundaries

| Cleaned file | Description |
|---|---|
| `boundaries/districts_clean.geojson` | District geometries |
| `boundaries/states_clean.geojson` | State geometries |

### Burned Area

| Cleaned file | Description |
|---|---|
| `burned_area/burned_district_season_clean.csv` | District-level seasonal burned area |
| `burned_area/burned_daily_by_state_clean.csv` | State-level daily burned area |
| `burned_area/burned_state_season_clean.csv` | State-level seasonal burned area |

## 3. Important Data-Handling Decisions

### CREAMS — Primary Target

- `fires_day` contains 155,894 missing values (approximately 78.57%).
- Missing daily counts were **not replaced with zero**.
- Daily-count availability is uneven across years and seasons.
- The target is based on reported daily event-list counts, not independently verified ground truth.
- `cum_bulletin_file` has substantial chronological inconsistencies with observation dates. Do not treat it as a reliable source-date mapping.
- The validation report contains multiple inconsistencies involving bulletin totals, district totals, cumulative counts, and dates.
- Suspicious district labels such as `Punjab` and `Kharar` should not automatically be treated as valid districts.
- Preserve the original cleaned dataset. Apply any modeling-specific exclusions in a separate modeling dataset.

### FIRMS

- FIRMS records represent satellite thermal-anomaly detections, not confirmed crop-residue-burning events.
- Multiple sensors may detect activity within the same district and date. Do not assume these represent separate individual fires.
- High detection values and statistical outliers were retained.
- `season_totals.csv` has 136 missing country values associated with `Outside mapped districts`. These were not imputed.

### Paddy

- Entirely empty metadata columns were preserved.
- Missing area and production values were retained.
- No statistical imputation was performed.

### Burned Area

- `burned_ha` was retained as a numeric measurement.
- The `complete` field was preserved as Boolean.
- In `state_season.csv`, `missing_months`, `months_used`, `through`, and `product` are entirely missing. These columns were preserved without fabricated values.
- Burned area is a measurement of burned land area, not a count of active-fire detections.

### Boundaries

- Both GeoJSON files were retained.
- One invalid district geometry exists: Karbi Anglong West, Assam.
- One invalid state geometry exists: Andaman and Nicobar Islands.
- Neither geometry was automatically repaired or removed.

## 4. Important Modeling Guidance

1. Use CREAMS district-level `fires_day` as the primary target, subject to valid target availability and geographic filtering.
2. Do not treat missing target values as zero.
3. Do not use future information or cumulative values that leak information beyond the prediction date.
4. Use chronological train, validation, and test splits.
5. Check district identifiers and names carefully before merging datasets.
6. Check whether each supplementary dataset has suitable temporal and geographic resolution for the prediction task.
7. FIRMS detections and MODIS burned area should be treated as supplementary measurements, not interchangeable target labels.
8. Keep the cleaned source datasets unchanged. Create separate feature-engineering and modeling datasets.
9. Investigate unresolved validation inconsistencies before making strong claims about data accuracy.
10. Document any exclusions, imputations, transformations, and merge decisions.

## 5. Next Stage

The cleaning stage is complete.

The next stage can begin with:

- Dataset integration and district mapping.
- Temporal alignment.
- Feature engineering.
- Missing-target and coverage analysis.
- Exploratory data analysis and visualizations.
- Model development and evaluation.
- Flask application development.

**Important:** The agent should use the cleaned files in `data/processed/cleaned/` and should not repeat the completed raw-data audits or modify the original raw files.

