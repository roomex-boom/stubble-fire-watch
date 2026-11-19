# Research Report: Machine Learning-Based Prediction of Agricultural Residue Burning Activity Across Districts of Northwestern India

**Project Title:** Machine Learning-Based Prediction of Agricultural Residue Burning Activity Across Districts of Northwestern India (Punjab & Haryana)  
**Authors:** Academic Capstone Team & Research Assistants  
**Date:** November 2026  
**Audited & Verified:** November 19, 2026  
**Status:** Peer-Review & Capstone Defense Ready  

---

## 1. Abstract

Post-monsoon agricultural crop residue burning (stubble burning) across the Northwestern Indian states of Punjab and Haryana generates severe atmospheric particulate pollution ($PM_{2.5}$ and $PM_{10}$), triggering annual public health emergencies across the Indo-Gangetic Plains. Accurate district-level, next-day forecasting of open-field burning activity is vital for state pollution control boards and agricultural authorities to strategically deploy in-situ machinery (super-seeders, balers) and enforcement squads. 

In this investigation, we design an end-to-end, reproducible machine learning architecture that integrates ground-monitored fire bulletins from ICAR-CREAMS, multi-sensor satellite thermal anomaly detections from NASA-FIRMS (MODIS, VIIRS, SNPP), district-level agricultural paddy statistics, and historical burned area measurements. Following an exhaustive technical and scientific audit, all predictive lag features were restructured to reflect **true calendar-day offsets ($t-1, t-2, \dots$)** rather than row shifts, FIRMS missingness was preserved by distinguishing confirmed zero detections from unmapped observations, and agricultural features were strictly aligned to the **prior finalized crop year ($t-1$)** to eliminate post-harvest production leakage. Evaluating five regression models under a strict chronological time-series protocol (Train: 2018–2022, Validation: 2023, Test: 2024–2026), our champion **Random Forest Regressor** achieves a Validation $R^2$ of **0.692** and a Mean Absolute Error (MAE) of **11.86 fires/day**, compared to a baseline Dummy Regressor MAE of **40.24 fires/day** (a 70.5% error reduction). On the prospective held-out test set (Seasons 2024–2026), the model achieves an MAE of **7.51 fires/day**. The trained pipeline is packaged into an audited, production-grade Flask web application featuring strict input validation gates and feature provenance tracking.

---

## 2. Introduction

Between late September and late November, following the harvesting of Kharif paddy (rice), approximately 2.5 to 3 million hectares of agricultural land in Punjab and Haryana must be rapidly prepared for the sowing of Rabi wheat. Farmers face a narrow temporal window of approximately 10 to 20 days. Mechanized harvesting via combine harvesters leaves behind 30 to 40 cm of loose rice straw. Because conventional tillage of this bulky residue delays wheat sowing and penalizes grain yields, open-field burning represents the most cost-effective and immediate disposal method from an individual farm perspective.

The resulting biomass combustion emits massive quantities of carbon monoxide ($CO$), nitrogen oxides ($NO_x$), methane ($CH_4$), volatile organic compounds (VOCs), and fine particulate matter ($PM_{2.5}$). Stagnant post-monsoon meteorological conditions trap these plumes within the planetary boundary layer, spreading hazardous smog across Northern India and the National Capital Region (NCR) of Delhi.

---

## 3. Problem Statement & Mathematical Formulation

While earth observation satellites (MODIS, VIIRS) provide retrospective detections of active fire hotspots, retrospective reporting fails to prevent emissions. Environmental regulators currently lack predictive systems that anticipate fire surges before they occur. The primary engineering challenge is to predict the **next-day ground-monitored agricultural fire count (`fires_day`)** at the administrative district level across Punjab (22 districts) and Haryana (21 districts), formulated mathematically as:

$$\hat{y}_{d, t} = f(\mathbf{x}_{d, \le t-1})$$

where $\hat{y}_{d, t}$ is the predicted active fire count in district $d$ on calendar date $t$, and $\mathbf{x}_{d, \le t-1}$ represents the feature vector containing strictly lagged satellite, ground, agricultural, and calendar observations up to calendar date $t-1$.

---

## 4. Research Objectives

1. Harmonize multi-agency datasets into an aligned spatial-temporal schema.
2. Ensure strict calendar-day lag alignment ($t-1$ day, not row indices) grouped by state and district.
3. Preserve satellite observation missingness, distinguishing confirmed zero detections from unmonitored observations.
4. Align agricultural paddy statistics to prior published crop years ($Y-1$) to eliminate post-harvest production leakage.
5. Empirically evaluate the statistical distributions, seasonal bell curves, and geographic hotspots of stubble burning.
6. Train and benchmark five machine learning regression algorithms using chronological time-series splitting.
7. Package the best model into an accessible, robust Flask web interface with strict validation gates.

---

## 5. Research Questions

- **RQ1:** Does integrating multi-sensor satellite thermal anomalies (MODIS, SNPP, VIIRS) significantly improve next-day fire prediction over simple historical autoregressive models?
- **RQ2:** To what extent does district-level Kharif rice cultivation acreage explain spatial variations in burning volume?
- **RQ3:** Can shallow, transparent machine learning algorithms (Random Forest, Gradient Boosting) achieve operational predictive accuracy without resorting to opaque deep neural networks?
- **RQ4:** How do model predictions generalize across unseen prospective calendar years?

---

## 6. Dataset Descriptions

Our study harmonizes five independent data streams spanning 2016 through 2026:

| Source | Agency / System | Temporal Granularity | Spatial Granularity | Key Variables |
| :--- | :--- | :--- | :--- | :--- |
| **CREAMS Daily** | ICAR-IARI | Daily (Post-Monsoon) | District-level | `fires_day` (target), `fires_cumulative`, `bulletin_file` |
| **FIRMS Detections** | NASA LANCE | Daily | District-level | `modis_detections`, `snpp_detections`, `viirs_detections`, `total_sensor_detections` |
| **Paddy Statistics** | Min. of Agriculture | Annual (1997–2024) | District-level | `kharif_area_ha`, `kharif_prod_t`, `total_area_ha` |
| **Burned Area** | MODIS MCD64A1 | Seasonal (2003–2025) | District-level | `burned_ha`, `complete` |
| **Boundaries** | Survey of India / LGD | Static GeoJSON | District / State | Polygon geometries, centroid coordinates, area ($km^2$) |

---

## 7. Data Cleaning Methodology

Data cleaning was executed in `notebooks/Data_cleaning.ipynb` following rigorous auditing (`Data_Auditing.ipynb`):
1. **Raw Data Preservation:** All raw files under `Data/Raw_data/` were preserved in read-only form.
2. **Date Standardization:** Inconsistent date representations across bulletins were parsed into ISO-8601 `YYYY-MM-DD`.
3. **Suspicious Label Resolution:** Ambiguous labels (e.g., `Punjab` or `Kharar` listed as districts) were flagged and separated.
4. **Target Integrity:** Missing daily counts in CREAMS were retained as missing rather than imputed as zero, preventing artificial zero-inflation.

---

## 8. Geographic Matching Methodology

1. **Scope Filtering:** Records were filtered to Indian Punjab and Haryana. Transboundary districts (Pakistani Punjab) were systematically removed.
2. **Spelling Normalization:** District names were normalized (lowercasing, punctuation removal) and aligned against verified Survey of India boundaries using verified aliases:
   - *Haryana:* `Gurugram` $\rightarrow$ `Gurgaon`, `Mahendergarh` $\rightarrow$ `Mahendragarh`, `Nuh` $\rightarrow$ `Mewat`.
   - *Punjab:* `SAS Nagar` $\rightarrow$ `Sahibzada Ajit Singh Nagar`, `SBS Nagar` $\rightarrow$ `Shahid Bhagat Singh Nagar`, `Sri Muktsar Sahib` $\rightarrow$ `Muktsar`, `Ferozepur` $\rightarrow$ `Firozpur`.
3. **Mahendragarh Duplicate Resolution:** Exactly 258 true duplicate records caused by dual spellings were removed; 43 unpaired valid records were retained.
4. **Boundary IDs vs. LGD Codes:** Boundary IDs (103–124) and Paddy LGD codes (58–610) were treated as separate reference systems and mapped independently.
5. **Valid Unmapped Districts:** Newer districts (e.g., Malerkotla carved in 2021, Charkhi Dadri in 2016) without separate boundary polygons (2,153 records) were preserved for tabular modeling using state-level centroid imputation.

---

## 9. Feature Engineering Methodology (Audited)

### A. True Calendar-Day Lag Alignment
Unlike naive index shifting, a true calendar lag for date $t$ must strictly correspond to calendar date $t - k\text{ days}$. If date $t - 1$ was not in the monitoring catalog (e.g., across multi-month seasonal breaks or unmonitored days), the lag is treated as missing (`NaN`), and an explicit indicator `lag1_is_missing = 1` is recorded. Grouping is strictly performed by `(state_creams, district_standardized)`.

### B. FIRMS Observation Availability
We explicitly distinguish three states of FIRMS satellite data:
1. `detected`: District boundary monitored and 1 or more thermal anomalies detected.
2. `zero_detection`: District boundary monitored, but satellite recorded zero thermal anomalies.
3. `unavailable_unmapped`: District lacks boundary polygon coordinates (newer districts), meaning satellite observations were unmapped rather than zero. Preserved with `sat_lag1_is_missing = 1`.

### C. Prior-Year Agricultural Alignment
To eliminate post-harvest production leakage, agricultural statistics for season $Y$ are strictly mapped from agricultural year $Y-1$ (`season_year = year + 1`). This guarantees that only finalized crop statistics already published prior to the post-monsoon harvest are used.

### Engineered Feature Categories (29 Features):
1. **Satellite Thermal Detections:**
   - 1-day, 2-day, and 3-day calendar detection lags (`sat_lag1`, `sat_lag2`, `sat_lag3`).
   - 3-day and 7-day calendar rolling means and 7-day rolling maximums (`sat_roll_mean_3d`, `sat_roll_mean_7d`, `sat_roll_max_7d`).
   - Sensor-specific 1-day calendar lags (`sat_modis_lag1`, `sat_viirs_lag1`, `sat_snpp_lag1`).
   - Missingness indicator: `sat_lag1_is_missing`.
2. **Historical Ground Persistence:**
   - 1-day lagged reported fires (`creams_lag1`).
   - Binary indicator of missingness (`creams_lag1_is_missing`).
   - 3-day and 7-day rolling ground fire means (`creams_roll_mean_3d`, `creams_roll_mean_7d`).
3. **Agricultural Capacity:**
   - District Kharif rice area (`kharif_area_ha`) and production (`kharif_prod_t`) from prior published agricultural year.
   - Total agricultural area (`total_area_ha`).
   - Prior-season burned area (`prior_season_burned_ha` from season $t-1$).
4. **Spatial Attributes:**
   - Centroid coordinates (`centroid_lat`, `centroid_lon`).
   - Projected boundary area (`boundary_area_sqkm`).
   - State binary indicator (`is_punjab`).
5. **Calendar & Seasonality:**
   - Day of year, month, day of month, day of week, weekend indicator.
   - Day of season (days elapsed since September 15).
   - Cyclical harmonic transformations: $\sin(2\pi \cdot \text{day} / 90)$ and $\cos(2\pi \cdot \text{day} / 90)$.
   - Peak window indicator (October 15 to November 20).

---

## 10. Model Descriptions

Five regression algorithms were trained using transparent Scikit-Learn pipelines:
1. **DummyRegressor (Baseline):** A naive baseline that constantly predicts the historical training mean.
2. **Linear Regression:** An ordinary least-squares multivariate linear baseline.
3. **Decision Tree Regressor:** A single CART tree constrained to a maximum depth of 6 to prevent overfitting.
4. **Random Forest Regressor (Champion):** A bagged ensemble of 100 decorrelated decision trees (max depth 10, bootstrapping enabled).
5. **Gradient Boosting Regressor:** Sequential boosting of 100 shallow trees (max depth 5, learning rate 0.08, Huber loss).

---

## 11. Experimental Setup

- **Supervised Modeling Subset:** Exactly 14,392 ground-monitored observation dates where verified target values exist.
- **Chronological Time-Series Partitioning:**
  - **Training Set (Seasons 2018–2022):** 8,114 samples.
  - **Validation Set (Season 2023):** 2,178 samples (used for hyperparameter selection).
  - **Untouched Test Set (Seasons 2024–2026):** 4,100 samples (used strictly for final evaluation).

---

## 12. Evaluation Metrics

Model performance was assessed using standard continuous regression metrics:
- **Mean Absolute Error (MAE):** $\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$
- **Root Mean Squared Error (RMSE):** $\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$
- **Coefficient of Determination ($R^2$):** $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$

---

## 13. Results & Empirical Leaderboard

### Table 1: Model Performance Benchmark Across Chronological Splits (Audited)

| Algorithm | Validation MAE | Validation RMSE | Validation $R^2$ | Test MAE (2024–26) | Test RMSE | Test $R^2$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest Regressor** | **11.86** | **26.20** | **0.692** | **7.51** | **15.30** | **0.324** | **Champion Model** |
| **Gradient Boosting Regressor** | 12.30 | 26.82 | 0.678 | 7.97 | 15.50 | 0.306 | Competitive |
| **Linear Regression** | 14.51 | 28.19 | 0.644 | 10.43 | 16.81 | 0.184 | Linear Baseline |
| **Decision Tree Regressor** | 13.40 | 30.20 | 0.591 | 8.05 | 16.77 | 0.188 | Single Tree |
| **Dummy Regressor (Baseline)** | 40.24 | 51.53 | -0.190 | 36.49 | 38.16 | -3.205 | Naive Mean |

### Key Experimental Findings:
1. **Random Forest Achieves Lowest Validation Error:** Random Forest reduces prediction error from 40.24 to 11.86 fires/day, capturing non-linear interactions between satellite momentum and calendar timing.
2. **Prospectively Robust Test Performance:** On prospective seasons (2024–2026), the test MAE is **7.51 fires/day**, reflecting lower aggregate burning intensity in recent post-monsoon campaigns.
3. **Feature Importance Hierarchy:** The top 3 predictive drivers identified by Gini importance are:
   - `sat_roll_mean_3d` (3-day rolling satellite detection average) — 38.9%
   - `creams_roll_mean_3d` (3-day rolling ground fire count average) — 23.8%
   - `day_of_season` (Seasonal progression counter) — 11.0%
   - `kharif_area_ha` (District rice acreage) — 7.2%

---

## 14. Discussion

The empirical findings confirm that agricultural residue burning is governed by a combination of **seasonal timing**, **recent fire momentum**, and **district agricultural capacity**. 

Satellite thermal anomaly lags provide continuous spatial intelligence that significantly augments ground persistence models. When satellite detections spike on day $t-1$, the probability of a ground fire surge on day $t$ increases dramatically. However, satellite counts cannot be used as a direct 1:1 substitute for ground events due to sensor resolution differences (e.g., 375m VIIRS vs 1km MODIS) and cloud obscuration.

District rice acreage (`kharif_area_ha`) explains long-term spatial burning volume ($r \approx 0.68$). High-intensity districts such as Sangrur, Firozpur, Bathinda, and Patiala maintain high baseline burning risks throughout October and November due to extensive monoculture paddy cultivation.

---

## 15. Scientific Limitations & Honesty

1. **Target Missingness:** The CREAMS daily target contains ~78% unmonitored days across the multi-year catalog. We did not fabricate labels or impute unmonitored days as zero.
2. **Satellite Anomalies $\neq$ Ground Fires:** NASA-FIRMS records thermal pixel anomalies, not independently verified individual crop burning incidents.
3. **Atmospheric & Weather Absence:** Due to the absence of spatially distributed weather receptors across all 43 rural districts (existing air datasets covered only 7 receptor cities), meteorological variables (wind speed, boundary layer height, relative humidity) were excluded to avoid fabricated geographic features.
4. **Generalization Across Shifts:** Policy interventions (e.g., government subsidies on balers and in-situ seeders) alter year-to-year burning baselines, explaining why model $R^2$ varies across differing harvest years.

---

## 16. Conclusion

This project successfully developed an operational machine learning pipeline for district-level agricultural residue burning prediction across Punjab and Haryana. By strictly enforcing calendar-day lags, multi-sensor feature lags, prior-year agricultural mapping, and non-negative prediction constraints, our Random Forest pipeline delivers actionable next-day fire forecasts (Validation MAE 11.86 fires/day; Test MAE 7.51 fires/day). The system is fully deployed in an audited Flask web interface with strict validation gates.

---

## 17. Future Work

1. **Meteorological Reanalysis Integration:** Incorporate ERA5 high-resolution reanalysis data (boundary layer height, surface wind speed, dew point).
2. **Crop Phenology Remote Sensing:** Ingest Sentinel-2 and Landsat-8 NDVI/NDWI vegetation indices to track field-by-field harvest harvesting progress.
3. **Air Quality Dispersion Coupling:** Interface next-day fire predictions with WRF-Chem or Gaussian plume dispersion models to forecast downwind $PM_{2.5}$ concentrations in Delhi-NCR.

---

## 18. References & Source Links

1. ICAR-IARI Consortium for Research on Agroecosystem Monitoring and Modeling (CREAMS): [https://creams.iari.res.in](https://creams.iari.res.in)
2. NASA Fire Information for Resource Management System (FIRMS): [https://firms.modaps.eosdis.nasa.gov](https://firms.modaps.eosdis.nasa.gov)
3. Directorate of Economics and Statistics, Ministry of Agriculture & Farmers Welfare, Government of India: [https://eands.dacnet.nic.in](https://eands.dacnet.nic.in)
4. MODIS Land Collection MCD64A1 Burned Area Product: [https://modis.gsfc.nasa.gov](https://modis.gsfc.nasa.gov)
5. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
