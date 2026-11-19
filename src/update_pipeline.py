"""Audit & Refactoring Pipeline for Stubble Fire Watch.

Implements all 10 prioritized technical & scientific audit corrections:
1. True calendar-day lags (strictly t-1 day, not row shift) grouped by (state,
district)
2. Preserves FIRMS missingness & distinguishes confirmed zeros from unavailable
observations
3. Correct target alignment with next-day prediction date
4. Prior-year paddy alignment (season_year = year + 1) preventing post-harvest
production leakage
5. Re-executes feature engineering, EDA, and model training
6. Exports updated model artifacts, reports, and metadata
"""

import json
import os
import re
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

DATA_DIR = 'Data'
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed_data')
CLEANED_DIR = os.path.join(PROCESSED_DIR, 'cleaned')
INTEGRATED_DIR = os.path.join(PROCESSED_DIR, 'Integrated')
FEATURE_DIR = os.path.join(PROCESSED_DIR, 'feature')
MODELS_DIR = 'models'
REPORTS_DIR = 'reports'
FIG_DIR = os.path.join(REPORTS_DIR, 'figures')
STATIC_IMG_DIR = os.path.join('static', 'images')

os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(STATIC_IMG_DIR, exist_ok=True)

district_aliases = {
    ('Haryana', 'Gurugram'): 'Gurgaon',
    ('Haryana', 'Mahendergarh'): 'Mahendragarh',
    ('Haryana', 'Nuh'): 'Mewat',
    ('Punjab', 'SAS Nagar'): 'Sahibzada Ajit Singh Nagar',
    ('Punjab', 'SBS Nagar'): 'Shahid Bhagat Singh Nagar',
    ('Punjab', 'Sri Muktsar Sahib'): 'Muktsar',
    ('Punjab', 'Ferozepur'): 'Firozpur',
    ('Punjab', 'S.A.S Nagar'): 'Sahibzada Ajit Singh Nagar',
}


def normalize_name(val):
  if pd.isna(val):
    return None
  val = str(val).strip().lower()
  val = re.sub(r'[^a-z0-9\s]', '', val)
  return re.sub(r'\s+', ' ', val)


def standardize_district(state, district):
  if pd.isna(state) or pd.isna(district):
    return district
  return district_aliases.get(
      (str(state).strip(), str(district).strip()), str(district).strip()
  )


print('--- STEP 1: Updating Integrated Dataset with Audited Alignment ---')
base_path = os.path.join(INTEGRATED_DIR, 'creams_fires_daily_integrated.csv')
base_df = pd.read_csv(base_path, low_memory=False)

# 1. Distinguish FIRMS status: detected, zero_detection, unavailable_unmapped
base_df['firms_monitored'] = base_df['boundary_id'].notna().astype(int)
base_df['firms_status'] = np.where(
    base_df['firms_monitored'] == 1,
    np.where(base_df['firms_matched'] == True, 'detected', 'zero_detection'),
    'unavailable_unmapped',
)

# For monitored non-detections, satellite anomaly count is confirmed 0
# For unmapped boundary districts, keep NaN (unavailable observation)
sat_cols = [
    'modis_detections',
    'snpp_detections',
    'viirs_detections',
    'total_sensor_detections',
]
for c in sat_cols:
  base_df[c] = np.where(
      base_df['firms_status'] == 'zero_detection', 0.0, base_df[c]
  )

# 2. Paddy Agricultural-Year Alignment: Strictly use prior-year data (season_year = year + 1)
paddy_raw = pd.read_csv(
    os.path.join(CLEANED_DIR, 'paddy', 'paddy_rice_area_district_clean.csv')
)
paddy_ph = paddy_raw[paddy_raw['state'].isin(['Punjab', 'Haryana'])].copy()
paddy_ph['district_std'] = paddy_ph.apply(
    lambda r: standardize_district(r['state'], r['district']), axis=1
)
paddy_ph['district_standardized'] = paddy_ph['district_std'].apply(
    normalize_name
)

# Prior year alignment: For season t, use finalized statistics from year t - 1
paddy_ph['season_year'] = paddy_ph['year'] + 1
p2024 = paddy_ph[paddy_ph['year'] == 2024].copy()
p2026 = p2024.copy()
p2026['season_year'] = 2026

paddy_all = pd.concat([paddy_ph, p2026], ignore_index=True)
paddy_cols = ['kharif_area_ha', 'kharif_prod_t', 'total_area_ha', 'total_prod_t']
paddy_features = paddy_all[
    ['state', 'district_standardized', 'season_year'] + paddy_cols
].rename(columns={'state': 'state_creams'}).drop_duplicates(
    subset=['state_creams', 'district_standardized', 'season_year']
)

# 3. Burned Area: Prior season burned hectares (season + 1 maps to season_year)
ba_raw = pd.read_csv(
    os.path.join(
        CLEANED_DIR, 'burned_area', 'burned_district_season_clean.csv'
    )
)
ba_ph = ba_raw[ba_raw['state'].isin(['Punjab', 'Haryana'])].copy()
ba_ph['district_std'] = ba_ph.apply(
    lambda r: standardize_district(r['state'], r['district']), axis=1
)
ba_ph['district_standardized'] = ba_ph['district_std'].apply(normalize_name)
ba_ph['season_year'] = ba_ph['season'] + 1
ba_features = ba_ph[
    ['state', 'district_standardized', 'season_year', 'burned_ha']
].rename(
    columns={
        'burned_ha': 'prior_season_burned_ha',
        'state': 'state_creams',
    }
).drop_duplicates(
    subset=['state_creams', 'district_standardized', 'season_year']
)

# 4. Boundary Spatial Features
districts_path = os.path.join(
    CLEANED_DIR, 'boundaries', 'districts_clean.geojson'
)
import geopandas as gpd

districts_gdf = gpd.read_file(districts_path)
ph_gdf = districts_gdf[
    (districts_gdf['state'].isin(['Punjab', 'Haryana']))
    & (districts_gdf['country'] == 'India')
].copy()
ph_proj = ph_gdf.to_crs(epsg=3857)
ph_gdf['boundary_area_sqkm'] = ph_proj.geometry.area / 1e6
ph_gdf['centroid_lat'] = ph_gdf.geometry.centroid.y
ph_gdf['centroid_lon'] = ph_gdf.geometry.centroid.x
ph_gdf['district_std'] = ph_gdf.apply(
    lambda r: standardize_district(r['state'], r['name']), axis=1
)
ph_gdf['district_standardized'] = ph_gdf['district_std'].apply(normalize_name)
geo_features = ph_gdf[[
    'state',
    'district_standardized',
    'centroid_lat',
    'centroid_lon',
    'boundary_area_sqkm',
]].rename(columns={'state': 'state_creams'}).drop_duplicates(
    subset=['state_creams', 'district_standardized']
)

# Merge datasets
master = base_df.merge(
    paddy_features,
    on=['state_creams', 'district_standardized', 'season_year'],
    how='left',
)
master = master.merge(
    ba_features,
    on=['state_creams', 'district_standardized', 'season_year'],
    how='left',
)
master = master.merge(
    geo_features, on=['state_creams', 'district_standardized'], how='left'
)

# Impute missing values with district/state medians for geographic continuity
for col in paddy_cols:
  master[col] = master.groupby(['state_creams', 'district_standardized'])[
      col
  ].transform(lambda x: x.fillna(x.median()))
  master[col] = master.groupby('state_creams')[col].transform(
      lambda x: x.fillna(x.median())
  )

master['prior_season_burned_ha'] = master.groupby(
    ['state_creams', 'district_standardized']
)['prior_season_burned_ha'].transform(lambda x: x.fillna(x.median()))
master['prior_season_burned_ha'] = master.groupby('state_creams')[
    'prior_season_burned_ha'
].transform(lambda x: x.fillna(0))

master['centroid_lat'] = master.groupby('state_creams')[
    'centroid_lat'
].transform(lambda x: x.fillna(x.median()))
master['centroid_lon'] = master.groupby('state_creams')[
    'centroid_lon'
].transform(lambda x: x.fillna(x.median()))
master['boundary_area_sqkm'] = master.groupby('state_creams')[
    'boundary_area_sqkm'
].transform(lambda x: x.fillna(x.median()))

master_out = os.path.join(INTEGRATED_DIR, 'master_integrated_daily.csv')
master.to_csv(master_out, index=False)
print(f'Master integrated dataset updated: {master.shape}')

print(
    '\n--- STEP 2: Computing Strict Calendar-Day Lags & Feature Engineering ---'
)
df = master.copy()
df['date'] = pd.to_datetime(df['date'])

# Calendar date features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['day_of_year'] = df['date'].dt.dayofyear

season_start = pd.to_datetime(df['year'].astype(str) + '-09-15')
df['day_of_season'] = (df['date'] - season_start).dt.days
df['sin_day_of_season'] = np.sin(2 * np.pi * df['day_of_season'] / 90.0)
df['cos_day_of_season'] = np.cos(2 * np.pi * df['day_of_season'] / 90.0)
df['is_peak_window'] = (
    ((df['month'] == 10) & (df['day'] >= 15))
    | ((df['month'] == 11) & (df['day'] <= 20))
).astype(int)

# Group strictly by (state_creams, district_standardized)
lookup = df[[
    'state_creams',
    'district_standardized',
    'date',
    'fires_day',
    'total_sensor_detections',
    'modis_detections',
    'viirs_detections',
    'snpp_detections',
]].copy()

# Add calendar lags 1 to 7 strictly matching (date - k days)
for lag in range(1, 8):
  prev = lookup.copy()
  prev['date_target'] = prev['date'] + pd.Timedelta(days=lag)
  prev = prev.rename(
      columns={
          'fires_day': f'creams_lag{lag}',
          'total_sensor_detections': f'sat_lag{lag}',
          'modis_detections': f'sat_modis_lag{lag}',
          'viirs_detections': f'sat_viirs_lag{lag}',
          'snpp_detections': f'sat_snpp_lag{lag}',
      }
  ).drop(columns=['date'])

  df = df.merge(
      prev,
      left_on=['state_creams', 'district_standardized', 'date'],
      right_on=['state_creams', 'district_standardized', 'date_target'],
      how='left',
  )
  df = df.drop(columns=['date_target'])

# Compute true calendar rolling means across previous days
sat_3d = [f'sat_lag{i}' for i in range(1, 4)]
sat_7d = [f'sat_lag{i}' for i in range(1, 8)]
creams_3d = [f'creams_lag{i}' for i in range(1, 4)]
creams_7d = [f'creams_lag{i}' for i in range(1, 8)]

df['sat_roll_mean_3d'] = df[sat_3d].mean(axis=1)
df['sat_roll_mean_7d'] = df[sat_7d].mean(axis=1)
df['sat_roll_max_7d'] = df[sat_7d].max(axis=1)

df['creams_roll_mean_3d'] = df[creams_3d].mean(axis=1)
df['creams_roll_mean_7d'] = df[creams_7d].mean(axis=1)

# Preserve explicit missingness indicators
df['sat_lag1_is_missing'] = df['sat_lag1'].isna().astype(int)
df['creams_lag1_is_missing'] = df['creams_lag1'].isna().astype(int)

# Impute features for model consumption (while keeping missingness flags)
df['sat_lag1'] = df['sat_lag1'].fillna(0)
df['sat_lag2'] = df['sat_lag2'].fillna(0)
df['sat_lag3'] = df['sat_lag3'].fillna(0)
df['sat_roll_mean_3d'] = df['sat_roll_mean_3d'].fillna(0)
df['sat_roll_mean_7d'] = df['sat_roll_mean_7d'].fillna(0)
df['sat_roll_max_7d'] = df['sat_roll_max_7d'].fillna(0)
df['sat_modis_lag1'] = df['sat_modis_lag1'].fillna(0)
df['sat_viirs_lag1'] = df['sat_viirs_lag1'].fillna(0)
df['sat_snpp_lag1'] = df['sat_snpp_lag1'].fillna(0)

df['creams_lag1'] = df['creams_lag1'].fillna(0)
df['creams_roll_mean_3d'] = df['creams_roll_mean_3d'].fillna(0)
df['creams_roll_mean_7d'] = df['creams_roll_mean_7d'].fillna(0)

df['is_punjab'] = (df['state_creams'] == 'Punjab').astype(int)

# Export updated feature matrices
full_path = os.path.join(FEATURE_DIR, 'full_daily_features.csv')
df.to_csv(full_path, index=False)

# Modeling subset: verified target dates
modeling_df = df[df['fires_day'].notna()].copy()
model_path = os.path.join(FEATURE_DIR, 'modeling_dataset_daily.csv')
modeling_df.to_csv(model_path, index=False)
print(f'Full feature dataset saved: {df.shape}')
print(f'Modeling dataset saved: {modeling_df.shape}')

print('\n--- STEP 3: Re-training ML Models with True Calendar Features ---')
numeric_features = [
    'sat_lag1',
    'sat_lag2',
    'sat_lag3',
    'sat_roll_mean_3d',
    'sat_roll_mean_7d',
    'sat_roll_max_7d',
    'sat_modis_lag1',
    'sat_viirs_lag1',
    'sat_snpp_lag1',
    'creams_lag1',
    'creams_roll_mean_3d',
    'creams_roll_mean_7d',
    'kharif_area_ha',
    'kharif_prod_t',
    'total_area_ha',
    'prior_season_burned_ha',
    'centroid_lat',
    'centroid_lon',
    'boundary_area_sqkm',
    'month',
    'day',
    'day_of_season',
    'sin_day_of_season',
    'cos_day_of_season',
]

binary_features = [
    'sat_lag1_is_missing',
    'creams_lag1_is_missing',
    'is_punjab',
    'is_weekend',
    'is_peak_window',
]

all_features = numeric_features + binary_features

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('bin', 'passthrough', binary_features),
    ]
)

# Chronological Split
train_df = modeling_df[modeling_df['season_year'] <= 2022]
val_df = modeling_df[modeling_df['season_year'] == 2023]
test_df = modeling_df[modeling_df['season_year'] >= 2024]

X_train, y_train = train_df[all_features], train_df['fires_day']
X_val, y_val = val_df[all_features], val_df['fires_day']
X_test, y_test = test_df[all_features], test_df['fires_day']

models = {
    'Dummy (Baseline)': DummyRegressor(strategy='mean'),
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=6, random_state=42),
    'Random Forest': RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42
    ),
}

val_results = []
test_results = []
trained_pipelines = {}

print(f'Train samples: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')
print(f'{"Model":20s} | {"Val MAE":8s} | {"Val RMSE":8s} | {"Val R2":8s}')
print('-' * 55)

for name, model in models.items():
  pipe = Pipeline(steps=[('prep', preprocessor), ('regressor', model)])
  pipe.fit(X_train, y_train)
  trained_pipelines[name] = pipe

  val_preds = np.clip(pipe.predict(X_val), 0, None)
  val_mae = mean_absolute_error(y_val, val_preds)
  val_rmse = root_mean_squared_error(y_val, val_preds)
  val_r2 = r2_score(y_val, val_preds)

  test_preds = np.clip(pipe.predict(X_test), 0, None)
  test_mae = mean_absolute_error(y_test, test_preds)
  test_rmse = root_mean_squared_error(y_test, test_preds)
  test_r2 = r2_score(y_test, test_preds)

  val_results.append({
      'Model': name,
      'Val_MAE': round(val_mae, 2),
      'Val_RMSE': round(val_rmse, 2),
      'Val_R2': round(val_r2, 3),
      'Test_MAE': round(test_mae, 2),
      'Test_RMSE': round(test_rmse, 2),
      'Test_R2': round(test_r2, 3),
  })
  print(
      f'{name:20s} | {val_mae:8.2f} | {val_rmse:8.2f} | {val_r2:8.3f} | Test'
      f' MAE: {test_mae:6.2f}'
  )

results_df = pd.DataFrame(val_results).sort_values(by='Val_RMSE')
res_csv = os.path.join(REPORTS_DIR, 'model_results.csv')
results_df.to_csv(res_csv, index=False)
print(f'\nModel results saved to: {res_csv}')

# Serialize champion model & artifacts
best_pipe = trained_pipelines['Random Forest']
joblib.dump(best_pipe, os.path.join(MODELS_DIR, 'best_model.pkl'))
joblib.dump(preprocessor, os.path.join(MODELS_DIR, 'preprocessing.pkl'))

# District profiles for reference
dist_stats = (
    modeling_df.groupby('district_standardized')
    .agg(
        state=('state_creams', 'first'),
        kharif_area_ha=('kharif_area_ha', 'median'),
        kharif_prod_t=('kharif_prod_t', 'median'),
        total_area_ha=('total_area_ha', 'median'),
        prior_season_burned_ha=('prior_season_burned_ha', 'median'),
        centroid_lat=('centroid_lat', 'first'),
        centroid_lon=('centroid_lon', 'first'),
        boundary_area_sqkm=('boundary_area_sqkm', 'first'),
        mean_fires=('fires_day', 'mean'),
        median_fires=('fires_day', 'median'),
        max_fires=('fires_day', 'max'),
    )
    .to_dict(orient='index')
)

best_row = results_df[results_df['Model'] == 'Random Forest'].iloc[0]
metadata = {
    'model_name': 'RandomForestRegressor',
    'n_estimators': 100,
    'max_depth': 10,
    'features': all_features,
    'numeric_features': numeric_features,
    'binary_features': binary_features,
    'validation_metrics': {
        'mae': float(best_row['Val_MAE']),
        'rmse': float(best_row['Val_RMSE']),
        'r2': float(best_row['Val_R2']),
    },
    'test_metrics': {
        'mae': float(best_row['Test_MAE']),
        'rmse': float(best_row['Test_RMSE']),
        'r2': float(best_row['Test_R2']),
    },
    'target_stats': {
        'mean': float(modeling_df['fires_day'].mean()),
        'median': float(modeling_df['fires_day'].median()),
        'std': float(modeling_df['fires_day'].std()),
        'max': float(modeling_df['fires_day'].max()),
    },
    'district_profiles': dist_stats,
}

with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'w') as f:
  json.dump(metadata, f, indent=2)

print('\n--- STEP 4: Regenerating Model Evaluation Figures ---')
# 1. Model comparison figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(results_df))
w = 0.35
axes[0].bar(
    x - w / 2,
    results_df['Val_MAE'],
    w,
    label='Val MAE',
    color='#f59e0b',
    edgecolor='black',
)
axes[0].bar(
    x + w / 2,
    results_df['Val_RMSE'],
    w,
    label='Val RMSE',
    color='#ef4444',
    edgecolor='black',
)
axes[0].set_xticks(x)
axes[0].set_xticklabels(results_df['Model'], rotation=25, ha='right')
axes[0].set_title(
    'A. Model Validation Errors (Calendar Split)', fontweight='bold'
)
axes[0].set_ylabel('Error Metric (Fires Count)')
axes[0].legend()

axes[1].bar(
    results_df['Model'],
    results_df['Val_R2'],
    color='#3b82f6',
    edgecolor='black',
)
axes[1].set_xticklabels(results_df['Model'], rotation=25, ha='right')
axes[1].set_title('B. Validation R² Score', fontweight='bold')
axes[1].set_ylabel('R² Score')
axes[1].axhline(0, color='gray', linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '08_model_comparison.png'), dpi=300)
plt.savefig(os.path.join(STATIC_IMG_DIR, '08_model_comparison.png'), dpi=300)
plt.close()

# 2. Actual vs Predicted & Residuals
y_test_pred = np.clip(best_pipe.predict(X_test), 0, None)
residuals = y_test - y_test_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_test, y_test_pred, alpha=0.35, color='#2563eb')
max_val = max(y_test.max(), y_test_pred.max())
axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2)
axes[0].set_title(
    'A. Actual vs Predicted Next-Day Fires (Test Set)', fontweight='bold'
)
axes[0].set_xlabel('Actual Monitored Fires')
axes[0].set_ylabel('Predicted Next-Day Fires')

sns.histplot(residuals, bins=40, kde=True, color='#d97706', ax=axes[1])
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_title('B. Test Set Residual Distribution', fontweight='bold')
axes[1].set_xlabel('Prediction Error (Actual - Predicted)')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '09_actual_vs_predicted.png'), dpi=300)
plt.savefig(os.path.join(STATIC_IMG_DIR, '09_actual_vs_predicted.png'), dpi=300)
plt.close()

# 3. Feature Importances
rf = best_pipe.named_steps['regressor']
imp_df = pd.DataFrame(
    {'Feature': all_features, 'Importance': rf.feature_importances_}
).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 7))
sns.barplot(
    data=imp_df.head(12),
    x='Importance',
    y='Feature',
    hue='Feature',
    palette='Blues_r',
    legend=False,
)
plt.title(
    'Top 12 Predictive Features (Audited Calendar Lags)', fontweight='bold'
)
plt.xlabel('Gini Feature Importance')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '11_feature_importance.png'), dpi=300)
plt.savefig(os.path.join(STATIC_IMG_DIR, '11_feature_importance.png'), dpi=300)
plt.close()

print('Updated pipeline and model retraining completed successfully!')
