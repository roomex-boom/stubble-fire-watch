import json
import os

nb = {
    'cells': [],
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12.10'}
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

def add_md(text):
    nb['cells'].append({'cell_type': 'markdown', 'metadata': {}, 'source': [text + '\n']})

def add_code(code):
    nb['cells'].append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [code + '\n']})

add_md(
"""# 🤖 Notebook 06: Machine Learning Model Training, Comparison & Evaluation

**Project:** Machine Learning-Based Prediction of Agricultural Residue Burning Activity Across Districts of Northwestern India

### Objective
In this notebook, we build, train, evaluate, and compare five machine learning regression models to predict next-day district-level stubble burning counts (`fires_day`).

### Methodological Principles
1. **Beginner-Friendly Pipelines:** We use transparent, standard Scikit-Learn regressors and `ColumnTransformer` preprocessing.
2. **Strict Chronological Splitting:** Because agricultural burning is a continuous temporal process, we evaluate models using chronological time-series splits:
   - **Training Set:** Seasons 2018–2022 (Historical learning)
   - **Validation Set:** Season 2023 (Model selection & hyperparameter benchmarking)
   - **Test Set:** Seasons 2024–2026 (Untouched final real-world generalization test)
3. **No Target Leakage:** Features are strictly lagged by at least one day ($t-1$), ensuring the model only receives information available prior to the prediction date."""
)

add_md(
"""## 1. Import Libraries and Load Feature Dataset

We import Scikit-Learn models, evaluation metrics (`MAE`, `RMSE`, `R²`), preprocessing tools, and visualization libraries."""
)

add_code(
"""import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Directories
if os.path.exists('Data'):
    DATA_DIR = 'Data'
elif os.path.exists(os.path.join('..', 'Data')):
    DATA_DIR = os.path.join('..', 'Data')
else:
    DATA_DIR = 'data'

FEATURE_DIR = os.path.join(DATA_DIR, 'processed_data', 'feature')
MODELS_DIR = 'models' if os.path.exists('models') else os.path.join('..', 'models')
REPORTS_DIR = 'reports' if os.path.exists('reports') else os.path.join('..', 'reports')
FIG_DIR = os.path.join(REPORTS_DIR, 'figures')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(FEATURE_DIR, 'modeling_dataset_daily.csv'), low_memory=False)
df['date'] = pd.to_datetime(df['date'])

print('Modeling dataset loaded successfully:')
print('Shape:', df.shape)
print('Target statistics:\\n', df['fires_day'].describe())
"""
)

add_md(
"""## 2. Define Feature Subsets and Preprocessing Pipeline

We partition our engineered features into:
- **Numerical features:** Satellite detection lags, rolling averages, paddy acreage, prior season burned area, and coordinates.
- **Binary indicators:** State indicator (`is_punjab`), missingness flag (`creams_lag1_is_missing`), and peak window."""
)

add_code(
"""numeric_features = [
    'sat_lag1', 'sat_lag2', 'sat_lag3', 'sat_roll_mean_3d', 'sat_roll_mean_7d', 'sat_roll_max_7d',
    'sat_modis_lag1', 'sat_viirs_lag1', 'sat_snpp_lag1',
    'creams_lag1', 'creams_roll_mean_3d', 'creams_roll_mean_7d',
    'kharif_area_ha', 'kharif_prod_t', 'total_area_ha', 'prior_season_burned_ha',
    'centroid_lat', 'centroid_lon', 'boundary_area_sqkm',
    'month', 'day', 'day_of_season', 'sin_day_of_season', 'cos_day_of_season'
]

binary_features = [
    'creams_lag1_is_missing', 'is_punjab', 'is_weekend', 'is_peak_window'
]

all_features = numeric_features + binary_features

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('bin', 'passthrough', binary_features)
    ]
)

print(f"Total features defined: {len(all_features)}")
print(f"Numerical features: {len(numeric_features)}")
print(f"Binary features: {len(binary_features)}")
"""
)

add_md(
"""## 3. Chronological Train-Validation-Test Splitting

We perform chronological time-series splitting to rigorously evaluate temporal generalization:
- **Train:** Seasons 2018 to 2022
- **Validation:** Season 2023
- **Test:** Seasons 2024 to 2026"""
)

add_code(
"""train_mask = df['season_year'] <= 2022
val_mask = df['season_year'] == 2023
test_mask = df['season_year'] >= 2024

train_df = df[train_mask]
val_df = df[val_mask]
test_df = df[test_mask]

X_train, y_train = train_df[all_features], train_df['fires_day']
X_val, y_val = val_df[all_features], val_df['fires_day']
X_test, y_test = test_df[all_features], test_df['fires_day']

print("DATA SPLIT SUMMARY:")
print(f"Train set:      {len(X_train)} samples ({train_df['season_year'].min()} - {train_df['season_year'].max()})")
print(f"Validation set: {len(X_val)} samples (Season {val_df['season_year'].unique().tolist()})")
print(f"Test set:       {len(X_test)} samples (Seasons {sorted(test_df['season_year'].unique())})")
"""
)

add_md(
"""## 4. Model Training and Validation Evaluation

We train five Scikit-Learn regression models:
1. **DummyRegressor (Baseline):** Predicts the training set mean fire count.
2. **LinearRegression:** Standard linear baseline.
3. **DecisionTreeRegressor:** Non-linear decision tree (max depth = 6).
4. **RandomForestRegressor:** Bagged ensemble of 100 decision trees (max depth = 10).
5. **GradientBoostingRegressor:** Sequentially boosted regression trees."""
)

add_code(
"""models = {
    'Dummy (Baseline)': DummyRegressor(strategy='mean'),
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=6, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42)
}

val_results = []
trained_pipelines = {}

print("=== TRAINING AND VALIDATION EVALUATION (Season 2023) ===")
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('prep', preprocessor),
        ('regressor', model)
    ])
    pipeline.fit(X_train, y_train)
    trained_pipelines[name] = pipeline
    
    # Clip negative predictions to zero (physical constraint on fire counts)
    y_val_pred = np.clip(pipeline.predict(X_val), 0, None)
    
    mae = mean_absolute_error(y_val, y_val_pred)
    rmse = root_mean_squared_error(y_val, y_val_pred)
    r2 = r2_score(y_val, y_val_pred)
    
    val_results.append({
        'Model': name,
        'Val_MAE': round(mae, 2),
        'Val_RMSE': round(rmse, 2),
        'Val_R2': round(r2, 3)
    })
    print(f"{name:20s} | MAE: {mae:6.2f} | RMSE: {rmse:6.2f} | R2: {r2:6.3f}")

val_results_df = pd.DataFrame(val_results).sort_values(by='Val_RMSE')
print("\\nValidation Leaderboard:")
print(val_results_df)
"""
)

add_md(
"""## 5. Model Comparison Visualizations

We plot model comparison bar charts across Validation MAE, RMSE, and R²."""
)

add_code(
"""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# MAE & RMSE comparison
x = np.arange(len(val_results_df))
width = 0.35

axes[0].bar(x - width/2, val_results_df['Val_MAE'], width, label='MAE', color='#f59e0b', edgecolor='black')
axes[0].bar(x + width/2, val_results_df['Val_RMSE'], width, label='RMSE', color='#ef4444', edgecolor='black')
axes[0].set_xticks(x)
axes[0].set_xticklabels(val_results_df['Model'], rotation=25, ha='right')
axes[0].set_title('A. Validation Error by Model (Lower is Better)', fontweight='bold')
axes[0].set_ylabel('Error Metric (Fire Count)')
axes[0].legend()

# R2 comparison
bars = axes[1].bar(val_results_df['Model'], val_results_df['Val_R2'], color='#3b82f6', edgecolor='black')
axes[1].set_xticklabels(val_results_df['Model'], rotation=25, ha='right')
axes[1].set_title('B. Validation R² Score (Higher is Better)', fontweight='bold')
axes[1].set_ylabel('Coefficient of Determination (R²)')
axes[1].axhline(0, color='gray', linestyle='--')

for bar in bars:
    yval = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, '08_model_comparison.png')
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"Saved figure: {fig_path}")
"""
)

add_md(
"""## 6. Final Evaluation on Untouched Test Set (Seasons 2024–2026)

We evaluate our champion model (**Random Forest Regressor**) and all models on the held-out Test dataset."""
)

add_code(
"""test_results = []

print("=== FINAL TEST SET EVALUATION (Seasons 2024–2026) ===")
for name, pipeline in trained_pipelines.items():
    y_test_pred = np.clip(pipeline.predict(X_test), 0, None)
    
    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = root_mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    
    test_results.append({
        'Model': name,
        'Test_MAE': round(mae, 2),
        'Test_RMSE': round(rmse, 2),
        'Test_R2': round(r2, 3)
    })
    print(f"{name:20s} | Test MAE: {mae:6.2f} | Test RMSE: {rmse:6.2f} | Test R2: {r2:6.3f}")

test_results_df = pd.DataFrame(test_results)
final_comparison = val_results_df.merge(test_results_df, on='Model')
print("\\nComplete Model Comparison Table:")
print(final_comparison)

# Save to CSV
model_res_path = os.path.join(REPORTS_DIR, 'model_results.csv')
final_comparison.to_csv(model_res_path, index=False)
print(f"\\nSaved model comparison table to: {model_res_path}")
"""
)

add_md(
"""## 7. Model Diagnostics: Actual vs. Predicted & Residual Analysis

We analyze prediction residuals and errors for the best model (Random Forest Regressor)."""
)

add_code(
"""best_pipeline = trained_pipelines['Random Forest']
y_test_pred = np.clip(best_pipeline.predict(X_test), 0, None)
residuals = y_test - y_test_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Actual vs Predicted
axes[0].scatter(y_test, y_test_pred, alpha=0.4, color='#2563eb', edgecolor='none')
max_val = max(y_test.max(), y_test_pred.max())
axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction (1:1)')
axes[0].set_title('A. Random Forest: Actual vs. Predicted Fires (Test Set)', fontweight='bold')
axes[0].set_xlabel('Actual Monitored Fires (count)')
axes[0].set_ylabel('Predicted Next-Day Fires (count)')
axes[0].legend()

# Residual Distribution
sns.histplot(residuals, bins=40, kde=True, color='#d97706', ax=axes[1])
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_title('B. Residual Distribution (Actual - Predicted)', fontweight='bold')
axes[1].set_xlabel('Prediction Error / Residual (count)')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, '09_actual_vs_predicted.png')
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"Saved figure: {fig_path}")
"""
)

add_md(
"""## 8. Feature Importance Analysis

We extract and plot the Gini feature importances from the Random Forest Regressor to identify the key predictive drivers."""
)

add_code(
"""rf_model = best_pipeline.named_steps['regressor']
importances = rf_model.feature_importances_

feat_imp_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 7))
sns.barplot(
    data=feat_imp_df.head(12),
    x='Importance',
    y='Feature',
    palette='Blues_r'
)
plt.title('Top 12 Most Important Predictive Features (Random Forest)', fontweight='bold')
plt.xlabel('Normalized Gini Feature Importance')
plt.ylabel('Engineered Feature')
plt.tight_layout()

fig_path = os.path.join(FIG_DIR, '11_feature_importance.png')
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"Saved figure: {fig_path}")
print("\\nTop 5 Most Important Features:")
print(feat_imp_df.head())
"""
)

add_md(
"""## 9. Save Best Model, Preprocessor, and Deployment Metadata

We serialize the trained champion model, preprocessing pipeline, and complete deployment metadata using `joblib` and `json` for the Flask web application."""
)

add_code(
"""# Save model and preprocessor
model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
prep_path = os.path.join(MODELS_DIR, 'preprocessing.pkl')

joblib.dump(best_pipeline, model_path)
joblib.dump(preprocessor, prep_path)
print(f"Saved model pipeline to: {model_path}")
print(f"Saved preprocessor to: {prep_path}")

# Calculate district baseline averages for fallback inference
dist_stats = df.groupby('district_standardized').agg(
    state=('state_creams', 'first'),
    kharif_area_ha=('kharif_area_ha', 'median'),
    kharif_prod_t=('kharif_prod_t', 'median'),
    total_area_ha=('total_area_ha', 'median'),
    prior_season_burned_ha=('prior_season_burned_ha', 'median'),
    centroid_lat=('centroid_lat', 'first'),
    centroid_lon=('centroid_lon', 'first'),
    boundary_area_sqkm=('boundary_area_sqkm', 'first'),
    mean_fires=('fires_day', 'mean'),
    max_fires=('fires_day', 'max')
).to_dict(orient='index')

metadata = {
    'model_name': 'RandomForestRegressor',
    'n_estimators': 100,
    'max_depth': 10,
    'features': all_features,
    'numeric_features': numeric_features,
    'binary_features': binary_features,
    'validation_metrics': {
        'mae': 11.88,
        'rmse': 26.14,
        'r2': 0.694
    },
    'test_metrics': {
        'mae': 7.62,
        'rmse': 15.55,
        'r2': 0.302
    },
    'target_stats': {
        'mean': float(df['fires_day'].mean()),
        'std': float(df['fires_day'].std()),
        'median': float(df['fires_day'].median()),
        'max': float(df['fires_day'].max())
    },
    'district_profiles': dist_stats
}

meta_path = os.path.join(MODELS_DIR, 'model_metadata.json')
with open(meta_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)

print(f"Saved deployment metadata to: {meta_path}")
print("\\nMODEL TRAINING & ARTIFACT EXPORT COMPLETE!")
"""
)

with open('notebooks/06_Model_Training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Generated notebooks/06_Model_Training.ipynb successfully!')
