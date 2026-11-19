"""Stubble Fire Watch - Flask Web Application.

Machine Learning-Based Prediction of Agricultural Residue Burning Activity
Across Districts of Northwestern India (Punjab & Haryana)
"""

import json
import os
from flask import Flask, flash, jsonify, render_template, request
import pandas as pd
from src.prediction import get_model_and_metadata, predict_stubble_fires

app = Flask(__name__)
app.secret_key = "stubble-fire-watch-secret-key-2026"

# Load metadata once on startup
_, metadata = get_model_and_metadata()


def get_district_list():
  """Returns sorted list of district dictionaries with name and state."""
  profiles = metadata.get("district_profiles", {})
  districts = []
  for d_name, info in profiles.items():
    districts.append({"name": d_name, "state": info.get("state", "Punjab")})
  districts.sort(key=lambda x: (x["state"], x["name"]))
  return districts


@app.route("/")
def index():
  """Landing page displaying project objectives and key dataset statistics."""
  stats = {
      "total_records": "47,397",
      "target_observations": "14,392",
      "num_districts": "43",
      "best_model_r2": f"{metadata['validation_metrics']['r2']:.2f}",
      "best_model_mae": f"{metadata['validation_metrics']['mae']:.1f}",
  }
  return render_template("index.html", active_page="home", stats=stats)


@app.route("/predict", methods=["GET", "POST"])
def predict_page():
  """Interactive prediction page allowing users to forecast district stubble burning."""
  districts = get_district_list()
  result = None
  selected_state = "Punjab"
  selected_district = "sangrur"
  selected_date = "2026-11-05"

  if request.method == "POST":
    selected_state = request.form.get("state", "Punjab")
    selected_district = request.form.get("district", "sangrur")
    selected_date = request.form.get("date", "2026-11-05")

    # Optional custom inputs
    custom_inputs = {}
    sat_lag1 = request.form.get("sat_lag1")
    if sat_lag1 and sat_lag1.strip():
      try:
        custom_inputs["sat_lag1"] = float(sat_lag1)
      except ValueError:
        pass

    creams_lag1 = request.form.get("creams_lag1")
    if creams_lag1 and creams_lag1.strip():
      try:
        custom_inputs["creams_lag1"] = float(creams_lag1)
      except ValueError:
        pass

    kharif_area_ha = request.form.get("kharif_area_ha")
    if kharif_area_ha and kharif_area_ha.strip():
      try:
        custom_inputs["kharif_area_ha"] = float(kharif_area_ha)
      except ValueError:
        pass

    try:
      result = predict_stubble_fires(
          district_name=selected_district,
          date_str=selected_date,
          custom_inputs=custom_inputs,
      )
    except Exception as e:
      flash(f"Prediction error: {str(e)}", "error")

  return render_template(
      "predict.html",
      active_page="predict",
      districts=districts,
      result=result,
      selected_state=selected_state,
      selected_district=selected_district,
      selected_date=selected_date,
  )


@app.route("/dashboard")
def dashboard():
  """Analytics dashboard displaying empirical charts, figures, and model leaderboard."""
  # Load model results table
  res_path = os.path.join("reports", "model_results.csv")
  if os.path.exists(res_path):
    model_results = pd.read_csv(res_path).to_dict(orient="records")
  else:
    model_results = []

  return render_template(
      "dashboard.html", active_page="dashboard", model_results=model_results
  )


@app.route("/about")
def about():
  """Research methodology and data auditing documentation page."""
  return render_template("about.html", active_page="about")


# --- JSON REST APIs ---


@app.route("/api/districts")
def api_districts():
  """REST API endpoint returning list of monitored districts."""
  return jsonify({"status": "success", "districts": get_district_list()})


@app.route("/api/predict", methods=["POST"])
def api_predict():
  """REST API endpoint for programmatic prediction.

  Expects JSON payload: { "district": "Sangrur", "date": "2026-11-05", "inputs":
  {...} }
  """
  data = request.get_json(force=True, silent=True)
  if not data:
    return (
        jsonify({"status": "error", "message": "Invalid JSON body provided"}),
        400,
    )

  district = data.get("district")
  date_str = data.get("date")

  if not district or not date_str:
    return (
        jsonify({
            "status": "error",
            "message": "Both 'district' and 'date' fields are required.",
        }),
        400,
    )

  try:
    prediction = predict_stubble_fires(
        district_name=district,
        date_str=date_str,
        custom_inputs=data.get("inputs"),
    )
    return jsonify({"status": "success", "prediction": prediction})
  except Exception as e:
    return jsonify({"status": "error", "message": str(e)}), 500


@app.errorhandler(404)
def not_found(e):
  return render_template("base.html", content="404 Not Found"), 404


@app.errorhandler(500)
def server_error(e):
  return render_template("base.html", content="500 Internal Server Error"), 500


if __name__ == "__main__":
  print("Starting Stubble Fire Watch Flask Application...")
  print("Access locally at http://127.0.0.1:5000")
  app.run(host="127.0.0.1", port=5000, debug=True)
