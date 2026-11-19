"""Comprehensive automated test suite for Stubble Fire Watch.

Tests model serialization, data preparation, inference engine, input validation
gates (unknown districts, invalid dates, negative inputs), input provenance
tracking, physics constraints (non-negativity), Flask web routes, and REST API
status codes.
"""

from datetime import datetime
import json
import os
import unittest
from app import app
from src.data_preparation import compute_date_features, normalize_district_name
from src.prediction import get_model_and_metadata, predict_stubble_fires, validate_prediction_inputs


class TestStubbleFireWatch(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.client = app.test_client()
    cls.model, cls.metadata = get_model_and_metadata()

  # --- Model & Serialization Tests ---

  def test_01_model_and_metadata_loaded(self):
    """Verify model pipeline and metadata are correctly serialized and loadable."""
    self.assertIsNotNone(self.model)
    self.assertIsNotNone(self.metadata)
    self.assertIn("features", self.metadata)
    self.assertEqual(len(self.metadata["features"]), 29)
    self.assertIn("district_profiles", self.metadata)
    self.assertIn("validation_metrics", self.metadata)

  # --- Input Validation Tests ---

  def test_02_unknown_district_rejection(self):
    """Verify that unknown/unmonitored districts raise ValueError and are rejected."""
    with self.assertRaises(ValueError) as ctx:
      predict_stubble_fires("Atlantis", "2026-11-05")
    self.assertIn("Unknown district 'Atlantis'", str(ctx.exception))

    with self.assertRaises(ValueError):
      predict_stubble_fires("", "2026-11-05")

  def test_03_invalid_date_rejection(self):
    """Verify that invalid date formats or impossible dates raise ValueError."""
    with self.assertRaises(ValueError) as ctx:
      predict_stubble_fires("Sangrur", "not-a-date")
    self.assertIn("Invalid date", str(ctx.exception))

    with self.assertRaises(ValueError) as ctx:
      predict_stubble_fires("Sangrur", "2026-02-31")
    self.assertIn("Invalid date", str(ctx.exception))

  def test_04_negative_input_rejection(self):
    """Verify that negative feature values are rejected with ValueError."""
    with self.assertRaises(ValueError) as ctx:
      predict_stubble_fires("Sangrur", "2026-11-05", {"sat_lag1": -10})
    self.assertIn("cannot be negative", str(ctx.exception))

    with self.assertRaises(ValueError) as ctx:
      predict_stubble_fires("Sangrur", "2026-11-05", {"kharif_area_ha": -500})
    self.assertIn("cannot be negative", str(ctx.exception))

  # --- Data Preparation & Temporal Alignment Tests ---

  def test_05_data_preparation_date_features(self):
    """Verify calendar feature computation and peak window detection."""
    peak_feats = compute_date_features("2026-11-05")
    self.assertEqual(peak_feats["month"], 11)
    self.assertEqual(peak_feats["is_peak_window"], 1)

    offpeak_feats = compute_date_features("2026-09-20")
    self.assertEqual(offpeak_feats["month"], 9)
    self.assertEqual(offpeak_feats["is_peak_window"], 0)

  def test_06_temporal_alignment_dates(self):
    """Verify target_date and feature_reference_date differ by exactly 1 calendar day."""
    res = predict_stubble_fires("Sangrur", "2026-11-05")
    t_date = datetime.strptime(res["target_date"], "%Y-%m-%d")
    ref_date = datetime.strptime(res["feature_reference_date"], "%Y-%m-%d")
    diff = (t_date - ref_date).days
    self.assertEqual(diff, 1)

  # --- Inference & Provenance Tests ---

  def test_07_input_provenance_tracking(self):
    """Verify user-provided inputs vs baseline estimates are accurately tracked."""
    # Custom input provided
    res_user = predict_stubble_fires(
        "Sangrur", "2026-11-05", {"sat_lag1": 45, "creams_lag1": 30}
    )
    self.assertEqual(
        res_user["provenance"]["sat_lag1"]["source"], "User-Provided"
    )
    self.assertEqual(
        res_user["provenance"]["creams_lag1"]["source"], "User-Provided"
    )

    # Defaults (no custom inputs provided)
    res_default = predict_stubble_fires("Sangrur", "2026-11-05")
    self.assertEqual(
        res_default["provenance"]["sat_lag1"]["source"],
        "Baseline Zero (No satellite anomaly reported)",
    )
    self.assertIn(
        "Unmonitored", res_default["provenance"]["creams_lag1"]["source"]
    )
    self.assertIn(
        "Official District Statistics",
        res_default["provenance"]["kharif_area_ha"]["source"],
    )

  def test_08_prediction_engine_valid_district(self):
    """Verify prediction output structure, count positivity, and risk categorization."""
    res = predict_stubble_fires(
        "Sangrur", "2026-11-05", {"sat_lag1": 100, "creams_lag1": 80}
    )
    self.assertIn("predicted_fires", res)
    self.assertGreaterEqual(res["predicted_fires"], 0.0)
    self.assertIn("risk", res)
    self.assertIn(
        res["risk"]["level"], ["Low", "Moderate", "High", "Severe"]
    )
    self.assertEqual(res["district"], "Sangrur")

  def test_09_prediction_non_negative_constraint(self):
    """Verify predictions are strictly clamped at 0 even under minimal conditions."""
    res = predict_stubble_fires(
        "Ambala",
        "2026-09-15",
        {"sat_lag1": 0, "creams_lag1": 0, "sat_roll_mean_3d": 0},
    )
    self.assertGreaterEqual(res["predicted_fires"], 0.0)

  # --- Flask Web Route Tests ---

  def test_10_flask_index_page(self):
    """Test landing page loads with status 200."""
    response = self.client.get("/")
    self.assertEqual(response.status_code, 200)
    self.assertIn(b"Stubble Fire Watch", response.data)

  def test_11_flask_predict_get(self):
    """Test prediction form loads with status 200."""
    response = self.client.get("/predict")
    self.assertEqual(response.status_code, 200)
    self.assertIn(b"District Fire Risk Predictor", response.data)

  def test_12_flask_predict_post_valid(self):
    """Test valid prediction form submission returns 200 and result table."""
    response = self.client.post(
        "/predict",
        data={
            "state": "Punjab",
            "district": "sangrur",
            "date": "2026-11-05",
            "sat_lag1": "75",
        },
    )
    self.assertEqual(response.status_code, 200)
    self.assertIn(b"Predicted Next-Day Fire Count", response.data)
    self.assertIn(b"Input Feature Provenance", response.data)

  def test_13_flask_dashboard_page(self):
    """Test analytics dashboard loads with status 200."""
    response = self.client.get("/dashboard")
    self.assertEqual(response.status_code, 200)
    self.assertIn(b"Machine Learning Regression Benchmark", response.data)

  def test_14_flask_about_page(self):
    """Test about & methodology page loads with status 200."""
    response = self.client.get("/about")
    self.assertEqual(response.status_code, 200)
    self.assertIn(b"Research Methodology", response.data)

  # --- REST API Tests ---

  def test_15_api_districts(self):
    """Test REST API returning district list."""
    response = self.client.get("/api/districts")
    self.assertEqual(response.status_code, 200)
    data = json.loads(response.data)
    self.assertEqual(data["status"], "success")
    self.assertGreaterEqual(len(data["districts"]), 43)

  def test_16_api_predict_json_valid_and_invalid(self):
    """Test REST API POST with valid payload (200) and invalid district/negative (400)."""
    # Valid payload
    payload_valid = {
        "district": "Bathinda",
        "date": "2026-11-02",
        "inputs": {"sat_lag1": 80, "creams_lag1": 60},
    }
    resp1 = self.client.post(
        "/api/predict",
        data=json.dumps(payload_valid),
        content_type="application/json",
    )
    self.assertEqual(resp1.status_code, 200)
    data1 = json.loads(resp1.data)
    self.assertEqual(data1["status"], "success")
    self.assertGreaterEqual(data1["prediction"]["predicted_fires"], 0.0)

    # Invalid district -> 400
    payload_invalid_dist = {"district": "GothamCity", "date": "2026-11-02"}
    resp2 = self.client.post(
        "/api/predict",
        data=json.dumps(payload_invalid_dist),
        content_type="application/json",
    )
    self.assertEqual(resp2.status_code, 400)
    data2 = json.loads(resp2.data)
    self.assertIn("Unknown district", data2["message"])

    # Negative input -> 400
    payload_negative = {
        "district": "Bathinda",
        "date": "2026-11-02",
        "inputs": {"sat_lag1": -25},
    }
    resp3 = self.client.post(
        "/api/predict",
        data=json.dumps(payload_negative),
        content_type="application/json",
    )
    self.assertEqual(resp3.status_code, 400)
    data3 = json.loads(resp3.data)
    self.assertIn("cannot be negative", data3["message"])


if __name__ == "__main__":
  unittest.main()
