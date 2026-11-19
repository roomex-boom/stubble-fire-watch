"""Automated test suite for Stubble Fire Watch.

Tests model loading, data preparation, inference engine, physics constraints,
Flask web routes, and REST API endpoints.
"""

import json
import os
import unittest
from app import app
from src.data_preparation import compute_date_features, normalize_district_name
from src.prediction import get_model_and_metadata, predict_stubble_fires


class TestStubbleFireWatch(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.client = app.test_client()
    cls.model, cls.metadata = get_model_and_metadata()

  def test_01_model_and_metadata_loaded(self):
    """Verify model pipeline and metadata are correctly serialized and loadable."""
    self.assertIsNotNone(self.model)
    self.assertIsNotNone(self.metadata)
    self.assertIn("features", self.metadata)
    self.assertEqual(len(self.metadata["features"]), 28)
    self.assertIn("district_profiles", self.metadata)

  def test_02_data_preparation_date_features(self):
    """Verify calendar feature computation and peak window detection."""
    # Test peak date
    peak_feats = compute_date_features("2026-11-05")
    self.assertEqual(peak_feats["month"], 11)
    self.assertEqual(peak_feats["is_peak_window"], 1)

    # Test off-peak date
    offpeak_feats = compute_date_features("2026-09-20")
    self.assertEqual(offpeak_feats["month"], 9)
    self.assertEqual(offpeak_feats["is_peak_window"], 0)

  def test_03_prediction_engine_valid_district(self):
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

  def test_04_prediction_non_negative_constraint(self):
    """Verify that predictions are strictly clamped at 0 even with zero inputs."""
    res = predict_stubble_fires(
        "Ambala",
        "2026-09-15",
        {"sat_lag1": 0, "creams_lag1": 0, "sat_roll_mean_3d": 0},
    )
    self.assertGreaterEqual(res["predicted_fires"], 0.0)

  def test_05_flask_index_page(self):
    """Test landing page loads with status 200."""
    response = self.client.get("/")
    self.assertEqual(response.status_code, 200)
    self.assertIn(b"Stubble Fire Watch", response.data)

  def test_06_flask_predict_get(self):
    """Test prediction form loads with status 200."""
    response = self.client.get("/predict")
    self.assertEqual(response.status_code, 200)
    self.assertIn(b"District Fire Risk Predictor", response.data)

  def test_07_flask_predict_post(self):
    """Test prediction form submission returns 200 and contains prediction result."""
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

  def test_08_flask_dashboard_page(self):
    """Test analytics dashboard loads with status 200."""
    response = self.client.get("/dashboard")
    self.assertEqual(response.status_code, 200)
    self.assertIn(b"Machine Learning Regression Benchmark", response.data)

  def test_09_flask_about_page(self):
    """Test about & methodology page loads with status 200."""
    response = self.client.get("/about")
    self.assertEqual(response.status_code, 200)
    self.assertIn(b"Research Methodology", response.data)

  def test_10_api_districts(self):
    """Test REST API returning district list."""
    response = self.client.get("/api/districts")
    self.assertEqual(response.status_code, 200)
    data = json.loads(response.data)
    self.assertEqual(data["status"], "success")
    self.assertGreater(len(data["districts"]), 40)

  def test_11_api_predict_json(self):
    """Test REST API POST prediction."""
    payload = {
        "district": "Bathinda",
        "date": "2026-11-02",
        "inputs": {"sat_lag1": 80, "creams_lag1": 60},
    }
    response = self.client.post(
        "/api/predict", data=json.dumps(payload), content_type="application/json"
    )
    self.assertEqual(response.status_code, 200)
    data = json.loads(response.data)
    self.assertEqual(data["status"], "success")
    self.assertGreaterEqual(data["prediction"]["predicted_fires"], 0.0)


if __name__ == "__main__":
  unittest.main()
