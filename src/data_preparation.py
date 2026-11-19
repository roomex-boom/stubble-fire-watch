"""Data preparation helper module for Stubble Fire Watch.

Handles loading of integrated datasets, date/calendar calculations, and feature
vector construction.
"""

import os
import re
import numpy as np
import pandas as pd


def normalize_district_name(value):
  """Standardizes district names by lowercasing and stripping non-alphanumeric

  characters.
  """
  if pd.isna(value):
    return ""
  val = str(value).strip().lower()
  val = re.sub(r"[^a-z0-9\s]", "", val)
  return re.sub(r"\s+", " ", val)


def compute_date_features(date_str):
  """Computes calendar and seasonal features for a given date string (YYYY-MM-DD)."""
  dt = pd.to_datetime(date_str)
  year = dt.year
  month = dt.month
  day = dt.day
  day_of_week = dt.dayofweek
  is_weekend = 1 if day_of_week in [5, 6] else 0
  day_of_year = dt.dayofyear

  season_start = pd.to_datetime(f"{year}-09-15")
  day_of_season = (dt - season_start).days
  sin_day_of_season = float(np.sin(2 * np.pi * day_of_season / 90.0))
  cos_day_of_season = float(np.cos(2 * np.pi * day_of_season / 90.0))

  is_peak_window = (
      1
      if ((month == 10 and day >= 15) or (month == 11 and day <= 20))
      else 0
  )

  return {
      "month": month,
      "day": day,
      "day_of_week": day_of_week,
      "is_weekend": is_weekend,
      "day_of_year": day_of_year,
      "day_of_season": day_of_season,
      "sin_day_of_season": sin_day_of_season,
      "cos_day_of_season": cos_day_of_season,
      "is_peak_window": is_peak_window,
  }
