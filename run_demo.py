"""Demo runner that exercises the trained classifier and reports summary metrics."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

... (trimmed) ...
