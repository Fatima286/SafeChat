import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
from sklearn.metrics import precision_score, recall_score

model = joblib.load("model.pkl")
X_val = sp.load_npz("tfidf/val_tfIdf.npz")
y_val = pd.read_csv("tfidf/val_hateScore.csv")["hate_speech_score"].values

pred = model.predict(X_val)

"""# 1. Look at the distributions
print("TRUE scores:\n", pd.Series(y_val).describe())
print("\nPREDICTED scores:\n", pd.Series(pred).describe())
print("\nPredicted percentiles:")
for q in [50, 75, 90, 95, 99]:
    print(f"  {q}th: {np.percentile(pred, q):.3f}")"""

# 2. Precision/recall at each candidate threshold
is_hate = y_val > 0.5
print(f"\nShare of truly hateful messages: {is_hate.mean():.1%}\n")
print("threshold | precision | recall | % flagged")
for t in np.arange(-1.0, 2.01, 0.25):
    flagged = pred >= t
    p = precision_score(is_hate, flagged, zero_division=0)
    r = recall_score(is_hate, flagged, zero_division=0)
    print(f"{t:9.2f} | {p:9.2f} | {r:6.2f} | {flagged.mean():8.1%}")