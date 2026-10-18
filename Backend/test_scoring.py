import numpy as np
import pandas as pd
import scipy.sparse as sp

from scoring import obj, score_message, ML_model

N = 20  # how many validation rows to check

df = pd.read_csv("tfidf/val_processed.csv")
print("Columns:", list(df.columns))

assert "text" in df.columns and "processed_text" in df.columns, (
    "Expected 'text' and 'processed_text' columns - check the printed column names"
)

sample = df.head(N).copy()
# empty strings are saved as blanks in CSV and read back as NaN
sample["processed_text"] = sample["processed_text"].fillna("")

# ---------- Part A: live preprocessing vs training preprocessing ----------
mismatches = 0
for i, row in sample.iterrows():
    live = " ".join(obj.preprocess_pipeline(row["text"]))
    saved = row["processed_text"]
    if live != saved:
        mismatches += 1
        print(f"\nMISMATCH at row {i}")
        print("  original:", repr(row["text"]))
        print("  live    :", repr(live))
        print("  saved   :", repr(saved))

print(f"\nPart A: {mismatches} mismatches out of {N}")

# ---------- Part B: live scores vs scores from the saved matrix ----------
X_val = sp.load_npz("tfidf/val_tfIdf.npz")
saved_scores = ML_model.predict(X_val[:N])
live_scores = np.array([score_message(t) for t in sample["text"]])

max_diff = np.abs(saved_scores - live_scores).max()
print(f"Part B: max difference = {max_diff:.2e}")
print("Part B: allclose =", np.allclose(saved_scores, live_scores, atol=1e-6))