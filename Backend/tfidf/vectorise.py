import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import scipy.sparse as sp
import os
 
OUTPUT_DIR = "tfidf"
os.makedirs(OUTPUT_DIR, exist_ok=True)

vectorizer=TfidfVectorizer(max_features=None,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.95)

#just takeout the processed text wala column

train_df=pd.read_csv("train_processed.csv").fillna("")
train_text=train_df["processed_text"]
train_hate_score=train_df["hate_speech_score"]

val_df=pd.read_csv("val_processed.csv").fillna("")
val_text=val_df["processed_text"]
val_hate_score=val_df["hate_speech_score"]

test_df=pd.read_csv("test_processed.csv").fillna("")
test_text=test_df["processed_text"]
test_hate_score=test_df["hate_speech_score"]

X_train=vectorizer.fit_transform(train_text)
print(X_train.shape[1])
X_val=vectorizer.transform(val_text)

X_test=vectorizer.transform(test_text)

""" SAVING """
#saving vectorizer object

joblib.dump(vectorizer,os.path.join(OUTPUT_DIR,"tfIdf_vectorizer.pkl"))  #.pkl are python pickle files. it is used to save and reload complex python object

#saving sparse matrix
sp.save_npz(os.path.join(OUTPUT_DIR,"train_tfIdf.npz"),X_train)
sp.save_npz(os.path.join(OUTPUT_DIR,"val_tfIdf.npz"),X_val)
sp.save_npz(os.path.join(OUTPUT_DIR,"test_tfIdf.npz"),X_test)

#saving the hate_speech_score in a separate csv to keep it aligned with the row order of the corresponding sparse matrix

train_hate_score.to_csv(os.path.join(OUTPUT_DIR,"train_hateScore.csv"),index=False)

val_hate_score.to_csv(os.path.join(OUTPUT_DIR,"val_hateScore.csv"),index=False)


test_hate_score.to_csv(os.path.join(OUTPUT_DIR,"test_hateScore.csv"),index=False)

