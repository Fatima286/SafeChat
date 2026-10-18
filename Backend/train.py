import scipy.sparse as sp
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.linear_model import Ridge
import joblib
#----------training------------
X_train=sp.load_npz("tfidf/train_tfIdf.npz")  # shape (n_samples, n_features)
Y_train=pd.read_csv("tfidf/train_hateScore.csv")["hate_speech_score"]  # shape (n_samples,)

model=Ridge(alpha=1.95)

#assert X_train.shape[0] == len(Y_train), "Row count mismatch between X and Y!"----------->PASSED

model.fit(X_train,Y_train)
joblib.dump(model, "model.pkl")
print("Saved model")
#--------validation-----------------
"""
X_val=sp.load_npz("tfidf/val_tfIdf.npz")
Y_val=pd.read_csv("tfidf/val_hateScore.csv")["hate_speech_score"]

alpha=[1.95,1.96,1.97,1.98,1.99,2.00 ]

best_mse=float("inf")
best_alpha=None

for a in alpha:
    model=Ridge(alpha=a)
    model.fit(X_train,Y_train)
    predicted_score=model.predict(X_val)
    
    mse=mean_squared_error(Y_val,predicted_score)

    print(f"alpha={a} val error={mse:.4f}")

    if mse<best_mse:
        best_alpha=a
        best_mse=mse
        print(f"Current best alpha={best_alpha}")

print(f"\n\nfinally best alpha: {best_alpha}")

#Final fit with best alpha
model=Ridge(alpha=best_alpha)
model.fit(X_train,Y_train)

#print(Y_train.describe())
#print(Y_train.std())  #TRYING TO REDUCE MSE
"""

#===========FINALISED ALPHA=1.95==============================#
print("Trained")

#Note: NOW THE MODEL HAS LEARNED THE TEST DATASET DO NOT CHANGE ANY HYPERPARAMETERS
#----------------Testing----------

def testing(model:Ridge):
    X_test=sp.load_npz("tfidf/test_tfIdf.npz")  # shape (n_samples, n_features)
    Y_test=pd.read_csv("tfidf/test_hateScore.csv")["hate_speech_score"]  # shape (n_samples,)
    predicted_score=model.predict(X_test)
    mse=mean_squared_error(Y_test,predicted_score)
    mae=mean_absolute_error(Y_test,predicted_score)
    r2=r2_score(Y_test,predicted_score)
    print("MSE: ",mse)
    print("MAE: ",mae)
    print("R2: ",r2)


testing(model)