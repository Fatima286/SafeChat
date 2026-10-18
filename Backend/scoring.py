import joblib
from config import MODEL_PATH, VECTORIZER_PATH,WARN_THRESHOLD,BLOCK_THRESHOLD
from dataprep.preprocess import Preprocess  #importing the class

#preprocessing object
obj=Preprocess()

vec_model=joblib.load(VECTORIZER_PATH)
ML_model=joblib.load(MODEL_PATH)


def message_to_features(text):
    tokens=obj.preprocess_pipeline(text)
    processed_text=' '.join(tokens)

    #applying the vectorization model
    vectorized_text=vec_model.transform([processed_text]) 
    return vectorized_text

def score_message(text):

    #call the fucntion above
    vectors=message_to_features(text)
    score=ML_model.predict(vectors)
    return float(score[0])

def decision(score):

    if score>=BLOCK_THRESHOLD:
        return 'block'
    elif score>=WARN_THRESHOLD:
        return 'warn'
    else:
        return 'allow'
    