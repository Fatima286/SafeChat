# SafeChat

Toxic message detection for chat. SafeChat scores each message with a TF-IDF + Ridge Regression model, then warns, blocks or mutes senders based on the score. The backend is built with FastAPI and the frontend with Streamlit.

## How it works

1. A message is preprocessed (tokenised and stemmed with NLTK).
2. It is converted to TF-IDF features (unigrams and bigrams).
3. A Ridge Regression model predicts a `hate_speech_score`.
4. The score is compared against two thresholds:

| Predicted score | Decision |
|---|---|
| below 0.25 | Allow |
| 0.25 to 1.5 | Warn (sender sees a warning, message is delivered) |
| above 1.5 | Block (message is not delivered) |

5. Each sender collects points (warn = 1, block = 2). At 5 points the sender is muted and further messages skip the model entirely.

The thresholds favour precision over recall, so innocent users are rarely flagged.

## Model

- Dataset: [Berkeley Measuring Hate Speech](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech)
- Features: TF-IDF (`ngram_range=(1,2)`, `min_df=10`, `max_df=0.95`)
- Model: Ridge Regression (alpha 1.95)
- Test results: MSE 2.00, MAE 1.11, R² 0.53

## Project structure

```
SafeChat/
├── Backend/
│   ├── dataprep/      # aggregation, preprocessing, train/val/test split
│   ├── tfidf/         # vectoriser and TF-IDF features
│   ├── train.py       # trains and saves the Ridge model
│   ├── Caliberate.py  # threshold calibration
│   ├── config.py      # thresholds, points, mute limit, paths
│   ├── scoring.py     # message -> score -> decision
│   ├── tracker.py     # per-sender points and muting
│   ├── main.py        # FastAPI app
│   └── requirements.txt
└── Frontend/
    └── app.py         # Streamlit two-person chat UI
```

## Setup

```bash
git clone https://github.com/<your-username>/SafeChat.git
cd SafeChat/Backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
pip install streamlit
```

## Data and model files

The dataset and the large generated files are not stored in this repo. To rebuild them:

1. Download the dataset and place it at `Backend/dataprep/Berkley.parquet`.
2. Run, from `Backend/`, in this order: `dataprep/aggregate.py`, `dataprep/preprocess.py`, `dataprep/split.py`, `tfidf/vectorise.py`, `train.py`.



## Run

Start the API (from `Backend/`):

```bash
uvicorn main:app --reload
```

Start the chat UI (from `Frontend/`, in a second terminal):

```bash
streamlit run app.py
```

### API endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Health check |
| `POST /message` | Score a message and get the decision |
| `GET /sender/{id}` | View a sender's points and mute status |
| `POST /sender/{id}/reset` | Reset a sender |

## Limitations

- The model sometimes scores long abusive messages lower than short single words.
- Sender state is stored in memory and resets when the server restarts.



## Author

Warisha Fatima, B.Tech Computer Science, Bhilai Institute of Technology, Durg