

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from tracker import process_score, sender_state
from config import MUTE_LIMIT

app = FastAPI(title="SafeChat API")

# Allow the Streamlit frontend (default port 8501) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- request / response shapes ----------

class MessageRequest(BaseModel):
    sender_id: str = Field(min_length=1, max_length=50)
    text: str = Field(min_length=1, max_length=2000)


class MessageResponse(BaseModel):
    label: str            # allow / warn / block / muted
    score: float | None   # None when the sender was already muted
    points: int
    mute: bool
    mute_limit: int       # lets the frontend show "3 / 5" without hard-coding 5


class SenderStatus(BaseModel):
    sender_id: str
    points: int
    mute: bool
    mute_limit: int


# ---------- endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok"}


# plain `def` (not `async def`): scoring is CPU work, and FastAPI runs plain
# `def` endpoints in a worker thread so the server stays responsive
@app.post("/message", response_model=MessageResponse)
def post_message(req: MessageRequest):
    sender_id = req.sender_id.strip()
    text = req.text.strip()

    # min_length lets "   " through, so reject whitespace-only input here
    if not sender_id or not text:
        raise HTTPException(status_code=400, detail="sender_id and text must not be blank")

    try:
        result = process_score(sender_id, text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # log sender, label and score only (not the message text)
    print(f"[{sender_id}] label={result['label']} score={result['score']} points={result['points']}")

    return {**result, "mute_limit": MUTE_LIMIT}


@app.get("/sender/{sender_id}", response_model=SenderStatus)
def get_sender(sender_id: str):
    # .get(...) only LOOKS UP the sender; it must not create a record
    points, muted = sender_state.get(sender_id, [0, False])
    return {"sender_id": sender_id, "points": points, "mute": muted, "mute_limit": MUTE_LIMIT}


# Demo/testing tool only: in a real system anyone could un-mute themselves
@app.post("/sender/{sender_id}/reset")
def reset_sender(sender_id: str):
    existed = sender_state.pop(sender_id, None) is not None
    return {"sender_id": sender_id, "reset": True, "existed": existed}