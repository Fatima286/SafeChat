import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"
PEOPLE = {"person1": "Person 1", "person2": "Person 2"}
DEFAULT_LIMIT = 5

st.set_page_config(page_title="SafeChat", page_icon="🛡️", layout="wide")


# ---------- API helpers ----------

def api_get_status(sender_id):
    """Ask the backend for a sender's points/mute state. Returns None if unreachable."""
    try:
        r = requests.get(f"{API_URL}/sender/{sender_id}", timeout=5)
        r.raise_for_status()
        d = r.json()
        return {"points": d["points"], "mute": d["mute"], "limit": d["mute_limit"]}
    except requests.RequestException:
        return None


def send_message(sender_id):
    """Button callback: runs BEFORE the script reruns, so both panels
    are drawn from the already-updated message list."""
    text = st.session_state.get(f"input_{sender_id}", "").strip()
    if not text:
        return

    try:
        r = requests.post(
            f"{API_URL}/message",
            json={"sender_id": sender_id, "text": text},
            timeout=15,
        )
    except requests.RequestException:
        st.session_state.last_error = "Backend not reachable. Is uvicorn running?"
        return

    if r.status_code != 200:
        try:
            detail = r.json().get("detail")
        except ValueError:
            detail = r.text
        if isinstance(detail, list) and detail:  # FastAPI validation errors are a list
            detail = detail[0].get("msg", detail)
        st.session_state.last_error = f"Request failed ({r.status_code}): {detail}"
        return

    data = r.json()
    st.session_state.messages.append(
        {"sender": sender_id, "text": text, "label": data["label"], "score": data["score"]}
    )
    st.session_state.status[sender_id] = {
        "points": data["points"],
        "mute": data["mute"],
        "limit": data["mute_limit"],
    }


def reset_all():
    """Sidebar callback: reset both senders on the backend and clear the chat."""
    for pid in PEOPLE:
        try:
            requests.post(f"{API_URL}/sender/{pid}/reset", timeout=5)
        except requests.RequestException:
            st.session_state.last_error = "Backend not reachable, reset failed."
            return
        limit = st.session_state.status[pid]["limit"]
        st.session_state.status[pid] = {"points": 0, "mute": False, "limit": limit}
    st.session_state.messages = []


# ---------- session state (created once per browser session) ----------

if "messages" not in st.session_state:
    st.session_state.messages = []      # shared history: sender, text, label, score
if "last_error" not in st.session_state:
    st.session_state.last_error = ""
if "status" not in st.session_state:
    st.session_state.status = {}
    for pid in PEOPLE:
        s = api_get_status(pid)         # the backend remembers points even if the page is refreshed
        if s is None:
            st.session_state.last_error = "Backend not reachable. Start uvicorn, then refresh the page."
            s = {"points": 0, "mute": False, "limit": DEFAULT_LIMIT}
        st.session_state.status[pid] = s


# ---------- one chat panel ----------

def draw_panel(me, other, show_scores):
    status = st.session_state.status[me]

    st.subheader(PEOPLE[me])
    st.caption(f"Warnings: {status['points']} / {status['limit']}")
    st.progress(min(status["points"] / status["limit"], 1.0))  # capped: points can pass the limit

    with st.container(height=420):
        for m in st.session_state.messages:
            if m["sender"] == me:
                # what the SENDER sees
                with st.chat_message("user", avatar="🧑"):
                    if m["label"] == "block":
                        st.caption("🚫 Message blocked, not delivered.")
                    elif m["label"] == "muted":
                        st.caption("🔇 Not delivered, you are muted.")
                    else:
                        st.write(m["text"])
                        if m["label"] == "warn":
                            st.caption("⚠️ Warning: this message may be hurtful.")
                    if show_scores and m["score"] is not None:
                        st.caption(f"score {m['score']:.2f} · {m['label']}")
            elif m["label"] in ("allow", "warn"):
                # what the RECIPIENT sees: only delivered messages
                with st.chat_message("assistant", avatar="💬"):
                    st.caption(PEOPLE[other])
                    st.write(m["text"])

    muted = status["mute"]
    if muted:
        st.error("🔇 You are muted and can't send messages.")

    with st.form(f"form_{me}", clear_on_submit=True):
        st.text_input(
            "Message",
            key=f"input_{me}",
            placeholder="Type a message...",
            label_visibility="collapsed",
            disabled=muted,
        )
        st.form_submit_button("Send", on_click=send_message, args=(me,), disabled=muted)


# ---------- page ----------

st.title("🛡️ SafeChat")

show_scores = st.sidebar.checkbox("Show scores (debug)", value=False)
st.sidebar.button("Reset chat and counters", on_click=reset_all)
st.sidebar.caption("Reset clears the history and both senders' warnings on the backend.")

if st.session_state.last_error:
    st.error(st.session_state.last_error)
    st.session_state.last_error = ""    # show it once

left, right = st.columns(2)
with left:
    with st.container(border=True):
        draw_panel("person1", "person2", show_scores)
with right:
    with st.container(border=True):
        draw_panel("person2", "person1", show_scores)