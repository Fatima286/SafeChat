import tracker
from tracker import process_score, sender_state
from scoring import score_message, decision
from config import WARN_POINT, BLOCK_POINT, MUTE_LIMIT

# Texts that sit well inside each zone (scores seen in your manual tests)
FRIENDLY = "I love you"                     # about -5.1  -> allow
WARN_TEXT = "That's how you breed assholes" # about  1.19 -> warn
BLOCK_TEXT = "Fuck"                         # about  2.03 -> block


def reset():
    sender_state.clear()


def check_texts():
    # if this fails, the model/thresholds changed, not the tracker
    assert decision(score_message(FRIENDLY)) == "allow"
    assert decision(score_message(WARN_TEXT)) == "warn"
    assert decision(score_message(BLOCK_TEXT)) == "block"


def mute_sender(sender):
    """Send warn messages until the sender is muted; return their points."""
    for _ in range(100):
        r = process_score(sender, WARN_TEXT)
        if r["mute"]:
            return r["points"]
    raise AssertionError("sender never got muted")


def test_friendly_adds_nothing():
    reset()
    for _ in range(10):
        r = process_score("alice", FRIENDLY)
        assert r["label"] == "allow"
        assert r["points"] == 0
        assert r["mute"] is False


def test_points_follow_labels():
    reset()
    r = process_score("bob", WARN_TEXT)
    assert r["points"] == WARN_POINT
    r = process_score("bob", BLOCK_TEXT)
    assert r["points"] == WARN_POINT + BLOCK_POINT


def test_mute_happens_exactly_at_limit():
    reset()
    previous = 0
    for _ in range(100):
        r = process_score("carol", WARN_TEXT)
        if r["mute"]:
            assert r["points"] >= MUTE_LIMIT
            assert previous < MUTE_LIMIT  # the message before did not reach the limit
            return
        previous = r["points"]
    raise AssertionError("never muted")


def test_muted_sender_rejected_without_scoring():
    reset()
    points_before = mute_sender("dave")

    original = tracker.score_message

    def boom(text):
        raise AssertionError("model was called for a muted sender")

    tracker.score_message = boom
    try:
        r = process_score("dave", FRIENDLY)
    finally:
        tracker.score_message = original

    assert r["label"] == "muted"
    assert r["mute"] is True
    assert r["points"] == points_before


def test_senders_are_independent():
    reset()
    mute_sender("erin")
    r = process_score("frank", FRIENDLY)
    assert r["points"] == 0 and r["mute"] is False
    r = process_score("frank", WARN_TEXT)
    assert r["points"] == WARN_POINT


def test_reset_gives_clean_start():
    reset()
    mute_sender("gina")
    reset()
    r = process_score("gina", FRIENDLY)
    assert r["mute"] is False and r["points"] == 0


if __name__ == "__main__":
    check_texts()
    tests = [
        test_friendly_adds_nothing,
        test_points_follow_labels,
        test_mute_happens_exactly_at_limit,
        test_muted_sender_rejected_without_scoring,
        test_senders_are_independent,
        test_reset_gives_clean_start,
    ]
    for t in tests:
        try:
            t()
            print("PASS", t.__name__)
        except Exception as e:
            print("FAIL", t.__name__, "->", repr(e))