from scoring import score_message, decision
from config import WARN_POINT, BLOCK_POINT, MUTE_LIMIT

# One entry per sender: {'sender_id': [points, muted]}
# e.g. {'alice': [3, False], 'bob': [5, True]}
# Think of it as a register with one line per sender.
sender_state = {}


def process_score(sender_id, text):
    # sender_id is only the NAME of the sender (like "alice").
    # An empty name would make all anonymous users share one record, so reject it.
    if not sender_id:
        raise ValueError("sender_id is required")

    # Find this sender's line in the register.
    # If it is a new sender, create their line: 0 points, not muted.
    if sender_id not in sender_state:
        sender_state[sender_id] = [0, False]

    # `record` is that sender's list: record[0] = points, record[1] = muted.
    # It is the very same list stored in the dictionary, so changing
    # record[0] or record[1] below automatically updates sender_state.
    record = sender_state[sender_id]

    # If already muted: don't run the model at all, and return the SAME
    # kind of result (a dictionary) as a normal message, so the API
    # never has to handle two different shapes.
    if record[1]:
        return {'label': 'muted', 'score': None,
                'points': record[0], 'mute': True}

    # Normal path: score the message and turn the score into a label.
    score = score_message(text)
    label = decision(score)

    # Add points depending on the label ("allow" adds nothing).
    if label == 'warn':
        record[0] += WARN_POINT
    elif label == 'block':
        record[0] += BLOCK_POINT

    # Mute once the points reach the limit. This message is still returned
    # with its own label; the mute applies from the sender's NEXT message.
    if record[0] >= MUTE_LIMIT:
        record[1] = True

    # The score is returned so tests, the frontend and logs can use it.
    return {'label': label, 'score': score,
            'points': record[0], 'mute': record[1]}