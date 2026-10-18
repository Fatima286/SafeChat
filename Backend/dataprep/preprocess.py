import re
import string
import emoji
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import pandas as pd


# Negation/intensifier words we deliberately KEEP even though they're
# standard NLTK stopwords -- removing these flips meaning for toxicity
# detection ("not funny" vs "funny").
NEGATION_WORDS = {
    "no", "not", "nor", "never", "none", "nothing", "nobody", "nowhere",
    "neither", "don't", "dont", "didn't", "didnt", "doesn't", "doesnt",
    "isn't", "isnt", "wasn't", "wasnt", "aren't", "arent", "weren't", "werent",
    "won't", "wont", "can't", "cant", "cannot", "couldn't", "couldnt",
    "shouldn't", "shouldnt", "wouldn't", "wouldnt", "very", "too", "so"
}

URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
MENTION_PATTERN = re.compile(r'@\w+')
ELONGATION_PATTERN = re.compile(r'(.)\1{2,}')  # 3+ repeated chars, e.g. "soooo"


class Preprocess:
    def __init__(self, use_stemming=True, remove_stopwords=True):
        self.stemmer = SnowballStemmer('english')
        self.stopwords = set(stopwords.words('english')) - NEGATION_WORDS
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords

    def extract_style_features(self, text):
        """Signals that get destroyed by lowercasing/punctuation stripping,
        so pull them out first as separate numeric features."""
        return {
            "num_exclamations": text.count('!'),
            "num_questions": text.count('?'),
            "is_all_caps": text.isupper() and len(text) > 3,
            "has_url": bool(URL_PATTERN.search(text)),
            "has_mention": bool(MENTION_PATTERN.search(text)),
        }

    def clean_text(self, text):
        text = URL_PATTERN.sub(' <URL> ', text)
        text = MENTION_PATTERN.sub(' <MENTION> ', text)
        text = emoji.demojize(text, delimiters=(" <emoji_", "> "))
        text = ELONGATION_PATTERN.sub(r'\1\1', text)  # "soooo" -> "soo"
        text = text.lower()
        return text

    def tokenize(self, text):
        tokens = word_tokenize(text)
        # keep placeholder tokens (url, mention, emoji_...) but drop
        # bare punctuation tokens (this also strips the < > around
        # placeholders since word_tokenize splits them off separately)
        tokens = [t for t in tokens if t not in string.punctuation]
        return tokens

    @staticmethod
    def _is_placeholder(token):
        return token in ("url", "mention") or token.startswith("emoji_")

    def remove_stops(self, tokens):
        if not self.remove_stopwords:
            return tokens
        return [t for t in tokens if self._is_placeholder(t) or t not in self.stopwords]

    def stem(self, tokens):
        if not self.use_stemming:
            return tokens
        # don't stem placeholder tokens -- stemming mangles them
        # (e.g. "emoji_enraged_face" -> "emoji_enraged_fac")
        return [t if self._is_placeholder(t) else self.stemmer.stem(t) for t in tokens]

    def preprocess_pipeline(self, text):
        """Single entry point: raw string in, cleaned token list out.
        Use with df['tokens'] = df['text'].apply(obj.preprocess_pipeline)"""
        if not isinstance(text, str) or not text.strip():
            return []
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens = self.remove_stops(tokens)
        tokens = self.stem(tokens)
        return tokens


def Main(split_df):



    obj=Preprocess()
    df = split_df
    df['num_exclamations'] = df['text'].str.count('!')
    df['num_questions']    = df['text'].str.count(r'\?')
    df['is_all_caps']      = df['text'].str.isupper() & (df['text'].str.len() > 3)
    df['has_url']          = df['text'].str.contains(URL_PATTERN)
    df['has_mention']      = df['text'].str.contains(MENTION_PATTERN)
    df['tokens'] = df['text'].apply(obj.preprocess_pipeline) # adds a column in the data by heading tokens
    df['processed_text'] = df['tokens'].apply(lambda toks: ' '.join(toks)) #processed_text column (string form, ready for vectorization).

    return df

    #print(df['processed_text'].head(10))
    #print(df['insult'].head(10))
    #print(df['tokens'].head(10))
    #df.to_csv("processed_data.csv", index=False)

