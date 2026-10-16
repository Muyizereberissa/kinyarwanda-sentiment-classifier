import re


def preprocess_text(text: str) -> str:
    """Light normalization for tweet-style Kinyarwanda text."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"http\S+", "<URL>", text)
    text = re.sub(r"\s+", " ", text)
    return text
