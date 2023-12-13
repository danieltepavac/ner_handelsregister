# Import necessary modules.
import nltk
from nltk.corpus import stopwords

# Download NLTK data for punctuation and stopwords.
nltk.download("punkt")  
nltk.download("stopwords")

# As well as for German. 
nltk.download("stopwords-de")

def preprocess_text(text: str) -> str:
    """Preprocesses text for clustering.

    Args:
        text (str): Text to preprocess.

    Returns:
        str: Preprocessed text.
    """

    # Tokenize input text.
    tokens = nltk.word_tokenize(text.lower(), language="german")

    # Remove punctuation and numbers.
    tokens = [word for word in tokens if word.isalpha()]

    # Remove stopwords.
    stop_words = set(stopwords.words("german"))
    tokens = [word for word in tokens if word not in stop_words]

    # Join the tokens back into a single string.
    preprocessed_text = " ".join(tokens)

    # Return preprocessed text.
    return preprocessed_text

