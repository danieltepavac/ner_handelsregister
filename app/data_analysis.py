import json 
from pathlib import Path
import re

from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

# Download NLTK data for punctuation and stopwords.
nltk.download("punkt")  
nltk.download("stopwords")

from sklearn.feature_extraction.text import CountVectorizer

from textblob import TextBlob

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# As well as for German. 
german_stop_words = stopwords.words('german')

vect = CountVectorizer(stop_words = german_stop_words) # Now use this in your pipeline

path = Path(Path(__file__).parent, "../data/1000_sample.json")

def open_json(path: str) -> json:
    with open(path, "r") as f:
        data = json.load(f)
    return data

DATA = open_json(path)

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

preprocessed_data = {filename: preprocess_text(text) for filename, text in DATA.items()}


def word_analysis(data: json) -> dict:

    results = {}

    total_words = sum(len(value.split()) for _, value in data.items())
    unique_words = len(set(word for _, value in data.items() for word in value.split()))
    average_word_length = sum(len(word) for _, value in data.items() for word in value.split()) / total_words

    results["total_words"] = total_words
    results["unique_words"] = unique_words
    results["avergae_word_length"] = average_word_length

    
    return results

result_data = word_analysis(DATA)
result_preprocessed_data = word_analysis(preprocessed_data)


def word_frequency(data) -> int: 

    word_counts = Counter(word for _, value in data.items() for word in value.split())

    most_common_words = word_counts.most_common(10)

    return most_common_words

word_frequency_text = word_frequency(DATA)
word_frequency_preprocessed_text = word_frequency(preprocessed_data)

print(word_frequency_text)
print(word_frequency_preprocessed_text)


def sentiment_analysis(data): 

    sentiments = [TextBlob(value).sentiment.polarity for _, value in data.items()]

    average_sentiment = sum(sentiments) / len(sentiments)

    return average_sentiment

sentiments = sentiment_analysis(DATA)

def generate_wordcloud(data):
    # Generate word cloud

    combined_text = ' '.join(data.values())

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)
    
    # Display the generated word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

print(generate_wordcloud(preprocessed_data))

def word_occurrence_patterns(data, num_collocations=10):

    combined_text = ' '.join(data.values())

    # Tokenize text into words
    words = re.findall(r'\b\w+\b', combined_text.lower())
    
    # Create bigrams from the list of words
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(words)
    
    # Filter out collocations that are stopwords or punctuation
    finder.apply_word_filter(lambda w: len(w) < 3 or w.lower() in nltk.corpus.stopwords.words('english'))
    
    # Get the top n collocations by PMI
    collocations = finder.nbest(bigram_measures.pmi, num_collocations)
    
    return collocations

collocations = word_occurrence_patterns(DATA)




