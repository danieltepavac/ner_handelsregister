import json
import matplotlib.pyplot as plt
import nltk
import re
from collections import Counter
from pathlib import Path


from nltk.corpus import stopwords

from wordcloud import WordCloud


# Import German stop words. 
german_stop_words = stopwords.words("german")

path = Path(Path(__file__).parent, "../data/1000_sample.json")



def open_json(path: str) -> json:
    """Open and read a json file.

    Args:
        path (str): Path to json file.

    Returns:
        json: Read in json file. 
    """ 
    with open(path, "r") as f:
        data = json.load(f)
    return data


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


def word_analysis(data: json) -> dict:
    """ Analyse the corpus based on word count. 

    Args:
        data (json): Data to be processed. 

    Returns:
        dict: Dictionary containing total word count, unique word count and average word length. 
    """    

    # Initialize empyt dictionary. 
    results = {}

    # Count total words, unique words and average word length. 
    total_words = sum(len(value.split()) for _, value in data.items())
    unique_words = len(set(word for _, value in data.items() for word in value.split()))
    average_word_length = sum(len(word) for _, value in data.items() for word in value.split()) / total_words

    # Save the results in the dictionary. 
    results["total_words"] = total_words
    results["unique_words"] = unique_words
    results["avergae_word_length"] = average_word_length

    # Return results dictionary. 
    return results


def word_frequency(data: json) -> list[int]: 
    """Count the occurence of each individual word.

    Args:
        data (json): Data to be processed. 

    Returns:
        list[int]: List containing the most common word counts. 
    """ 
    # Count the occurence of each indivivual words.     
    word_counts = Counter(word for _, value in data.items() for word in value.split())

    # Save most common words. 
    most_common_words = word_counts.most_common(10)

    # Return most common words. 
    return most_common_words


def generate_wordcloud(data: json) -> WordCloud:
    """Generate a wordcloud over preprocessed text. 

    Args:
        data (json): Data used to generate a wordcloud

    Returns:
        WordCloud: A Wordcloud.
    """    
    # Combine text data.
    combined_text = " ".join(data.values())

    # Generate word cloud with Arial font.
    wordcloud = WordCloud(width=800, height=400, background_color="white",
                          max_words=200, colormap="plasma",
                          max_font_size=100).generate(combined_text)

    # Display the generated word cloud. 
    plt.figure(figsize=(10, 5))
    # Show the plot and fill it when it is resized. 
    plt.imshow(wordcloud, interpolation="bilinear")
    # Disable the coordinate system. 
    plt.axis("off")
    plt.show()

    plt.savefig(Path(Path(__file__).parent, "../result/data_analysis/wordcloud.png"))



def main(): 

    DATA = open_json(path)
    preprocessed_data = {filename: preprocess_text(text) for filename, text in DATA.items()}

    result_data = word_analysis(DATA)
    result_preprocessed_data = word_analysis(preprocessed_data)

    word_frequencies = word_frequency(DATA)
    word_frequency_preprocessed = word_frequency(preprocessed_data)

    results_data = {
        "word_counts": result_data,
        "word_frequencies": word_frequencies
    }

    results_preprocessed = {
        "word_counts": result_preprocessed_data,
        "word_frequencies": word_frequency_preprocessed
    }
    
    data_path = Path(Path(__file__).parent, "../result/data_analysis/data_results.json")
    preprocessed_path = Path(Path(__file__).parent, "../result/data_analysis/preprocessed_data_results.json")

    with open(data_path, "w", encoding="utf-8") as f: 
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    with open(preprocessed_path, "w", encoding="utf-8") as f: 
        json.dump(results_preprocessed, f, indent=2, ensure_ascii=False)

    generate_wordcloud(preprocessed_data)


if __name__ == "__main__": 
    main()





