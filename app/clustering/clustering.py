# Import own scripts.
from utils.read_ndjson import read_ndjson
from preprocessing import preprocess_text

# Import tqdm for progress bar.
from tqdm import tqdm

# Import json.
import json

# Import gensim for Doc2Vec, sklearn for KMeans and silhouette score.
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Import matplotlib and PCA from sklearn for visualization.
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Save file_path and read it in. 
file_path = "1000_sample.ndjson"
sample = read_ndjson(file_path)

# Preprocess the text of the sample and save it in a new dictionary with key: filename and text as value.
preprocessed_sample = {filename: preprocess_text(text) for filename, text in sample.items()}

def doc2vec(preprocessed_dict: dict) -> list: 
    """Apply doc2vec to preprocessed sample dict. 

    Args:
        preprocessed_dict (dict): Preprocessed sample dict.

    Returns:
        list: List of vectors of each document.
    """
    # Create a list of tagged documents.
    documents = [TaggedDocument(words, [filename]) for filename, words in preprocessed_dict.items()]

    # Create doc2vec model. Best parameters found: vector_size: 100, window: 5, min_count: 5.
    doc2vec_model = Doc2Vec(documents, vector_size=100, window=5, min_count=5, workers=4)
    
    # Train model. 
    with tqdm(total=50, desc="Training Doc2Vec") as pbar:
        for epoch in range(50):
            doc2vec_model.build_vocab(documents, progress_per=1)
            doc2vec_model.train(documents, total_examples=doc2vec_model.corpus_count, epochs=1)
            pbar.update(1)
    
    # Create a list of document vectors.
    document_vectors = [doc2vec_model.dv[filename] for filename in sample.keys()]

    # Return document vectors.
    return document_vectors

def create_cluster_labels(document_vectors: list, num_clusters: int) -> list: 
    """ Train a KMeans model and return cluster labels. 

    Args:
        document_vectors (list): List of document vectors.
        num_clusters (int): Number of wanted clusters.

    Returns:
        list: List of cluster_labels for each document
    """

    # Build KMeans model with number of clusters and random state as arguments.
    kmeans = KMeans(n_clusters=num_clusters, random_state= 42)
    # Train Kmeans model.
    kmeans.fit(document_vectors)
    
    # Save cluster labels from model.
    cluster_labels = kmeans.labels_

    # Return cluster_labels.
    return cluster_labels
 
def create_file_clusters(cluster_labels: list, preprocessed_sample: dict) -> dict:
    """ Create a dictionary with cluster_label as key and filenames as values.

    Args:
        cluster_labels (list): List of cluster labels.
        preprocessed_sample (dict): Preprocessed sample dict for retrieving filenames.

    Returns:
        dict: Dictionary with cluster label as key and filenames as values.
    """

    # Create empty dictionary. 
    file_clusters = {}

    # Iterate over keys of preprocessed dict and cluster labels. 
    for filename, cluster_label in zip(preprocessed_sample.keys(), cluster_labels):
        # Create instance of one cluster label in dictionary as well as empty list.
        if cluster_label not in file_clusters:
            file_clusters[cluster_label] = []

        # Append filename to according cluster.
        file_clusters[cluster_label].append(filename)
    
    # Return file_clusters. 
    return file_clusters

def create_text_clusters(cluster_labels: list, preprocessed_sample: dict) -> dict:
    """ Create a dictionary with cluster_label as key and text as values.

    Args:
        cluster_labels (list): List of cluster labels.
        preprocessed_sample (dict): Preprocessed sample dict for retrieving text.

    Returns:
        dict: Dictionary with cluster label as key and text as values.
    """

    # Create empty dictionary.
    text_clusters = {}

    # Iterate over values of preprocessed dict and cluster labels. 
    for text, cluster_label in zip(preprocessed_sample.values(), cluster_labels): 
        # Create instance of one cluster label in dictionary as well as empty list.
        if cluster_label not in text_clusters: 
            text_clusters[cluster_label] = []
        
        # Append filename to according cluster.
        text_clusters[cluster_label].append(text)

    # Return text_clusters.
    return text_clusters

def create_clustered_dictionary(cluster_labels: list, preprocessed_sample: dict) -> None:
    """Create two dictionaries for each cluster.

    Args:
        cluster_labels (list): List of cluster labels. 
        preprocessed_sample (dict): Dictionary with filename as key and the value is preprocessed text.
    """

    # Create two empty dicts.
    clustered_dict_0 = {}
    clustered_dict_1 = {}

    # Iterate over preprocessed_sample and cluster_labels.
    for (filename, text), cluster_label in zip(preprocessed_sample.items(), cluster_labels): 
        # If the cluster_label is 0, add this instance to dict 0.
        if cluster_label == 0: 
            clustered_dict_0[filename] = text
        # If the cluster label is 1, add this instance to dict 1. 
        elif cluster_label == 1: 
            clustered_dict_1[filename] = text
    
    # Save dict 0 in a JSON-file. 
    with open("clustered_dict_0.json", "w", encoding="utf-8") as file: 
        json.dump(clustered_dict_0, file, indent=2, ensure_ascii=False)
    
    # Save dict 1 in a JSON-file. 
    with open("clustered_dict_1.json", "w", encoding="utf-8") as file: 
        json.dump(clustered_dict_1, file, indent=2, ensure_ascii=False)

def compute_silhoutte_score(document_vectors: list, cluster_labels: list) -> int:
    """ Compute silhouette score.

    Args:
        document_vectors (list): List of document vectors.
        cluster_labels (list): List of cluster labels. 

    Returns:
        int: Silhouette Score. 
    """

    # Compute average of silhouette score. 
    silhouette_avg = silhouette_score(document_vectors, cluster_labels)

    # Return average silhouette score. 
    return silhouette_avg

def visualize_clusters(document_vectors: list, cluster_labels: list, num_clusters: int, save_path: str) -> None: 
    """Visualize clusters in a scatter plot. 

    Args:
        document_vectors (list): List of document vectors.
        cluster_labels (list): List of cluster labels.
        num_clusters (int): Number of wanted clusters.
        save_path (str): Path where plot should be saved at.
    """

    # Create PCA to educe dimensionality into 2-dimensional space. 
    pca = PCA(n_components=2)
    # Apply PCA on document vectors. 
    reduced_vectors = pca.fit_transform(document_vectors)

    # Colors for clusters.
    colors = ["r", "g"]
    # Iterate over number of clusters.
    for cluster_id in range(num_clusters):
        # Bolean mask to check if cluster label belongs to current ID (True) or not (False).
        cluster_mask = cluster_labels == cluster_id
        # Create scatter plot. Components are x and y axis respectively. Label is the point in the scatter plot and the c shows the color of the label. 
        plt.scatter(reduced_vectors[cluster_mask, 0], reduced_vectors[cluster_mask, 1], label=f"Cluster {cluster_id}", c=colors[cluster_id])

    #Title and labels of plot. Legend will be shown as well. 
    plt.title("Clusters of Handelsregisterdaten")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()

    # Save figure in save path. 
    plt.savefig(save_path, dpi=300, bbox_inches="tight")


def main(): 
    document_vectors = doc2vec(preprocessed_sample)
    cluster_labels = create_cluster_labels(document_vectors, num_clusters=2)

    silhouette_score = compute_silhoutte_score(document_vectors, cluster_labels)

    visualize_clusters(document_vectors, cluster_labels, num_clusters=2, save_path="cluster_plot.png")

    create_clustered_dictionary(cluster_labels, preprocessed_sample)

    file_clusters = create_file_clusters(cluster_labels, preprocessed_sample)
    return silhouette_score, file_clusters

if __name__ == "__main__": 
    score, file_clusters = main()
    print(f"Silhouette Score: {score}")
    print(len(file_clusters[0]), len(file_clusters[1]))
