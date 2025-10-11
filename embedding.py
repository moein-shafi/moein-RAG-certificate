import numpy as np

from sentence_transformers import SentenceTransformer
from numpy.linalg import norm

def cosine_similarity(vec_a, vec_b):
    """
    Compute cosine similarity between two vectors.
    Range: -1 (opposite directions) to 1 (same direction).
    """
    return np.dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b))


def main():
    # Initialize a pre-trained embedding model from Sentence Transformers.
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    sentences = [
        "RAG stands for Retrieval Augmented Generation.",
        "A Large Language Model is a Generative AI model for text generation.",
        "RAG enhance text generation of LLMs by incorporating external data",
        "Bananas are yellow fruits.",
        "Apples are good for your health.",
        "What's monkey's favorite food?"
    ]

    embeddings = model.encode(sentences)
    print(embeddings.shape)  # e.g., (6, 384), depending on the model
    # print(embeddings[0])     # A sample embedding for the first sentence

    for i, sent_i in enumerate(sentences):
        for j, sent_j in enumerate(sentences[i+1:], start=i+1):
            sim_score = cosine_similarity(embeddings[i], embeddings[j])
            print(f"Similarity('{sent_i}' , '{sent_j}') = {sim_score:.4f}")

if __name__ == "__main__":
    main()