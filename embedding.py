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
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    sentences = [
        "The Eiffel Tower is one of the most famous landmarks in Paris.",
        "Quantum computing promises to revolutionize technology with its speed.",
        "The Amazon rainforest is home to a vast diversity of wildlife.",
        "Meditation can significantly reduce stress and improve mental health.",
        "The Great Wall of China stretches over 13,000 miles."
    ]

    sentence_embeddings = model.encode(sentences)

    query = "What are some notable landmarks around the world?"
    query_embedding = model.encode([query])[0]

    similarities = []
    for idx, emb in enumerate(sentence_embeddings):
        sim = cosine_similarity(query_embedding, emb)
        similarities.append((sentences[idx], sim))

    similarities.sort(key=lambda x: x[1], reverse=True)

    print(f"Query: {query}\n")
    print("Sentences ranked by similarity:")
    for sent, score in similarities:
        print(f"Score: {score:.4f} | Sentence: {sent}")


if __name__ == "__main__":
    main()