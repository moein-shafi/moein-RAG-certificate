import numpy as np
from numpy.linalg import norm


VOCAB = {
    "rag": 0,
    "stands": 1,
    "for": 2,
    "retrieval": 3,
    "augmented": 4,
    "generation": 5,
    "a": 6,
    "large": 7,
    "language": 8,
    "model": 9,
    "is": 10,
    "generative": 11,
    "ai": 12,
    "text": 13,
    "enhance": 14,
    "of": 15,
    "llms": 16,
    "by": 17,
    "incorporating": 18,
    "external": 19,
    "data": 20,
    "bananas": 21,
    "are": 22,
    "yellow": 23,
    "fruits": 24,
    "apples": 25,
    "good": 26,
    "your": 27,
    "health": 28,
    "what's": 29,
    "monkey's": 30,
    "favorite": 31,
    "food": 32
}


def bow_vectorize(text, vocab=VOCAB):
    """
    Convert a text into a Bag-of-Words vector, using a shared vocabulary.
    Each element counts how many times a particular token or bigram appears.
    """
    vector = np.zeros(len(vocab), dtype=int)
    words = [word.strip(".,!?") for word in text.lower().split()]
    for i in range(len(words)):
        if words[i] in vocab:
            vector[vocab[words[i]]] += 1
        # TODO: Count bigrams in the vector
        if i < len(words) - 1:
            bigram = f"{words[i]} {words[i + 1]}"
            if bigram in vocab:
                vector[vocab[bigram]] += 1
    return vector



# def bow_vectorize(text, vocab):
#     """
#     Convert a text into a Bag-of-Words vector by counting how many times 
#     each token from our vocabulary appears in the text.
#     """
#     vector = np.zeros(len(vocab), dtype=int)
#     for word in text.lower().split():
#         # Remove punctuation for consistency
#         clean_word = word.strip(".,!?")
#         if clean_word in vocab:
#             vector[vocab[clean_word]] += 1
#         # TODO: Count bigrams in the vector
#         if ' ' in clean_word:
#             bigrams = clean_word.split(' ')
#             for i in range(len(bigrams) - 1):
#                 bigram = f"{bigrams[i]} {bigrams[i + 1]}"
#                 if bigram in vocab:
#                     vector[vocab[bigram]] += 1

#     return vector

def bow_search(query, docs):
    """
    Rank documents by lexical overlap using the BOW technique. 
    The dot product between the query vector and each document vector 
    indicates how many words they share.
    """
    query_vec = bow_vectorize(query, VOCAB)
    scores = []
    for i, doc in enumerate(docs):
        doc_vec = bow_vectorize(doc, VOCAB)
        score = np.dot(query_vec, doc_vec)  # Higher score = more overlap
        scores.append((i, score))
    # Sort by descending overlap
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def cos_sim(a, b):
    """
    Compute cosine similarity between two vectors, 
    indicating how similar they are.
    """
    return np.dot(a, b) / (norm(a) * norm(b))

def embedding_search(query, docs, model):
    """
    Rank documents by comparing how semantically close they are 
    to the query in the embedding space using cosine similarity.
    """
    # Encode both the query and documents into embeddings
    query_emb = model.encode([query])[0]
    doc_embs = model.encode(docs)

    scores = []
    for i, emb in enumerate(doc_embs):
        score = cos_sim(query_emb, emb)
        scores.append((i, score))
    # Sort by semantic similarity in descending order
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def build_vocab(docs):
    """
    Dynamically build a vocabulary from the given docs.
    Each new word or bigram in the corpus is an entry in the vocabulary.
    """
    unique_tokens = set()
    for doc in docs:
        words = [word.strip(".,!?") for word in doc.lower().split()]
        for i in range(len(words)):
            if words[i]:
                unique_tokens.add(words[i])

            # TODO: Add bigrams to the vocabulary
            if i < len(words) - 1 and words[i] and words[i+1]:
                bigram = f"{words[i]} {words[i+1]}"
                unique_tokens.add(bigram)
    return {token: idx for idx, token in enumerate(sorted(unique_tokens))}


def main():
    from sentence_transformers import SentenceTransformer

    # Sample documents
    documents = [
        "RAG stands for Retrieval Augmented Generation.",
        "A Large Language Model is a Generative AI model for text generation.",
        "RAG enhance text generation of LLMs by incorporating external data",
        "Bananas are yellow fruits.",
        "Apples are good for your health.",
        "What's monkey's favorite food?"
    ]

    # Sample query
    query = "What is RAG in AI?"

    # Perform Bag-of-Words search
    bow_results = bow_search(query, documents)
    print("Bag-of-Words Search Results:")
    for idx, score in bow_results:
        print(f"Doc {idx} (Score: {score}): {documents[idx]}")

    # Initialize embedding model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Perform Embedding-based search
    emb_results = embedding_search(query, documents, model)
    print("\nEmbedding-based Search Results:")
    for idx, score in emb_results:
        print(f"Doc {idx} (Score: {score:.4f}): {documents[idx]}")


if __name__ == "__main__":
    main()