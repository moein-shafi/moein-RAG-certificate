import numpy as np
import string


def build_vocab(docs):
    # 1. Initialize an empty set for unique words
    unique_words = set()
    # 2. Iterate through each document and its words
    for doc in docs:
        for word in doc.split():
            # 3. Clean words by converting to lowercase and removing punctuation
            clean_word = word.lower().strip(string.punctuation)
            if clean_word:
                # 4. Add clean words to the set
                unique_words.add(clean_word)
    # 5. Return a dictionary mapping words to indices (use enumerate)
    return {word: idx for idx, word in enumerate(sorted(unique_words))}


def bow_vectorize(text, vocab):
    # 1. Create a zero vector with length equal to vocabulary size
    vector = np.zeros(len(vocab), dtype=int)
    # 2. Process each word in the text (lowercase and clean)
    for word in text.split():
        clean_word = word.lower().strip(string.punctuation)
        # 3. If word exists in vocabulary, increment its count in the vector
        if clean_word in vocab:
            vector[vocab[clean_word]] += 1
    # 4. Return the BOW vector
    return vector


if __name__ == "__main__":
    example_texts = [
        "RAG stands for retrieval augmented generation, and retrieval is a key component of RAG.",
        "Data is crucial for retrieval processes, and without data, retrieval systems cannot function effectively."
    ]

    vocab = build_vocab(example_texts)
    print("Vocabulary:", vocab)
    for text in example_texts:
        vector = bow_vectorize(text, vocab)
        print(f"Text: {text}\nBOW Vector: {vector}\n")
