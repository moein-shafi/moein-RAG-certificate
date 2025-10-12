import os
import json
import re


def chunk_text(text, chunk_size=30):
    """
    Splits the given text into chunks of size 'chunk_size', preserving sentence boundaries.
    Returns a list of chunk strings.
    """
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def load_and_chunk_dataset(file_path, chunk_size=30):
    """
    Loads a dataset from JSON 'file_path', then splits each document into smaller chunks.
    Metadata such as 'doc_id' and 'category' is included with each chunk.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    all_chunks = []
    for doc_id, doc in enumerate(data):
        doc_text = doc["content"]
        doc_category = doc.get("category", "general")
        doc_chunks = chunk_text(doc_text, chunk_size)
        for chunk_id, chunk_str in enumerate(doc_chunks):
            all_chunks.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "category": doc_category,
                "text": chunk_str
            })
    return all_chunks

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    dataset_file = os.path.join(current_dir, "data", "corpus.json")
    chunked_docs = load_and_chunk_dataset(dataset_file, chunk_size=30)
    print("Loaded and chunked", len(chunked_docs), "chunks from dataset.")
    for c in chunked_docs:
        print(c)

