import os
import json

def chunk_text(text, chunk_size=10, overlap=2):
    """
    Splits the given text into chunks of size 'chunk_size' with a specified overlap.
    Returns a list of chunk strings.
    
    Args:
        text (str): The input text to chunk
        chunk_size (int): The size of each chunk in words
        overlap (int): Number of words to overlap between chunks
    """
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = words[i:i + chunk_size]
        if chunk:  # Ensure the chunk is not empty
            chunks.append(" ".join(chunk))
    
    return chunks

def load_and_chunk_dataset(file_path, chunk_size=30, overlap=5):
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
        doc_chunks = chunk_text(doc_text, chunk_size, overlap)
        
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
    
    # Test the overlapping chunks
    chunked_docs = load_and_chunk_dataset(dataset_file, chunk_size=10, overlap=2)
    print("Loaded and chunked", len(chunked_docs), "chunks from dataset.")
    
    # Print chunks to see the overlap in action
    for i, chunk in enumerate(chunked_docs):
        print(f"\nChunk {i}:")
        print(chunk["text"])