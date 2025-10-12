# Note that this method is a simplified approach and may not handle punctuation or 
# sentence boundaries effectively, causing issues with context preservation; 
# for more advanced chunking, consider using NLP libraries like NLTK or spaCy 
# that can respect sentence boundaries. Consider this a basic starting point.
def chunk_text(text, chunk_size=10):
    """
    Splits the given text into smaller chunks, each containing
    up to 'chunk_size' words. Returns a list of these chunk strings.
    """
    words = text.split()  # Tokenize by splitting on whitespace
    # Construct chunks by stepping through the words list in increments of chunk_size
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


def load_and_chunk_dataset(data, chunk_size=10):
    """
    Iterates over a structured dataset of documents, splits each into chunks,
    and associates metadata (doc_id and chunk_id) with every piece.
    """
    all_chunks = []
    for doc in data:
        doc_id = doc["id"] 
        doc_text = doc["content"]

        # Create smaller text segments from the original document
        doc_chunks = chunk_text(doc_text, chunk_size)

        # Label each chunk with its source identifier
        for chunk_id, chunk_str in enumerate(doc_chunks):
            all_chunks.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "text": chunk_str
            })

    return all_chunks


def main():
    # Example usage
    sample_data = [
        {"id": "doc1", "content": "This is the first document. It has several sentences. Each sentence will help form chunks."},
        {"id": "doc2", "content": "Here is another document! It also contains multiple sentences? Yes, indeed it does."}
    ]
    chunked_data = load_and_chunk_dataset(sample_data, chunk_size=10)
    for chunk in chunked_data:
        print(chunk)