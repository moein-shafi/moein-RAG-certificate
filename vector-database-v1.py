import os
import json
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions


def chunk_text(text, chunk_size=50):
    """
    Splits the given text into chunks of size 'chunk_size'.
    Returns a list of chunk strings.
    """
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


def load_and_chunk_dataset(file_path, chunk_size=50):
    """
    Loads a dataset from JSON 'file_path', then splits each document into smaller chunks.
    Metadata such as 'doc_id' and 'category' is included with each chunk.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_chunks = []
    for doc in data:
        doc_text = doc["content"]
        category = doc.get("category", "unknown")
        doc_id = doc.get("id", "unknown")
        
        doc_chunks = chunk_text(doc_text, chunk_size)
        for chunk_index, chunk_str in enumerate(doc_chunks):
            chunk_dict = {
                "doc_id": doc_id,
                "chunk_id": chunk_index,
                "category": category,
                "text": chunk_str
            }
            all_chunks.append(chunk_dict)
            
            
    return all_chunks


def build_chroma_collection(chunks, collection_name="rag_collection"):
    """
    Builds or retrieves a ChromaDB collection, embedding each chunk using a SentenceTransformer.
    Adds all chunks in the 'chunks' list to the collection for fast retrieval.
    """
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

    client = Client(Settings())
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embed_func
    )

    texts = [c["text"] for c in chunks]
    ids = [f"chunk_{c['doc_id']}_{c['chunk_id']}" for c in chunks]
    metadatas = [
        {"doc_id": c["doc_id"], "chunk_id": c["chunk_id"], "category": c["category"]}
        for c in chunks
    ]

    collection.add(documents=texts, metadatas=metadatas, ids=ids)
    return collection


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    dataset_file = os.path.join(current_dir, "data", "corpus.json")

    chunked_docs = load_and_chunk_dataset(dataset_file)
    collection = build_chroma_collection(chunked_docs)

    total_docs = collection.count()
    print("ChromaDB collection created with", total_docs, "documents.")