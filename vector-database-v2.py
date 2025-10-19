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
    with open(file_path, "r") as f:
        data = json.load(f)

    all_chunks = []
    for doc_id, doc in enumerate(data):
        doc_text = doc["content"]
        doc_category = doc.get("category", "general")  # Default to "general" if no category.
        doc_chunks = chunk_text(doc_text, chunk_size)

        for chunk_id, chunk_str in enumerate(doc_chunks):
            all_chunks.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "category": doc_category,
                "text": chunk_str
            })
    return all_chunks


def build_chroma_collection(chunks, collection_name="rag_collection"):
    """
    Builds or retrieves a ChromaDB collection, embedding each chunk using a SentenceTransformer.
    Adds all chunks in the 'chunks' list to the collection for fast retrieval.
    """
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    client = Client(Settings())
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embed_func)

    texts = [c["text"] for c in chunks]
    ids = [f"chunk_{c['doc_id']}_{c['chunk_id']}" for c in chunks]
    metadatas = [
        {"doc_id": c["doc_id"], "chunk_id": c["chunk_id"], "category": c["category"]}
        for c in chunks
    ]

    collection.add(documents=texts, metadatas=metadatas, ids=ids)
    return collection


def delete_documents_with_keyword(collection, keyword):
    """
    Deletes all documents from the given ChromaDB 'collection' whose text contains 'keyword'.
    """
    all_docs = collection.get()
    texts = all_docs["documents"]
    ids = all_docs["ids"]
    ids_to_delete = [ids[i] for i, text in enumerate(texts) if keyword in text]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    dataset_file = os.path.join(current_dir, "data", "corpus.json")

    # Load and chunk the dataset, then build the initial collection.
    chunked_docs = load_and_chunk_dataset(dataset_file)
    collection = build_chroma_collection(chunked_docs)
    initial_count = collection.count()
    print("ChromaDB collection created with", initial_count, "documents.")

    # Add a new document containing "Bananas".
    new_document = {
        "doc_id": initial_count + 1,
        "chunk_id": 0,
        "category": "food",
        "text": "Bananas are yellow fruits rich in potassium."
    }
    doc_id_str = f"chunk_{new_document['doc_id']}_{new_document['chunk_id']}"

    collection.add(
        documents=[new_document["text"]],
        metadatas=[{
            "doc_id": new_document["doc_id"],
            "chunk_id": new_document["chunk_id"],
            "category": new_document["category"]
        }],
        ids=[doc_id_str]
    )

    updated_count = collection.count()
    print("After adding keyword document, collection has", updated_count, "documents.")

    # Now delete all documents containing the keyword "Bananas".
    delete_documents_with_keyword(collection, "Bananas")

    final_count = collection.count()
    print("After deleting documents with 'Bananas', collection has", final_count, "documents.")