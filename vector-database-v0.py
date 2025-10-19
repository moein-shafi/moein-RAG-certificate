from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions


"""
This module sets up a ChromaDB vector database collection using sentence embeddings.
Embedding Setup: We define a SentenceTransformerEmbeddingFunction to generate vectors 
for the text chunks. The model we're using, all-MiniLM-L6-v2, is a lightweight but powerful 
sentence transformer that maps sentences to a 384-dimensional dense vector space. 
It's popular for RAG applications because it balances efficiency (small size, fast inference) 
with strong semantic understanding capabilities.

Client Configuration: Client(Settings()) connects to ChromaDB with default settings. 
By default, ChromaDB creates an in-memory store for quick experimentation. You can customize 
its behavior (for example, specifying a file path for persistence or enabling other features) 
by passing additional parameters to Settings().

Collection Management: get_or_create_collection checks if a collection named "rag_collection" 
exists; if not, it creates a new one. A collection in ChromaDB is a logical container that 
groups related documents and their embeddings together, similar to a table in a traditional 
database but optimized for vector similarity operations. Collections allow you to organize 
your vector data into separate namespaces, making it possible to maintain multiple distinct 
sets of documents with different embedding models or for different use cases.
"""


def build_chroma_collection(chunks):
    # Use a Sentence Transformer model for embeddings
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    
    # Create a ChromaDB client with default settings
    client = Client(Settings())

    # Either get an existing collection or create a new one
    collection = client.get_or_create_collection(
        name="rag_collection",
        embedding_function=embed_func
    )

    # Prepare the data: texts, IDs, and metadata 
    texts = [c["content"] for c in chunks]
    ids = [f"chunk_{c['doc_id']}_{c['chunk_id']}" for c in chunks]
    metadatas = [
        {"doc_id": chunk["doc_id"], 
        "chunk_id": chunk["chunk_id"], 
        "category": chunk["category"]}
        for chunk in chunks
    ]

    # Add the documents (chunks) to the collection
    collection.add(documents=texts, metadatas=metadatas, ids=ids)
    return collection


def reset_collection(collection):
    # Delete all documents in the collection
    collection.delete()


def main():
    # Example chunks to be added to the collection
    example_chunks = [
        {"doc_id": 1, "chunk_id": 1, "content": "This is the first chunk of document one.", "category": "A"},
        {"doc_id": 1, "chunk_id": 2, "content": "This is the second chunk of document one.", "category": "A"},
        {"doc_id": 2, "chunk_id": 1, "content": "This is the first chunk of document two.", "category": "B"},
        {"doc_id": 2, "chunk_id": 2, "content": "This is the second chunk of document two.", "category": "B"},
    ]

    # Build the ChromaDB collection with the example chunks
    collection = build_chroma_collection(example_chunks)
    print("Collection built with ID:", collection.name)

    # Example chunks to showcase adding them to a new collection
    example_chunks = [
        {"doc_id": 0, "chunk_id": 0, "category": "ai", "content": "RAG stands for Retrieval-Augmented Generation."},
        {"doc_id": 0, "chunk_id": 1, "category": "ai", "content": "A crucial component of a RAG pipeline is the Vector Database."},
        {"doc_id": 1, "chunk_id": 0, "category": "finance", "content": "Accurate data is essential in finance."},
    ]
    collection = build_chroma_collection(example_chunks)

    # Prepare a new chunk to add
    new_document = {
        "doc_id": 2,
        "chunk_id": 0,
        "category": "food",
        "content": "Bananas are yellow fruits rich in potassium."
    }

    # Construct a unique ID for the new document
    # Format: "chunk_{doc_id}_{chunk_id}" (e.g., "chunk_2_0")
    doc_id = f"chunk_{new_document['doc_id']}_{new_document['chunk_id']}"

    # Add the new chunk to the existing collection
    collection.add(
        documents=[new_document["content"]],  # The text content to be embedded
        metadatas=[{                          # Metadata for filtering and context
            "doc_id": new_document["doc_id"],
            "chunk_id": new_document["chunk_id"],
            "category": new_document["category"]
        }],
        ids=[doc_id]                          # Unique identifier for this chunk
    )

    # If needed, remove the chunk by its unique ID
    # For example, if the information about bananas becomes outdated
    collection.delete(ids=[doc_id])  # Using the same ID: "chunk_2_0"


if __name__ == "__main__":
    main()