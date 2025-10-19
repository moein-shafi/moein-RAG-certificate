import json
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions


def metadata_enhanced_search(query, collection, categories=None, top_k=3):
    """
    Demonstrates how to filter results by metadata. If 'categories' is specified,
    we only retrieve documents whose category is in the provided list.
    If the initial search yields no results, automatically rerun the
    query without any category filter as a fallback.
    """
    # Build the initial where clause for category filtering
    where_clause = {"category": {"$in": categories}} if categories else None

    # Perform the initial search with category filter (if provided)
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where_clause
    )

    if not results['documents'] or not results['documents'][0]:
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=None
        )

    # Build a result list
    retrieved_chunks = []
    for i in range(len(results['documents'][0])):
        retrieved_chunks.append({
            "chunk": results['documents'][0][i],
            "doc_id": results['metadatas'][0][i]['doc_id'],
            "category": results['metadatas'][0][i].get('category', "General"),
            "distance": results['distances'][0][i]
        })
    return retrieved_chunks


if __name__ == "__main__":
    # Load sample data from JSON file
    with open('data/corpus.json', 'r') as f:
        sample_chunks = json.load(f)

    # Build a ChromaDB collection
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    client = Client(Settings())
    collection = client.create_collection("metadata_demo_collection", embedding_function=embed_func)

    # Clear any existing data and add fresh documents
    existing_ids = collection.get().get('ids', [])
    if existing_ids:
        collection.delete(ids=existing_ids)

    texts = [c["content"] for c in sample_chunks]
    ids = [f"doc_{c['id']}" for c in sample_chunks]
    metadatas = [{
        "doc_id": c["id"],
        "category": c["category"],
        "title": c["title"],
        "date": c["date"]
    } for c in sample_chunks]

    collection.add(documents=texts, metadatas=metadatas, ids=ids)

    # Single query to compare results with and without metadata filtering
    query_input = "Recent advancements in AI and their impact on teaching"

    # Search WITHOUT category filtering
    print("======== WITHOUT CATEGORY FILTER ========")
    no_filter_results = metadata_enhanced_search(query_input, collection, categories=None, top_k=3)
    for res in no_filter_results:
        print(f"Doc ID: {res['doc_id']}, Category: {res['category']}, Distance: {res['distance']:.4f}")
        print(f"Chunk: {res['chunk']}\n")

    # Search WITH a strict category filter, potentially yielding zero results
    very_strict_category = "NonExistentCategory"
    print(f"======== WITH VERY STRICT CATEGORY FILTER ({very_strict_category}) ========")
    strict_filter_results = metadata_enhanced_search(query_input, collection, categories=[very_strict_category], top_k=3)
    for res in strict_filter_results:
        print(f"Doc ID: {res['doc_id']}, Category: {res['category']}, Distance: {res['distance']:.4f}")
        print(f"Chunk: {res['chunk']}\n")