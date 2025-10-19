import json
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from datetime import datetime


def metadata_enhanced_search(query, collection, categories=None, min_date=None, top_k=3):
    """
    Filters documents by category and a minimum publication date.
    If a list of categories is provided, only documents in those categories are returned.
    If a min_date is provided, only documents with date >= min_date are returned.
    Both filters are combined such that documents must satisfy all provided conditions.
    """

    if min_date:
        min_date = int(datetime.strptime(min_date, "%Y-%m-%d").timestamp())

    where_clause = None
    if categories and min_date:
        where_clause = {
            "category": {"$in": categories},
            "date": {"$gte": min_date}
        }
    elif categories:
        where_clause = {"category": {"$in": categories}}
    elif min_date:
        where_clause = {"date": {"$gte": min_date}}

    # Execute the query using the ChromaDB collection
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where_clause
    )

    # Compile the retrieved documents into a list.
    retrieved_chunks = []
    for i in range(len(results['documents'][0])):
        retrieved_chunks.append({
            "chunk": results['documents'][0][i],
            "doc_id": results['metadatas'][0][i]['doc_id'],
            "category": results['metadatas'][0][i].get('category', "General"),
            "distance": results['distances'][0][i],
            "date": datetime.fromtimestamp(results['metadatas'][0][i].get('date', 0)).isoformat()
        })

    return retrieved_chunks


if __name__ == "__main__":
    # Load sample data from JSON file
    with open("data/corpus.json", "r") as f:
        sample_chunks = json.load(f)

    # Create a ChromaDB client and embedder
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    client = Client(Settings())

    # Create or retrieve the collection
    collection = client.create_collection(
        "metadata_demo_collection",
        embedding_function=embed_func
    )

    # Clear any existing documents in the collection
    existing_ids = collection.get().get("ids", [])
    if existing_ids:
        collection.delete(ids=existing_ids)

    # Prepare data and add to the collection
    texts = [doc["content"] for doc in sample_chunks]
    doc_ids = [f"doc_{doc['id']}" for doc in sample_chunks]
    metadatas = []
    for doc in sample_chunks:
        # TODO: Convert date to timestamp and store it
        metadatas.append({
            "doc_id": doc["id"],
            "category": doc.get("category", "General"),
            "title": doc["title"],
            "date": doc["date"]  # Store date as a timestamp
        })

    collection.add(documents=texts, metadatas=metadatas, ids=doc_ids)

    # Demonstrate searches
    query_input = "Recent advancements in AI and their impact on teaching"

    print("======== WITHOUT ANY FILTER ========")
    no_filter_results = metadata_enhanced_search(query_input, collection, categories=None, min_date=None, top_k=3)
    for res in no_filter_results:
        print(f"Doc ID: {res['doc_id']} | Category: {res['category']} | Date: {res['date']} | Distance: {res['distance']:.4f}")
        print(f"Chunk: {res['chunk']}\n")

    print("======== WITH CATEGORY FILTER (Education) ONLY ========")
    cat_only_results = metadata_enhanced_search(query_input, collection, categories=["Education"], min_date=None, top_k=3)
    for res in cat_only_results:
        print(f"Doc ID: {res['doc_id']} | Category: {res['category']} | Date: {res['date']} | Distance: {res['distance']:.4f}")
        print(f"Chunk: {res['chunk']}\n")

    print("======== WITH CATEGORY FILTER (Education) AND DATE FILTER (>= 2022-01-01) ========")
    cat_and_date_results = metadata_enhanced_search(query_input, collection, categories=["Education"], min_date="2022-01-01", top_k=3)
    for res in cat_and_date_results:
        print(f"Doc ID: {res['doc_id']} | Category: {res['category']} | Date: {res['date']} | Distance: {res['distance']:.4f}")
        print(f"Chunk: {res['chunk']}\n")