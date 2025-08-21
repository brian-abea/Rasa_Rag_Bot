from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

def test_vector_search(query, top_k=3):
    client = PersistentClient(path="./chroma_persist")
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    try:
        collection = client.get_collection("rag_docs", embedding_function=embedding_function)
    except Exception:
        print("❌ Collection 'rag_docs' not found.")
        return

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = embedder.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[embedding], n_results=top_k)

    for i, doc in enumerate(results['documents'][0]):
        print(f"\n🔹 Match #{i+1}")
        print(f"📄 Source: {results['metadatas'][0][i].get('source', 'Unknown')}")
        print(f"📖 Text:\n{doc}")
        print("-" * 60)

if __name__ == "__main__":
    query = input("🧪 Test a query: ")
    test_vector_search(query)
