import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_persist")
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

try:
    collection = client.get_collection(name="rag_docs", embedding_function=embedding_function)
except chromadb.errors.NotFoundError:
    print("❌ Vector DB collection not found.")
    exit()

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def query_rag(question, top_k=3):
    embedding = embedder.encode([question])[0].tolist()
    results = collection.query(query_embeddings=[embedding], n_results=top_k)

    for i, doc in enumerate(results['documents'][0]):
        print(f"\n🔹 Match #{i+1}")
        print(f"📄 Source: {results['metadatas'][0][i].get('source', 'N/A')}")
        print(f"📖 Text:\n{doc}")
        print("-" * 50)

if __name__ == "__main__":
    while True:
        q = input("\n❓ Ask your question (or 'exit'): ")
        if q.lower() == 'exit':
            break
        query_rag(q)
