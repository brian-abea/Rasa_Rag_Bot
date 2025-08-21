import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="./chroma_persist")

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

try:
    collection = client.get_collection(name="rag_docs", embedding_function=embedding_function)
except chromadb.errors.NotFoundError:
    print("❌ Collection 'rag_docs' not found. Please run the embedding script first.")
    exit()

def query_rag(question, top_k=3):
    # Embed the question using the same embedding function to keep consistency
    embedding = embedding_function.embed([question])[0]

    try:
        results = collection.query(query_embeddings=[embedding], n_results=top_k)
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return

    docs = results.get('documents', [[]])[0]
    metadatas = results.get('metadatas', [[]])[0]

    if not docs:
        print("⚠️ No relevant documents found.")
        return

    for i, doc in enumerate(docs):
        source = metadatas[i].get('source', 'N/A') if metadatas else 'N/A'
        print(f"\n🔹 Match #{i+1}")
        print(f"📄 Source: {source}")
        print(f"📖 Text:\n{doc}")
        print("-" * 60)

if __name__ == "__main__":
    query = input("🧠 Ask your question: ")
    query_rag(query)
