# Rasa_bot/rag/export_chunks.py

from chromadb import PersistentClient
import os

# ✅ Use full path to the chroma_persist directory
PERSIST_DIR = os.path.abspath("chroma_persist")
COLLECTION_NAME = "rag_docs"
EXPORT_FILE = "exported_chunks.txt"

# Connect to ChromaDB
client = PersistentClient(path=PERSIST_DIR)
collection = client.get_collection(COLLECTION_NAME)

# Fetch all documents
results = collection.get(include=["documents", "metadatas"])


# Write to a text file
with open(EXPORT_FILE, "w", encoding="utf-8") as f:
    for idx, (doc, meta, id_) in enumerate(zip(results["documents"], results["metadatas"], results["ids"])):
        f.write(f"--- Chunk #{idx + 1} ---\n")
        f.write(f"ID: {id_}\n")
        f.write(f"Source File: {meta.get('source')}\n")
        f.write(f"Content:\n{doc}\n\n")

print(f"✅ Exported {len(results['documents'])} chunks to {EXPORT_FILE}")
