import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions

def split_text_structured(text, chunk_size=1500, chunk_overlap=30):
    """
    Split text into chunks that respect bullets, headings, subheadings,
    and avoid breaking mid sentence or word.
    """
    separators = [
        "\n# ",    # Heading 1
        "\n## ",   # Heading 2
        "\n### ",  # Heading 3
        "\n- ",    # Bullet point
        "\n* ",    # Bullet alternative
        "\n• ",    # Bullet alternative
        "\n\n",    # Paragraph
        ". ",      # Sentence end
        "! ",
        "? ",
        ", ",      # Comma to avoid phrase break
        " ",       # Space - last resort to split words
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    return splitter.split_text(text)

def get_embedding_function():
    """
    Returns the embedding function to use with Chroma.
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

def build_vector_db(documents, metadatas, persist_dir="./chroma_persist", collection_name="rag_docs"):
    """
    Build or update a Chroma vector database with the given documents and metadata.
    """
    client = chromadb.PersistentClient(path=persist_dir)
    embed_fn = get_embedding_function()

    try:
        collection = client.get_collection(name=collection_name, embedding_function=embed_fn)
    except chromadb.errors.NotFoundError:
        collection = client.create_collection(name=collection_name, embedding_function=embed_fn)

    existing = collection.count()
    ids = [str(uuid.uuid4()) for _ in documents]

    if documents:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        print(f"✅ Added {len(documents)} new chunks (total now: {existing + len(documents)}).")
    else:
        print("⚠️ No new documents to embed.")
