from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def create_vector_store(docs: list) -> Chroma:
    """Store document chunks in ChromaDB."""
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print(f"Stored {len(docs)} chunks in vector store.")
    return vector_store


def load_vector_store() -> Chroma:
    """Load an existing vector store from disk."""
    vector_store = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    return vector_store
