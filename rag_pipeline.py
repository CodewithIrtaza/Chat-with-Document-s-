import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from utils.pdf_loader import load_pdf
from utils.splitter import split_text
from utils.vector_store import create_vector_store, load_vector_store

load_dotenv()


# --- Step 1: Build vector store from PDFs ---
def build_knowledge_base(pdf_paths: list[str]):
    """Load PDFs → Split → Store in ChromaDB."""
    docs = load_pdf(pdf_paths)
    chunks = split_text(docs)
    vector_store = create_vector_store(chunks)
    return vector_store


# --- Step 2: Ask a question ---
def ask(question: str, vector_store=None):
    """Search relevant chunks → Send to LLM → Get answer + sources."""

    # Load vector store if not provided
    if vector_store is None:
        vector_store = load_vector_store()

    # Search for relevant chunks
    results = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in results])

    # Get source pages
    sources = []
    for doc in results:
        page = doc.metadata.get("page", None)
        if page is not None:
            sources.append(page + 1)  # 0-indexed → 1-indexed
    sources = sorted(set(sources))

    # Setup LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based on the context below. 
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""
    )

    # Get answer
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content, sources

