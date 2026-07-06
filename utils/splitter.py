from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def split_text(texts: list) -> list:
    if not texts:
        print("No documents to split")
        return []
    
    docs = []
    for text in texts:
        if isinstance(text, Document):
            docs.append(text)
        elif isinstance(text, dict):
            docs.append(Document(page_content=text["page_content"]))
        elif isinstance(text, str):
            docs.append(Document(page_content=text))
        else:
            print(f"Unsupported type: {type(text)}, skipping.")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(docs)
