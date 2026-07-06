from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path: list[str]) -> list:
    docs = []
    for path in file_path:
        try:
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
            print(f"Loaded: {path}")
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return docs
