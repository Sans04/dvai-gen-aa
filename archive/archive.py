from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path

# PDF-Verzeichnis
pdf_dir = Path("./leitgedanken")

# 1. Alle PDFs laden
all_docs = []
for pdf_path in pdf_dir.glob("*.pdf"):
    print(f"Lade {pdf_path.name} ...")
    loader = PyMuPDFLoader(str(pdf_path))
    all_docs.extend(loader.load())

# 2. Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
chunks = splitter.split_documents(all_docs)

# 3. Embedding
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Vektorstore erzeugen und speichern
db = FAISS.from_documents(chunks, embeddings)
db.save_local("vector_index")
print("Vektorindex gespeichert.")
