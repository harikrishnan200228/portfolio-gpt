"""
ingest.py — Run this ONCE to load your documents into ChromaDB.
Uses HuggingFace free embeddings — NO OpenAI API key needed!
Usage: python ingest.py
"""

import os
import shutil
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DIR = "./chroma_db"
DOCS_DIR   = "./docs"

# ── 1. Load documents ─────────────────────────────────────────────────────────
print("Loading documents...")
all_docs = []

# Resume PDF
resume_path = os.path.join(DOCS_DIR, "resume.pdf")
if os.path.exists(resume_path):
    loader = PyPDFLoader(resume_path)
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = "resume.pdf"
    all_docs.extend(docs)
    print(f"  Loaded resume.pdf ({len(docs)} pages)")

# Plain text files
for filename in ["about.txt", "skills.txt", "bio.txt"]:
    path = os.path.join(DOCS_DIR, filename)
    if os.path.exists(path):
        loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = filename
        all_docs.extend(docs)
        print(f"  Loaded {filename}")

# Markdown project files
md_dir = os.path.join(DOCS_DIR, "projects")
if os.path.exists(md_dir):
    for fname in os.listdir(md_dir):
        if fname.endswith(".md"):
            fpath = os.path.join(md_dir, fname)
            loader = TextLoader(fpath, encoding="utf-8")
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = fname
            all_docs.extend(docs)
            print(f"  Loaded projects/{fname}")

if not all_docs:
    print("\nNo documents found in ./docs/")
    print("Add files: docs/about.txt, docs/skills.txt, docs/projects/*.md")
    exit(1)

print(f"\nTotal documents loaded: {len(all_docs)}")

# ── 2. Chunk documents ────────────────────────────────────────────────────────
print("\nChunking documents...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_documents(all_docs)
print(f"Created {len(chunks)} chunks")

# ── 3. Embed using FREE HuggingFace model ─────────────────────────────────────
print("\nLoading FREE HuggingFace embedding model...")
print("(Downloading once ~90MB — no API key needed!)")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",   # free, fast, runs locally
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# ── 4. Save to ChromaDB ───────────────────────────────────────────────────────
if os.path.exists(CHROMA_DIR):
    shutil.rmtree(CHROMA_DIR)
    print("  Cleared old ChromaDB")

print("Saving to ChromaDB...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
)

print(f"\nDone! {len(chunks)} chunks saved to {CHROMA_DIR}/")

# ── 5. Quick test ─────────────────────────────────────────────────────────────
print("\nRunning quick test...")
results = vectorstore.similarity_search("skills and experience", k=2)
for i, r in enumerate(results):
    print(f"  Result {i+1}: [{r.metadata.get('source','?')}] {r.page_content[:80]}...")

print("\nIngestion complete! Now run: streamlit run app.py")
