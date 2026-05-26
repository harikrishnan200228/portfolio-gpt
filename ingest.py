

import os
import re
import shutil
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
 
CHROMA_DIR = "./chroma_db"
DOCS_DIR   = "./docs"
 
# ── Section-based splitter for structured docs ────────────────────────────────
def split_by_sections(text: str, source: str) -> list:
    """Split document by markdown headings or keyword sections."""
    # Split on lines that look like section headers
    section_pattern = re.compile(
        r'\n(?=(?:#{1,3}\s|'
        r'(?:Experience|Education|Skills|Projects|Summary|'
        r'Contact|Achievements|Certifications|Languages|'
        r'About|Work History|Technical Skills)\s*:?\s*\n))',
        re.IGNORECASE
    )
    sections = section_pattern.split(text)
    docs = []
    for i, section in enumerate(sections):
        section = section.strip()
        if not section or len(section) < 30:
            continue
        # Extract section title for metadata
        first_line = section.split('\n')[0].strip().lstrip('#').strip()
        docs.append(Document(
            page_content=section,
            metadata={
                "source": source,
                "section": first_line,
                "chunk_type": "section",
            }
        ))
    return docs if docs else None  # Return None if no sections found
 
# ── Fixed-size splitter for unstructured docs ─────────────────────────────────
def split_fixed(text: str, source: str) -> list:
    """Standard fixed-size chunking with overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = splitter.create_documents(
        texts=[text],
        metadatas=[{"source": source, "chunk_type": "fixed"}]
    )
    return docs
 
# ── Smart loader — picks the right strategy ───────────────────────────────────
def load_and_chunk(filepath: str, source_name: str) -> list:
    """Load a file and chunk it using the best strategy."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"  Could not read {source_name}: {e}")
        return []
 
    # Try section-based first
    sections = split_by_sections(text, source_name)
    if sections and len(sections) >= 2:
        print(f"  Section-based chunking → {len(sections)} sections")
        return sections
 
    # Fallback to fixed-size
    chunks = split_fixed(text, source_name)
    print(f"  Fixed-size chunking → {len(chunks)} chunks")
    return chunks
 
# ── 1. Load all documents ─────────────────────────────────────────────────────
print("=" * 50)
print("Portfolio GPT — Smart Ingestion")
print("=" * 50)
all_chunks = []
 
# Resume PDF — fixed size (PDF text is unstructured)
resume_path = os.path.join(DOCS_DIR, "resume.pdf")
if os.path.exists(resume_path):
    loader = PyPDFLoader(resume_path)
    pages  = loader.load()
    full_text = "\n\n".join([p.page_content for p in pages])
    # Try section-based on PDF text too
    sections = split_by_sections(full_text, "resume.pdf")
    if sections and len(sections) >= 2:
        all_chunks.extend(sections)
        print(f"  resume.pdf → section-based → {len(sections)} sections")
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks   = splitter.split_documents(pages)
        for c in chunks:
            c.metadata["source"] = "resume.pdf"
            c.metadata["chunk_type"] = "fixed"
        all_chunks.extend(chunks)
        print(f"  resume.pdf → fixed-size → {len(chunks)} chunks")
 
# Structured docs (about, skills, bio) — section-based
for filename in ["about.txt", "skills.txt", "bio.txt"]:
    path = os.path.join(DOCS_DIR, filename)
    if os.path.exists(path):
        print(f"\nLoading {filename}...")
        chunks = load_and_chunk(path, filename)
        all_chunks.extend(chunks)
 
# Project files — fixed size (unstructured descriptions)
projects_dir = os.path.join(DOCS_DIR, "projects")
if os.path.exists(projects_dir):
    for fname in os.listdir(projects_dir):
        if fname.endswith((".md", ".txt")):
            fpath = os.path.join(projects_dir, fname)
            print(f"\nLoading projects/{fname}...")
            chunks = load_and_chunk(fpath, fname)
            all_chunks.extend(chunks)
 
if not all_chunks:
    print("\nNo documents found in ./docs/")
    print("Add: docs/about.txt, docs/skills.txt, docs/projects/*.md")
    exit(1)
 
# ── 2. Print chunking summary ─────────────────────────────────────────────────
section_count = sum(1 for c in all_chunks if c.metadata.get("chunk_type") == "section")
fixed_count   = sum(1 for c in all_chunks if c.metadata.get("chunk_type") == "fixed")
 
print(f"\n{'='*50}")
print(f"Chunking Summary:")
print(f"  Section-based chunks : {section_count}")
print(f"  Fixed-size chunks    : {fixed_count}")
print(f"  Total chunks         : {len(all_chunks)}")
print(f"{'='*50}")
 
# ── 3. Embed and save ─────────────────────────────────────────────────────────
print("\nLoading HuggingFace embedding model...")
print("(Free — runs locally, no API key needed)")
 
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
 
if os.path.exists(CHROMA_DIR):
    shutil.rmtree(CHROMA_DIR)
    print("Cleared old ChromaDB")
 
print("Saving to ChromaDB...")
vectorstore = Chroma.from_documents(
    documents=all_chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
)
 
print(f"\nDone! {len(all_chunks)} chunks saved.")
 
# ── 4. Test query ─────────────────────────────────────────────────────────────
print("\nRunning test query: 'skills and experience'")
results = vectorstore.similarity_search("skills and experience", k=3)
for i, r in enumerate(results):
    chunk_type = r.metadata.get('chunk_type', 'unknown')
    section    = r.metadata.get('section', '')
    source     = r.metadata.get('source', '?')
    label      = f"{source} [{chunk_type}]" + (f" — {section}" if section else "")
    print(f"  Result {i+1}: {label}")
    print(f"    {r.page_content[:80]}...")
 
print("\nIngestion complete! Run: streamlit run app.py")
 