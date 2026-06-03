import os
import pickle
import numpy as np

from transformers import pipeline

from langchain_community.document_loaders import (
    PyPDFLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)

from langchain_community.vectorstores import (
    FAISS
)

from rank_bm25 import BM25Okapi

from sentence_transformers import CrossEncoder


# ============================================================
# PATHS
# ============================================================

VECTOR_DB_PATH = "vectorstore"

CHUNKS_PATH = "chunks.pkl"

PDF_FOLDER = "pdfs"

os.makedirs(VECTOR_DB_PATH, exist_ok=True)

os.makedirs(PDF_FOLDER, exist_ok=True)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading Embedding Model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding Model Loaded")


# ============================================================
# LOAD CROSS ENCODER
# ============================================================

print("\nLoading Cross Encoder...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Cross Encoder Loaded")


# ============================================================
# GLOBAL VARIABLES
# ============================================================

all_chunks = []

bm25 = None

vector_db = None


# ============================================================
# TEXT SPLITTER
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)


# ============================================================
# LOAD VECTOR DATABASE
# ============================================================

def load_vector_db():

    global vector_db

    if os.path.exists(f"{VECTOR_DB_PATH}/index.faiss"):

        vector_db = FAISS.load_local(
            VECTOR_DB_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )

        print("\nExisting Vector DB Loaded")

    else:

        print("\nNo Existing Vector DB Found")


# ============================================================
# SAVE CHUNKS
# ============================================================

def save_chunks():

    with open(CHUNKS_PATH, "wb") as f:

        pickle.dump(all_chunks, f)


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks():

    global all_chunks

    if os.path.exists(CHUNKS_PATH):

        with open(CHUNKS_PATH, "rb") as f:

            all_chunks = pickle.load(f)

        print(f"\nLoaded {len(all_chunks)} Chunks")


# ============================================================
# BUILD BM25
# ============================================================

def rebuild_bm25():

    global bm25

    texts = [
        doc.page_content
        for doc in all_chunks
    ]

    tokenized = [
        text.split()
        for text in texts
    ]

    bm25 = BM25Okapi(tokenized)

    print("\nBM25 Ready")


# ============================================================
# ADD PDF
# ============================================================

def add_pdf(pdf_path):

    global vector_db
    global all_chunks

    print(f"\nLoading PDF: {pdf_path}")

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    print(f"Pages Loaded: {len(documents)}")

    chunks = splitter.split_documents(documents)

    print(f"Chunks Created: {len(chunks)}")

    for chunk in chunks:

        chunk.metadata["source"] = os.path.basename(pdf_path)

    all_chunks.extend(chunks)

    if vector_db is None:

        vector_db = FAISS.from_documents(
            chunks,
            embedding_model
        )

    else:

        vector_db.add_documents(chunks)

    vector_db.save_local(VECTOR_DB_PATH)

    save_chunks()

    rebuild_bm25()

    print(f"\nPDF Added: {pdf_path}")


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def hybrid_retrieval(query, k=10):

    dense_docs = vector_db.similarity_search(
        query,
        k=k
    )

    tokenized_query = query.split()

    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:k]

    sparse_docs = [
        all_chunks[i]
        for i in top_indices
    ]

    combined = dense_docs + sparse_docs

    unique_docs = []

    seen = set()

    for doc in combined:

        if doc.page_content not in seen:

            seen.add(doc.page_content)

            unique_docs.append(doc)

    return unique_docs


# ============================================================
# RERANK DOCUMENTS
# ============================================================

def rerank_documents(query, docs, top_k=4, min_relevance_score=0.3, verbose=True):

    pairs = [
        (query, doc.page_content)
        for doc in docs
    ]

    scores = reranker.predict(pairs)

    scored_docs = list(zip(scores, docs))

    scored_docs.sort(
        key=lambda x: x[0],
        reverse=True
    )

    final_docs = []

    chunk_count = 0
    for idx, (score, doc) in enumerate(scored_docs[:top_k]):
        
        # Skip documents below minimum relevance threshold
        if score < min_relevance_score:
            continue

        chunk_count += 1

        final_docs.append((score, doc))

    # Print retrieval metrics instead of individual chunks
    if verbose and final_docs:
        print("\n================ RETRIEVAL METRICS ================\n")
        print(f"Total Chunks Retrieved: {len(final_docs)}")
        avg_score = sum(score for score, _ in final_docs) / len(final_docs)
        min_score = min(score for score, _ in final_docs)
        max_score = max(score for score, _ in final_docs)
        print(f"Average Relevance Score: {avg_score:.4f}")
        print(f"Min Score: {min_score:.4f}")
        print(f"Max Score: {max_score:.4f}")
        print("\n================================================\n")

    return final_docs


# ============================================================
# LOAD LLM
# ============================================================

print("\nLoading LLM...")

llm = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=128,
    temperature=0.2
)

print("LLM Loaded")


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(query, scored_docs, min_relevance_score=0.3):
    
    # Check if we have any relevant documents
    if not scored_docs:
        return "I can only answer questions based on the provided PDFs. Your question doesn't match the PDF content."
    
    # Get average relevance score
    avg_score = sum(score for score, _ in scored_docs) / len(scored_docs)
    
    if avg_score < min_relevance_score:
        return "I can only answer questions based on the provided PDFs. Your question doesn't match the PDF content."

    context = "\n\n".join([
        doc.page_content[:300]
        for score, doc in scored_docs[:3]
    ])

    prompt = f"""
Answer the question briefly using the context.

Rules:
- Maximum 7 lines
- Give direct answer
- Do not copy context
- If the context doesn't contain information to answer the question, say "I cannot answer this based on the provided PDFs"

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm(prompt)

    answer = response[0]["generated_text"]

    if "Answer:" in answer:

        answer = answer.split("Answer:")[-1]

    return answer.strip()


# ============================================================
# CRAG PIPELINE (RETURNS ANSWER + SCORED DOCS)
# ============================================================

def CRAG(query):

    global vector_db

    if vector_db is None:

        return "No PDFs loaded into vector database.", []

    # RETRIEVAL
    docs = hybrid_retrieval(query)

    # RERANKING
    scored_docs = rerank_documents(query, docs, min_relevance_score=0.3, verbose=True)

    # GENERATION
    answer = generate_answer(query, scored_docs, min_relevance_score=0.3)

    return answer, scored_docs


# ============================================================
# INTERACTIVE QUERY INTERFACE
# ============================================================

def query_with_scores(query):
    """Ask a question and get answer with retrieval metrics"""
    
    print(f"\n\n{'='*60}")
    print(f"QUESTION: {query}")
    print(f"{'='*60}")
    
    answer, scored_docs = CRAG(query)
    
    print(f"\n{'='*60}")
    print("ANSWER:")
    print(f"{'='*60}")
    print(answer)
    
    return answer, scored_docs


# ============================================================
# INITIALIZE CRAG ENGINE
# ============================================================

def initialize_crag():
    """Initialize vector DB, chunks, and BM25"""
    
    load_vector_db()
    
    load_chunks()
    
    if len(all_chunks) > 0:
        rebuild_bm25()
    
    # Load PDFs from pdfs/ folder
    pdf_files = []
    
    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            pdf_files.append(os.path.join(PDF_FOLDER, file))
    
    print(f"\nFound {len(pdf_files)} PDFs")
    
    # Process PDFs
    for pdf in pdf_files:
        add_pdf(pdf)


# Auto-initialize on import
initialize_crag()
