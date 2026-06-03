# CRAG - Contextual Retrieval Augmented Generation

A modular Python-based system that combines PDF retrieval with LLM-powered question answering. The system uses hybrid retrieval (dense + sparse), cross-encoder reranking, and provides both interactive and batch evaluation modes.

---

## 📋 Project Structure

```
CRAG_project/
├── crag_engine.py          # Core CRAG engine (all models and functions)
├── interactive_crag.py     # Interactive LLM mode (ask questions in real-time)
├── evaluate_crag.py        # Batch evaluation (process Excel file)
├── CRAG.py                 # Backwards compatibility wrapper
├── requirements.txt        # Python dependencies
├── GroundTruth.xlsx        # Input: Questions to evaluate
├── CRAG_Output.xlsx        # Output: Results with answers & chunk scores
├── pdfs/                   # Input: PDF documents (add your PDFs here)
├── vectorstore/            # Vector database (FAISS index)
│   └── index.faiss
├── chunks.pkl              # Serialized document chunks
└── README.md               # This file
```

---

## 🚀 Quick Start

### 1. Setup

#### Install Python Environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Add Your PDFs
Place PDF files in the `pdfs/` folder. They will be automatically loaded and indexed.

### 3. Run One of Three Modes

---

## 📁 Executable Files

### **1. crag_engine.py** - Core Engine
**Purpose:** Loads models, builds vector database, and provides all CRAG functions

**What it does:**
- Loads embedding model (sentence-transformers/all-MiniLM-L6-v2)
- Loads cross-encoder for reranking (cross-encoder/ms-marco-MiniLM-L-6-v2)
- Loads LLM (TinyLlama-1.1B-Chat-v1.0)
- Creates FAISS vector database from PDFs
- Builds BM25 index for sparse retrieval
- Auto-initializes on import

**Key Functions:**
- `initialize_crag()` - Setup vector DB and load PDFs
- `hybrid_retrieval(query, k=10)` - Dense + sparse search
- `rerank_documents(query, docs, top_k=4)` - Rerank with cross-encoder
- `generate_answer(query, scored_docs)` - Generate answer with LLM
- `CRAG(query)` - Full pipeline: retrieve → rerank → generate
- `query_with_scores(query)` - User-friendly interface with metrics

**Run directly to test:**
```powershell
python crag_engine.py
```
Output: Model loading messages + "CRAG Engine Initialized"

---

### **2. interactive_crag.py** - Interactive Mode
**Purpose:** Chat-like interface for real-time question answering

**How to use:**
```powershell
python interactive_crag.py
```

**Output Example:**
```
============================================================
CRAG - CONTEXTUAL RETRIEVAL AUGMENTED GENERATION
Interactive LLM Mode
============================================================

Type your questions below. Type 'exit' to quit.

>>> Ask a question: What is machine learning?

============================================================
QUESTION: What is machine learning?
============================================================

================ RETRIEVAL METRICS ================

Total Chunks Retrieved: 3
Average Relevance Score: 0.7845
Min Score: 0.7234
Max Score: 0.8456

================================================

============================================================
ANSWER:
============================================================
Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without explicit programming...

>>> Ask a question: exit
Goodbye!
```

**Features:**
- Real-time interactive queries
- Shows retrieval metrics (total chunks, avg/min/max scores)
- Direct LLM-generated answers
- Type 'exit', 'quit', or 'q' to close
- Handles errors gracefully
- Press Ctrl+C to exit anytime

---

### **3. evaluate_crag.py** - Batch Evaluation
**Purpose:** Process batch of questions from Excel and generate results file

**How to use:**
```powershell
python evaluate_crag.py
```

**Input:**
- `GroundTruth.xlsx` with column "question"
```
| question                          |
|-----------------------------------|
| What is CRAG?                     |
| How does retrieval work?          |
| Explain the reranking process     |
```

**Output:**
- `CRAG_Output.xlsx` with columns:
  - `slno.` - Serial number
  - `question` - Original question
  - `Retrieved chunks` - Full text of retrieved documents
  - `Chunk Scores` - Relevance scores for each chunk
  - `answer (recieved from llm)` - LLM-generated answer

**Example Output Row:**
```
Sl | Question | Retrieved chunks | Chunk Scores | Answer
---|----------|------------------|--------------|-------
1  | What is... | Text of chunk 1... | Chunk 1: 0.8234 | Machine learning is...
                | Text of chunk 2... | Chunk 2: 0.7645 |
                | Text of chunk 3... | Chunk 3: 0.6890 |
```

**Features:**
- Processes all questions from input Excel
- Shows progress for each question
- Removes illegal characters for Excel compatibility
- Generates comprehensive results file
- Captures chunk scores for each answer
- Error handling for file permissions

**Troubleshooting:**
```
Error: Cannot write to CRAG_Output.xlsx
The file is likely open in Excel. Please close it and run again.
```

---

### **4. CRAG.py** - Compatibility Wrapper
**Purpose:** Backwards compatibility - imports all functions from crag_engine

**Use when:**
- You want to import CRAG functions in your own scripts
- You need all models/functions in one namespace

**Example:**
```python
from CRAG import query_with_scores, hybrid_retrieval

# Use functions directly
answer, scored_docs = query_with_scores("Your question here")
```

---

## 🔧 Configuration

### Paths (in `crag_engine.py`)
```python
VECTOR_DB_PATH = "vectorstore"    # Where to store FAISS index
CHUNKS_PATH = "chunks.pkl"        # Serialized chunks file
PDF_FOLDER = "pdfs"               # Where to read PDFs from
```

### Models (in `crag_engine.py`)
```python
# Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# LLM
llm = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=128,
    temperature=0.2
)
```

### Text Splitting (in `crag_engine.py`)
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,           # Size of each chunk
    chunk_overlap=50          # Overlap between chunks
)
```

### Retrieval Parameters (in functions)
```python
hybrid_retrieval(query, k=10)  # Retrieve top 10 chunks
rerank_documents(query, docs, top_k=4, min_relevance_score=0.3)
generate_answer(query, scored_docs, min_relevance_score=0.3)
```

### Excel Path (in `evaluate_crag.py`)
```python
GROUND_TRUTH_PATH = r"C:\Users\gavij\Desktop\CRAG_project\GroundTruth.xlsx"
OUTPUT_EXCEL_PATH = "CRAG_Output.xlsx"
```

---

## 📊 How It Works

### Pipeline Flow

```
INPUT QUESTION
    ↓
[RETRIEVAL] 
    ├─ Dense Retrieval (FAISS embeddings)
    └─ Sparse Retrieval (BM25)
    ↓
[RERANKING]
    └─ Cross-Encoder scores documents
    ↓
[GENERATION]
    └─ LLM generates answer from top chunks
    ↓
OUTPUT (Answer + Retrieval Metrics)
```

### Retrieval Metrics

After retrieval, the system shows:
- **Total Chunks Retrieved** - Number of relevant documents found
- **Average Relevance Score** - Mean score across all chunks
- **Min/Max Score** - Range of relevance scores

---

## 📦 Dependencies

See `requirements.txt` for all packages:

| Package | Purpose |
|---------|---------|
| `langchain` | LLM framework |
| `langchain-community` | PDF loaders & vectorstores |
| `sentence-transformers` | Cross-encoder reranking |
| `transformers` | LLM pipeline |
| `faiss-cpu` | Vector similarity search |
| `rank-bm25` | Sparse retrieval |
| `pypdf` | PDF parsing |
| `pandas` | Excel handling |
| `numpy` | Numerical operations |

---

## 💡 Usage Examples

### Example 1: Interactive Query
```powershell
$ python interactive_crag.py

>>> Ask a question: Explain deep learning

============================================================
QUESTION: Explain deep learning
============================================================

================ RETRIEVAL METRICS ================

Total Chunks Retrieved: 4
Average Relevance Score: 0.8234
Min Score: 0.7890
Max Score: 0.8567

================================================

============================================================
ANSWER:
============================================================
Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers...
```

### Example 2: Batch Processing
```powershell
$ python evaluate_crag.py

Loading Ground Truth Excel...
Columns Found: ['question']
Sample Data: (showing first rows)

Found 3 PDFs

Starting Evaluation...

Processing Question 1
Question: What is CRAG?
Answer: CRAG stands for Contextual Retrieval Augmented Generation...

Processing Question 2
Question: How does retrieval work?
Answer: Retrieval works by searching through document chunks...

Processing Question 3
Question: Explain reranking?
Answer: Reranking uses a cross-encoder to score document relevance...

Output Excel Saved: CRAG_Output.xlsx
```

### Example 3: Programmatic Use
```python
from CRAG import query_with_scores, hybrid_retrieval, generate_answer

# Ask a question with metrics
answer, scored_docs = query_with_scores("Your question?")

# Custom pipeline
docs = hybrid_retrieval("Your question?", k=5)
scored_docs = rerank_documents("Your question?", docs, top_k=3)
answer = generate_answer("Your question?", scored_docs)

print(f"Number of chunks: {len(scored_docs)}")
for score, doc in scored_docs:
    print(f"Score: {score:.4f} | Source: {doc.metadata['source']}")
```

---

## 🐛 Troubleshooting

### Issue: "No Existing Vector DB Found"
**Solution:** Add PDF files to the `pdfs/` folder. They will be indexed on first run.

### Issue: "No PDFs loaded into vector database"
**Solution:** 
1. Ensure PDFs are in `pdfs/` folder
2. Run the script again - it will load and index them
3. Wait for "PDF Added" messages

### Issue: "Cannot write to CRAG_Output.xlsx"
**Solution:** Close the Excel file and run `evaluate_crag.py` again

### Issue: Model download is very slow
**Solution:** Models are downloaded on first run only. Subsequent runs are fast.

### Issue: "PermissionError" when saving results
**Solution:** Make sure `CRAG_Output.xlsx` is not open in Excel

### Issue: Empty or incorrect answers
**Solution:** 
- Check that PDFs contain relevant content
- Lower `min_relevance_score` threshold (default 0.3)
- Increase `top_k` for more chunks to consider

---

## 🔄 Data Flow

### First Run
1. `crag_engine.py` auto-initializes
2. Loads embedding, reranker, and LLM models
3. Scans `pdfs/` folder for PDF files
4. Processes each PDF and creates chunks
5. Builds FAISS vector index
6. Builds BM25 sparse index
7. Saves chunks to `chunks.pkl`
8. Saves vector index to `vectorstore/index.faiss`

### Subsequent Runs
1. Loads existing vector index
2. Loads existing chunks
3. Rebuilds BM25 index (fast)
4. Checks for new PDFs in `pdfs/` folder
5. Processes only new PDFs (if any)

---

## 📝 Output Files

### CRAG_Output.xlsx
Generated by `evaluate_crag.py`:
- **Column A (slno.)** - Question number
- **Column B (question)** - Original question
- **Column C (Retrieved chunks)** - Full text of retrieved documents
- **Column D (Chunk Scores)** - Scores for each chunk
- **Column E (answer)** - LLM-generated answer

### chunks.pkl
Binary file containing all document chunks - auto-generated

### vectorstore/index.faiss
FAISS vector database - auto-generated

---

## ⚙️ Advanced Customization

### Change Chunk Size
In `crag_engine.py`:
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,    # Increase for longer chunks
    chunk_overlap=100
)
```

### Change Number of Retrieved Chunks
In functions:
```python
hybrid_retrieval(query, k=20)  # Get 20 instead of 10
```

### Change Reranking Threshold
In functions:
```python
rerank_documents(query, docs, top_k=5, min_relevance_score=0.5)
```

### Use Different LLM
In `crag_engine.py`:
```python
llm = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B",  # Your model
    max_new_tokens=256,
    temperature=0.3
)
```

---

## 📚 File Details

### crag_engine.py
- **Lines:** ~450
- **Functions:** 15+
- **Models Loaded:** 3 (embedding, reranker, LLM)
- **Auto-runs:** Yes (initialization on import)

### interactive_crag.py
- **Lines:** ~45
- **Purpose:** User-facing interactive interface
- **Loop:** Continuous until 'exit' command

### evaluate_crag.py
- **Lines:** ~180
- **Purpose:** Batch processing from Excel
- **Input:** GroundTruth.xlsx
- **Output:** CRAG_Output.xlsx

### CRAG.py
- **Lines:** ~40
- **Purpose:** Backwards compatibility wrapper
- **Auto-runs:** No

---

## 🎯 Performance Tips

1. **First run is slow** - Models are downloaded and PDFs indexed
2. **Subsequent runs are fast** - Uses cached models and indices
3. **Increase batch size** - For more concurrent queries
4. **Use GPU** - Install CUDA for faster inference
5. **Smaller chunks** - Faster retrieval but less context
6. **Larger chunks** - Slower retrieval but more context

---

## 📞 Support

For issues or questions:
1. Check the **Troubleshooting** section
2. Verify PDF files are in `pdfs/` folder
3. Check that `GroundTruth.xlsx` has a "question" column
4. Ensure all dependencies are installed: `pip install -r requirements.txt`

---

**Last Updated:** May 21, 2026


