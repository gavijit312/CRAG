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
