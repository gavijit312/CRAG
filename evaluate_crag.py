import os
import sys
import pandas as pd

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import CRAG engine
from crag_engine import hybrid_retrieval, rerank_documents, generate_answer

# ============================================================
# CONFIGURATION
# ============================================================

GROUND_TRUTH_PATH = r"C:\Users\gavij\Desktop\CRAG_project\GroundTruth.xlsx"

OUTPUT_EXCEL_PATH = "CRAG_Output.xlsx"


# ============================================================
# UTILITY FUNCTION - CLEAN TEXT FOR EXCEL
# ============================================================

def clean_text_for_excel(text):
    """Remove illegal characters for Excel"""
    if not isinstance(text, str):
        return str(text)
    
    # Remove control characters and other illegal characters
    illegal_chars = [chr(i) for i in range(0, 32) if i not in (9, 10, 13)]  # Keep tab, newline, carriage return
    
    for char in illegal_chars:
        text = text.replace(char, '')
    
    # Remove other problematic characters
    text = text.replace('\x00', '')  # Null character
    text = text.replace('\x01', '')
    text = text.replace('\x02', '')
    
    return text


# ============================================================
# LOAD GROUND TRUTH EXCEL
# ============================================================

print("\nLoading Ground Truth Excel...")

df = pd.read_excel(GROUND_TRUTH_PATH)

# ============================================================
# FIX COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip().str.lower()

print("\nColumns Found:")
print(df.columns.tolist())

print("\nSample Data:")
print(df.head())

# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = ["question"]

for col in required_columns:

    if col not in df.columns:

        raise Exception(
            f"\nRequired column '{col}' not found in Excel file"
        )


# ============================================================
# OUTPUT STORAGE
# ============================================================

output_rows = []


# ============================================================
# RUN EVALUATION
# ============================================================

print("\nStarting Evaluation...\n")

slno = 1

for index, row in df.iterrows():

    # ========================================================
    # GET QUESTION
    # ========================================================

    question = str(row["question"])

    print(f"\nProcessing Question {slno}")

    # ========================================================
    # CRAG PIPELINE
    # ========================================================

    # RETRIEVAL
    docs = hybrid_retrieval(question)

    # RERANKING
    scored_docs = rerank_documents(question, docs, min_relevance_score=0.3, verbose=False)

    # ========================================================
    # CONTEXTS AND CHUNK SCORES
    # ========================================================

    retrieved_chunks = [
        clean_text_for_excel(doc.page_content)
        for score, doc in scored_docs
    ]

    combined_chunks = "\n\n".join(retrieved_chunks)

    # Create chunk scores string
    chunk_scores_str = "\n".join([
        f"Chunk {idx+1}: {score:.4f} | Source: {doc.metadata['source']}"
        for idx, (score, doc) in enumerate(scored_docs)
    ])

    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    answer = generate_answer(question, scored_docs, min_relevance_score=0.3)
    answer = clean_text_for_excel(answer)

    print(f"\nQuestion: {question}")

    print(f"\nAnswer: {answer}")

    # ========================================================
    # SAVE OUTPUT ROW
    # ========================================================

    output_rows.append({

        "slno.": slno,

        "question": clean_text_for_excel(question),

        "Retrieved chunks": combined_chunks,
        
        "Chunk Scores": chunk_scores_str,

        "answer (recieved from llm)": answer
    })

    slno += 1


# ============================================================
# SAVE OUTPUT EXCEL
# ============================================================

output_df = pd.DataFrame(output_rows)

try:
    output_df.to_excel(
        OUTPUT_EXCEL_PATH,
        index=False
    )
    print(f"\n\nOutput Excel Saved: {OUTPUT_EXCEL_PATH}\n")

except PermissionError:
    print(f"\n\nError: Cannot write to {OUTPUT_EXCEL_PATH}")
    print("The file is likely open in Excel. Please close it and run the script again.\n")
    
except Exception as e:
    print(f"\n\nError saving Excel file: {e}\n")
