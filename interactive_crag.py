import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import CRAG engine
from crag_engine import query_with_scores, vector_db

# ============================================================
# MAIN INTERACTIVE LOOP
# ============================================================

print("\n" + "="*70)
print("CRAG - CORRECTIVE RETRIEVAL AUGMENTED GENERATION")
print("Interactive LLM Mode")
print("="*70)
print("\nType your questions below. Type 'exit' to quit.\n")

while True:
    try:
        question = input("\n>>> Ask a question: ").strip()
        
        if question.lower() in ['exit', 'quit', 'q']:
            print("\nGoodbye!")
            break
        
        if not question:
            print("Please enter a question.")
            continue
        
        # Query with scores
        answer, scored_docs = query_with_scores(question)
        
    except KeyboardInterrupt:
        print("\n\nExiting...")
        break
    except Exception as e:
        print(f"\nError: {e}")
        continue
