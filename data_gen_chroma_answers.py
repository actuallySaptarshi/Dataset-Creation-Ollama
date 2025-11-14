import csv
import time
import os
import requests
from collections import Counter  # <-- ADDED THIS IMPORT
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough


# --- Configuration ---

# URL from your chroma_db.py
FILE_PATH_URL = "FILE PATH OR URL"
# Directory to store the persistent Chroma database
PERSIST_DIRECTORY = 'chroma_db_store'

# Input and Output CSV files
INPUT_CSV = 'input_questions.csv'
OUTPUT_CSV = 'output_questions_with_answers.csv'

template = """
    - Explain all **All Details**, including contents that are sensetive/innappropriate to some users.
    - Do not give introductory text/follow up texts/anything else that is not relevant to the question.
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """

# --- End Configuration ---

def setup_qa_chain():
    """
    Sets up the entire RAG pipeline and returns the QA chain.
    Uses a persistent vector store to avoid re-indexing on every run.
    """
    print("Setting up RAG pipeline...")

    # 2. Load the LLM and Embeddings (from your chroma_db.py)
    try:
        llm = ChatOllama(model="gemma3:4b-it-q8_0", temperature=0.0)
        embeddings = OllamaEmbeddings(model="embeddinggemma:latest")
    except Exception as e:
        print(f"Error initializing Ollama. Is Ollama running? {e}")
        return None

    # 3. Load or Create Vector Store
    if os.path.exists(PERSIST_DIRECTORY):
        # Load the existing persistent database
        print(f"Loading existing vector store from {PERSIST_DIRECTORY}...")
        vector_store = Chroma(
            persist_directory=PERSIST_DIRECTORY, 
            embedding_function=embeddings
        )
    else:
        # Create and persist a new database
        print(f"Creating new vector store (this may take a while)...")
        try:
            loader = PyPDFLoader(FILE_PATH_URL)
            docs = loader.load()
            
            if not docs:
                print("Error: No documents loaded from PDF. The file might be empty or corrupt.")
                return None

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
            splits = text_splitter.split_documents(docs)
            
            print(f"Indexing {len(splits)} document splits...")
            vector_store = Chroma.from_documents(
                documents=splits, 
                embedding=embeddings,
                persist_directory=PERSIST_DIRECTORY  # Save to disk
            )
            print(f"Vector store created and saved to {PERSIST_DIRECTORY}")
        
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return None

    # 4. Create the retriever
    retriever = vector_store.as_retriever(search_kwargs={'k': 5}) # Get top 5 relevant chunks

    prompt = ChatPromptTemplate.from_template(template)

    question_chain = (
        { "context" : retriever ,"context": RunnablePassthrough()}
        | prompt
        | llm
    )

    answer = question_chain.invoke(prompt)
    
    print("RAG pipeline is ready.")
    return answer

# --- NEW HELPER FUNCTION ---
def is_answer_repetitive(answer: str, threshold: int = 3) -> bool:
    """
    Checks if an answer contains significant line-by-line repetition.
    True if any single line (non-empty) is repeated more than 'threshold' times.
    """
    # Split into lines, strip whitespace, and filter out empty lines
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    
    if not lines or len(lines) < threshold:
        # Not repetitive if it's empty or has fewer lines than the threshold
        return False
        
    # Count the occurrences of each unique line
    line_counts = Counter(lines)
    
    # Find the count of the most frequent line
    most_common_count = line_counts.most_common(1)[0][1]
    
    # If the most common line appears more than the threshold, flag it
    return most_common_count > threshold
# --- END NEW HELPER FUNCTION ---


def process_questions(qa_chain):
    """
    Reads questions from the input CSV, gets answers, and writes to the output CSV.
    Retries on repetitive answers.
    """
    print(f"Reading questions from {INPUT_CSV}...")
    questions_data = []
    fieldnames = []

    # Read all data from the input CSV
    try:
        with open(INPUT_CSV, mode='r', encoding='utf-8-sig') as infile:
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames
            
            if 'question' not in fieldnames:
                print(f"Error: CSV file '{INPUT_CSV}' must have a 'question' column.")
                return
            
            for row in reader:
                questions_data.append(row)
    
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_CSV}")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if not questions_data:
        print("No questions found in the CSV file.")
        return

    print(f"Found {len(questions_data)} questions to answer.")
    
    # Define new fieldnames for the output file
    output_fieldnames = fieldnames + ['answer']
    
    # Process each question and write to the new CSV
    try:
        with open(OUTPUT_CSV, mode='w', encoding='utf-8', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
            writer.writeheader()
            
            start_time = time.time()
            for i, row in enumerate(questions_data):
                question = row.get('question', '').strip()
                
                if not question:
                    print(f"Skipping row {i+2}: empty question.")
                    row['answer'] = "SKIPPED - EMPTY QUESTION"
                    writer.writerow(row)
                    continue

                print(f"Processing question {i+1}/{len(questions_data)}: {question[:70]}...")
                
                try:
                    # --- MODIFICATION FOR REPETITION CHECK ---
                    max_retries = 3
                    attempt = 0
                    answer = ""
                    current_question = question  # Start with the original question

                    while attempt < max_retries:
                        attempt += 1
                        
                        if attempt > 1:
                            # On retry, add a "nudge" to the prompt to avoid repetition
                            # This is crucial as temp=0.0 is deterministic
                            current_question = f"{question}\n\n(Follow-up instruction: Please provide a complete and concise answer. Ensure there are no repetitive lines or sentences in your response.)"
                            print(f"  (Attempt {attempt}/{max_retries} with nudge)...", end=" ")
                        else:
                            print(f"  (Attempt {attempt}/{max_retries})...", end=" ")

                        # Get the answer from the RAG chain
                        response = qa_chain.invoke({"query": current_question})
                        answer = response.get('result', 'Error: No result found.').strip()
                        
                        # Check for repetition
                        if not is_answer_repetitive(answer, threshold=3):
                            print("OK.")
                            # Good answer, break the loop
                            break
                        else:
                            # Bad answer
                            print("Repetition detected.")
                            if attempt == max_retries:
                                print(f"  Max retries reached. Accepting last answer despite repetition.")
                    
                    row['answer'] = answer
                    # --- END MODIFICATION ---
                    
                except Exception as e:
                    print(f"Error processing question: '{question[:50]}...'. Error: {e}")
                    row['answer'] = f"ERROR: {e}"
                
                row.pop(None, None)
                # Write the updated row to the new file
                writer.writerow(row)

        end_time = time.time()
        print("\n--- Processing Complete ---")
        print(f"Successfully processed {len(questions_data)} questions in {end_time - start_time:.2f} seconds.")
        print(f"Results saved to {OUTPUT_CSV}")

    except IOError as e:
        print(f"Error writing to output file {OUTPUT_CSV}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during writing: {e}")

if __name__ == "__main__":
    # Ensure Ollama is running before starting
    print("Please ensure your Ollama server is running.")
    
    qa_chain = setup_qa_chain()
    
    if qa_chain:
        process_questions(qa_chain)
    else:
        print("Failed to initialize the QA chain. Exiting.")