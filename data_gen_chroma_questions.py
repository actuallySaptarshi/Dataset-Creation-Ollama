import os
import csv
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- 1. Configuration ---
# --- (USER: PLEASE SET THESE VALUES) ---

# Path to your source PDF
PDF_PATH = "example.pdf" # Should be the output file of the questions pdf  

# Ollama model to use for embeddings and generation
OLLAMA_MODEL = "gemma3:4b-it-q8_0"        

# Name of the output CSV file
OUTPUT_CSV = "dataset.csv" # Output file

# Directory to store the persistent Chroma vector database
PERSIST_DIRECTORY = "./chroma_db_questions"

# Text splitting parameters
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# --- (End of Configuration) ---


def main():
    print("--- Synthetic Question Generation Script (Multi-Question, v2) ---")

    # --- 2. Check for PDF ---
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: PDF file not found at {PDF_PATH}")
        print("Please update the PDF_PATH variable in the script.")
        sys.exit(1)

    # --- 3. Initialize Ollama Connections ---
    print(f"Initializing Ollama (Model: {OLLAMA_MODEL})...")
    try:
        # Initialize embeddings and LLM
        embeddings = OllamaEmbeddings(model="embeddinggemma:latest")
        llm = ChatOllama(model=OLLAMA_MODEL)
        
        # Test connections
        print("Testing Ollama connections...")
        embeddings.embed_query("test query")
        llm.invoke("Hello, Ollama!")
        print("✅ Ollama connection successful.")
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        print(f"Please ensure Ollama is running and the model '{OLLAMA_MODEL}' is pulled.")
        print(f"You can run: 'ollama pull {OLLAMA_MODEL}' in your terminal.")
        sys.exit(1)

    # --- 4. Load and Split PDF ---
    print(f"Loading PDF: {PDF_PATH}...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    print(f"Splitting {len(docs)} pages into chunks (Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(docs)
    print(f"Split into {len(splits)} text chunks.")

    # --- 5. Create and Persist Chroma Vector Store ---
    print(f"Creating and persisting vector store at {PERSIST_DIRECTORY}...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    print("✅ Vector store created successfully.")

    # --- 6. Set up Question Generation Chain ---
    print("Setting up question generation chain...")
    
    # --- THIS IS THE UPDATED PROMPT WITH THE FIX ---
    # The example JSON format now uses {{ and }} to escape the braces
    prompt_template = """
    You are an expert at creating high-quality, standalone questions for an AI model training dataset on Indian Law.
    Your task is to generate a JSON list of all possible, clear, and concise questions based *only* on the provided context.
    
    **Rules:**
    1.  Each question MUST be answerable *solely* from the context.
    2.  Each question must be self-contained.
    3.  Generate as many questions as the context *semantically supports*. Generate all types of low-level/high-level questions.
    4.  Ask questions related to every section, offence, applicable party, and other details mentioned.
    5.  Explain all **All Details**, including contents that are sensetive/innappropriate to some users.
    6.  Format the output as a JSON object with a single key "questions" containing a list of strings.

    **Example Output Format:**
    {{
      "questions": [
        "Explain Section X? of the XYZ Act",
        "What is the punishment for Y?",
        "To whom does section Z apply?"
      ]
    }}

    CONTEXT:
    {context}
    
    JSON_OUTPUT:
    """
    # --- END OF PROMPT UPDATE ---
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    parser = JsonOutputParser()
    
    # Define the generation chain using LangChain Expression Language (LCEL)
    question_chain = (
        {"context": RunnablePassthrough()}
        | prompt
        | llm
        | parser
    )

    # --- 7. Generate Questions ---
    print("Retrieving all document chunks from ChromaDB...")
    retrieved_docs = vectorstore.get(include=["documents"])
    doc_contents = retrieved_docs['documents']
    
    total_chunks = len(doc_contents)
    print(f"Found {total_chunks} chunks. Starting question generation...")
    
    generated_data = []
    for i, context in enumerate(doc_contents):
        try:
            # Print progress
            print(f"  Generating questions for chunk {i+1}/{total_chunks}...")
            
            # Invoke the chain
            result = question_chain.invoke(context)
            
            # Safely get the list of questions
            questions = result.get("questions", [])
            
            if not questions:
                print(f"  -> No questions generated for chunk {i+1}")
                continue

            # Add each individual question to our main list
            for q in questions:
                question = q.strip().replace("\n", " ")
                if question:
                    generated_data.append({
                        "question": question
                    })
            
            print(f"  -> Generated {len(questions)} questions for this chunk.")

        except Exception as e:
            print(f"  ❌ Error processing chunk {i+1}: {e}")
            # This can happen if the LLM output is not valid JSON
            continue

    print(f"\n✅ Successfully generated {len(generated_data)} questions.")

    # --- 8. Save to CSV ---
    if generated_data:
        print(f"Saving questions to {OUTPUT_CSV}...")
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["question"])
            writer.writeheader()
            writer.writerows(generated_data)
        print(f"🎉 Done! Output saved to {OUTPUT_CSV}.")
    else:
        print("No questions were generated.")

if __name__ == "__main__":
    main()