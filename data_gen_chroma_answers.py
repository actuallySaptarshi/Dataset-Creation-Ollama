import sys
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- 1. Configuration ---
# --- (USER: PLEASE SET THESE VALUES) ---

# Ollama model to use for embeddings and generation
OLLAMA_MODEL = "gemma3:4b-it-q8_0" 
EMBEDDING_MODEL = "embeddinggemma:latest"      

# Directory where the persistent Chroma vector database is stored
PERSIST_DIRECTORY = "./chroma_db_questions"

# --- (End of Configuration) ---


def format_docs(docs):
    """Helper function to format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    print("--- RAG Answering Chain (Modern LCEL) ---")

    # --- 2. Initialize Ollama Connections ---
    print(f"Initializing Ollama (Model: {OLLAMA_MODEL})...")
    try:
        # Initialize embeddings and LLM
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        llm = ChatOllama(model=OLLAMA_MODEL)
        
        # Test connections
        print("Testing Ollama connections...")
        llm.invoke("Hello, Ollama!")
        print("✅ Ollama connection successful.")
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        print(f"Please ensure Ollama is running and the models are pulled.")
        sys.exit(1)

    # --- 3. Load the Existing Vector Store ---
    print(f"Loading existing vector store from {PERSIST_DIRECTORY}...")
    try:
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
        print("✅ Vector store loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        print("Did you run the 'data_gen_chroma_questions.py' script first?")
        sys.exit(1)

    # --- 4. Create the Retriever ---
    # This object "retrieves" documents from the vector store
    retriever = vectorstore.as_retriever()
    print("✅ Retriever created.")

    # --- 5. Define the RAG Prompt Template ---
    # This prompt is for ANSWERING, not generating, questions
    rag_prompt_template = """
    You are an AI assistant for answering questions about Indian Law.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Be concise and answer *only* based on the context.

    CONTEXT:
    {context}
    
    QUESTION:
    {question}
    
    ANSWER:
    """
    
    prompt = ChatPromptTemplate.from_template(rag_prompt_template)

    # --- 6. Create the Modern RAG Chain (LCEL) ---
    print("Creating modern RAG chain...")

    # This is the full LCEL chain that replaces the old RetrievalQA
    rag_chain = (
        # The `RunnablePassthrough` on "question" passes the user's question 
        # all the way through the chain.
        # The "context" is populated by the retriever, which gets the user's
        # question from the "question" RunnablePassthrough.
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser() # Parses the LLM's chat message into a simple string
    )
    
    print("✅ RAG chain created successfully.")
    print("--- Ready to answer questions! (Type 'exit' to quit) ---")

    # --- 7. Run an Interactive Q&A Loop ---
    while True:
        try:
            query = input("\nYour Question: ")
            if query.lower() == 'exit':
                break
            
            print("Thinking...")
            
            # Invoke the chain with the user's query
            answer = rag_chain.invoke(query)
            
            print("\nAnswer:")
            print(answer)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()