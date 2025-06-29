# Importing necessary modules for building the question-answering system

# Load a predefined question-answering chain from LangChain
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from pypdf import PdfReader


def process_text(text):
    """
    Splits raw text into smaller chunks, converts them into embeddings, and stores them in a FAISS index.
    This allows for fast similarity search based on user queries.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,      # Each chunk will have up to 1000 characters
        chunk_overlap=200,
        length_function=len
    )

    # Split the input text into chunks
    chunks = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    # Store the vectorized chunks in a FAISS vector store
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base


def summarizer(pdf_path, query):
    """
    Takes a PDF file path and a query, extracts the text from the PDF,
    processes the text into chunks and embeddings, and returns a relevant answer to the query.
    """
    # Load and read the PDF file
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""


    knowledge_base = process_text(text)

    if query:

        docs = knowledge_base.similarity_search(query)
        llm = ChatOpenAI(model="gpt-4", temperature=0.1)
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)
        return response


#  Entry point for running as a script

if __name__ == "__main__":
    import sys

    pdf_path = input("Enter path to the PDF file: ").strip()
    query = input("Enter your question/query about the PDF: ").strip()

    if pdf_path and query:
        result = summarizer(pdf_path, query)
        print("\nAnswer:\n", result)
    else:
        print("Please provide both PDF path and a query.")
