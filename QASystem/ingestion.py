# Import necessary modules from langchain_community, langchain, and other libraries
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)  # To split text into manageable chunks
from langchain_community.vectorstores import (
    FAISS,
)  # For creating and managing a FAISS vector store
from langchain_aws import BedrockEmbeddings  # To generate embeddings using AWS Bedrock
import io
from aws_clients import AWSClients
from PyPDF2 import PdfReader

s3 = AWSClients.get_s3_client()
bedrock = AWSClients.get_bedrock_client()


# Create an instance of BedrockEmbeddings using the specified model and client
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0", client=bedrock
)


class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


# Function to ingest data from PDF files in a directory
def data_ingestion(bucket_name, file_key="latestdoc"):
    # Load PDF documents from the specified directory
    # loader = PyPDFDirectoryLoader("./data")
    # documents = loader.load()

    # # Split the loaded documents into chunks using a recursive character text splitter
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000)
    # docs = text_splitter.split_documents(documents)

    # # Return the split documents
    # return docs
    file_obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    file_content = file_obj["Body"].read()

    # Load the PDF using PyPDF2
    pdf_reader = PdfReader(io.BytesIO(file_content))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Create a document structure (assuming LangChain requires this format)
    documents = [Document(page_content=text, metadata={"source": file_key})]

    # Split the loaded document into chunks using a recursive character text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Return the split documents
    return docs


# Function to create and save a FAISS vector store from the provided documents
def get_vector_store(docs):
    # Create a FAISS vector store from the documents using the Bedrock embeddings
    vector_store_faiss = FAISS.from_documents(docs, bedrock_embeddings)

    # Save the vector store locally
    vector_store_faiss.save_local("faiss_index")

    # Return the created vector store
    return vector_store_faiss


# Main execution block
if __name__ == "__main__":
    # Ingest data from PDF files
    docs = data_ingestion("rag-app-live")

    # Create and save the FAISS vector store from the ingested documents
    get_vector_store(docs)
