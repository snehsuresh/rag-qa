# Import necessary modules from langchain_community, langchain, and other libraries
from langchain_community.document_loaders import PyPDFDirectoryLoader  # To load PDF documents from a directory
from langchain.text_splitter import RecursiveCharacterTextSplitter  # To split text into manageable chunks
from langchain_community.vectorstores import FAISS  # For creating and managing a FAISS vector store
from langchain_community.embeddings import BedrockEmbeddings  # To generate embeddings using AWS Bedrock
from langchain_community.llms import Bedrock  # To interact with Bedrock models

import json  # Provides methods for working with JSON data
import os  # Provides a way to interact with the operating system
import sys  # Provides access to some variables used or maintained by the interpreter
import boto3  # AWS SDK for Python, used to interact with AWS services

# Create a Boto3 client for the Bedrock runtime service
bedrock = boto3.client(service_name="bedrock-runtime")

# Create an instance of BedrockEmbeddings using the specified model and client
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

# Function to ingest data from PDF files in a directory
def data_ingestion():
    # Load PDF documents from the specified directory
    loader = PyPDFDirectoryLoader("./data")
    documents = loader.load()
    
    # Split the loaded documents into chunks using a recursive character text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000)
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
if __name__ == '__main__':
    # Ingest data from PDF files
    docs = data_ingestion()
    
    # Create and save the FAISS vector store from the ingested documents
    get_vector_store(docs)
