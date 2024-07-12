# Import necessary modules
import json  # Provides methods for working with JSON data
import os  # Provides a way to interact with the operating system
import sys  # Provides access to some variables used or maintained by the interpreter
import boto3  # AWS SDK for Python, used to interact with AWS services
import streamlit as st  # Library for creating web applications

# Import necessary modules from langchain_community and langchain
from langchain_community.embeddings import BedrockEmbeddings  # To generate embeddings using AWS Bedrock
from langchain.llms.bedrock import Bedrock  # For using Bedrock models
from langchain.prompts import PromptTemplate  # For creating prompt templates
from langchain.chains import RetrievalQA  # For creating a retrieval-based QA system
from langchain.vectorstores import FAISS  # For managing a FAISS vector store

# Import custom modules for data ingestion and vector store creation
from QASystem.ingestion import data_ingestion, get_vector_store
# Import custom modules for LLM and response retrieval
from QASystem.retrievalandgeneration import get_llama2_llm, get_response_llm

# Create a Boto3 client for the Bedrock runtime service
bedrock = boto3.client(service_name="bedrock-runtime")

# Create an instance of BedrockEmbeddings using the specified model and client
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

# Main function to run the Streamlit application
def main():
    # Set the configuration for the Streamlit page
    st.set_page_config("QA with Doc")
    
    # Set the header for the Streamlit app
    st.header("QA with Doc using langchain and AWS Bedrock")
    
    # Create a text input box for the user to ask a question
    user_question = st.text_input("Ask a question from the PDF files")
    
    # Create a sidebar for additional options
    with st.sidebar:
        st.title("Update or Create the Vector Store")
        
        # Button to update the vector store
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                # Ingest data from PDF files and create/update the vector store
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")
                
        # Button to use the Llama model for QA
        if st.button("Llama Model"):
            with st.spinner("Processing..."):
                # Load the FAISS index from local storage
                faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                
                # Get the Bedrock LLM instance
                llm = get_llama2_llm()
                
                # Get the response from the QA system and display it
                st.write(get_response_llm(llm, faiss_index, user_question))
                st.success("Done")

# Execute the main function when the script is run
if __name__ == "__main__":
    # This is the main method
    main()
