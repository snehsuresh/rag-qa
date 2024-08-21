# Import necessary modules from langchain and other libraries
from langchain.chains import RetrievalQA  # For creating a retrieval-based QA system
from langchain_community.vectorstores import FAISS  # For managing a FAISS vector store
from langchain_community.llms.bedrock import Bedrock  # For using Bedrock models
import boto3  # AWS SDK for Python, used to interact with AWS services
import os
from langchain.prompts import PromptTemplate  # For creating prompt templates
from QASystem.ingestion import (
    get_vector_store,
    data_ingestion,
)  # Custom module imports for data ingestion and vector store creation
from langchain_aws import BedrockEmbeddings  # To generate embeddings using AWS Bedrock

access_key = os.getenv("AWS_ACCESS_KEY_ID")
secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

# Create a Boto3 client for the Bedrock runtime service
bedrock = boto3.client(
    service_name="bedrock-runtime",
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_access_key,
)


# Create an instance of BedrockEmbeddings using the specified model and client
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0", client=bedrock
)

# Define a prompt template for the QA system
prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least summarize with 
250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:
"""

# Create a PromptTemplate instance with the defined template and input variables
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


# Function to create and return a Bedrock LLM instance
def get_llama2_llm():
    llm = Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock)
    return llm


# Function to get a response from the LLM using the QA system
def get_response_llm(llm, vectorstore_faiss, query):
    print("===========>")
    # Create a RetrievalQA instance using the provided LLM, vector store, and prompt template
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},  # Retrieve top 3 similar documents
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    # Get the answer from the QA system for the provided query
    print("Getting query")
    answer = qa({"query": query})
    return answer["result"]


# Main execution block
if __name__ == "__main__":
    # Load the FAISS index from the local storage
    faiss_index = FAISS.load_local(
        "faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True
    )

    # Define the query to be answered by the QA system
    query = "What is RAG token?"

    # Get the Bedrock LLM instance
    llm = get_llama2_llm()

    # Get the response from the QA system and print it
    print(get_response_llm(llm, faiss_index, query))
