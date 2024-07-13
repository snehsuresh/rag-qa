# Deploying RAG App Using AWS (Lambda, ECR), Langchain, Docker

## Introduction

This project demonstrates deploying a Retrieval-Augmented Generation (RAG) application using AWS services such as Lambda and ECR, along with Langchain, HuggingFace, and Docker. The RAG approach enhances the capabilities of large language models (LLMs) by integrating a retrieval mechanism that fetches relevant information from a knowledge base to provide more accurate responses.

## Handling LLM Inability to Generate Output

If the LLM model cannot generate an output for a given input prompt, you have two options:
1. Retrieval-Augmented Generation (RAG)
2. Finetuning

## Retrieval-Augmented Generation (RAG) - Few-shot Prompting

1. **Collect data from data sources and store it in a knowledge base.**
2. **Create chunks of data.**
3. **Convert chunks into embeddings.**
4. **Store embeddings in a vector database.**
5. **User query is sent to the database, performs a similarity search, and finds the relevant response (context).**
6. **Pass the relevant response along with the prompt input to the LLM model.**

### Steps Breakdown
- **Step 1-4**: Data Ingestion
- **Step 5**: Retrieval
- **Step 6**: Generation

### AWS Configuration
To start, configure your AWS CLI:
```sh
aws configure
```

## Deployment
1. **Create a Docker Image of the App**
2. **Push the Image to ECR**
3. **Consume the Image from ECR**
4. **Use services like EC2, Lambda Function, ECS, AppRunner, etc.**
5. **Start an App Runner Service.**
6. **Configure it to run your Streamlit service.**

## CI/CD
### Automate Image Creation
**On every git push, update the image in the ECR.**

### Configure App Runner
**Set it up to fetch the latest image from ECR and automatically deploy it.**

## Using HuggingFace and SageMaker
1. **Fetch an LLM from HuggingFace (Example: Llama2)**
2. **Push the Model to HuggingFace's Model Hub.**
3. **Deploy the Model to AWS SageMaker Studio.**
4. **Create an instance and deploy the model.**

### SageMaker Deployment
1. **Service Quota Check**: Check your service quota for the required instance size (e.g., ml.g5.12xlarge) for your Llama model.
2. **Request the required instance size** through AWS Service Quota service (Note: Can be expensive).
3. **After approval, access the instance in SageMaker.**

#### Deployment Process
1. **Follow the steps outlined in the HuggingFace blog on SageMaker deployment.**
2. **Check Deployment Endpoint**
3. **Go to the SageMaker homepage, navigate to Deployment -> Endpoint to find your deployed application.**
