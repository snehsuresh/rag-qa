# Deploying RAG app using AWS(Lambda, ECR), Langchain, HuggingFace, Docker

## If on any given input promts if the LLM model is not able to generate output. You have 2 options
1. RAG
2. Finetuning

## Retrieval-augmented generation (RAG) - Few short prompting
1. Collect data from data scource and store in knowledge base
2. Create chunks of data 
3. Convert chunks into embeddings
4. Store embeddings to vector database
5. User query is sent to database, it does a similarity search, you find relevant response (context)
6. Along with promts imput, you also pass to the LLM model the relevant response.

### Step 1 till 4 is data ingestion
### Step 5 is retreival
### Step 6 is generation

`aws configure`

