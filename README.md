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

Step 1 - 4 is data ingestion
Step 5 is retieval
Step 6 is generation

`aws configure`



# Deployment
1. Create a Docker image of the app
2. Push the image to ECR instead of dockerhub
3. Consume it from ECR through any of the sercices like EC2, Lambda function, ECS, AppRunner etc..
4. Start an App Runner service and configure it to your streamlit service.

# CI/CD
1. On every change create a new image
2. Configure App Runner in such a way that it fetches image from EC2 and automatically deploys it.

# Doing the same with huggingface and sagemaker
1. You fetch an LLM like llama2 from huggigface
2. Push it to huggingface hub.
3. Deploy it to AWS Sagemaker Studio. (Create an instance)

# Sagemaker
1. You need to check your service quota for the required instance size for your llama model. 
2. Let's say you need ml.g5.12xlarge size instance. Request for it through Service Quora service of AWS. (Beware: Too pricey)
3. Then go back to sagemaker and you will be able to access it
4. Checkout https://huggingface.co/blog/sagemaker-huggingface-llm
5. After deployment, go to sagemaker homepage and check the deployment -> endpoint to find your deployed app.




