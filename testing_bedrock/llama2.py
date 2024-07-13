# Import necessary modules
import os  # Provides a way to interact with the operating system
import json  # Provides methods for working with JSON data
import sys  # Provides access to some variables used or maintained by the interpreter
import boto3  # AWS SDK for Python, used to interact with AWS services

# Print a message to confirm successful imports
print("imported successfully...")

# Define the prompt to be used as input for the model
prompt = """
        When will RCB win the IPL tournament?
"""

# Create a Boto3 client for the Bedrock runtime service
bedrock = boto3.client(service_name="bedrock-runtime")

# Prepare the payload with the necessary parameters
payload = {
    # Include the prompt in a specific format
    "prompt": "[INST]" + prompt + "[/INST]",
    # Maximum length of the generated response (in characters)
    "max_gen_len": 512,
    # Temperature controls the creativity of the response (lower values make the output more deterministic)
    "temperature": 0.3,
    # Top-p controls the sampling method (probability mass to consider)
    "top_p": 0.9
}

# Convert the payload dictionary to a JSON string
body = json.dumps(payload)

# Specify the model ID to be used for generating the response
model_id = "meta.llama2-70b-chat-v1"

# Invoke the model using the Bedrock client
response = bedrock.invoke_model(
    body=body,  # JSON string containing the payload
    modelId=model_id,  # ID of the model to be used
    accept="application/json",  # Expected response format (JSON)
    contentType="application/json"  # Content type of the request (JSON)
)

# Read the body of the response and parse it as JSON
response_body = json.loads(response.get("body").read())

# Extract the generated text from the response
response_text = response_body["generation"]

# Print the generated response
print(response_text)
