# Import necessary modules
import boto3  # AWS SDK for Python, used to interact with AWS services
import json  # Provides methods for working with JSON data
import base64  # Provides methods for base64 encoding and decoding
import os  # Provides a way to interact with the operating system

# Define the prompt for image generation
prompt = """
provide me one 4k hd image of person who is standing over the mount everest peak.
"""

# Create a list containing the prompt and its weight
prompt_template = [{"text": prompt, "weight": 1}]

# Create a Boto3 client for the Bedrock runtime service
bedrock = boto3.client(service_name="bedrock-runtime")

# Prepare the payload with the necessary parameters
payload = {
    "text_prompts": prompt_template,  # The prompt for the image generation
    "cfg_scale": 10,  # Classifier-free guidance scale for controlling image quality and adherence to the prompt
    "seed": 0,  # Seed for reproducibility of the generated image
    "steps": 50,  # Number of steps for the image generation process
    "width": 512,  # Width of the generated image
    "height": 512  # Height of the generated image
}

# Convert the payload dictionary to a JSON string
body = json.dumps(payload)

# Specify the model ID to be used for generating the image
model_id = "stability.stable-diffusion-xl-v0"

# Invoke the model using the Bedrock client
response = bedrock.invoke_model(
    body=body,  # JSON string containing the payload
    modelId=model_id,  # ID of the model to be used
    accept="application/json",  # Expected response format (JSON)
    contentType="application/json"  # Content type of the request (JSON)
)

# Read the body of the response and parse it as JSON
response_body = json.loads(response.get("body").read())
print(response_body)

# Extract the encoded image data from the response
artifacts = response_body.get("artifacts")[0]
image_encoded = artifacts.get("base64").encode('utf-8')

# Decode the base64 encoded image data to bytes
image_bytes = base64.b64decode(image_encoded)

# Define the output directory for saving the generated image
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Define the file name for the generated image
file_name = f"{output_dir}/generated-img.png"

# Write the image bytes to a file
with open(file_name, "wb") as f:
    f.write(image_bytes)
