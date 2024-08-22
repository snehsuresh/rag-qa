from flask import Flask, request, render_template, redirect, url_for, flash
import os
import boto3
from langchain_aws import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from QASystem.ingestion import data_ingestion, get_vector_store
from QASystem.retrievalandgeneration import get_llama2_llm, get_response_llm
from aws_clients import AWSClients

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed for flash messages


S3_BUCKET = os.getenv("S3_BUCKET_NAME")
# Create a Boto3 client for the Bedrock runtime service
s3 = AWSClients.get_s3_client()
bedrock = AWSClients.get_bedrock_client()


# Create an instance of BedrockEmbeddings using the specified model and client
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0", client=bedrock
)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" in request.files:
            uploaded_files = request.files.getlist("file")
            for uploaded_file in uploaded_files:
                if uploaded_file.filename != "":
                    new_filename = "latestdoc"
                    # Upload the file to S3 with the new filename
                    s3.upload_fileobj(uploaded_file, S3_BUCKET, new_filename)
                    # file_path = os.path.join(
                #     app.config["UPLOAD_FOLDER"], uploaded_file.filename
                # )
                # uploaded_file.save(file_path)
            flash("Uploaded files successfully!", "success")

        if "update_vectors" in request.form:
            with app.app_context():
                docs = data_ingestion(S3_BUCKET)
                get_vector_store(docs)
                flash("Vector store updated successfully!", "success")

    return render_template("index.html")


@app.route("/ask", methods=["GET", "POST"])
def answer():
    print("Asked")
    if request.method == "POST":
        user_question = request.form["question"]
        faiss_index = FAISS.load_local(
            "faiss_index",
            bedrock_embeddings,
            allow_dangerous_deserialization=True,
        )
        print("faiss index", faiss_index)
        llm = get_llama2_llm()
        response = get_response_llm(llm, faiss_index, user_question)
        return render_template("index.html", response=response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=82)
