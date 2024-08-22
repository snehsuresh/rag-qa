from setuptools import find_packages, setup

setup(
    name="rag-chatbot",
    version="0.0.1",
    author="sneh",
    author_email="snehsuresh02@gmail.com",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchainhub",
        "bs4",
        "tiktoken",
        "openai",
        "boto3",
        "langchain_community",
        "chromadb",
        "awscli",
        "streamlit",
        "pypdf",
        "faiss-cpu",
    ],
)
