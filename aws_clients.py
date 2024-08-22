# aws_clients.py
import boto3
import os


class AWSClients:
    _s3_client = None
    _bedrock_client = None

    @staticmethod
    def get_s3_client():
        if AWSClients._s3_client is None:
            AWSClients._s3_client = boto3.client(
                "s3",
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_REGION"),
            )
        return AWSClients._s3_client

    @staticmethod
    def get_bedrock_client():
        if AWSClients._bedrock_client is None:
            AWSClients._bedrock_client = boto3.client(
                "bedrock-runtime",
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_REGION"),
            )
        return AWSClients._bedrock_client
