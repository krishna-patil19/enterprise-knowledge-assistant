"""List available Bedrock foundation models in the configured region."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

import boto3

region = os.getenv("AWS_REGION", "us-east-1")
aws_key = os.getenv("AWS_ACCESS_KEY_ID", "")
aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY", "")

print(f"Region: {region}")
print("=" * 70)

try:
    if aws_key and aws_secret and len(aws_key) > 10:
        client = boto3.client(
            "bedrock", region_name=region,
            aws_access_key_id=aws_key, aws_secret_access_key=aws_secret
        )
    else:
        client = boto3.client("bedrock", region_name=region)

    # List foundation models
    response = client.list_foundation_models()
    models = response.get("modelSummaries", [])
    
    # Filter for relevant models
    print(f"\nTotal models available: {len(models)}")
    
    print("\n--- EMBEDDING MODELS (Amazon Titan) ---")
    for m in models:
        if "titan" in m["modelId"].lower() and "embed" in m["modelId"].lower():
            print(f"  ID: {m['modelId']}")
            print(f"     Name: {m.get('modelName', 'N/A')}")
            print(f"     Status: {m.get('modelLifecycle', {}).get('status', 'N/A')}")
            print()
    
    print("--- LLM MODELS (Anthropic Claude) ---")
    for m in models:
        if "anthropic" in m["modelId"].lower() or "claude" in m["modelId"].lower():
            print(f"  ID: {m['modelId']}")
            print(f"     Name: {m.get('modelName', 'N/A')}")
            print(f"     Status: {m.get('modelLifecycle', {}).get('status', 'N/A')}")
            print()

    print("--- OTHER LLM OPTIONS (Amazon Nova, Meta Llama) ---")
    for m in models:
        mid = m["modelId"].lower()
        if ("nova" in mid or "llama" in mid or "mistral" in mid) and "embed" not in mid:
            print(f"  ID: {m['modelId']}")
            print(f"     Name: {m.get('modelName', 'N/A')}")
            print()

except Exception as e:
    print(f"[FAIL] Could not list models: {str(e)}")
    print("\nThis likely means your IAM user also lacks 'bedrock:ListFoundationModels' permission.")
    print("You need to add Bedrock permissions to your IAM user first.")
