"""Quick test to verify AWS Bedrock connectivity for both Embedding and LLM services."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("AWS BEDROCK CONNECTIVITY TEST")
print("=" * 60)

# --- Test 1: Embedding (Titan Embed v2) ---
print("\n[TEST 1] Embedding with Titan Embed v2...")
try:
    from backend.services import EmbeddingService
    embedding = EmbeddingService.get_embedding("Hello, this is a test query for ClickHouse documentation.")
    dim = len(embedding)
    print(f"  [OK] Embedding generated successfully! Dimension: {dim}")
    if dim == 1024:
        print("  [OK] Correct dimension for Bedrock Titan Embed v2 (1024)")
    else:
        print(f"  [WARN] Unexpected dimension: {dim} (expected 1024 for Bedrock)")
except Exception as e:
    print(f"  [FAIL] Embedding FAILED: {str(e)}")

# --- Test 2: LLM Completion (Claude 3 Haiku) ---
print("\n[TEST 2] LLM Completion with Claude 3 Haiku...")
try:
    from backend.services import LLMService
    response = LLMService.chat_completion(
        system_prompt="You are a helpful assistant. Reply in one sentence.",
        user_prompt="What is ClickHouse?"
    )
    print(f"  [OK] LLM responded successfully!")
    print(f"  Response: {response[:200]}...")
except Exception as e:
    print(f"  [FAIL] LLM Completion FAILED: {str(e)}")

# --- Test 3: LLM Streaming ---
print("\n[TEST 3] LLM Streaming with Claude 3 Haiku...")
try:
    tokens = []
    for token in LLMService.chat_completion_stream(
        system_prompt="You are a helpful assistant. Reply in one sentence.",
        user_prompt="What is a MergeTree engine?"
    ):
        tokens.append(token)
    full_response = "".join(tokens)
    print(f"  [OK] Streaming works! Received {len(tokens)} chunks.")
    print(f"  Response: {full_response[:200]}...")
except Exception as e:
    print(f"  [FAIL] LLM Streaming FAILED: {str(e)}")

print("\n" + "=" * 60)
print("ALL TESTS COMPLETE")
print("=" * 60)
