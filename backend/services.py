# Model and API Services for Enterprise Engineering Knowledge Assistant
# File: backend/services.py
# Supports dual providers: AWS Bedrock (default) and OpenAI (fallback)

import os
import re
import json
import logging
import numpy as np
from typing import List, Dict, Any, Generator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load keys from .env file in workspace root if present
# ---------------------------------------------------------------------------
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(env_path):
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, _, val = line.partition("=")
                    val_cleaned = val.strip().strip('"').strip("'")
                    if key.strip() and val_cleaned:
                        os.environ[key.strip()] = val_cleaned
    except Exception as e:
        logger.warning(f"Failed to read .env file: {str(e)}")

# ---------------------------------------------------------------------------
# Provider Configuration
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "bedrock").lower()  # "bedrock" or "openai"

# OpenAI config
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if OPENAI_API_KEY.startswith("your_") or len(OPENAI_API_KEY) < 20:
    OPENAI_API_KEY = ""

# AWS Bedrock config
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
BEDROCK_LLM_MODEL = os.environ.get("BEDROCK_LLM_MODEL", "anthropic.claude-3-haiku-20240307")
BEDROCK_EMBED_MODEL = os.environ.get("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")

# Check if AWS credentials are available (explicit keys in .env OR AWS CLI profile)
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
_HAS_EXPLICIT_KEYS = bool(AWS_ACCESS_KEY and AWS_SECRET_KEY and len(AWS_ACCESS_KEY) > 10)

# Also check if AWS CLI profile exists (~/.aws/credentials)
_HAS_AWS_CLI_PROFILE = os.path.exists(os.path.expanduser("~/.aws/credentials"))

HAS_AWS_CREDS = _HAS_EXPLICIT_KEYS or _HAS_AWS_CLI_PROFILE

if _HAS_EXPLICIT_KEYS:
    logger.info(f"LLM Provider: {LLM_PROVIDER} | AWS Region: {AWS_REGION} | Auth: Explicit keys in .env")
elif _HAS_AWS_CLI_PROFILE:
    logger.info(f"LLM Provider: {LLM_PROVIDER} | AWS Region: {AWS_REGION} | Auth: AWS CLI profile (~/.aws/credentials)")
else:
    logger.info(f"LLM Provider: {LLM_PROVIDER} | AWS Region: {AWS_REGION} | Auth: No AWS credentials found")


def _get_bedrock_client(service_name: str = "bedrock-runtime"):
    """Creates a Boto3 Bedrock Runtime client.
    Uses explicit keys from .env if available, otherwise falls back to AWS CLI profile.
    """
    import boto3
    if _HAS_EXPLICIT_KEYS:
        return boto3.client(
            service_name=service_name,
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
        )
    else:
        # Use default credential chain (AWS CLI profile, env vars, IAM role, etc.)
        return boto3.client(
            service_name=service_name,
            region_name=AWS_REGION,
        )


# ===========================================================================
# Embedding Service
# ===========================================================================
class EmbeddingService:
    """
    Abstracted service to generate dense text embeddings.
    Supports: AWS Bedrock Titan Embed v2 (1024-dim) or OpenAI text-embedding-3-small (1536-dim).
    Falls back to a deterministic local hashing vector generator if no API credentials are present.
    """

    # Dimension depends on provider
    EMBEDDING_DIM = 1024 if LLM_PROVIDER == "bedrock" else 1536

    @classmethod
    def get_embedding(cls, text: str) -> List[float]:
        if LLM_PROVIDER == "bedrock" and HAS_AWS_CREDS:
            return cls._get_bedrock_embedding(text)
        elif LLM_PROVIDER == "openai" and OPENAI_API_KEY:
            return cls._get_openai_embedding(text)
        else:
            logger.warning("No API credentials found. Using Local Mock Embeddings.")
            return cls._generate_mock_embedding(text)

    @classmethod
    def get_embeddings(cls, texts: List[str]) -> List[List[float]]:
        # Bedrock Titan doesn't have a batch endpoint, so we loop
        return [cls.get_embedding(t) for t in texts]

    # --- Bedrock Titan Embed v2 ---
    @classmethod
    def _get_bedrock_embedding(cls, text: str) -> List[float]:
        try:
            client = _get_bedrock_client()
            body = json.dumps({
                "inputText": text[:8000],  # Titan v2 max input ~8k chars
                "dimensions": 1024,
                "normalize": True
            })
            response = client.invoke_model(
                modelId=BEDROCK_EMBED_MODEL,
                contentType="application/json",
                accept="application/json",
                body=body,
            )
            result = json.loads(response["body"].read())
            return result["embedding"]
        except Exception as e:
            logger.error(f"Bedrock Embedding API call failed: {str(e)}. Falling back to mock.")
            return cls._generate_mock_embedding(text)

    # --- OpenAI ---
    @classmethod
    def _get_openai_embedding(cls, text: str) -> List[float]:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            # OpenAI's limit is 8192 tokens (~30k chars). Safely truncate.
            safe_text = text[:24000]
            
            response = client.embeddings.create(
                input=[safe_text],
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI Embedding API call failed: {str(e)}. Falling back to mock.")
            return cls._generate_mock_embedding(text)

    # --- Mock (offline) ---
    @classmethod
    def _generate_mock_embedding(cls, text: str) -> List[float]:
        # Generates a consistent vector of the configured dimension based on text hash
        h = hash(text) & 0xffffffff
        rng = np.random.default_rng(h)
        vec = rng.standard_normal(cls.EMBEDDING_DIM)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


# ===========================================================================
# LLM Service
# ===========================================================================
class LLMService:
    """
    Abstracted service for LLM completions and streaming.
    Supports: AWS Bedrock Claude 3 Haiku or OpenAI GPT-4o-mini.
    Falls back to a smart local rule-based response generator for offline POC.
    """

    @classmethod
    def chat_completion(cls, system_prompt: str, user_prompt: str) -> str:
        if LLM_PROVIDER == "bedrock" and HAS_AWS_CREDS:
            return cls._bedrock_chat_completion(system_prompt, user_prompt)
        elif LLM_PROVIDER == "openai" and OPENAI_API_KEY:
            return cls._openai_chat_completion(system_prompt, user_prompt)
        else:
            logger.warning("No API credentials found. Using offline response simulator.")
            return cls._generate_mock_completion(system_prompt, user_prompt)

    @classmethod
    def chat_completion_stream(cls, system_prompt: str, user_prompt: str) -> Generator[str, None, None]:
        if LLM_PROVIDER == "bedrock" and HAS_AWS_CREDS:
            yield from cls._bedrock_chat_completion_stream(system_prompt, user_prompt)
        elif LLM_PROVIDER == "openai" and OPENAI_API_KEY:
            yield from cls._openai_chat_completion_stream(system_prompt, user_prompt)
        else:
            logger.warning("No API credentials found. Using offline streaming simulator.")
            yield from cls._generate_mock_completion_stream(system_prompt, user_prompt)

    # -----------------------------------------------------------------------
    # AWS Bedrock — Claude 3 Haiku
    # -----------------------------------------------------------------------
    @classmethod
    def _bedrock_chat_completion(cls, system_prompt: str, user_prompt: str) -> str:
        try:
            client = _get_bedrock_client()
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2048,
                "temperature": 0.1,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user_prompt}
                ]
            })
            response = client.invoke_model(
                modelId=BEDROCK_LLM_MODEL,
                contentType="application/json",
                accept="application/json",
                body=body,
            )
            result = json.loads(response["body"].read())
            return result["content"][0]["text"]
        except Exception as e:
            logger.error(f"Bedrock LLM Call failed: {str(e)}")
            return cls._generate_mock_completion(system_prompt, user_prompt)

    @classmethod
    def _bedrock_chat_completion_stream(cls, system_prompt: str, user_prompt: str) -> Generator[str, None, None]:
        try:
            client = _get_bedrock_client()
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2048,
                "temperature": 0.1,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user_prompt}
                ]
            })
            response = client.invoke_model_with_response_stream(
                modelId=BEDROCK_LLM_MODEL,
                contentType="application/json",
                accept="application/json",
                body=body,
            )
            for event in response["body"]:
                chunk = json.loads(event["chunk"]["bytes"])
                if chunk.get("type") == "content_block_delta":
                    delta = chunk.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield delta["text"]
        except Exception as e:
            logger.error(f"Bedrock LLM Stream failed: {str(e)}")
            yield f"\n[SYSTEM NOTICE: Bedrock API call failed: {str(e)}. Falling back to offline simulator...]\n"
            yield from cls._generate_mock_completion_stream(system_prompt, user_prompt)

    # -----------------------------------------------------------------------
    # OpenAI — GPT-4o-mini (fallback provider)
    # -----------------------------------------------------------------------
    @classmethod
    def _openai_chat_completion(cls, system_prompt: str, user_prompt: str) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI LLM Call failed: {str(e)}")
            return cls._generate_mock_completion(system_prompt, user_prompt)

    @classmethod
    def _openai_chat_completion_stream(cls, system_prompt: str, user_prompt: str) -> Generator[str, None, None]:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                stream=True
            )
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI LLM Stream failed: {str(e)}")
            yield f"\n[SYSTEM NOTICE: OpenAI API call failed: {str(e)}. Falling back to offline simulator...]\n"
            yield from cls._generate_mock_completion_stream(system_prompt, user_prompt)

    # -----------------------------------------------------------------------
    # Offline Mock (no credentials)
    # -----------------------------------------------------------------------
    @classmethod
    def _generate_mock_completion(cls, system_prompt: str, user_prompt: str) -> str:
        # Rules-based completion generator that reads the injected context to answer queries logically.
        # This makes the offline mode feel extremely interactive and realistic.
        context_matches = re.findall(r'\[Context Chunk \d+\]\s*(.*?)(?=\[Context Chunk|\Z)', user_prompt, re.DOTALL)
        
        # Analyze user query
        query = user_prompt.lower()
        
        response = "### Enterprise Knowledge Assistant (Offline Simulation Mode)\n\n"
        
        if "sumstate" in query or "aggregatingmergetree" in query:
            response += (
                "Based on the ClickHouse configurations and codebase:\n\n"
                "1. **`AggregatingMergeTree`** is utilized in the table `analytics.sales_aggregates_local` "
                "(defined in `analytics_queries.sql`). It is designed to optimize time-series transactional aggregates.\n"
                "2. **`sumState`** is defined inside the Materialized View `analytics.mv_sales_aggregates` "
                "to accumulate intermediate sums for the `revenue` and `units` columns:\n"
                "   ```sql\n"
                "   sumState(revenue) AS revenue_sum,\n"
                "   sumState(units) AS units_sum\n"
                "   ```\n"
                "3. In Python, the loader pipeline **`etl_pipeline.py`** interacts with this by query execution:\n"
                "   - It inserts data into `analytics.raw_sales` which triggers the materialized view.\n"
                "   - It aggregates reporting states using `sumMerge(revenue_sum)` and `uniqMerge(customers_uniq)`.\n\n"
                "Let me know if you would like me to show the code mappings!"
            )
        elif "pii" in query or "email" in query:
            response += (
                "The **PII Protection Scanner** detects and redacts customer sensitive data before embeddings are generated.\n"
                "It uses regex rules to scan for emails (`[^@]+@[^@]+\\.[^@]+`), phone numbers, and secrets/keys. "
                "If sensitive data is found, it is automatically redacted (e.g. `[REDACTED_EMAIL]`) to satisfy secure embedding rules."
            )
        else:
            response += (
                "Here is the relevant information retrieved from your engineering documentation:\n\n"
            )
            if context_matches:
                response += f"Retrieved {len(context_matches)} relevant chunks from your simulated S3 bucket. Here is a summary of the context:\n\n"
                for i, chunk in enumerate(context_matches[:3]):
                    first_lines = "\n".join(chunk.strip().splitlines()[:5])
                    response += f"**Context Block {i+1}:**\n```\n{first_lines}...\n```\n\n"
            else:
                response += "No direct database context chunks were found. Please make sure the ingestion pipeline has indexed your files."
                
        return response

    @classmethod
    def _generate_mock_completion_stream(cls, system_prompt: str, user_prompt: str) -> Generator[str, None, None]:
        full_text = cls._generate_mock_completion(system_prompt, user_prompt)
        # Split text into small words/sentences and yield slowly to simulate streaming
        import time
        words = full_text.split(" ")
        for i in range(0, len(words), 3):
            chunk = " ".join(words[i:i+3]) + " "
            time.sleep(0.02)
            yield chunk
