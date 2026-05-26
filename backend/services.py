# Model and API Services for Enterprise Engineering Knowledge Assistant
# File: backend/services.py

import os
import re
import logging
import numpy as np
from typing import List, Dict, Any, Generator
from openai import OpenAI

logger = logging.getLogger(__name__)

# Load keys from .env file in workspace root if present
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

# Try to fetch API key from environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if OPENAI_API_KEY.startswith("your_") or len(OPENAI_API_KEY) < 20:
    logger.info("OPENAI_API_KEY is empty or a placeholder. Forcing local simulation mode.")
    OPENAI_API_KEY = ""

class EmbeddingService:
    """
    Abstrated service to generate dense text embeddings.
    Defaults to OpenAI text-embedding-3-small (1536-dimensional).
    Falls back to a deterministic local hashing vector generator if no API key is present.
    """
    @classmethod
    def get_embedding(cls, text: str) -> List[float]:
        if not OPENAI_API_KEY:
            # Deterministic mock embedding generator for offline testing
            # Generates a pseudo-random unit vector of dimension 1536 based on text hash
            logger.warning("OPENAI_API_KEY not found in environment. Using Local Mock Embeddings.")
            return cls._generate_mock_embedding(text)
            
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.embeddings.create(
                input=[text],
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI Embedding API call failed: {str(e)}. Falling back to mock embeddings.")
            return cls._generate_mock_embedding(text)

    @classmethod
    def get_embeddings(cls, texts: List[str]) -> List[List[float]]:
        if not OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not found in environment. Using Local Mock Embeddings.")
            return [cls._generate_mock_embedding(t) for t in texts]
            
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.embeddings.create(
                input=texts,
                model="text-embedding-3-small"
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"OpenAI Multi-Embedding API call failed: {str(e)}")
            return [cls._generate_mock_embedding(t) for t in texts]

    @classmethod
    def _generate_mock_embedding(cls, text: str) -> List[float]:
        # Generates a 1536-dimension float vector that is consistent for identical texts
        # We seed standard numpy random using the hash of the text
        h = hash(text) & 0xffffffff
        rng = np.random.default_rng(h)
        vec = rng.standard_normal(1536)
        # Normalize to unit vector
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


class LLMService:
    """
    Abstracted service for LLM completions and streaming.
    Defaults to OpenAI gpt-4o-mini for technical reasoning.
    Falls back to a smart local rule-based response generator for offline POC.
    """
    @classmethod
    def chat_completion(cls, system_prompt: str, user_prompt: str) -> str:
        if not OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not found. Using offline response simulator.")
            return cls._generate_mock_completion(system_prompt, user_prompt)
            
        try:
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
    def chat_completion_stream(cls, system_prompt: str, user_prompt: str) -> Generator[str, None, None]:
        if not OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not found. Using offline streaming simulator.")
            yield from cls._generate_mock_completion_stream(system_prompt, user_prompt)
            return

        try:
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
