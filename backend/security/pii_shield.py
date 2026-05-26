# PII Protection and Security Shield for Enterprise Engineering Knowledge Assistant
# File: backend/security/pii_shield.py

import re
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)

# PII & Secret Patterns
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b')
PHONE_REGEX = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
CREDIT_CARD_REGEX = re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b')
# SSN/National ID Pattern
SSN_REGEX = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')

# Secret Keys (OpenAI API key, AWS secret, general passwords)
API_KEY_REGEX = re.compile(r'(?i)\b(?:api_key|secret|password|passwd|private_key|token|auth_token)\b\s*[:=]\s*["\']([a-zA-Z0-9_\-\.\+\/]{16,})["\']')
OPENAI_KEY_REGEX = re.compile(r'\b(?:sk-[a-zA-Z0-9]{32,48})\b')

class PIIShield:
    """
    Security module to sanitize technical chunks before embedding generation.
    Ensures absolute compliance with "no customer data embeddings" and "secure architecture" rules.
    """
    
    @classmethod
    def scan_and_censor(cls, text: str) -> Tuple[str, bool]:
        """
        Scans chunk text for standard PII patterns and secret keys.
        Returns (censored_text, pii_detected).
        """
        censored = text
        detected = False
        
        # 1. Emails
        if EMAIL_REGEX.search(censored):
            censored = EMAIL_REGEX.sub("[REDACTED_EMAIL]", censored)
            detected = True
            
        # 2. Phone Numbers
        if PHONE_REGEX.search(censored):
            censored = PHONE_REGEX.sub("[REDACTED_PHONE]", censored)
            detected = True
            
        # 3. Credit Cards
        if CREDIT_CARD_REGEX.search(censored):
            censored = CREDIT_CARD_REGEX.sub("[REDACTED_CREDIT_CARD]", censored)
            detected = True
            
        # 4. SSN
        if SSN_REGEX.search(censored):
            censored = SSN_REGEX.sub("[REDACTED_ID]", censored)
            detected = True
            
        # 5. OpenAI Keys
        if OPENAI_KEY_REGEX.search(censored):
            censored = OPENAI_KEY_REGEX.sub("[REDACTED_OPENAI_KEY]", censored)
            detected = True
            
        # 6. Generic Secrets
        if API_KEY_REGEX.search(censored):
            censored = API_KEY_REGEX.sub(lambda m: m.group(0).replace(m.group(1), "[REDACTED_SECRET]"), censored)
            detected = True
            
        if detected:
            logger.info("PII or Secret detected and redacted from engineering chunk.")
            
        return censored, detected

    @classmethod
    def is_customer_data_payload(cls, content: str) -> bool:
        """
        Heuristic filter to check if the chunk represents a raw customer database payload
        (e.g., transactional data, raw JSON records, customer logs) instead of technical system logic/schema.
        """
        content_lower = content.lower()
        
        # Heuristic 1: Raw user JSON lists (often found in transactional backups)
        # Check if there is an abundance of common user/customer transaction keys
        user_keys = ["first_name", "last_name", "credit_card", "card_number", "billing_address", "purchase_amount", "transaction_id", "email_address"]
        matches = sum(1 for key in user_keys if key in content_lower)
        if matches >= 3 and (content.strip().startswith("[") or content.strip().startswith("{")):
            logger.warning("Chunk identified as raw user transactional data payload. Blocking embedding.")
            return True
            
        # Heuristic 2: Excessive raw email dumps
        emails = EMAIL_REGEX.findall(content)
        if len(emails) > 5:
            logger.warning("Excessive email counts found in chunk. Classified as user dump. Blocking embedding.")
            return True
            
        return False
