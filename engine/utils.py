



import os
from typing import Any

from sentence_transformers import SentenceTransformer

TEXT_DIR = "./text-data"
FAISS_DIR = "./FAISS-DB"
import faiss
import json

LOCAL_EMBED_MODEL_PATH = os.path.expanduser("D:\\sentence-transformers\\all-MiniLM-L6-v2")
DEFAULT_EMBED_MODEL = LOCAL_EMBED_MODEL_PATH


SUPPORTED_FIELDS = {
    "policy_id",
    "policy_type",
    "issue_date",
    "expiry_date",
    "policy_term",
    "nominee",
    "policyholder_details",
    "date_of_birth",
    "email",
    "phone",
    "nominee_details",
}

CANONICAL_HEADERS = {
    "policy_id": "Policy ID",
    "policy_type": "Policy Type",
    "issue_date": "Issue Date",
    "expiry_date": "Expiry Date",
    "policy_term": "Policy Term",
    "nominee": "Nominee",
    "policyholder_details": "Policyholder Details",
    "date_of_birth": "Date of Birth",
    "email": "Email",
    "phone": "Phone",
    "nominee_details": "Nominee Details",
}

RAW_REGEX_PATTERNS = {
    "policy_id": [
        r"Policy\s*ID\s*[:\-]\s*([A-Z0-9]{12})",
        r"\b([A-Z0-9]{12})\b",
    ],
    "policy_type": [
        r"Policy\s*Type\s*[:\-]\s*(.+)",
    ],
    "issue_date": [
        r"Issue\s*Date\s*[:\-]\s*(.+)",
    ],
    "expiry_date": [
        r"Expiry\s*Date\s*[:\-]\s*(.+)",
    ],
    "policy_term": [
        r"Policy\s*Term\s*[:\-]\s*(.+)",
    ],
    "nominee": [
        r"Nominee\s*[:\-]\s*(.+)",
    ],
    "policyholder_details": [
        r"Policyholder\s*Details\s*[:\-]\s*(.+)",
    ],
    "date_of_birth": [
        r"Date\s*of\s*Birth\s*[:\-]\s*(.+)",
        r"DOB\s*[:\-]\s*(.+)",
    ],
    "email": [
        r"Email\s*[:\-]\s*([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})",
    ],
    "phone": [
        r"Phone\s*[:\-]\s*([\+\d\-\(\)\s]{7,})",
        r"Mobile\s*[:\-]\s*([\+\d\-\(\)\s]{7,})",
    ],
    "nominee_details": [
        r"Nominee\s*Details\s*[:\-]\s*(.+)",
    ],
}


def metadata_path(name: str) -> str:
    return os.path.join(FAISS_DIR, name)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def load_embedding_model(model_path: str = DEFAULT_EMBED_MODEL, device: str = "cpu") -> SentenceTransformer:
    """
    Load local SentenceTransformer model from filesystem path, for example:
    ~/sentence-transformers/all-MiniLM-L6-v2
    """
    expanded_path = os.path.expanduser(model_path)

    if not os.path.exists(expanded_path):
        raise FileNotFoundError(
            f"Embedding model path not found: {expanded_path}\n"
            f"Expected local model directory such as: ~/sentence-transformers/all-MiniLM-L6-v2"
        )

    try:
        model = SentenceTransformer(expanded_path, device=device)
    except TypeError:
        # fallback for older sentence-transformers versions
        model = SentenceTransformer(expanded_path)

    return model

