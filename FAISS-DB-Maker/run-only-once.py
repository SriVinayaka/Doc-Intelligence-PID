



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integrated PDF -> text -> structured fields -> embeddings -> FAISS pipeline
for personal insurance policy PDFs.

Features:
- Reads PDFs from ./pdf-policy-documents/
- Saves raw extracted text into ./text-data/<index>.txt
- Uses ~/Qwen/Qwen2.5-VL-7B-Instruct/ for field extraction (optional but integrated)
- Uses local sentence-transformers model: ~/sentence-transformers/all-MiniLM-L6-v2
- Stores semantic embeddings in ./FAISS-DB/
- Supports exact lookup + semantic search

Tested conceptually for folder/file naming pattern:
<index>_<12 characters policy-ID>_<First Name>_<Last Name>.pdf
"""

import os
import re
import json
import math
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm

# ---------------------------
# External packages
# ---------------------------

from pypdf import PdfReader
import fitz  # PyMuPDF
from PIL import Image

import faiss
from sentence_transformers import SentenceTransformer

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# ============================================================
# CONFIG
# ============================================================

PDF_DIR = "./pdf-policy-documents"
TEXT_DIR = "./text-data"
FAISS_DIR = "./FAISS-DB"

QWEN_MODEL_PATH = os.path.expanduser("D:\\Qwen\\Qwen2.5-VL-7B-Instruct")
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


# ============================================================
# UTILS
# ============================================================

def ensure_dirs():
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(TEXT_DIR, exist_ok=True)
    os.makedirs(FAISS_DIR, exist_ok=True)

def parse_filename_metadata(filename: str) -> Dict[str, str]:
    """
    Expected:
    <index>_<12 chars policy-ID>_<First Name>_<Last Name>.pdf
    """
    base = os.path.basename(filename)
    m = re.match(r"^(\d+)_([A-Z0-9]{12})_([^_]+)_([^_]+)\.pdf$", base, re.IGNORECASE)
    if not m:
        return {
            "index": "",
            "policy_id_from_filename": "",
            "first_name": "",
            "last_name": "",
        }
    return {
        "index": m.group(1),
        "policy_id_from_filename": m.group(2),
        "first_name": m.group(3),
        "last_name": m.group(4),
    }

def clean_value(value: str) -> str:
    if not value:
        return ""
    value = value.strip()
    value = re.sub(r"\s+", " ", value)
    value = value.replace(" ,", ",")
    return value

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(end - overlap, 0)
    return chunks

def metadata_path(name: str) -> str:
    return os.path.join(FAISS_DIR, name)

def save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

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


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Basic text extraction for text-based PDFs.
    """
    text_parts = []
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text:
                text_parts.append(page_text)
    except Exception as e:
        print(f"[WARN] pypdf failed on {pdf_path}: {e}")
    return "\n".join(text_parts).strip()

def render_pdf_pages_to_images(pdf_path: str, max_pages: int = 2) -> List[Image.Image]:
    """
    Render first N pages as PIL images for Qwen2.5-VL.
    """
    images = []
    doc = fitz.open(pdf_path)
    try:
        for i in range(min(max_pages, len(doc))):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    finally:
        doc.close()
    return images


# ============================================================
# RAW REGEX EXTRACTION
# ============================================================

def regex_extract_fields(text: str, filename_meta: Dict[str, str]) -> Dict[str, str]:
    out = {
        "policy_id": filename_meta.get("policy_id_from_filename", "") or "",
        "policy_type": "",
        "issue_date": "",
        "expiry_date": "",
        "policy_term": "",
        "nominee": "",
        "policyholder_details": "",
        "date_of_birth": "",
        "email": "",
        "phone": "",
        "nominee_details": "",
    }

    for field, patterns in RAW_REGEX_PATTERNS.items():
        current = out.get(field, "")
        if current:
            continue
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
            if m:
                out[field] = clean_value(m.group(1))
                break

    # helpful fallback for policyholder details if name is in filename
    if not out["policyholder_details"]:
        name = f"{filename_meta.get('first_name', '')} {filename_meta.get('last_name', '')}".strip()
        if name:
            out["policyholder_details"] = name

    return out


# ============================================================
# QWEN2.5-VL LOADER + STRUCTURED EXTRACTION
# ============================================================

class QwenFieldExtractor:
    def __init__(self, model_path: str):
        self.model_path = os.path.expanduser(model_path)
        self.model = None
        self.processor = None
        self.device = "cpu"

    def load(self):
        if self.model is None or self.processor is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Qwen model path not found: {self.model_path}")

            self.processor = AutoProcessor.from_pretrained(self.model_path)

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32
            )
            self.model.to(self.device)
            self.model.eval()

    def extract_fields_from_images_and_text(
        self,
        images: List[Image.Image],
        raw_text: str,
        filename_meta: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Ask Qwen for structured JSON extraction using the first pages + raw text.
        """
        self.load()

        prompt = f"""
Extract the following insurance-policy fields from the provided policy document.
Return ONLY valid JSON with these exact keys:

{{
  "policy_id": "",
  "policy_type": "",
  "issue_date": "",
  "expiry_date": "",
  "policy_term": "",
  "nominee": "",
  "policyholder_details": "",
  "date_of_birth": "",
  "email": "",
  "phone": "",
  "nominee_details": ""
}}

Rules:
- If a value is not found, return an empty string.
- Keep values concise.
- policy_id should prefer a 12-character alphanumeric policy ID.
- Here is filename metadata that may help:
  index: {filename_meta.get("index", "")}
  policy_id_from_filename: {filename_meta.get("policy_id_from_filename", "")}
  first_name: {filename_meta.get("first_name", "")}
  last_name: {filename_meta.get("last_name", "")}

Raw extracted text:
{raw_text[:10000]}
""".strip()

        content = [{"type": "text", "text": prompt}]
        for img in images:
            content.insert(0, {"type": "image", "image": img})

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        text_input = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_input],
            images=images if images else None,
            padding=True,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )

        input_len = inputs["input_ids"].shape[1]
        trimmed = generated_ids[:, input_len:]

        output_text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        json_obj = self._safe_parse_json(output_text)
        if json_obj is None:
            return {}

        return {k: clean_value(str(json_obj.get(k, ""))) for k in CANONICAL_HEADERS.keys()}

    @staticmethod
    def _safe_parse_json(text: str):
        text = text.strip()

        try:
            return json.loads(text)
        except Exception:
            pass

        fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except Exception:
                pass

        obj_match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if obj_match:
            try:
                return json.loads(obj_match.group(1))
            except Exception:
                pass

        return None


# ============================================================
# FIELD MERGING
# ============================================================

def merge_fields(regex_fields: Dict[str, str], qwen_fields: Dict[str, str], filename_meta: Dict[str, str]) -> Dict[str, str]:
    merged = {}
    for key in CANONICAL_HEADERS.keys():
        merged[key] = clean_value(qwen_fields.get(key, "")) or clean_value(regex_fields.get(key, ""))

    # final fallback to filename policy id
    if not merged["policy_id"]:
        merged["policy_id"] = filename_meta.get("policy_id_from_filename", "") or ""

    # final fallback to policyholder details
    if not merged["policyholder_details"]:
        name = f"{filename_meta.get('first_name', '')} {filename_meta.get('last_name', '')}".strip()
        if name:
            merged["policyholder_details"] = name

    return merged


# ============================================================
# INDEX BUILDER
# ============================================================

def build_documents_for_embedding(fields: Dict[str, str], raw_text: str) -> List[str]:
    """
    Creates chunkable canonical text for FAISS embeddings.
    """
    header_text = "\n".join(
        f"{CANONICAL_HEADERS[k]}: {fields.get(k, '')}" for k in CANONICAL_HEADERS.keys()
    )
    joint = f"{header_text}\n\nFull Document Text:\n{raw_text}".strip()
    return chunk_text(joint, chunk_size=1200, overlap=200)

def build_index(use_qwen: bool = True, index_type: str = "ivf", embed_model_path: str = DEFAULT_EMBED_MODEL):
    ensure_dirs()

    pdf_files = sorted([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")])
    if not pdf_files:
        print(f"[ERROR] No PDFs found in {PDF_DIR}")
        return

    qwen_extractor = QwenFieldExtractor(QWEN_MODEL_PATH) if use_qwen else None

    print(f"[INFO] Loading embedding model from: {os.path.expanduser(embed_model_path)}")
    embed_model = load_embedding_model(embed_model_path, device="cpu")

    all_chunk_texts: List[str] = []
    all_chunk_meta: List[Dict[str, Any]] = []
    document_records: List[Dict[str, Any]] = []
    policy_lookup: Dict[str, Dict[str, Any]] = {}

    for filename in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(PDF_DIR, filename)
        filename_meta = parse_filename_metadata(filename)

        raw_text = extract_text_from_pdf(pdf_path)

        # save ./text-data/<index>.txt
        txt_index = filename_meta.get("index", "") or Path(filename).stem.split("_")[0]
        txt_path = os.path.join(TEXT_DIR, f"{txt_index}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(raw_text)

        regex_fields = regex_extract_fields(raw_text, filename_meta)

        qwen_fields = {}
        if use_qwen:
            try:
                images = render_pdf_pages_to_images(pdf_path, max_pages=2)
                qwen_fields = qwen_extractor.extract_fields_from_images_and_text(
                    images=images,
                    raw_text=raw_text,
                    filename_meta=filename_meta
                )
            except Exception as e:
                print(f"[WARN] Qwen extraction failed for {filename}: {e}")
                qwen_fields = {}

        fields = merge_fields(regex_fields, qwen_fields, filename_meta)

        record = {
            "filename": filename,
            "pdf_path": pdf_path,
            "text_path": txt_path,
            "index": filename_meta.get("index", ""),
            **fields,
        }
        document_records.append(record)

        if record["policy_id"]:
            policy_lookup[record["policy_id"]] = record

        chunk_texts = build_documents_for_embedding(fields, raw_text)
        for i, chunk in enumerate(chunk_texts):
            all_chunk_texts.append(chunk)
            all_chunk_meta.append({
                "filename": filename,
                "chunk_id": i,
                "index": record["index"],
                "policy_id": record["policy_id"],
                "text_path": txt_path,
                **fields,
            })

    if not all_chunk_texts:
        print("[ERROR] No text chunks generated.")
        return

    print("[INFO] Generating embeddings...")
    embeddings = embed_model.encode(
        all_chunk_texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    dim = embeddings.shape[1]
    n = embeddings.shape[0]

    print(f"[INFO] Total chunks: {n}, embedding dim: {dim}")

    if index_type.lower() == "ivf" and n >= 100:
        nlist = max(10, min(100, int(math.sqrt(n))))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        try:
            index.nprobe = min(16, nlist)
        except Exception:
            pass
        index_kind = "ivf"
    else:
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        index_kind = "flat"

    faiss.write_index(index, metadata_path("index.faiss"))
    save_json(metadata_path("chunk_metadata.json"), all_chunk_meta)
    save_json(metadata_path("document_records.json"), document_records)
    save_json(metadata_path("policy_lookup.json"), policy_lookup)

    config = {
        "pdf_dir": PDF_DIR,
        "text_dir": TEXT_DIR,
        "faiss_dir": FAISS_DIR,
        "qwen_model_path": QWEN_MODEL_PATH,
        "embed_model": os.path.expanduser(embed_model_path),
        "index_type": index_kind,
        "total_documents": len(document_records),
        "total_chunks": len(all_chunk_texts),
    }
    save_json(metadata_path("config.json"), config)

    print("\n[SUCCESS] Index built successfully.")
    print(f"FAISS index: {metadata_path('index.faiss')}")
    print(f"Chunk metadata: {metadata_path('chunk_metadata.json')}")
    print(f"Document records: {metadata_path('document_records.json')}")
    print(f"Policy lookup: {metadata_path('policy_lookup.json')}")


# ============================================================
# QUERY ENGINE
# ============================================================

class PolicySearchEngine:
    def __init__(self, embed_model_path: str = ""):
        self.index = faiss.read_index(metadata_path("index.faiss"))
        self.chunk_metadata = load_json(metadata_path("chunk_metadata.json"))
        self.document_records = load_json(metadata_path("document_records.json"))
        self.policy_lookup = load_json(metadata_path("policy_lookup.json"))

        config_path = metadata_path("config.json")
        config = load_json(config_path) if os.path.exists(config_path) else {}
        saved_model_path = config.get("embed_model", DEFAULT_EMBED_MODEL)

        self.embed_model_path = embed_model_path or saved_model_path
        print(f"[INFO] Loading embedding model from: {os.path.expanduser(self.embed_model_path)}")
        self.embed_model = load_embedding_model(self.embed_model_path, device="cpu")

    def exact_policy_lookup(self, policy_id: str) -> Dict[str, Any]:
        return self.policy_lookup.get(policy_id, {})

    def exact_field_lookup(self, field: str, value: str) -> List[Dict[str, Any]]:
        field = field.strip().lower()
        if field not in SUPPORTED_FIELDS:
            raise ValueError(f"Unsupported field: {field}")

        value_norm = value.strip().lower()
        results = []
        seen = set()

        for doc in self.document_records:
            candidate = str(doc.get(field, "")).strip().lower()
            if value_norm and value_norm in candidate:
                key = doc.get("filename")
                if key not in seen:
                    seen.add(key)
                    results.append(doc)

        return results

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q_vec = self.embed_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(q_vec, top_k)
        results = []

        seen = set()
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue

            meta = self.chunk_metadata[idx]
            fname = meta.get("filename")
            if fname in seen:
                continue
            seen.add(fname)

            doc = self._get_document_record(fname)
            if doc:
                doc_copy = dict(doc)
                doc_copy["_score"] = float(score)
                results.append(doc_copy)

        return results

    def hybrid_query(self, query: str, field: str = "", value: str = "", top_k: int = 5) -> Dict[str, Any]:
        """
        Strategy:
        1. If exact policy_id supplied -> fast direct lookup
        2. If field+value supplied -> exact field filter
        3. Always include semantic fallback
        """
        search_text = (query or value or "").strip()

        result = {
            "exact_policy": None,
            "field_matches": [],
            "semantic_matches": [],
        }

        if field == "policy_id" and value:
            exact = self.exact_policy_lookup(value)
            if exact:
                result["exact_policy"] = exact

        if field and value and field in SUPPORTED_FIELDS:
            result["field_matches"] = self.exact_field_lookup(field, value)

        if search_text:
            result["semantic_matches"] = self.semantic_search(query=search_text, top_k=top_k)

        return result

    def _get_document_record(self, filename: str) -> Dict[str, Any]:
        for doc in self.document_records:
            if doc.get("filename") == filename:
                return doc
        return {}

    @staticmethod
    def pretty_print(results: Dict[str, Any]):
        print("=" * 90)

        if results.get("exact_policy"):
            print("EXACT POLICY MATCH")
            print("-" * 90)
            print(json.dumps(results["exact_policy"], indent=2, ensure_ascii=False))
            print("=" * 90)

        if results.get("field_matches"):
            print("FIELD MATCHES")
            print("-" * 90)
            for i, item in enumerate(results["field_matches"], start=1):
                print(f"[{i}] {item.get('filename')}")
                print(json.dumps(item, indent=2, ensure_ascii=False))
                print("-" * 90)

        if results.get("semantic_matches"):
            print("SEMANTIC MATCHES")
            print("-" * 90)
            for i, item in enumerate(results["semantic_matches"], start=1):
                print(f"[{i}] {item.get('filename')} | score={item.get('_score', 0):.4f}")
                print(json.dumps(item, indent=2, ensure_ascii=False))
                print("-" * 90)

        if not results.get("exact_policy") and not results.get("field_matches") and not results.get("semantic_matches"):
            print("No matches found.")
            print("-" * 90)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Policy PDF -> Qwen -> FAISS pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build
    p_build = subparsers.add_parser("build", help="Build FAISS index from PDFs")
    p_build.add_argument("--no-qwen", action="store_true", help="Disable Qwen structured extraction")
    p_build.add_argument("--index-type", choices=["flat", "ivf"], default="ivf", help="FAISS index type")
    p_build.add_argument(
        "--embed-model",
        type=str,
        default=DEFAULT_EMBED_MODEL,
        help="Local path to SentenceTransformer model (default: ~/sentence-transformers/all-MiniLM-L6-v2)"
    )

    # query
    p_query = subparsers.add_parser("query", help="Query the local FAISS/metadata DB")
    p_query.add_argument("--query", type=str, default="", help="General natural-language query")
    p_query.add_argument("--field", type=str, default="", help="Structured field name")
    p_query.add_argument("--value", type=str, default="", help="Value for structured field")
    p_query.add_argument("--top-k", type=int, default=5, help="Top semantic matches")
    p_query.add_argument(
        "--embed-model",
        type=str,
        default="",
        help="Optional override for local embedding model path"
    )

    args = parser.parse_args()

    if args.command == "build":
        build_index(
            use_qwen=not args.no_qwen,
            index_type=args.index_type,
            embed_model_path=args.embed_model
        )

    elif args.command == "query":
        engine = PolicySearchEngine(embed_model_path=args.embed_model)
        results = engine.hybrid_query(
            query=args.query,
            field=args.field.lower().strip(),
            value=args.value.strip(),
            top_k=args.top_k
        )
        engine.pretty_print(results)


if __name__ == "__main__":
    main()




