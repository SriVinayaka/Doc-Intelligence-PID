



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Policy Search Engine

Supports exact-first unique search for:
01. Policy ID
02. Policy Type
03. Issue Date
04. Expiry Date
05. Policy Term
06. Nominee
07. Policyholder Details
08. Date of Birth
09. Email
10. Phone
11. Nominee Details

Uses semantic search only as a fallback for non-structured, broad queries.
"""

import os
import re
import json
import ast
from typing import Dict, Any, List, Tuple

import faiss

import phonenumbers
from phonenumbers import NumberParseException, PhoneNumberFormat, region_code_for_number

# Primary/default region for local numbers without '+'
PHONE_DEFAULT_REGION = "IN"

# Heuristic candidate regions for non-E.164 numbers when region is unknown.
# This part is my suggested fallback strategy, not something libphonenumber does automatically.
PHONE_REGION_CANDIDATES = (
    "IN", "US", "GB", "AE", "SG", "AU", "CA", "DE", "FR"
)



try:
    from engine.utils import (
        DEFAULT_EMBED_MODEL,
        SUPPORTED_FIELDS,
        metadata_path,
        load_json,
        load_embedding_model,
    )
except ImportError:
    from .utils import (
        DEFAULT_EMBED_MODEL,
        SUPPORTED_FIELDS,
        metadata_path,
        load_json,
        load_embedding_model,
    )


FIELD_QUERY_ALIASES = {
    "policy_id": [
        "policy id",
        "policy-id",
        "policyid",
        "id",
    ],
    "policy_type": [
        "policy type",
        "type",
    ],
    "issue_date": [
        "issue date",
        "issued on",
        "issued date",
    ],
    "expiry_date": [
        "expiry date",
        "expiration date",
        "expires on",
        "expiry",
    ],
    "policy_term": [
        "policy term",
        "term",
    ],
    "nominee": [
        "nominee",
    ],
    "policyholder_details": [
        "policyholder details",
        "policy holder details",
        "policyholder",
        "policy holder",
        "holder",
        "policy for",
    ],
    "date_of_birth": [
        "date of birth",
        "dob",
        "birth date",
    ],
    "email": [
        "email",
        "mail",
    ],
    "phone": [
        "phone",
        "mobile",
        "contact",
        "phone number",
        "mobile number",
    ],
    "nominee_details": [
        "nominee details",
        "nominee detail",
        "relationship",
    ],
}


class PolicySearchEngine:
    def __init__(self, embed_model_path: str = ""):
        self._validate_required_files()

        self.index = faiss.read_index(metadata_path("index.faiss"))
        self.chunk_metadata = load_json(metadata_path("chunk_metadata.json"))
        self.document_records = load_json(metadata_path("document_records.json"))
        self.policy_lookup = load_json(metadata_path("policy_lookup.json"))

        config_path = metadata_path("config.json")
        config = load_json(config_path) if os.path.exists(config_path) else {}
        saved_model_path = config.get("embed_model", DEFAULT_EMBED_MODEL)

        self.embed_model_path = os.path.expanduser(embed_model_path or saved_model_path)
        print(f"[INFO] Loading embedding model from: {self.embed_model_path}")
        self.embed_model = load_embedding_model(self.embed_model_path, device="cpu")

    # ============================================================
    # Validation
    # ============================================================

    @staticmethod
    def _required_paths() -> List[str]:
        return [
            metadata_path("index.faiss"),
            metadata_path("chunk_metadata.json"),
            metadata_path("document_records.json"),
            metadata_path("policy_lookup.json"),
        ]

    def _validate_required_files(self):
        missing = [p for p in self._required_paths() if not os.path.exists(p)]
        if missing:
            missing_text = "\n".join(f" - {p}" for p in missing)
            raise FileNotFoundError(
                "PolicySearchEngine initialisation failed because required files are missing:\n"
                f"{missing_text}\n\n"
                "Please build the index first."
            )

    # ============================================================
    # Normalisation helpers
    # ============================================================

    @staticmethod
    def _clean_text(text: Any) -> str:
        text = str(text or "").strip()
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _normalise_for_match(text: Any) -> str:
        text = str(text or "").lower()
        text = text.replace("_", " ")
        text = re.sub(r'["“”\'`]', "", text)
        text = re.sub(r"[^a-z0-9@\+\-\.\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _safe_parse_dict(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value

        if not value or not isinstance(value, str):
            return {}

        text = value.strip()
        if not text:
            return {}

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        return {}

    @staticmethod
    def _extract_name_from_filename(filename: str) -> str:
        """
        Expected format:
        <index>_<policyid>_<First>_<Last>.pdf
        """
        base = os.path.basename(filename or "")
        m = re.match(r"^\d+_[A-Z0-9]{12}_([^_]+)_([^_]+)\.pdf$", base, re.IGNORECASE)
        if not m:
            return ""
        return f"{m.group(1)} {m.group(2)}".strip()

    @staticmethod
    def _dedupe_by_filename(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        unique = []
        for item in items:
            fname = item.get("filename", "")
            if fname and fname not in seen:
                seen.add(fname)
                unique.append(item)
        return unique

    # ============================================================
    # Phone helpers (STRICT FIX)
    # ============================================================

    def _extract_first_phone_like(self, text: str) -> str:
        """
        Extract a phone-like fragment from free text, preserving + if present.
        Examples matched:
        +91 7722170430
        +917722170430
        7722170430
        091-7722170430
        """
        text = self._clean_text(text)
        if not text:
            return ""

        m = re.search(r"(\+?\d[\d\-\s\(\)]{7,}\d)", text)
        if m:
            return m.group(1).strip()

        # fallback if already mostly numeric
        if re.fullmatch(r"\+?\d[\d\-\s\(\)]{7,}", text):
            return text.strip()

        return ""

    def _normalize_phone(self, phone: str) -> str:
        """
        Canonical numeric phone representation.

        Examples:
        +91 7722170430  -> 917722170430
        +917722170430   -> 917722170430
        7722170430      -> 917722170430
        0917722170430   -> 917722170430
        """
        phone = self._extract_first_phone_like(phone)

        # keep digits only
        digits = re.sub(r"\D", "", str(phone or ""))

        if not digits:
            return ""

        # remove all leading zeros
        digits = digits.lstrip("0")

        # indian canonicalisation:
        # 10 digits -> prefix 91
        if len(digits) == 10:
            digits = "91" + digits

        # 12 digits already starting 91 -> keep
        # other lengths -> keep as-is after cleanup
        return digits

    # ============================================================
    # Parsed field helpers
    # ============================================================

    def _policyholder_parsed(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        return self._safe_parse_dict(doc.get("policyholder_details", ""))

    def _nominee_parsed(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        return self._safe_parse_dict(doc.get("nominee_details", ""))

    def _policyholder_full_name(self, doc: Dict[str, Any]) -> str:
        parsed = self._policyholder_parsed(doc)
        full_name = self._clean_text(parsed.get("full_name", ""))
        if full_name:
            return full_name

        filename_name = self._extract_name_from_filename(doc.get("filename", ""))
        if filename_name:
            return filename_name

        raw = self._clean_text(doc.get("policyholder_details", ""))
        if raw and "full_name" not in raw.lower():
            return raw

        return ""

    def _nominee_name(self, doc: Dict[str, Any]) -> str:
        parsed = self._nominee_parsed(doc)
        nominee_name = self._clean_text(parsed.get("nominee_name", ""))
        if nominee_name:
            return nominee_name

        nominee = self._clean_text(doc.get("nominee", ""))
        if nominee:
            return nominee

        raw = self._clean_text(doc.get("nominee_details", ""))
        return raw

    def _get_email_value(self, doc: Dict[str, Any]) -> str:
        parsed = self._policyholder_parsed(doc)

        email = self._clean_text(doc.get("email", ""))
        if email:
            return email

        email = self._clean_text(parsed.get("email", ""))
        if email:
            return email

        return ""

    def _get_phone_value(self, doc: Dict[str, Any]) -> str:
        """
        Strict phone extraction only from phone fields.
        No concatenation.
        """
        parsed = self._policyholder_parsed(doc)

        phone = self._clean_text(doc.get("phone", ""))
        if phone:
            return phone

        phone = self._clean_text(parsed.get("phone", ""))
        if phone:
            return phone

        return ""



    # ============================================================
    # Phone helpers - libphonenumber based newly added
    # ============================================================

    def _clean_phone_input(self, phone: str) -> str:
        """
        Light cleanup before parsing. Preserve '+' if present.
        """
        phone = str(phone or "").strip()
        phone = phone.replace("tel:", "").strip()
        phone = re.sub(r"\s+", " ", phone)
        return phone

    def _phone_parse_candidates(self, phone: str) -> List[phonenumbers.PhoneNumber]:
        """
        Parse a phone number into one or more valid candidates.

        Rules:
        - If input starts with '+', parse with region=None (globally unique style).
        - If input starts with '00', convert to '+' and parse with region=None.
        - Otherwise:
            1. try PHONE_DEFAULT_REGION
            2. try PHONE_REGION_CANDIDATES as a heuristic fallback
        - Keep only numbers that are possible/valid enough for comparison.
        """
        phone = self._clean_phone_input(phone)
        if not phone:
            return []

        candidates: List[phonenumbers.PhoneNumber] = []
        seen_e164 = set()

        def _add_if_good(parsed):
            try:
                if parsed is None:
                    return
                # Prefer valid numbers; allow possible numbers as a fallback
                if not (phonenumbers.is_valid_number(parsed) or phonenumbers.is_possible_number(parsed)):
                    return
                e164 = phonenumbers.format_number(parsed, PhoneNumberFormat.E164)
                if e164 not in seen_e164:
                    seen_e164.add(e164)
                    candidates.append(parsed)
            except Exception:
                return

        # Case 1: explicit international number with '+'
        if phone.startswith("+"):
            try:
                parsed = phonenumbers.parse(phone, None)
                _add_if_good(parsed)
            except NumberParseException:
                pass
            return candidates

        # Case 2: international prefix '00...' -> convert to '+...'
        digits_only = re.sub(r"\D", "", phone)
        if digits_only.startswith("00") and len(digits_only) > 2:
            maybe_plus = "+" + digits_only[2:]
            try:
                parsed = phonenumbers.parse(maybe_plus, None)
                _add_if_good(parsed)
            except NumberParseException:
                pass
            return candidates

        # Case 3: local/national format -> need a region
        # First try the configured default region
        try:
            parsed = phonenumbers.parse(phone, PHONE_DEFAULT_REGION)
            _add_if_good(parsed)
        except NumberParseException:
            pass

        # Heuristic fallback across candidate regions
        # This is a suggested strategy for your app logic.
        for region in PHONE_REGION_CANDIDATES:
            try:
                parsed = phonenumbers.parse(phone, region)
                _add_if_good(parsed)
            except NumberParseException:
                continue

        return candidates

    def _normalise_phone_for_compare(self, phone: str) -> str:
        """
        Return the best canonical comparison form.

        We use E.164 without punctuation, e.g.:
        +91 9823104620 -> +919823104620
        """
        candidates = self._phone_parse_candidates(phone)
        if not candidates:
            return ""

        # Return first candidate's E.164
        try:
            return phonenumbers.format_number(candidates[0], PhoneNumberFormat.E164)
        except Exception:
            return ""

    def _phone_equivalence_set(self, phone: str) -> set:
        """
        Build a set of canonical representations for matching.

        Includes:
        - E.164 (+919823104620)
        - digits-only E.164 (919823104620)
        - INTERNATIONAL (+91 9823104620)
        - NATIONAL formatting if available
        - national significant number digits only
        """
        out = set()
        for parsed in self._phone_parse_candidates(phone):
            try:
                e164 = phonenumbers.format_number(parsed, PhoneNumberFormat.E164)
                intl = phonenumbers.format_number(parsed, PhoneNumberFormat.INTERNATIONAL)
                nat = phonenumbers.format_number(parsed, PhoneNumberFormat.NATIONAL)
                nsn = str(parsed.national_number)

                out.add(e164)
                out.add(re.sub(r"\D", "", e164))
                out.add(intl)
                out.add(re.sub(r"\D", "", intl))
                out.add(nat)
                out.add(re.sub(r"\D", "", nat))
                out.add(nsn)
            except Exception:
                continue
        return {x for x in out if x}

    def _format_phone_display(self, phone: str) -> str:
        """
        Pretty display format, e.g. '+91 9823104620' when possible.
        """
        candidates = self._phone_parse_candidates(phone)
        if not candidates:
            return self._clean_text(phone)

        try:
            return phonenumbers.format_number(candidates[0], PhoneNumberFormat.INTERNATIONAL)
        except Exception:
            return self._clean_text(phone)

    def _detect_country_for_phone(self, phone: str) -> str:
        """
        Return ISO region code like 'IN', 'GB', 'US' when resolvable.
        """
        candidates = self._phone_parse_candidates(phone)
        if not candidates:
            return ""
        try:
            region = region_code_for_number(candidates[0])
            return region or ""
        except Exception:
            return ""

    def _get_phone_value(self, doc: Dict[str, Any]) -> str:
        """
        Strict phone extraction only from phone-related fields.
        """
        parsed = self._policyholder_parsed(doc)

        # top-level field first
        phone = self._clean_text(doc.get("phone", ""))
        if phone:
            return phone

        # fallback inside policyholder_details
        phone = self._clean_text(parsed.get("phone", ""))
        if phone:
            return phone

        return ""



    # ============================================================
    # Field search text (STRICT ISOLATION)
    # ============================================================

    def _get_field_search_text(self, doc: Dict[str, Any], field: str) -> str:
        if field == "policy_id":
            return self._clean_text(doc.get("policy_id", ""))

        if field == "policy_type":
            return self._clean_text(doc.get("policy_type", ""))

        if field == "issue_date":
            return self._clean_text(doc.get("issue_date", ""))

        if field == "expiry_date":
            return self._clean_text(doc.get("expiry_date", ""))

        if field == "policy_term":
            return self._clean_text(doc.get("policy_term", ""))

        if field == "nominee":
            return self._nominee_name(doc)

        if field == "policyholder_details":
            return self._policyholder_full_name(doc)

        if field == "date_of_birth":
            parsed = self._policyholder_parsed(doc)
            return self._clean_text(doc.get("date_of_birth", "")) or self._clean_text(parsed.get("date_of_birth", ""))

        if field == "email":
            return self._get_email_value(doc)

        if field == "phone":
            return self._get_phone_value(doc)

        if field == "nominee_details":
            parsed = self._nominee_parsed(doc)
            values = [
                self._clean_text(parsed.get("nominee_name", "")),
                self._clean_text(parsed.get("relationship", "")),
                self._clean_text(doc.get("nominee_details", "")),
                self._clean_text(doc.get("nominee", "")),
            ]
            return " | ".join(v for v in values if v)

        return self._clean_text(doc.get(field, ""))

    # ============================================================
    # Query parsing
    # ============================================================

    def _normalise_user_query(self, query: str) -> str:
        q = self._clean_text(query)
        q = re.sub(r"^\s*search\s*[:\-]?\s*", "", q, flags=re.IGNORECASE)
        return q.strip()

    def _infer_field_and_value_from_query(self, query: str) -> Tuple[str, str]:
        """
        Examples:
        - policy id QODPXNF7C3DD
        - policy holder "Ucchal Pillay"
        - email oniswamy@example.org
        - phone +917722170430
        """
        q = self._normalise_user_query(query).strip()

        # Keep the original for phone/email parsing, but make a stripped version for aliases
        q_unquoted = q.strip().strip('"').strip("'").strip()

        # Alias-based prefix parsing
        for field, aliases in FIELD_QUERY_ALIASES.items():
            for alias in sorted(aliases, key=len, reverse=True):
                pattern = rf"^{re.escape(alias)}\s+(.+)$"
                m = re.match(pattern, q_unquoted, flags=re.IGNORECASE)
                if m:
                    value = self._clean_text(m.group(1)).strip('"').strip("'").strip()
                    if value:
                        return field, value

        # Policy ID by shape
        m = re.search(r"\b([A-Z0-9]{12})\b", q, flags=re.IGNORECASE)
        if m:
            return "policy_id", m.group(1)

        # Email by shape
        m = re.search(r"([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})", q)
        if m:
            return "email", m.group(1)

        # Phone by query prefix
        m = re.match(r"^(?:phone|mobile|contact|phone number|mobile number)\s+(.+)$", q, flags=re.IGNORECASE)
        if m:
            return "phone", self._clean_text(m.group(1))

        # Phone as standalone query
        phone_candidate = self._extract_first_phone_like(q)
        if phone_candidate:
            return "phone", phone_candidate

        return "", ""

    # ============================================================
    # Exact policy lookup
    # ============================================================

    def exact_policy_lookup(self, policy_id: str) -> Dict[str, Any]:
        policy_id = self._clean_text(policy_id)
        if not policy_id:
            return {}
        return self.policy_lookup.get(policy_id, {})

    # ============================================================
    # Exact field lookup
    # ============================================================

    def exact_field_lookup(self, field: str, value: str) -> List[Dict[str, Any]]:
        """
        Exact-first logic:
        - For strict fields (policyholder, phone, email, etc.), do exact/strict match only
        - If exact matches exist, return ONLY those
        - Allow limited broader matching only for broader categorical fields
        - Always return unique results by filename
        """
        field = self._clean_text(field).lower()
        value = self._clean_text(value)

        if not field or not value:
            return []

        if field not in SUPPORTED_FIELDS:
            raise ValueError(f"Unsupported field: {field}")

        target = self._normalise_for_match(value)
        if not target and field != "phone":
            return []

        exact_matches: List[Dict[str, Any]] = []
        strong_matches: List[Dict[str, Any]] = []
        weak_matches: List[Dict[str, Any]] = []

        for doc in self.document_records:
            doc_copy = dict(doc)
            doc_copy["_match_type"] = f"exact_field:{field}"
            doc_copy["_matched_field"] = field




            # ----------------------------------------------------
            # STRICT PHONE MATCH (libphonenumber based)
            # ----------------------------------------------------
            if field == "phone":
                target_set = self._phone_equivalence_set(value)
                candidate_phone_raw = self._get_phone_value(doc)
                candidate_set = self._phone_equivalence_set(candidate_phone_raw)

                doc_copy["_matched_value"] = self._format_phone_display(candidate_phone_raw)
                doc_copy["_detected_country"] = self._detect_country_for_phone(candidate_phone_raw)

                # Exact intersection means same normalised number
                if target_set and candidate_set and target_set.intersection(candidate_set):
                    doc_copy["_score"] = 1.0
                    exact_matches.append(doc_copy)

                # No fuzzy fallback for phone numbers
                continue




            # ----------------------------------------------------
            # STRICT PHONE MATCH (FIXED)
            # ----------------------------------------------------
            # if field == "phone":
            #     target_phone = self._normalize_phone(value)
            #     candidate_phone_raw = self._get_phone_value(doc)
            #     candidate_phone = self._normalize_phone(candidate_phone_raw)
            #
            #     doc_copy["_matched_value"] = candidate_phone_raw
            #
            #     if target_phone and candidate_phone and candidate_phone == target_phone:
            #         doc_copy["_score"] = 1.0
            #         exact_matches.append(doc_copy)
            #
            #     # absolutely no fuzzy/semantic-like fallback within field lookup
            #     continue

            # ----------------------------------------------------
            # STRICT EMAIL MATCH
            # ----------------------------------------------------
            if field == "email":
                candidate_email = self._normalise_for_match(self._get_email_value(doc))
                doc_copy["_matched_value"] = self._get_email_value(doc)

                if candidate_email and candidate_email == target:
                    doc_copy["_score"] = 1.0
                    exact_matches.append(doc_copy)
                    continue

                # contains only if exact absent system-wide
                if candidate_email and target in candidate_email:
                    doc_copy["_score"] = 0.9
                    strong_matches.append(doc_copy)
                continue

            # ----------------------------------------------------
            # STRICT POLICYHOLDER NAME MATCH
            # ----------------------------------------------------
            if field == "policyholder_details":
                full_name = self._normalise_for_match(self._policyholder_full_name(doc))
                doc_copy["_matched_value"] = self._policyholder_full_name(doc)

                if full_name and full_name == target:
                    doc_copy["_score"] = 1.0
                    exact_matches.append(doc_copy)
                    continue

                # Allow substring only for multi-word full name query
                if len(target.split()) >= 2 and full_name and target in full_name:
                    doc_copy["_score"] = 0.95
                    strong_matches.append(doc_copy)
                continue

            # ----------------------------------------------------
            # STRICT NOMINEE NAME MATCH
            # ----------------------------------------------------
            if field == "nominee":
                nominee_name = self._normalise_for_match(self._nominee_name(doc))
                doc_copy["_matched_value"] = self._nominee_name(doc)

                if nominee_name and nominee_name == target:
                    doc_copy["_score"] = 1.0
                    exact_matches.append(doc_copy)
                    continue

                if len(target.split()) >= 2 and nominee_name and target in nominee_name:
                    doc_copy["_score"] = 0.95
                    strong_matches.append(doc_copy)
                continue

            # ----------------------------------------------------
            # NOMINEE DETAILS MATCH
            # ----------------------------------------------------
            if field == "nominee_details":
                candidate = self._get_field_search_text(doc, field)
                candidate_norm = self._normalise_for_match(candidate)
                doc_copy["_matched_value"] = candidate

                if candidate_norm and candidate_norm == target:
                    doc_copy["_score"] = 1.0
                    exact_matches.append(doc_copy)
                    continue

                if candidate_norm and target in candidate_norm:
                    doc_copy["_score"] = 0.9
                    strong_matches.append(doc_copy)
                continue

            # ----------------------------------------------------
            # GENERIC FIELDS
            # ----------------------------------------------------
            candidate = self._get_field_search_text(doc, field)
            candidate_norm = self._normalise_for_match(candidate)
            doc_copy["_matched_value"] = candidate

            if not candidate_norm:
                continue

            # exact wins
            if candidate_norm == target:
                doc_copy["_score"] = 1.0
                exact_matches.append(doc_copy)
                continue

            # strong contains for broader text fields
            if target in candidate_norm:
                doc_copy["_score"] = 0.9
                strong_matches.append(doc_copy)
                continue

            # limited weak token overlap ONLY for broader categorical fields
            if field in {"policy_type", "policy_term"}:
                target_tokens = set(target.split())
                candidate_tokens = set(candidate_norm.split())
                overlap = len(target_tokens.intersection(candidate_tokens))
                if target_tokens and overlap == len(target_tokens):
                    doc_copy["_score"] = 0.8
                    weak_matches.append(doc_copy)
                    continue
                if len(target_tokens) >= 2 and overlap >= len(target_tokens) - 1:
                    doc_copy["_score"] = 0.7
                    weak_matches.append(doc_copy)
                    continue

        exact_matches = self._dedupe_by_filename(exact_matches)
        strong_matches = self._dedupe_by_filename(strong_matches)
        weak_matches = self._dedupe_by_filename(weak_matches)

        if exact_matches:
            exact_matches.sort(key=lambda d: (-float(d.get("_score", 0.0)), d.get("filename", "")))
            return exact_matches

        if strong_matches:
            strong_matches.sort(key=lambda d: (-float(d.get("_score", 0.0)), d.get("filename", "")))
            return strong_matches

        weak_matches.sort(key=lambda d: (-float(d.get("_score", 0.0)), d.get("filename", "")))
        return weak_matches

    # ============================================================
    # Semantic search
    # ============================================================

    def _expand_query(self, query: str) -> str:
        q = self._clean_text(query)
        if not q:
            return ""

        return (
            f"Insurance policy search query: {q}\n"
            f"Relevant concepts include policy id, policyholder, nominee, "
            f"policy type, issue date, expiry date, policy term, email, phone, and nominee details."
        )

    def _keyword_overlap_score(self, query: str, doc: Dict[str, Any]) -> int:
        q_words = set(w for w in self._normalise_for_match(query).split() if w.strip())
        if not q_words:
            return 0

        text = self._normalise_for_match(json.dumps(doc, ensure_ascii=False))
        return sum(1 for w in q_words if w in text)

    def _get_document_record(self, filename: str) -> Dict[str, Any]:
        for doc in self.document_records:
            if doc.get("filename") == filename:
                return doc
        return {}

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query = self._clean_text(query)
        if not query:
            return []

        expanded_query = self._expand_query(query)

        q_vec = self.embed_model.encode(
            [expanded_query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        internal_top_k = max(top_k * 3, top_k)
        scores, indices = self.index.search(q_vec, internal_top_k)

        best_docs: Dict[str, Dict[str, Any]] = {}

        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            if idx >= len(self.chunk_metadata):
                continue

            meta = self.chunk_metadata[idx]
            fname = meta.get("filename", "")
            if not fname:
                continue

            current = best_docs.get(fname)
            if current is None or float(score) > float(current["score"]):
                best_docs[fname] = {
                    "score": float(score),
                    "meta": meta,
                }

        results: List[Dict[str, Any]] = []

        for fname, data in best_docs.items():
            doc = self._get_document_record(fname)
            if not doc:
                continue

            doc_copy = dict(doc)
            doc_copy["_score"] = float(data["score"])
            doc_copy["_keyword_overlap"] = self._keyword_overlap_score(query, doc_copy)
            doc_copy["_match_type"] = "semantic"
            results.append(doc_copy)

        results = self._dedupe_by_filename(results)
        results.sort(
            key=lambda x: (
                -float(x.get("_keyword_overlap", 0)),
                -float(x.get("_score", 0.0))
            )
        )

        return results[:top_k]

    # ============================================================
    # Hybrid query
    # ============================================================

    def hybrid_query(
        self,
        query: str,
        field: str = "",
        value: str = "",
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Strategy:
        1. Explicit field + value -> exact field lookup first
        2. Else infer field + value from natural query
        3. If structured query has no exact match, return empty
        4. Use semantic search only for broad non-structured queries
        """
        query = self._clean_text(query)
        field = self._clean_text(field).lower()
        value = self._clean_text(value)

        result: Dict[str, Any] = {
            "exact_policy": None,
            "field_matches": [],
            "semantic_matches": [],
        }

        # --------------------------------------------
        # 1. Explicit field/value route
        # --------------------------------------------
        if field and value:
            if field not in SUPPORTED_FIELDS:
                raise ValueError(f"Unsupported field: {field}")

            if field == "policy_id":
                exact = self.exact_policy_lookup(value)
                if exact:
                    result["exact_policy"] = exact
                    return result

            matches = self.exact_field_lookup(field, value)
            if matches:
                result["field_matches"] = matches[:top_k]
                return result

            # structured query with no match => do not fallback
            return result

        # --------------------------------------------
        # 2. Infer structured field/value from query
        # --------------------------------------------
        inferred_field, inferred_value = self._infer_field_and_value_from_query(query)

        if inferred_field and inferred_value:
            if inferred_field == "policy_id":
                exact = self.exact_policy_lookup(inferred_value)
                if exact:
                    result["exact_policy"] = exact
                    return result

            matches = self.exact_field_lookup(inferred_field, inferred_value)
            if matches:
                result["field_matches"] = matches[:top_k]
                return result

            # structured query with no match => do not fallback
            return result

        # --------------------------------------------
        # 3. Query may itself be a pure policyholder name
        # --------------------------------------------
        query_norm = self._normalise_for_match(query)
        if re.fullmatch(r"[a-z]+\s+[a-z]+(?:\s+[a-z]+)?", query_norm):
            matches = self.exact_field_lookup("policyholder_details", query)
            if matches:
                result["field_matches"] = matches[:top_k]
                return result

            # pure person-name query with no match => no semantic fallback
            return result

        # --------------------------------------------
        # 4. Semantic fallback only for real broad queries
        # --------------------------------------------
        search_text = query or value
        if search_text and len(search_text.split()) >= 3:
            result["semantic_matches"] = self.semantic_search(search_text, top_k=top_k)

        return result

    # ============================================================
    # Pretty print helper
    # ============================================================

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
                print(
                    f"[{i}] {item.get('filename')} | "
                    f"match_type={item.get('_match_type', '')} | "
                    f"score={item.get('_score', 0):.4f}"
                )
                print(json.dumps(item, indent=2, ensure_ascii=False))
                print("-" * 90)

        if results.get("semantic_matches"):
            print("SEMANTIC MATCHES")
            print("-" * 90)
            for i, item in enumerate(results["semantic_matches"], start=1):
                print(
                    f"[{i}] {item.get('filename')} | "
                    f"score={item.get('_score', 0):.4f} | "
                    f"keyword_overlap={item.get('_keyword_overlap', 0)}"
                )
                print(json.dumps(item, indent=2, ensure_ascii=False))
                print("-" * 90)

        if (
            not results.get("exact_policy")
            and not results.get("field_matches")
            and not results.get("semantic_matches")
        ):
            print("No matches found.")
            print("-" * 90)



