import os
import json
import re
import logging
import argparse
from typing import Any, Dict, Iterable, Optional, Tuple
from datetime import datetime, date
from tenacity import retry, stop_after_attempt, wait_fixed
from langsmith import traceable
from pymongo import MongoClient
from pymongo.collection import Collection
from pydantic import ValidationError
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from memory.finance_profile import FinanceProfile

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("finance_memory_builder")

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://admin:admin123@127.0.0.1:27017/admin")
DATABASE_NAME = os.getenv("DATABASE_NAME", "finpal_db")
SOURCE_COLLECTION_NAME = os.getenv("COLLECTION_NAME", "finance_profiles")
TARGET_COLLECTION_NAME = os.getenv("TARGET_COLLECTION_NAME", "finance_memory")


class FinanceMemoryBuilder:
    def __init__(self, llm: Optional[ChatGoogleGenerativeAI] = None):
        """Initialize the FinanceMemoryBuilder with LLM and MongoDB collections"""
        self.llm = llm or self._get_llm()
        self.source_col, self.target_col = self._get_collections()
        self.system_prompt = self._build_system_prompt()

    @traceable(name="finance_memory_builder_llm_call")
    def llm_call(self, prompt: str):
        """Call the LLM with retry logic"""
        return self.llm.invoke(prompt)

    @staticmethod
    def _get_llm() -> ChatGoogleGenerativeAI:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY is not set")
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.2,
        )

    @staticmethod
    def _get_collections() -> Tuple[Collection, Collection]:
        client = MongoClient(MONGODB_URI)
        db = client[DATABASE_NAME]
        return db[SOURCE_COLLECTION_NAME], db[TARGET_COLLECTION_NAME]

    @staticmethod
    def _build_system_prompt() -> str:
        schema_json = json.dumps(FinanceProfile.model_json_schema()["properties"], indent=2)
        return f"""
You are a **financial data analyst** that extracts and transforms banking data into structured financial profiles.
You are given:
1. The **FinanceProfile schema** (see below).
2. The **user’s latest financial data** from `data.json`.

Your objectives:
- Use the schema provided below, all keys must always be present in the final output, in the same order.
- Populate or update fields only when data is explicitly present or can be inferred with high confidence.
- Perform all necessary calculations.
- If relevant information exists that is not covered by the schema but valuable for long-term financial planning, add them as new fields appended after the original schema keys using snake_case.
- Any missing or uncertain values must remain null.
- Dates must follow ISO 8601 format.
- Output must be valid JSON with exactly three top-level keys:
  1. "FinanceProfile"
  2. "calculations"
  3. "profile_summary"

### FinanceProfile Schema:
{schema_json}

User Banking Data:
{{input_json}}

Return ONLY this JSON structure:
{{
  "FinanceProfile": {{ ... }},
  "calculations": {{ "field_name": "detailed explanation", ... }},
  "profile_summary": "Comprehensive paragraph."
}}
"""

    def fetch_user_docs(self, user_id: Optional[str] = None) -> Iterable[Dict[str, Any]]:
        if user_id:
            query = {
                "$or": [
                    {"user_id": user_id},
                    {"Person.PersonID": int(user_id) if user_id.isdigit() else user_id},
                    {"Customer.CustomerID": int(user_id) if user_id.isdigit() else user_id},
                    {"Account.AccountID": int(user_id) if user_id.isdigit() else user_id},
                ]
            }
        else:
            query = {}
        cursor = self.source_col.find(query)
        for doc in cursor:
            doc = dict(doc)
            doc.pop("_id", None)
            yield doc

    @staticmethod
    def _json_default_serializer(obj: Any) -> str:
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        try:
            return str(obj)
        except Exception:
            return repr(obj)

    def _build_prompt_for_doc(self, doc: Dict[str, Any]) -> str:
        input_json = json.dumps(
            doc,
            ensure_ascii=False,
            indent=2,
            default=self._json_default_serializer,
        )
        return self.system_prompt.replace("{input_json}", input_json)

    @staticmethod
    def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
        try:
            text_clean = re.sub(r"```(json)?", "", text).strip()
            first = text_clean.find("{")
            last = text_clean.rfind("}")
            if first != -1 and last != -1 and last > first:
                json_str = text_clean[first : last + 1]
                return json.loads(json_str)
        except Exception as e:
            logger.warning(f"JSON parsing failed: {e}")

        try:
            json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
            json_matches = re.findall(json_pattern, text, re.DOTALL)
            for match in json_matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, dict) and ("FinanceProfile" in parsed or len(parsed) > 10):
                        return parsed
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            logger.warning(f"Fallback JSON extraction failed: {e}")

        try:
            text_clean = re.sub(r"```(json)?", "", text).strip()
            first = text_clean.find("{")
            last = text_clean.rfind("}")
            if first != -1 and last != -1 and last > first:
                json_str = text_clean[first : last + 1]
                json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)
                json_str = re.sub(r"([}\]])(\s*)([\"\\w])", r"\1,\2\3", json_str)
                json_str = re.sub(r":\s*([^\",{}\[\]]+)(?=\s*[,}])", r': "\1"', json_str)
                return json.loads(json_str)
        except Exception as e:
            logger.warning(f"JSON fixing strategy failed: {e}")

        logger.error(f"Failed to extract JSON. Content preview: {text[:500]}...")
        return None

    @staticmethod
    def _validate_finance_profile(fp_obj: Dict[str, Any]) -> Dict[str, Any]:
        try:
            fp_valid = FinanceProfile(**fp_obj)
            return fp_valid.model_dump(mode="json", exclude_unset=True)
        except ValidationError as ve:
            logger.error(f"FinanceProfile validation failed: {ve}")
            return fp_obj

    def _upsert_finance_memory(
        self,
        user_id: str,
        finance: Dict[str, Any],
        calculations: Dict[str, Any],
        profile_summary: str,
    ) -> None:
        self.target_col.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "user_id": user_id,
                    "finance": finance,
                    "calculations": calculations,
                    "profile_summary": profile_summary,
                }
            },
            upsert=True,
        )

    def _extract_user_id(self, user_doc: Dict[str, Any]) -> str:
        possible_ids = [
            user_doc.get("user_id"),
            user_doc.get("id"),
            user_doc.get("Person", {}).get("PersonID"),
            user_doc.get("Customer", {}).get("CustomerID"),
            user_doc.get("Account", {}).get("AccountID"),
        ]
        for uid in possible_ids:
            if uid is not None:
                return str(uid)
        return "unknown"

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def process_user(self, user_doc: Dict[str, Any]) -> None:
        user_id = self._extract_user_id(user_doc)
        person_info = user_doc.get("Person", {})
        customer_info = user_doc.get("Customer", {})
        user_name = f"{person_info.get('FirstName', '')} {person_info.get('LastName', '')}".strip()
        customer_type = customer_info.get("CustomerType", "Unknown")
        logger.info(f"Processing user_id={user_id} ({user_name}, {customer_type})")

        prompt = self._build_prompt_for_doc(user_doc)
        try:
            response = self.llm_call(prompt)
            content = getattr(response, "content", "") if response else ""
        except Exception as e:
            logger.error(f"LLM invocation failed for user {user_id}: {e}")
            return

        if not content:
            logger.error(f"Empty response from LLM for user {user_id}")
            return

        json_obj = self._extract_json_block(content)
        if not json_obj:
            logger.error(f"No valid JSON found for user {user_id}")
            return

        required_keys = ["FinanceProfile", "calculations", "profile_summary"]
        missing_keys = [key for key in required_keys if key not in json_obj]
        if missing_keys:
            logger.error(f"Missing keys {missing_keys} in JSON response for user {user_id}")
            return

        fp_validated = self._validate_finance_profile(json_obj["FinanceProfile"])
        calculations = json_obj.get("calculations", {})
        profile_summary = json_obj.get("profile_summary", "")

        try:
            self._upsert_finance_memory(user_id, fp_validated, calculations, profile_summary)
            logger.info(f"Successfully upserted finance memory for user {user_id} ({user_name})")
        except Exception as e:
            logger.error(f"Failed to upsert finance memory for user {user_id}: {e}")

    def process_users(self, user_id: Optional[str] = None) -> int:
        docs = self.fetch_user_docs(user_id)
        count = 0
        for doc in docs:
            self.process_user(doc)
            count += 1
        return count


def main():
    parser = argparse.ArgumentParser(
        description="Build finance memory from source profiles using LLM"
    )
    parser.add_argument(
        "--user_id",
        type=str,
        default="",
        help="Process a single user by PersonID/CustomerID; if omitted, process all users",
    )
    args = parser.parse_args()
    builder = FinanceMemoryBuilder()
    user_id = args.user_id if args.user_id else None
    count = builder.process_users(user_id)
    logger.info(f"Processed {count} user(s)")


if __name__ == "__main__":
    main()
