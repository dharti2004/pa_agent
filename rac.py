import os
import json
import logging
import argparse
from typing import Any, Dict, Optional
from datetime import datetime
import re
from langsmith import traceable
from pymongo import MongoClient
from pymongo.collection import Collection
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("finance_suggestions")

SUGGESTIONS_PROMPT = """
You are a **professional financial advisor** who provides personalized recommendations based on a user’s long-term financial profile and their current situation.

Your task:
- Use the given data to create a **personalized recommendation** as if you were speaking directly to the user.
- Do **not** repeat back all the facts they already know — instead, focus on **what they should do next** to improve their financial health.
- Make the advice **specific to them** based on their profile, debt levels, savings, income, location, and goals.
- Use a **supportive, encouraging, and human tone** that makes them feel understood and motivated.
- Focus on **clear, practical steps** they can take.
- Avoid sounding robotic or overly formal — keep it conversational.
- Output a JSON object with two keys:
  - `short_msg`: A concise, notification-like message summarizing the core advice.
  - `suggestion`: The full, detailed personalized advice as one cohesive paragraph.
---
Input:
FinanceProfile:
{finance_json}

Profile Summary:
{profile_summary}

Output:
"""

def _json_default_serializer(obj: Any) -> Any:
    """Handle non-serializable types for JSON."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def build_prompt(finance: Dict[str, Any], profile_summary: str) -> str:
    finance_json = json.dumps(
        finance or {}, ensure_ascii=False, indent=2, default=_json_default_serializer
    )
    return (
        SUGGESTIONS_PROMPT.replace("{finance_json}", finance_json)
        .replace("{profile_summary}", profile_summary or "")
    )
def clean_llm_output(content: str) -> str:
    """
    Remove Markdown code fences (```json ... ```).
    """
    if not content:
        return ""
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.DOTALL)


@traceable(name="finance_suggestions_llm_call")
def llm_call(llm, prompt: str):
    return llm.invoke(prompt)

def process_user(
    memory_col: Collection, llm: ChatGoogleGenerativeAI, memory_doc: Dict[str, Any]
) -> Optional[Dict[str, str]]:
    user_id = memory_doc.get("user_id") or "unknown"
    finance = memory_doc.get("finance") or {}
    profile_summary = memory_doc.get("profile_summary", "")

    logger.info(f"Generating suggestion JSON for user_id={user_id}")

    prompt = build_prompt(finance, profile_summary) + "\nRespond ONLY with valid JSON. Do not use ```json fences."

    try:
        resp = llm_call(llm, prompt)
        raw_content = (getattr(resp, "content", "") or "").strip()
        content = clean_llm_output(raw_content)
    except Exception as e:
        logger.error(f"LLM invocation failed for user {user_id}: {e}")
        return None

    if not content:
        logger.error(f"Empty suggestion content for user {user_id}")
        return None

    try:
        suggestion_json = json.loads(content)
        if not isinstance(suggestion_json, dict) or "short_msg" not in suggestion_json or "suggestion" not in suggestion_json:
            logger.error(f"Invalid JSON structure for user {user_id}: {content}")
            return None

        memory_col.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "suggestions": suggestion_json["suggestion"],
                    "short_msg": suggestion_json["short_msg"],
                    "suggestions_last_updated": datetime.utcnow(),
                }
            },
            upsert=False,
        )
        logger.info(f"Suggestions updated for user_id={user_id}")
        return suggestion_json

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON for user {user_id}: {e}\nRaw Content: {raw_content}")
        return None
    except Exception as e:
        logger.error(f"Failed to update suggestions for user {user_id}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate finance suggestions for users.")
    parser.add_argument("--user_id", type=str, help="User ID to process (optional).")
    args = parser.parse_args()

    mongo_uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("DATABASE_NAME")
    target_collection = os.getenv("TARGET_COLLECTION_NAME")

    if not mongo_uri or not db_name or not target_collection:
        logger.error("Missing required environment variables. Please check your .env file.")
        return

    client = MongoClient(mongo_uri)
    db = client[db_name]
    memory_col = db[target_collection]

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.4)

    if args.user_id:
        doc = memory_col.find_one({"user_id": args.user_id})
        if not doc:
            logger.error(f"No finance memory found for user_id={args.user_id}")
            return
        suggestion_json = process_user(memory_col, llm, doc)
        if suggestion_json:
            print(json.dumps(suggestion_json, indent=2, ensure_ascii=False))
    else:
        cursor = memory_col.find({})
        for doc in cursor:
            user_id = doc.get("user_id", "unknown")
            suggestion_json = process_user(memory_col, llm, doc)
            if suggestion_json:
                print(f"\nUser {user_id} suggestion:\n")
                print(json.dumps(suggestion_json, indent=2, ensure_ascii=False))
                print("\n" + ("-" * 40) + "\n")

if __name__ == "__main__":
    main()
