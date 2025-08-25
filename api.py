import os
import logging
from typing import Optional, Dict, Any
from test import FinanceMemoryBuilder
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from langchain_google_genai import ChatGoogleGenerativeAI
import uvicorn
from rac import process_user as generate_suggestions
builder = FinanceMemoryBuilder()


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("suggestions_api")


def get_mongo_collection() -> Collection:
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("DATABASE_NAME")
    target_collection = os.getenv("TARGET_COLLECTION_NAME")

    if not mongo_uri or not db_name or not target_collection:
        raise EnvironmentError(
            "Missing MONGO_URI, DATABASE_NAME, or TARGET_COLLECTION_NAME. Configure your .env."
        )

    client = MongoClient(mongo_uri)
    db = client[db_name]
    return db[target_collection]


def get_llm() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is not set")
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0.4)


app = FastAPI(title="Finance Suggestions API", version="1.1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def initialize_dependencies() -> None:
    try:
        app.state.memory_col = get_mongo_collection()
        app.state.llm = get_llm()
        app.state.init_error = None
        logger.info("Dependencies initialized")
    except Exception as e:
        app.state.init_error = str(e)
        logger.error(f"Initialization error: {e}")


@app.on_event("startup")
def startup_event():
    initialize_dependencies()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "initialized": getattr(app.state, "init_error", None) is None}

@app.post("/memory/{user_id}")
def build_user_memory(user_id: str) -> Dict[str, Any]:
    try:
        docs = list(builder.fetch_user_docs(user_id))
        if not docs:
            raise HTTPException(status_code=404, detail=f"No source data found for user_id={user_id}")
        
        user_doc = docs[0]  
        builder.process_user(user_doc)

        memory_doc: Optional[Dict[str, Any]] = builder.target_col.find_one({"user_id": user_id})
        if not memory_doc:
            raise HTTPException(status_code=500, detail=f"Memory not created for user_id={user_id}")
        
        return {
            "finance": memory_doc.get("finance", {}),
            "calculations": memory_doc.get("calculations", {}),
            "profile_summary": memory_doc.get("profile_summary", "")
        }

    except Exception as e:
        logger.error(f"Error while building memory for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
    

@app.post("/suggestions/{user_id}")
def generate_user_suggestions(user_id: str):
    if getattr(app.state, "init_error", None) is not None:
        raise HTTPException(status_code=500, detail=f"Service not configured: {app.state.init_error}")

    memory_col: Collection = app.state.memory_col
    llm: ChatGoogleGenerativeAI = app.state.llm

    doc = memory_col.find_one({"user_id": user_id})
    if not doc:
        raise HTTPException(status_code=404, detail=f"No finance memory found for user_id={user_id}")

    suggestion_json: Optional[Dict[str, str]] = generate_suggestions(memory_col, llm, doc)
    if not suggestion_json:
        raise HTTPException(status_code=500, detail="Failed to generate suggestions")

    return {
        "short_msg": suggestion_json.get("short_msg", ""),
        "suggestion": suggestion_json.get("suggestion", "")
    }


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False) 