import uvicorn
import json
import logging
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.concurrency import run_in_threadpool
from utils.helper import chat, parse_file, get_user_profiles, set_last_file_id, get_last_file_id
from utils.rag import store_file_embeddings

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger("pymongo").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

app = FastAPI()


@app.post("/upload")
async def upload_endpoint(
    user_id: str = Form(...),
    file: UploadFile = File(...),
):
    try:
        file_bytes = await file.read()
        _ = await run_in_threadpool(parse_file, file_bytes, file.filename)
        result = await run_in_threadpool(store_file_embeddings, file_bytes, file.filename, user_id, None)
        file_id = result.get("file_id")
        if file_id:
            await run_in_threadpool(set_last_file_id, user_id, file_id)
        return {"status": "ok", "message": "File ingested to vector store", "file_id": file_id}
    except Exception as e:
        logger.error(f"Upload handling error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process uploaded file")


@app.post("/chat")
async def chat_endpoint(
    user_id: str = Form(...),
    messages: str = Form(...),
    file_id: str = Form("")
):
    try:
        messages_list = json.loads(messages)
        if not isinstance(messages_list, list):
            raise ValueError("'messages' must be a JSON array")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON format for 'messages' field.")

    effective_file_id = (file_id or "").strip()
    if not effective_file_id:
        effective_file_id = await run_in_threadpool(get_last_file_id, user_id)

    graph_input = {
        "messages": messages_list,
        "user_id": user_id,
        "file_id": effective_file_id,
    }

    config = {"configurable": {"thread_id": user_id}}

    try:
        final_state = await run_in_threadpool(chat.invoke, graph_input, config)
    except Exception as e:
        logger.error("Chat pipeline error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error in chat pipeline")

    response_message = final_state["messages"][-1]

    return {"response": getattr(response_message, "content", str(response_message))}


@app.get("/memory/{user_id}")
async def get_user_memory(user_id: str):
    try:
        profiles = get_user_profiles(user_id)
        return {"user_id": user_id, **profiles}
    except Exception as e:
        logger.error(f"Error getting user memory for {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch user memory")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)