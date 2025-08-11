import uvicorn
import json
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import Optional
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from memory.finance_profile import FinanceProfile
from memory.travel_profile import TravelProfile
from helper import chat, parse_file, manager

app = FastAPI()

@app.post("/chat")
async def chat_endpoint(
    user_id: str = Form(...),
    messages: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    try:
        messages_list = json.loads(messages)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for 'messages' field.")

    parsed_file_data = ""
    if file:
        file_bytes = await file.read()
        parsed_file_data = await run_in_threadpool(parse_file, file_bytes, file.filename)

    graph_input = {
        "messages": messages_list,
        "user_id": user_id,
        "parsed_file_data": parsed_file_data,
    }

    config = {"configurable": {"user_id": user_id}}
    
    final_state = await chat.ainvoke(graph_input, config=config)
    
    response_message = final_state['messages'][-1]
    
    return {"response": response_message}

@app.get("/memory/{user_id}")
async def get_user_memory(user_id: str):
    """
    Returns stored FinanceProfile and TravelProfile for the given user.
    """
    underlying_store = manager.store  

    profile_memories = underlying_store.search(("users", user_id, "profile"))

    finance_data = []
    travel_data = []

    for mem in (profile_memories or []):
        value = mem.value
        if isinstance(value, FinanceProfile):
            finance_data.append(value.model_dump(mode="json", exclude_unset=True))
        elif isinstance(value, TravelProfile):
            travel_data.append(value.model_dump(mode="json", exclude_unset=True))

    return {
        "user_id": user_id,
        "finance_memories": finance_data,
        "travel_memories": travel_data
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)