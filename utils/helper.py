import os
import json
import tempfile
import logging
from typing import List, Annotated, TypedDict, Optional

from llama_index.core import Document as LlamaDocument 
from llama_parse import LlamaParse 
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import ConfigurationError
from langgraph.checkpoint.mongodb import MongoDBSaver
from utils.rag import semantic_search_by_file

from memory.finance_profile import FinanceProfile
from memory.travel_profile import TravelProfile
from utils.tools import tools, tool_map


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.6,
        )
    return _llm

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/pa_agent")
MONGODB_DB = os.getenv("MONGODB_DB", "pa_agent")
MEMORY_COLLECTION = os.getenv("MEMORY_COLLECTION", "memory")

_mongo_client: Optional[MongoClient] = None

def _get_db():
    global _mongo_client
    try:
        if _mongo_client is None:
            _mongo_client = MongoClient(MONGODB_URI)
        try:
            db = _mongo_client.get_default_database()
        except ConfigurationError:
            db = _mongo_client[MONGODB_DB]
        logger.info("Connected to MongoDB and selected database")
        return db
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}", exc_info=True)
        raise


def _get_collection() -> Collection:
    db = _get_db()
    return db[MEMORY_COLLECTION]


def get_user_profiles(user_id: str) -> dict:
    try:
        col = _get_collection()
        doc = col.find_one({"user_id": user_id}) or {}
        doc.pop("_id", None)
        return {"finance": doc.get("finance"), "travel": doc.get("travel")}
    except Exception as e:
        logger.error(f"Error fetching user profiles from MongoDB: {e}", exc_info=True)
        return {"finance": None, "travel": None}


def set_last_file_id(user_id: str, file_id: str) -> None:
    try:
        col = _get_collection()
        col.update_one({"user_id": user_id}, {"$set": {"user_id": user_id, "last_file_id": file_id}}, upsert=True)
        logger.info(f"Set last_file_id for user {user_id}: {file_id}")
    except Exception as e:
        logger.error(f"Error setting last_file_id for user {user_id}: {e}", exc_info=True)


def get_last_file_id(user_id: str) -> str:
    try:
        col = _get_collection()
        doc = col.find_one({"user_id": user_id}, {"last_file_id": 1}) or {}
        return doc.get("last_file_id", "") or ""
    except Exception as e:
        logger.error(f"Error getting last_file_id for user {user_id}: {e}", exc_info=True)
        return ""


def upsert_profiles(user_id: str, finance: Optional[FinanceProfile] = None, travel: Optional[TravelProfile] = None) -> None:
    try:
        col = _get_collection()
        set_doc = {"user_id": user_id}
        if finance is not None:
            fdict = finance.model_dump(mode="json", exclude_unset=True, exclude_none=True)
            if fdict:
                set_doc["finance"] = fdict
        if travel is not None:
            tdict = travel.model_dump(mode="json", exclude_unset=True, exclude_none=True)
            if tdict:
                set_doc["travel"] = tdict
        if len(set_doc) > 1:
            col.update_one({"user_id": user_id}, {"$set": set_doc}, upsert=True)
    except Exception as e:
        logger.error(f"Error upserting profiles for user {user_id}: {e}", exc_info=True)


def parse_file(file_bytes: bytes, filename: str) -> str:
    logger.info(f"Attempting to parse file: {filename}")
    api_key = os.getenv("PARSE_KEY")
    if not api_key:
        logger.error("FATAL: LlamaParse API key ('PARSE_KEY') not found.")
        raise EnvironmentError("PARSE_KEY environment variable not set.")
    tmp_path = None
    try:
        file_extension = os.path.splitext(filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        logger.debug(f"Temporary file created at: {tmp_path}")
        parser = LlamaParse(api_key=api_key, result_type="text", verbose=True)
        documents: List[LlamaDocument] = parser.load_data([tmp_path])
        if not documents or not documents[0].text.strip():
            logger.warning(f"LlamaParse returned NO TEXT content for file '{filename}'.")
            return ""
        parsed_text = "\n\n".join(doc.text for doc in documents)
        logger.info(f"Successfully parsed file '{filename}'. Extracted {len(parsed_text)} characters.")
        return parsed_text
    except Exception as e:
        logger.error(f"An error occurred during LlamaParse: {e}", exc_info=True)
        return ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.debug(f"Temporary file deleted: {tmp_path}")


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    file_id: str


def build_system_prompt(state: AgentState):
    user_id = state["user_id"]
    file_id = get_last_file_id(user_id)

    logger.info(f"Building system prompt for user_id: {user_id} with file_id: {file_id}")

    profiles = get_user_profiles(user_id)

    profile_context_str = ""
    if profiles.get("finance") or profiles.get("travel"):
        profile_context_str = f"\n<User Profile>:\n{json.dumps(profiles, ensure_ascii=False)}\n</User Profile>\n"
        logger.info(f"User profile context included in prompt for user {user_id}")

    latest_user_query = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) and getattr(msg, "content", None):
            latest_user_query = str(msg.content)
            break


    retrieved_snippets = []
    if file_id and latest_user_query:
        try:
            retrieved_snippets = semantic_search_by_file(latest_user_query, user_id=user_id, file_id=file_id, top_k=1)
        except Exception as e:
            logger.warning(f"Prefetch file-scoped snippets failed: {e}")

    instructions = (
        "You are a highly capable and helpful assistant.\n"
        "- Always ground your answer in the user's profile if relevant.\n"
        "- Use the provided File Context snippets from the uploaded file to answer.\n"
        "- If additional external or up-to-date information is required beyond the snippets, call `web_search`.\n"
        "- Cite the sources or mention when document snippets were used.\n"
        "- The final answer must reflect the latest memory (profile) and any provided context.\n"
    )

    file_block = ""
    if file_id:
        args_line = {
            "query": latest_user_query or "",
            "user_id": user_id,
            "file_id": file_id,
        }
        snippets_text = "\n\n".join(retrieved_snippets) if retrieved_snippets else ""
        file_block = (
            "\n<File Context>\n"
            f"tool_args: {json.dumps(args_line, ensure_ascii=False)}\n"
            "retrieved_snippets:\n<<<\n"
            f"{snippets_text}\n"
            ">>>\n"
            "</File Context>\n"
        )

    system_prompt = f"{instructions}{profile_context_str}{file_block}"
    logger.debug(f"System prompt built:\n{system_prompt}")
    return system_prompt


def agent_node(state: AgentState):
    logger.info(
        f"Agent node called with {len(state['messages'])} messages for user {state['user_id']}"
    )
    system_prompt = build_system_prompt(state)
    messages_with_prompt = [{"role": "system", "content": system_prompt}, *state["messages"]]
    logger.debug(f"Invoking LLM with messages:\n{messages_with_prompt}")
    response = get_llm().bind_tools(tools).invoke(messages_with_prompt)
    logger.info(
        f"LLM response received, length {len(getattr(response, 'content', '') or str(response))} characters"
    )
    return {"messages": [response]}


def tool_node(state: AgentState):
    last_message = state["messages"][-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        logger.info("No tool calls found in last message, ending tool node.")
        return {"messages": []}

    logger.info(
        f"Tool node executing tools: {[tool['name'] for tool in last_message.tool_calls]}"
    )
    tool_messages: List[ToolMessage] = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        if tool_name in tool_map:
            try:
                logger.info(f"Calling tool '{tool_name}' with args {tool_call['args']}")
                result = tool_map[tool_name].invoke(tool_call["args"]) 
                logger.info(
                    f"Tool '{tool_name}' returned result length: {len(str(result))}"
                )
                tool_messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_call["id"])  
                )
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: {e}", tool_call_id=tool_call["id"]  
                    )
                )
    logger.debug(f"Tool messages prepared: {tool_messages}")
    return {"messages": tool_messages}


def update_memory_from_messages(user_id: str, messages: List) -> None:
    try:
        if not messages:
            logger.info("No user messages to analyze for memory.")
            return
        human_texts = [
            getattr(m, "content", "")
            for m in messages
            if isinstance(m, HumanMessage) and getattr(m, "content", None)
        ]
        if not human_texts:
            logger.info("No human messages found for memory extraction.")
            return

        finance_fields = ", ".join(FinanceProfile.model_fields.keys())
        travel_fields = ", ".join(TravelProfile.model_fields.keys())
        extraction_prompt = (
    "Return STRICT JSON with optional keys 'finance_profile' and 'travel_profile'.\n"
    "Use EXACT field names only from the schemas below. Omit a key if nothing to update.\n"
    f"FinanceProfile fields: {finance_fields}.\n"
    f"TravelProfile fields: {travel_fields}.\n"
    "For any list-typed fields, ALWAYS return an array (e.g., notes_specific_questions_asked: [\"...\"]).\n"
    )
        extraction_messages = [
            {"role": "system", "content": extraction_prompt},
            {"role": "user", "content": "\n\n".join(human_texts)},
        ]
        raw = get_llm().invoke(extraction_messages)
        text = getattr(raw, "content", "") if raw else ""
        logger.debug(f"Raw extraction output: {text}")

        data = {}
        try:
            data = json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(text[start : end + 1])
                except Exception as inner:
                    logger.warning(
                        f"Failed to parse JSON from extraction output: {inner}"
                    )
        if not isinstance(data, dict):
            logger.info("No valid JSON extracted for memory update.")
            return

        finance_obj = None
        travel_obj = None
        fp = data.get("finance_profile")
        tp = data.get("travel_profile")
        if isinstance(fp, dict) and fp:
            try:
                finance_obj = FinanceProfile.model_validate(fp)  
            except Exception as e:
                logger.warning(f"FinanceProfile validation failed: {e}")
        if isinstance(tp, dict) and tp:
            try:
                travel_obj = TravelProfile.model_validate(tp)  
            except Exception as e:
                logger.warning(f"TravelProfile validation failed: {e}")

        if finance_obj or travel_obj:
            upsert_profiles(user_id=user_id, finance=finance_obj, travel=travel_obj)
        else:
            logger.info("No profile updates detected from messages.")
    except Exception as e:
        logger.error(
            f"Unexpected error in update_memory_from_messages for user {user_id}: {e}",
            exc_info=True,
        )

def memory_update_node(state: AgentState):
    user_id = state["user_id"]
    logger.info(f"Memory update node called for user {user_id}")
    user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    try:
        update_memory_from_messages(user_id=user_id, messages=user_messages)
        logger.info(f"Memory update for user {user_id} completed successfully.")
    except Exception as e:
        logger.error(f"Error during memory update for user {user_id}: {e}", exc_info=True)


workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("update_memory", memory_update_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    lambda x: "tools" if hasattr(x["messages"][-1], "tool_calls") and x["messages"][-1].tool_calls else "update_memory",
    {"tools": "tools", "update_memory": "update_memory"},
)
workflow.add_edge("tools", "agent")
workflow.add_edge("update_memory", END)

if _mongo_client is None:
    _mongo_client = MongoClient(MONGODB_URI)
checkpointer = MongoDBSaver(_mongo_client, db_name=MONGODB_DB, collection_name="langgraph_checkpoints")

chat = workflow.compile(checkpointer=checkpointer)
logger.info("LangGraph agent compiled with MongoDBSaver persistence")