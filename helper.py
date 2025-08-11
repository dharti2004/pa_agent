import os
import tempfile
import logging
from typing import List, Annotated, TypedDict
from langchain_core.messages import HumanMessage, ToolMessage
from dotenv import load_dotenv
from llama_index.core import Document as LlamaDocument
from llama_parse import LlamaParse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from ddgs import DDGS
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage
from memory.finance_profile import FinanceProfile
from memory.travel_profile import TravelProfile
from langgraph.store.memory import InMemoryStore
from langmem import create_memory_store_manager
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
def embed_fn(texts):
    logger.debug(f"Embedding {len(texts)} texts")
    embeddings = embed_model.encode(texts, convert_to_numpy=True).tolist()
    logger.debug(f"Generated embeddings of shape: {len(embeddings)}")
    return embeddings

store = InMemoryStore(index={"dims": 384, "embed": embed_fn})

manager = create_memory_store_manager(
    "google_genai:gemini-2.5-flash",
    namespace=("users", "{user_id}", "profile"),
    schemas=[FinanceProfile, TravelProfile],
    instructions="Extract all user information and events as triples.",
    enable_inserts=True,
    enable_deletes=True,
    store=store
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.6)

@tool
def web_search(query: str) -> str:
    """
    Searches the web for real-time information using DDGS (DuckDuckGo Search).
    Returns a top 5 results including title, snippet, and source URL.
    """
    logger.info(f"Executing web search for query: '{query}'")
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                logger.debug(f"Web search hit: Title='{r['title']}', URL={r['href']}")
                results.append(f"Title: {r['title']}\nContent: {r['body']}\nSource: {r['href']}")
        if not results:
            logger.info("No results found for the query.")
            return "No results found for the query."
        return "\n\n".join(results)
    except Exception as e:
        logger.error(f"An error occurred during web search: {e}", exc_info=True)
        return f"An error occurred during web search: {str(e)}"

tools = [web_search]
tool_map = {tool.name: tool for tool in tools}

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
    parsed_file_data: str

def build_system_prompt(state: AgentState):
    user_id = state['user_id']
    parsed_file_data = state['parsed_file_data']
    
    logger.info(f"Building system prompt for user_id: {user_id}")
    
    profile_results = store.search(("users", user_id, "profile"))
    
    profile_context_str = ""
    if profile_results:
        profiles_dict = {}
        for result in profile_results:
            profile_name = result.value.__class__.__name__
            profiles_dict[profile_name] = result.value.model_dump(mode="json", exclude_unset=True)
        profile_context_str = f"\n<User Profile>:\n{profiles_dict}\n</User Profile>\n"
        logger.info(f"User profile context included in prompt for user {user_id}")
    else:
        logger.info(f"No existing profile found for user {user_id}.")

    file_context = f"\n<File Content>\n{parsed_file_data}\n</File Content>\n" if parsed_file_data else ""
    if file_context:
        logger.info(f"File content included in prompt for user {user_id}, length {len(parsed_file_data)} chars")

    instructions = """You are a highly capable and helpful assistant. 
Your primary goal is to provide accurate, relevant, and personalized responses.

Follow these guidelines strictly:

1. **Analyze Context**  
   Carefully examine the information provided in the `<User Profile>` and `<File Content>` sections.  
   Use this context to tailor your answers specifically to the user.  
   Reference it directly if it is relevant to the question.

2. **Use Tools When Needed**  
   - If the user's question requires **current, real-time, or external information**, always use the `web_search` tool.  
   - After receiving the search results, **summarize and integrate** them into your final answer instead of just pasting them.  
   - Clearly cite the key sources or mention where the information came from.

3. **Maintain Tone**  
   Be friendly, helpful, and concise. Avoid unnecessary verbosity while ensuring completeness.

4. **Final Output Rule**  
   The final response to the user must **directly answer their query** using all available context, including search results when used.  
"""
    
    system_prompt = f"{instructions}{profile_context_str}{file_context}"
    logger.debug(f"System prompt built:\n{system_prompt}")
    return system_prompt

def agent_node(state: AgentState):
    """Builds prompt and invokes the conversational LLM."""
    logger.info(f"Agent node called with {len(state['messages'])} messages for user {state['user_id']}")
    system_prompt = build_system_prompt(state)
    messages_with_prompt = [{"role": "system", "content": system_prompt}, *state['messages']]
    logger.debug(f"Invoking LLM with messages:\n{messages_with_prompt}")
    response = llm.bind_tools(tools).invoke(messages_with_prompt)
    logger.info(f"LLM response received, length {len(response.content)} characters")
    return {"messages": [response]}

def tool_node(state: AgentState):
    """Executes tools if called."""
    last_message = state['messages'][-1]
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        logger.info("No tool calls found in last message, ending tool node.")
        return {"messages": []}

    logger.info(f"Tool node executing tools: {[tool['name'] for tool in last_message.tool_calls]}")
    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        if tool_name in tool_map:
            try:
                logger.info(f"Calling tool '{tool_name}' with args {tool_call['args']}")
                result = tool_map[tool_name].invoke(tool_call["args"])
                logger.info(f"Tool '{tool_name}' returned result length: {len(str(result))}")
                tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
                tool_messages.append(ToolMessage(content=f"Error: {e}", tool_call_id=tool_call["id"]))
    
    logger.debug(f"Tool messages prepared: {tool_messages}")
    return {"messages": tool_messages}

async def memory_update_node(state: AgentState):
    """
    Updates the user's memory profile synchronously (awaited) so it's available immediately.
    """
    user_id = state["user_id"]
    logger.info(f"Memory update node called for user {user_id}")

    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if not user_messages:
        logger.info("No user messages in this turn to analyze for memory.")
        return

    logger.info(f"Triggering memory update for user_id: {user_id}, messages count: {len(user_messages)}")
    try:
        await manager.ainvoke(
            {"messages": user_messages},
            config={"configurable": {"user_id": user_id}}
        )
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
    lambda x: "tools" if hasattr(x['messages'][-1], 'tool_calls') and x['messages'][-1].tool_calls else "update_memory",
    {
        "tools": "tools", 
        "update_memory": "update_memory" 
    }
)
workflow.add_edge("tools", "agent") 
workflow.add_edge("update_memory", END)

chat = workflow.compile()
logger.info("LangGraph agent compiled")