import os
import time
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.messages import trim_messages, HumanMessage
from langchain.chat_models import init_chat_model

from shared_store import url_time
from tools import (
    get_rendered_html, download_file, post_request,
    run_code, add_dependencies, ocr_image_tool, transcribe_audio, encode_image_to_base64
)

load_dotenv()

USER_EMAIL = os.getenv("EMAIL")
USER_SECRET = os.getenv("SECRET")

MAX_RECURSION_DEPTH = 5000
TOKEN_LIMIT = 60000


# -------------------------------------------------
# STATE DEFINITION
# -------------------------------------------------
class QuizAgentState(TypedDict):
    messages: Annotated[List, add_messages]


AVAILABLE_TOOLS = [
    run_code, get_rendered_html, download_file,
    post_request, add_dependencies, ocr_image_tool, transcribe_audio, encode_image_to_base64
]


# -------------------------------------------------
# LANGUAGE MODEL INITIALIZATION
# -------------------------------------------------
api_rate_limiter = InMemoryRateLimiter(
    requests_per_second=4 / 60,
    check_every_n_seconds=1,
    max_bucket_size=4
)

language_model = init_chat_model(
    model_provider="google_genai",
    model="gemini-2.5-flash",
    rate_limiter=api_rate_limiter
).bind_tools(AVAILABLE_TOOLS)


# -------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------
QUIZ_SYSTEM_PROMPT = f"""
You are an autonomous quiz-solving agent.

Your job is to:
1. Load each quiz page from the given URL.
2. Extract instructions, parameters, and submit endpoint.
3. Solve tasks exactly.
4. Submit answers ONLY to the correct endpoint.
5. Follow new URLs until none remain, then output END.

Rules:
- For base64 generation of an image NEVER use your own code, always use the "encode_image_to_base64" tool that's provided
- Never hallucinate URLs or fields.
- Never shorten endpoints.
- Always inspect server response.
- Never stop early.
- Use tools for HTML, downloading, rendering, OCR, or running code.
- Include:
    email = {USER_EMAIL}
    secret = {USER_SECRET}
"""


# -------------------------------------------------
# NEW NODE: HANDLE MALFORMED JSON
# -------------------------------------------------
def handle_json_error_node(state: QuizAgentState):
    """
    If the LLM generates invalid JSON, this node sends a correction message
    so the LLM can try again.
    """
    print("--- DETECTED MALFORMED JSON. ASKING AGENT TO RETRY ---")
    return {
        "messages": [
            {
                "role": "user", 
                "content": "SYSTEM ERROR: Your last tool call was Malformed (Invalid JSON). Please rewrite the code and try again. Ensure you escape newlines and quotes correctly inside the JSON."
            }
        ]
    }


# -------------------------------------------------
# AGENT NODE
# -------------------------------------------------
def quiz_processing_node(state: QuizAgentState):
    # --- TIME HANDLING START ---
    current_time = time.time()
    current_url = os.getenv("url")
    
    # SAFE GET: Prevents crash if url is None or not in dict
    previous_time = url_time.get(current_url) 
    time_offset = os.getenv("offset", "0")

    if previous_time is not None:
        previous_time = float(previous_time)
        time_difference = current_time - previous_time

        if time_difference >= 180 or (time_offset != "0" and (current_time - float(time_offset)) > 90):
            print(f"Timeout exceeded ({time_difference}s) — instructing LLM to purposely submit wrong answer.")

            timeout_instruction = """
            You have exceeded the time limit for this task (over 180 seconds).
            Immediately call the `post_request` tool and submit a WRONG answer for the CURRENT quiz.
            """

            # Using HumanMessage (as you correctly implemented)
            timeout_message = HumanMessage(content=timeout_instruction)

            # We invoke the LLM immediately with this new instruction
            result = language_model.invoke(state["messages"] + [timeout_message])
            return {"messages": [result]}
    # --- TIME HANDLING END ---

    trimmed_context = trim_messages(
        messages=state["messages"],
        max_tokens=TOKEN_LIMIT,
        strategy="last",
        include_system=True,
        start_on="human",
        token_counter=language_model, 
    )
    
    # Better check: Does it have a HumanMessage?
    has_human_message = any(msg.type == "human" for msg in trimmed_context)
    
    if not has_human_message:
        print("WARNING: Context was trimmed too far. Injecting state reminder.")
        # We remind the agent of the current URL from the environment
        current_quiz_url = os.getenv("url", "Unknown URL")
        context_reminder = HumanMessage(content=f"Context cleared due to length. Continue processing URL: {current_quiz_url}")
        
        # We append this to the trimmed list (temporarily for this invoke)
        trimmed_context.append(context_reminder)
    # ----------------------------------------

    print(f"--- INVOKING AGENT (Context: {len(trimmed_context)} items) ---")
    
    result = language_model.invoke(trimmed_context)

    return {"messages": [result]}


# -------------------------------------------------
# ROUTING LOGIC (UPDATED FOR MALFORMED CALLS)
# -------------------------------------------------
def determine_next_step(state):
    last_message = state["messages"][-1]
    
    # 1. CHECK FOR MALFORMED FUNCTION CALLS
    if "finish_reason" in last_message.response_metadata:
        if last_message.response_metadata["finish_reason"] == "MALFORMED_FUNCTION_CALL":
            return "handle_malformed"

    # 2. CHECK FOR VALID TOOLS
    available_tool_calls = getattr(last_message, "tool_calls", None)
    if available_tool_calls:
        print("Route → tools")
        return "tools"

    # 3. CHECK FOR END
    message_content = getattr(last_message, "content", None)
    if isinstance(message_content, str) and message_content.strip() == "END":
        return END

    if isinstance(message_content, list) and len(message_content) and isinstance(message_content[0], dict):
        if message_content[0].get("text", "").strip() == "END":
            return END

    print("Route → agent")
    return "agent"


# -------------------------------------------------
# WORKFLOW GRAPH
# -------------------------------------------------
workflow_graph = StateGraph(QuizAgentState)

# Add Nodes
workflow_graph.add_node("agent", quiz_processing_node)
workflow_graph.add_node("tools", ToolNode(AVAILABLE_TOOLS))
workflow_graph.add_node("handle_malformed", handle_json_error_node) # Add the repair node

# Add Edges
workflow_graph.add_edge(START, "agent")
workflow_graph.add_edge("tools", "agent")
workflow_graph.add_edge("handle_malformed", "agent") # Retry loop

# Conditional Edges
workflow_graph.add_conditional_edges(
    "agent", 
    determine_next_step,
    {
        "tools": "tools",
        "agent": "agent",
        "handle_malformed": "handle_malformed", # Map the new route
        END: END
    }
)

compiled_app = workflow_graph.compile()


# -------------------------------------------------
# EXECUTION RUNNER
# -------------------------------------------------
def run_agent(quiz_url: str):
    # system message is seeded ONCE here
    initial_message_stack = [
        {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
        {"role": "user", "content": quiz_url}
    ]

    compiled_app.invoke(
        {"messages": initial_message_stack},
        config={"recursion_limit": MAX_RECURSION_DEPTH}
    )

    print("Tasks completed successfully!")