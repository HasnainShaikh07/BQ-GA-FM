import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "openai/gpt-oss-20b"
MAX_ITERATIONS = 5          # always cap the tool loop
TOOL_VERSION = "good"       # switch to "bad" for the experiment

# ---------------------------------------------------------------
# 1. TOOL IMPLEMENTATION (plain Python, the model never sees this)
# ---------------------------------------------------------------
BOOKS = {
    "dune":       {"central": 2, "north": 0, "east": 1},
    "the hobbit": {"central": 0, "north": 0, "east": 0},
}

def check_book_availability(title, branch=None):
    copies = BOOKS.get(title.strip().lower())
    if copies is None:
        return json.dumps({"is_error": True,
                           "error": f"No book titled '{title}' found in the catalog."})
    if branch is not None:
        if branch not in copies:
            return json.dumps({"is_error": True,
                               "error": f"Unknown branch '{branch}'. Valid: {list(copies)}"})
        copies = {branch: copies[branch]}
    return json.dumps({"title": title, "copies_available_by_branch": copies})

# ---------------------------------------------------------------
# 2. TOOL DEFINITIONS (this is what the model actually sees)
# ---------------------------------------------------------------
GOOD_TOOLS = [{
    "type": "function",
    "name": "check_book_availability",
    "description": (
        "Check whether a specific book is currently available to borrow at "
        "Riverside Library. Returns copies available per branch. Use when the "
        "user asks if a book is in, available, or on the shelf. Do NOT use for "
        "recommendations or opening hours."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string",
                      "description": "Exact book title, e.g. 'Dune'"},
            "branch": {"type": "string", "enum": ["central", "north", "east"],
                       "description": "Optional. Only include if the user names a specific branch."},
        },
        "required": ["title"],
    },
}]

BAD_TOOLS = [{
    "type": "function",
    "name": "get_data",
    "description": "Gets data",
    "parameters": {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    },
}]

TOOLS = GOOD_TOOLS if TOOL_VERSION == "good" else BAD_TOOLS

# Registry: tool name -> real function
REGISTRY = {
    "check_book_availability": check_book_availability,
    "get_data": lambda q: check_book_availability(q),
}

# ---------------------------------------------------------------
# 3. SYSTEM INSTRUCTIONS (last lesson's skill)
# ---------------------------------------------------------------
INSTRUCTIONS = """You are the catalog assistant for Riverside Library.
- Use the availability tool only when the user names a specific book.
  If they ask about "that book" without a title, ask which book they mean.
- You do not have information about opening hours or book recommendations.
  Say so and suggest asking library staff.
- State only availability facts returned by the tool. If the tool returns
  an error, tell the user you couldn't find that title and ask them to check
  the spelling. Never guess.
- Keep answers to 1-3 sentences."""

# ---------------------------------------------------------------
# 4. APPLICATION LOGIC: execute safely + the orchestration loop
# ---------------------------------------------------------------
def execute_tool_call(call) -> str:
    """Model-generated arguments are untrusted input: parse and validate."""
    try:
        func = REGISTRY[call.name]            # unknown tool -> KeyError
        args = json.loads(call.arguments)     # arguments arrive as a JSON string
        return func(**args)
    except Exception as e:
        # Return the error to the model so it can recover instead of crashing
        return json.dumps({"is_error": True, "error": f"{type(e).__name__}: {e}"})

def run_agent(user_message: str) -> str:
    input_items = [{"role": "user", "content": user_message}]

    for step in range(1, MAX_ITERATIONS + 1):
        response = client.responses.create(
            model=MODEL,
            instructions=INSTRUCTIONS,   # resent on every request
            input=input_items,
            tools=TOOLS,
        )
        input_items += response.output   # keep the model's items in the history

        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:                    # no tool requested -> final answer
            return response.output_text

        for call in calls:
            print(f"  [step {step}] model requested: {call.name}({call.arguments})")
            result = execute_tool_call(call)
            print(f"  [step {step}] tool returned:   {result}")
            input_items.append({
                "type": "function_call_output",
                "call_id": call.call_id,  # pairs the result with the request
                "output": result,
            })

    return "Stopped: too many tool-call steps."

if __name__ == "__main__":
    tests = [
        "Is Dune available?",
        "Do you have The Hobbit at the north branch?",
        "Is Neuromancer in stock?",
        "What time does the library open?",
        "Is that book available?",
    ]
    for question in tests:
        print(f"\nUSER: {question}")
        print(f"BOT:  {run_agent(question)}")