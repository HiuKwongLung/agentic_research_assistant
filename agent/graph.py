from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from agent.state import AgentState
from tools.web_search import web_search
from tools.report_writer import write_report

# --- Load model and bind tools ---
TOOLS = [web_search, write_report]

model = ChatOllama(model="qwen3:8b")
model_with_tools = model.bind_tools(TOOLS)

# --- Nodes ---
def agent_node(state: AgentState) -> dict:
    """The LLM decides what to do next based on the conversation so far."""
    response = model_with_tools.invoke(state.messages)

    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    """Check the last message to decide whether to call a tool or stop."""
    last_message = state.messages[-1]
    
    # If the LLM made tool calls, route to the tool node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # Otherwise the LLM is done
    return END

# --- Build graph ---
tool_node = ToolNode(TOOLS)
graph_builder = StateGraph(AgentState)

graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)

graph_builder.set_entry_point("agent")

graph_builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END,
    }
)

# Always go back to the agent to decide next step after tools run
graph_builder.add_edge("tools", "agent")

graph = graph_builder.compile()

# --- Run the agent ---
def run_agent(topic:str) -> str:
    """Entry point to runt he agent with a researh topic."""
    initial_state = AgentState(
        topic=topic,
        messages=[
            HumanMessage(content=(
                f"Please research the following topic: '{topic}'. "
                f"Search the web for information, then write a report about what you found."
            ))
        ]
    )

    final_state = graph.invoke(initial_state)

    # Return the last message content as the result
    return final_state["messages"][-1].content

# --- Test ---
if __name__ == "__main__":
    result = run_agent("What are the health effects of drinking too much coffee?")
    print(result)