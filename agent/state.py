from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from dataclasses import dataclass, field

@dataclass
class AgentState:
    # Research topic provided by the user
    topic: str = ""
    
    # String output from web search tool
    search_results: str = ""
    
    # String output from RAG tool
    rag_results: str = ""
    
    # Markdown report from report writter tool
    report: str = ""
    
    # Full conversation history
    messages: Annotated[list[BaseMessage], add_messages] = field(default_factory=list)