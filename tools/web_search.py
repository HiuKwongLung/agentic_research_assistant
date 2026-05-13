from langchain_core.tools import tool
from ddgs import DDGS

@tool
def web_search(query: str) -> str:
    """
    Search the web for a given query using DuckDuckGo.
    Returns a formatted string of the top results with title, URL, and snippet.
    """
    try:
        results = _run_search(query)

        if not results:
            return "Noresults found."
        
        return _format_results(results)

    except Exception as e:
        return f"Search failed: {str(e)}"
    
def _run_search(query:str, max_results:int = 5) -> list[dict]:
    """Run a DuckDuckGo search and return raw results."""
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))
    
def _format_results(results:list[dict]) -> str:
    """Format raw DDG results into a readable string for the agent."""
    formatted = []
    
    for i, r in enumerate(results, start=1):
        title = r.get("title", "No title")
        url = r.get("href", "No URL")
        snippet = r.get("body", "No snippet")
        
        formatted.append(
            f"[{i}] {title}\n"
            f"  URL: {url}\n"
            f"  {snippet}"
            )
        
    return "\n\n".join(formatted)
    

# ---Test---
if __name__ == "__main__":
    query = "LangGraph agent tutorial"
    print(f"Searching for: {query}\n")
    print(web_search.invoke(query))