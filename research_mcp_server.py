# fastmcp app for mcp server(tool provider for agent)
from fastmcp import FastMCP

mcp = FastMCP("Research Tools")


@mcp.tool()
def get_research_sources() -> list[str]:
    """Return allowed research sources."""
    return [
        "Wikipedia",
        "Google",
        "Youtube",
    ]
