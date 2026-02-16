try:
    from fastmcp import FastMCP
except Exception:  # pragma: no cover - fallback for older MCP SDKs
    from mcp.server.fastmcp import FastMCP

from MCP.tools import register_tools


def create_app():
    mcp = FastMCP("RAG Agent MCP")
    register_tools(mcp)
    return mcp


if __name__ == "__main__":
    create_app().run()
