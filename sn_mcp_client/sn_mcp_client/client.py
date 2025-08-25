import os
import json
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

# Try to import SSE client (if available in MCP library)
try:
    from mcp.client.sse import sse_client
    HAS_SSE = True
except ImportError:
    HAS_SSE = False


class MCPConfig:
    """Holds how to launch the server and its environment."""
    def __init__(self, mode: str, command: Optional[str], args: List[str],
                 env: Optional[Dict[str, str]] = None):
        self.mode = mode  # "stdio" or "sse"
        self.command = command
        self.args = args or []
        self.env = env or {}

    @staticmethod
    def load(path: str) -> "MCPConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print("Loaded config:", data)

        mode = data.get("mode")
        if not mode:
            # Auto-detect: if command exists → stdio, else → sse
            mode = "stdio" if "command" in data else "sse"

        return MCPConfig(
            mode=mode,
            command=data.get("command"),
            args=data.get("args", []),
            env=data.get("env", {}),
        )


@asynccontextmanager
async def session_from_config(cfg: MCPConfig):
    """Start the MCP server via stdio or connect via SSE based on config."""
    # Ensure environment variables are set
    if cfg.env:
        os.environ.update(cfg.env)

    if cfg.mode == "sse":
        if not HAS_SSE:
            raise RuntimeError("SSE mode requested but `mcp.client.sse` not available.")
        mcp_url = cfg.env.get("MCP_SERVER_URL", "http://localhost:8080")
        print(f"Connecting to MCP Server via SSE at {mcp_url}")
        async with sse_client(mcp_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
    else:
        # Default: stdio mode (spawn the process)
        print(f"Starting MCP server via stdio: {cfg.command} {' '.join(cfg.args)}")
        server_params = StdioServerParameters(
            command=cfg.command,
            args=cfg.args,
            env=cfg.env
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


async def list_tools(cfg: MCPConfig):
    async with session_from_config(cfg) as session:
        tools = await session.list_tools()
        return tools


async def call_tool(cfg: MCPConfig, tool_name: str, args: Dict[str, Any]):
    async with session_from_config(cfg) as session:
        return await session.call_tool(tool_name, args)


def tool_to_dict(tool: Any) -> Dict[str, Any]:
    """Convert a tool descriptor to a plain dict for pretty printing."""
    out: Dict[str, Any] = {}
    for attr in ("name", "description", "input_schema"):
        if hasattr(tool, attr):
            out[attr] = getattr(tool, attr)
    if not out:
        try:
            return json.loads(json.dumps(tool, default=lambda o: getattr(o, "__dict__", str(o))))
        except Exception:
            return {"repr": repr(tool)}
    return out


def choose_arg_key(schema: Dict[str, Any], candidates: List[str], default_key: str) -> str:
    """From a JSON schema, pick the first candidate key in properties, else fallback."""
    props: Dict[str, Any] = {}
    if schema and isinstance(schema, dict):
        props = schema.get("properties", {}) or {}
    for k in candidates:
        if k in props:
            return k
    return default_key
