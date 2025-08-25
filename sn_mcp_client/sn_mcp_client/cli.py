import argparse
import asyncio
import json
from typing import Any, Dict

from .client import MCPConfig, list_tools, call_tool, tool_to_dict, choose_arg_key


def pretty_print(obj: Any):
    try:
        print(json.dumps(obj, indent=2, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        print(str(obj))


def cmd_discover(cfg_path: str):
    cfg = MCPConfig.load(cfg_path)
    tools = asyncio.run(list_tools(cfg))
    print("Discovered tools (with input schemas when available):\n")
    for t in tools:
        tdict = tool_to_dict(t)
        pretty_print(tdict)
        print("-" * 60)


def cmd_tools(cfg_path: str):
    cfg = MCPConfig.load(cfg_path)
    tools = asyncio.run(list_tools(cfg))
    print("Tools:")
    for t in tools:
        name = getattr(t, "name", str(t))
        print(f" - {name}")


def cmd_call(cfg_path: str, tool: str, args_json: str):
    cfg = MCPConfig.load(cfg_path)
    try:
        args: Dict[str, Any] = json.loads(args_json) if args_json else {}
    except json.JSONDecodeError as e:
        raise SystemExit(f"--args must be valid JSON: {e}")
    result = asyncio.run(call_tool(cfg, tool, args))
    pretty_print(result)


def cmd_list_kbs(cfg_path: str):
    cfg = MCPConfig.load(cfg_path)
    result = asyncio.run(call_tool(cfg, "list_knowledge_bases", {}))
    pretty_print(result)


def cmd_list_articles(cfg_path: str, kb: str):
    cfg = MCPConfig.load(cfg_path)
    # Discover schema to choose the right key automatically
    tools = asyncio.run(list_tools(cfg))
    target = next((t for t in tools if getattr(t, "name", "") == "list_articles"), None)
    key = "kb_id"
    if target:
        key = choose_arg_key(getattr(target, "input_schema", {}) or {},
                             ["kb_sys_id", "kb_id", "knowledge_base_id", "id", "kb"],
                             "kb_id")
    args = {key: kb}
    result = asyncio.run(call_tool(cfg, "list_articles", args))
    pretty_print(result)


def cmd_get_article(cfg_path: str, article_id: str):
    cfg = MCPConfig.load(cfg_path)
    # Discover schema to choose the right key automatically
    tools = asyncio.run(list_tools(cfg))
    target = next((t for t in tools if getattr(t, "name", "") == "get_article"), None)
    key = "id"
    if target:
        key = choose_arg_key(getattr(target, "input_schema", {}) or {},
                             ["sys_id", "id", "article_id", "record_id"],
                             "id")
    args = {key: article_id}
    result = asyncio.run(call_tool(cfg, "get_article", args))
    pretty_print(result)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sn-mcp", description="Python MCP client for ServiceNow MCP server (stdio).")
    p.add_argument("--config", default="config.client.json", help="Path to client config JSON (default: config.client.json)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("discover", help="List tools with input schemas (JSON)")
    sub.add_parser("tools", help="List tool names only")

    s = sub.add_parser("call", help="Call a tool with raw JSON args")
    s.add_argument("--tool", required=True, help="Tool name")
    s.add_argument("--args", default="{}", help="JSON string with arguments")

    sub.add_parser("list-kbs", help="Shortcut to call 'list_knowledge_bases'")
    s = sub.add_parser("list-articles", help="Shortcut to call 'list_articles'")
    s.add_argument("--kb", required=True, help="Knowledge Base sys_id")
    s = sub.add_parser("get-article", help="Shortcut to call 'get_article'")
    s.add_argument("--id", required=True, help="Article sys_id")

    return p


def main():
    parser = build_parser()
    ns = parser.parse_args()
    cfg_path = ns.config

    if ns.cmd == "discover":
        cmd_discover(cfg_path)
    elif ns.cmd == "tools":
        cmd_tools(cfg_path)
    elif ns.cmd == "call":
        cmd_call(cfg_path, ns.tool, ns.args)
    elif ns.cmd == "list-kbs":
        cmd_list_kbs(cfg_path)
    elif ns.cmd == "list-articles":
        cmd_list_articles(cfg_path, ns.kb)
    elif ns.cmd == "get-article":
        cmd_get_article(cfg_path, ns.id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
