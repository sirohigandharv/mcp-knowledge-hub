# sn-mcp-client

A minimal **Python MCP client** that connects (via stdio) to your **ServiceNow MCP server** and lets you:
- Discover tools and their input schemas
- Call tools generically with JSON arguments
- Use shortcuts for common actions:
  - `list_knowledge_bases`
  - `list_articles` for a Knowledge Base
  - `get_article` by sys_id

> ⚠️ Never commit secrets. Use `config.client.json` locally; it's git-ignored. Share `config.client.example.json` only.

## 1) Prereqs

- Python 3.9+
- Your ServiceNow MCP server is accessible via a command + args (same values you use in Claude Desktop config).
- Instance URL and credentials.

## 2) Setup

```bash
# from project root
python -m venv .venv
conda create --name sn_mcp_client python=3.13
# Windows:
.venv\Scripts\activate
conda activate sn_mcp_client
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
# or
pip install -e .
```

Copy the example config and fill your values:

```bash
# Windows PowerShell
Copy-Item config.client.example.json config.client.json
# macOS/Linux
# cp config.client.example.json config.client.json
```

Edit `config.client.json`:
```json
{
  "command": "C:/Users/your_user/servicenow-mcp/.venv/Scripts/python.exe",
  "args": ["-m", "servicenow_mcp.cli"],
  "env": {
    "SERVICENOW_INSTANCE_URL": "https://your-instance.service-now.com",
    "SERVICENOW_USERNAME": "your_username",
    "SERVICENOW_PASSWORD": "your_password",
    "SERVICENOW_AUTH_TYPE": "basic",
    "MCP_TOOL_PACKAG": "knowledge_author"
  }
}
```

> The client will spawn this server process and pass these environment variables.

## 3) Run

Use the CLI entrypoint:
```bash
python -m sn_mcp_client.cli --help
# or if installed:
sn-mcp --help
```

### Discover tools & schemas
```bash
python -m sn_mcp_client.cli discover
```

### List tool names only
```bash
python -m sn_mcp_client.cli tools
```

### Call any tool with JSON args
```bash
python -m sn_mcp_client.cli call --tool list_knowledge_bases --args "{}"
```

### Convenience commands

List Knowledge Bases:
```bash
python -m sn_mcp_client.cli list-kbs
```

List Articles in a KB:
```bash
python -m sn_mcp_client.cli list-articles --kb 429b07aa0b231200263a089b37673a97
```

Get Article by ID (sys_id):
```bash
python -m sn_mcp_client.cli get-article --id 0acc7cbe53e6001089a6ddeeff7b1208
```

> If a convenience command fails due to differing parameter names, run `discover` to see the tool's JSON schema and then use `call` with `--args` to pass the exact keys.

## 4) How it works

- This client uses **stdio** transport and performs an MCP handshake automatically.
- It spawns your ServiceNow MCP server using `command` + `args` from `config.client.json` and injects `env` variables.
- Output is printed as readable JSON.

## 5) Summary of commands

- python -m sn_mcp_client.cli tools
- python -m sn_mcp_client.cli discover
- python -m sn_mcp_client.cli list-kbs
- python -m sn_mcp_client.cli list-articles --kb 429b07aa0b231200263a089b37673a97
- python -m sn_mcp_client.cli get-article --id 0acc7cbe53e6001089a6ddeeff7b1208


Enjoy!



## 6) Web server

- uvicorn sn_mcp_client.api_server:app --reload --port 8000

### Run with openAI key

- set OPENAI_API_KEY=<your_open_api_key>

- uvicorn sn_mcp_client.api_server:app --reload --port 8000 

### Use 

- http://127.0.0.1:8000/tools
- http://127.0.0.1:8000/discover
- http://127.0.0.1:8000/list-kbs
- http://127.0.0.1:8000/list-articles?kb=429b07aa0b231200263a089b37673a97
- http://127.0.0.1:8000/get-article?article_id=0acc7cbe53e6001089a6ddeeff7b1208
