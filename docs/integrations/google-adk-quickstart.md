# Google ADK quickstart

Premium boundary: core OSS. This document covers the public `synapt server` MCP surface and how to consume it from Google ADK.

## Compatibility result

`synapt server` works with Google ADK's MCP client surface.

Verified locally on April 19, 2026 with:

- `google-adk==1.31.0`
- `mcp==1.27.0`
- local stdio connection through `google.adk.tools.mcp_tool.McpToolset`

What was exercised:

- ADK discovered 29 tools from `synapt server`
- tool discovery included core recall operations such as `recall_search`, `recall_save`, and `recall_channel`
- direct execution of `recall_quick` through the ADK MCP wrapper succeeded

What was not fully exercised here:

- Google Cloud API Registry deployment
- Application Default Credentials against a real registered MCP server in Google Cloud

The ADK `ApiRegistry` connector exists in the package and is the right distribution path for a hosted deployment, but it requires a Google Cloud project, a registered MCP server, and ADC with the relevant registry permissions.

## Local quickstart

1. Create and activate a virtual environment.

```bash
cd synapt
python3 -m venv .venv
source .venv/bin/activate
```

2. Install `synapt` and Google ADK.

```bash
python -m pip install -U pip
python -m pip install -e ".[test]" google-adk
```

3. Run the compatibility probe.

```bash
python scripts/google_adk_probe.py
```

Expected result:

- ADK lists the available `synapt` tools
- `recall_quick` executes successfully through the MCP wrapper

## Minimal ADK agent example

```python
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="synapt_agent",
    instruction="Use recall tools when they help answer the user.",
    tools=[
        McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command="synapt",
                    args=["server"],
                ),
                timeout=20.0,
            )
        )
    ],
)
```

## Cloud API Registry path

For Google Cloud deployment, ADK exposes `ApiRegistry` as the registry-backed connector:

```python
from google.adk.integrations.api_registry import ApiRegistry

api_registry = ApiRegistry(api_registry_project_id="your-project-id", location="global")
toolset = api_registry.get_toolset(
    "projects/your-project-id/locations/global/mcpServers/your-mcp-server-name"
)
```

That path was not exercised in this workspace because it depends on:

- a real Google Cloud project
- a registered MCP server entry in Cloud API Registry
- Application Default Credentials with registry access

So the current recommendation is:

- use the local stdio connector for development and compatibility testing
- use `ApiRegistry` only once the hosted MCP deployment exists and ADC can be validated in CI or a staging project
