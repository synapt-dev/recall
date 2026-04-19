#!/usr/bin/env python3
"""Minimal Google ADK probe against the local synapt MCP server."""

import asyncio
import uuid

from google.adk.agents import LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.tool_context import ToolContext
from mcp import StdioServerParameters


async def main() -> None:
    toolset = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(command="synapt", args=["server"]),
            timeout=20.0,
        )
    )
    try:
        tools = await toolset.get_tools()
        names = [getattr(tool, "name", type(tool).__name__) for tool in tools]
        print(f"Discovered {len(names)} tools")
        print(", ".join(names[:20]))

        target = next(tool for tool in tools if getattr(tool, "name", "") == "recall_quick")
        agent = LlmAgent(model="gemini-2.5-flash", name="probe_agent", instruction="Probe tools")
        sessions = InMemorySessionService()
        session = await sessions.create_session(app_name="synapt-adk-probe", user_id="local")
        context = ToolContext(
            InvocationContext(
                invocation_id=str(uuid.uuid4()),
                agent=agent,
                session_service=sessions,
                session=session,
            )
        )
        result = await target.run_async(args={"query": "search"}, tool_context=context)
        print("recall_quick call succeeded")
        print(str(result)[:800])
    finally:
        await toolset.close()


if __name__ == "__main__":
    asyncio.run(main())
