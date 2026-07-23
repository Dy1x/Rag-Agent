import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER_MODULE = "mcp_servers.mysql_mcp_server"


async def call_mysql_mcp_tool_async(tool_name: str, arguments: dict) -> str:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", SERVER_MODULE],
        cwd=str(PROJECT_ROOT),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(
                tool_name,
                arguments=arguments,
            )

            if not result.content:
                return json.dumps(
                    {
                        "found": False,
                        "message": "MCP工具无返回内容",
                        "data": None,
                    },
                    ensure_ascii=False,
                )

            return result.content[0].text


def call_mysql_mcp_tool(tool_name: str, arguments: dict) -> str:
    return asyncio.run(
        call_mysql_mcp_tool_async(tool_name, arguments)
    )