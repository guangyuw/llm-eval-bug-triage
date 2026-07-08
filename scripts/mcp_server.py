"""Minimal MCP server: the project's semantic data catalog.

Step 4 extraction is grounded to this catalog instead of prompt-inlined
strings — the field definitions and the official component taxonomy live in
ONE place and are served over MCP (stdio). The notebook is the MCP client.

Run standalone:  python mcp_server.py
"""
import json

import pandas as pd
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("bug-catalog")

FIELD_CATALOG = {
    "fields": {
        "area": "the affected subsystem/topic (short label)",
        "severity": "impact level, one of the allowed enums",
        "kind": "defect category, one of the allowed enums",
    },
    "enums": {
        "severity": ["low", "med", "high"],
        "kind": ["crash", "ui", "performance", "security", "data", "other"],
    },
}


@mcp.tool()
def get_field_catalog() -> str:
    """Field semantics + allowed enums for bug extraction (data dictionary)."""
    return json.dumps(FIELD_CATALOG)


@mcp.tool()
def get_component_taxonomy() -> str:
    """Official component names observed in the corpus (bugs.csv)."""
    comps = pd.read_csv("bugs.csv", usecols=["component"])["component"].dropna().unique()
    return json.dumps(sorted(map(str, comps)))


if __name__ == "__main__":
    mcp.run()
