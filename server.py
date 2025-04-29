import sys
from pathlib import Path

import duckdb
from mcp.server.fastmcp import FastMCP

BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR / "packages" / "src"))

from utils.search import Search

BASE_FILE_NAME = "fulltext-20250427"

mcp = FastMCP(
    "国土交通白書2024",
    dependencies=[
        "duckdb",
        "lindera-py",
        "sentence-transformers",
        "sentencepiece",
        "torch",
        "transformers",
    ],
)


@mcp.tool(
    name="search_mlit_whitepaper",
    description="国土交通白書2024から情報を検索します",
)
def search(query: str) -> list[str]:
    db_file = BASE_DIR / f"{BASE_FILE_NAME}.duckdb"
    conn = duckdb.connect(db_file, read_only=True)
    search = Search()

    return [
        result[2] for result in search.hybrid_search(conn, query) if result[1] > 0.2
    ]
