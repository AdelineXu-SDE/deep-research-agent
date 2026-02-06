"""
RAG retrieval tools for financial research.

This module implements hybrid retrieval over SEC filings using Qdrant
(dense embeddings + sparse BM25), optional metadata filtering extracted
from user queries, and live market data access via Yahoo Finance.
"""


import sys
import subprocess
from langchain_core.tools import tool
from scripts.schema import ChunkMetadata
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os


load_dotenv()


# re-ranking for better result

# metadata filtering

# metadata extraction from LLM

DEBUG = os.getenv("DEBUG_RAG", "false").lower() == "true"

# Configuration
COLLECTION_NAME = "financial_docs"
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"

RERANKER_MODEL = "BAAI/bge-reranker-base"

# ### Initialize LLM and Vector Store

# Initialize LLM
llm = ChatGoogleGenerativeAI(model=LLM_MODEL)

# Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

# Sparse embeddings
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

# Connect to existing collection
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    collection_name=COLLECTION_NAME,
    url="http://localhost:6333",
    retrieval_mode=RetrievalMode.HYBRID,
)


def normalize_filters(filters: dict) -> dict:
    if "company_name" in filters and isinstance(filters["company_name"], str):
        filters["company_name"] = filters["company_name"].lower()

    if "doc_type" in filters and isinstance(filters["doc_type"], str):
        s = filters["doc_type"].lower().replace("_", "-")
        s = s.replace("10q", "10-q").replace("10k",
                                             "10-k").replace("8k", "8-k")
        filters["doc_type"] = s

    if "fiscal_quarter" in filters and isinstance(filters["fiscal_quarter"], str):
        filters["fiscal_quarter"] = filters["fiscal_quarter"].lower()
        # 如果是 "q3" 以外的怪格式，尽量提取 q[1-4]
        import re
        m = re.search(r"q([1-4])", filters["fiscal_quarter"])
        filters["fiscal_quarter"] = f"q{m.group(1)}" if m else filters["fiscal_quarter"]

    if "fiscal_year" in filters:
        try:
            filters["fiscal_year"] = int(filters["fiscal_year"])
        except Exception:
            pass

    # 去掉 None
    return {k: v for k, v in filters.items() if v is not None}

# ### Filter Extraction with LLM


def extract_filters(user_query: str):

    prompt = f"""
            Extract metadata filters from the query. Return None for fields not mentioned.

                <USER QUERY STARTS>
                {user_query}
                </USER QUERY ENDS>

                #### EXAMPLES
                COMPANY MAPPINGS:
                - Amazon/AMZN -> amazon
                - Google/Alphabet/GOOGL/GOOG -> google
                - Apple/AAPL -> apple
                - Microsoft/MSFT -> microsoft
                - Tesla/TSLA -> tesla
                - Nvidia/NVDA -> nvidia
                - Meta/Facebook/FB -> meta

                DOC TYPE:
                - Annual report -> 10-k
                - Quarterly report -> 10-q
                - Current report -> 8-k

                EXAMPLES:
                "Amazon Q3 2024 revenue" -> {{"company_name": "amazon", "doc_type": "10-q", "fiscal_year": 2024, "fiscal_quarter": "q3"}}
                "Apple 2023 annual report" -> {{"company_name": "apple", "doc_type": "10-k", "fiscal_year": 2023}}
                "Tesla profitability" -> {{"company_name": "tesla"}}

                Extract metadata based on the user query only:
            """

    structurerd_llm = llm.with_structured_output(ChunkMetadata)

    metadata = structurerd_llm.invoke(prompt)

    if metadata:
        filters = metadata.model_dump(exclude_none=True)
    else:
        filters = {}

    return normalize_filters(filters)


@tool
def hybrid_search(query: str, k: int = 5):
    """
    Search historical financial documents (SEC filings: 10-K, 10-Q, 8-K) using hybrid search.

    **IMPORTANT: This is the PRIMARY tool for financial research.**
    **ALWAYS call this tool FIRST for ANY financial question unless:**
    - User explicitly asks for "current", "live", "real-time", or "latest" market data
    - User asks about current stock prices or today's market information

    This tool searches through:
    - Historical SEC filings (10-K annual reports, 10-Q quarterly reports)
    - Financial statements, revenue, expenses, cash flow data
    - Company performance metrics from past quarters and years
    - Automatically extracts filters (company, year, quarter, doc type) from your query

    Use this for queries about:
    - Historical revenue, profit, expenses ("What was Amazon's revenue in Q1 2024?")
    - Year-over-year or quarter-over-quarter comparisons
    - Financial metrics from SEC filings
    - Any historical financial data

    Args:
        query: Natural language search query (e.g., "Amazon Q1 2024 revenue")
        k: Number of results to return (default: 5)

    Returns:
        List of Document objects with page content and metadata (source_file, page_number, etc.)
    """

    filters = extract_filters(query)
    # if DEBUG:
    # print("DEBUG filters:", filters)

    qdrant_filter = None

    if filters:
        condition = [
            FieldCondition(key=f"metadata.{key}",
                           match=MatchValue(value=value))
            for key, value in filters.items()
        ]

        qdrant_filter = Filter(must=condition)

    results = vector_store.similarity_search(
        query=query, k=k, filter=qdrant_filter)

    return results


@tool
def live_finance_researcher(query: str):
    """
    Research live stock data using Yahoo Finance MCP.

    Use this tool to get:
    - Current stock prices and real-time market data
    - Latest financial news
    - Stock recommendations and analyst ratings
    - Option chains and expiration dates
    - Recent stock actions (splits, dividends)

    Args:
        query: The financial research question about current market data

    Returns:
        Research results from Yahoo Finance
    """

    code = f"""
import asyncio
from scripts.yahoo_mcp import finance_research
asyncio.run(finance_research("{query}"))
"""
    result = subprocess.run([sys.executable, '-c', code],
                            capture_output=True, text=True)

    return result.stdout


@tool
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"
