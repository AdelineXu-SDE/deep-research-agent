# 🧠 Deep Research Agent (Financial Domain)

A production-oriented **agentic deep research system** for financial analysis, built with **LangChain / LangGraph / DeepAgents**, supporting **multi-step planning, sub-agent delegation, tool-augmented reasoning, and persistent memory**.

This project demonstrates **two parallel implementations** of the same research capability:

- A **LangGraph-based agent system** (low-level, high control)
- A **DeepAgents-based agent system** (high-level, faster composition)

Both are designed to solve **real-world financial research tasks**, such as:

- Extracting company revenue from SEC filings
- Combining live financial data with RAG
- Producing structured, verifiable research reports

---

## ✨ Key Features

- 🧩 **Multi-agent architecture**
  - Orchestrator agent + specialized financial research sub-agents
- 🗂 **Tool-augmented reasoning**
  - Hybrid RAG search (vector + keyword)
  - Live financial data (Yahoo Finance MCP)
  - File-system based research artifacts
- 💾 **Persistent memory**
  - SQLite-based checkpointing across sessions
- 🧠 **Explicit planning & verification**
  - Task decomposition → research → synthesis → verification
- 🔀 **Two architectural styles**
  - LangGraph (fine-grained control)
  - DeepAgents (rapid agent composition)

---

## 🏗 Project Architecture

### High-level Flow

```text
User Query
   ↓
Orchestrator Agent
   ├─ Plans research steps
   ├─ Delegates tasks
   ↓
Financial Research Sub-Agent(s)
   ├─ Retrieve SEC / financial data
   ├─ Run hybrid RAG search
   ├─ Use live finance tools
   ↓
Synthesis & Verification
   ↓
Final Research Report (Markdown)

```

## 🧠 Two Implementations

To explore different levels of agent abstraction and control, this project provides two parallel implementations of the same financial research capability:

### 1️⃣ LangGraph-Based Agent (Low-Level Control)

- 📂 **Entrypoint**
  - `apps/finance_langgraph.py`

- ⚙️ **Characteristics**
  - Explicit graph-style orchestration
  - Custom agent state schema
  - Fine-grained control over:
    - Tool calls
    - Agent transitions
    - Memory checkpoints

- 🎯 **Best for**
  - Complex workflows
  - Custom agent logic
  - Research / infra-level agent systems

---

### 2️⃣ DeepAgents-Based Agent (High-Level Composition)

- 📂 **Entrypoint**
  - `apps/finance_deepagents.py`

- ⚙️ **Characteristics**
  - Built on top of LangGraph
  - Built-in support for:
    - Sub-agents
    - File-system backends
    - Planning & todo tracking
  - Less boilerplate, faster iteration

- 🎯 **Best for**
  - Rapid prototyping
  - Production-ready agent scaffolding
  - Clear separation of orchestration vs execution

## 📁 Project Structure

```text
deep-research-agent/
├── apps/                    # Runnable agent entrypoints
│   ├── finance_langgraph.py
│   └── finance_deepagents.py
├── scripts/                 # Core logic, tools, and offline pipelines
│   ├── agent_utils.py
│   ├── deep_prompts.py
│   ├── prompts.py
│   ├── file_tools.py
│   ├── rag_tools.py
│   ├── schema.py
│   ├── yahoo_mcp.py
│   └── ingest_qdrant.py     # Build HYBRID Qdrant index from SEC filings
├── data/                    # SQLite checkpoints (gitignored)
├── research_outputs/        # Generated research artifacts (gitignored)
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 How to Run

### 1️⃣ Setup Environment

- 🐍 **Create virtual environment**
  - `python -m venv .venv`
  - `source .venv/bin/activate`

- 📦 **Install dependencies**
  - `pip install -r requirements.txt`

- 🔐 **Configure environment variables**
  - Create `.env` file from template
    - `cp .env.example .env`
  - Fill in required keys:
    - `GOOGLE_API_KEY`
    - `QDRANT_API_KEY`
    - `QDRANT_URL`
    - _(Optional)_ `LANGSMITH_API_KEY`

## 📄 Example Output

- 📊 **Key Result**
  - Amazon Q1 2024 Revenue: **$143.313B**
  - Year-over-Year Growth: **+13%**

- 📚 **Sources**
  - Amazon Form 10-Q (Q1 2024)

- 🗂 **Generated Artifacts**
  - Research plan (`research_plan.md`)
  - Sub-agent research notes (`researcher/*.md`)
  - Final report (`final_report.md`)
