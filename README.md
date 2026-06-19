# ContextFabric AI

**Preparing operational data for AI before AI consumes it.**

ContextFabric AI is a personal, open industrial-AI prototype that transforms raw multi-vendor operational signals into trusted, AI-ready context.

It demonstrates how raw PLC, SCADA, historian, and vendor tag names can be converted into reusable operational understanding for downstream AI applications such as Copilots, Digital Twins, Reliability Analytics, Production Intelligence, Energy AI, Quality Systems, and Agentic Operations.

## Why this exists

Industrial AI applications often struggle because raw machine tags do not carry enough semantic, asset, business, or quality context. Similar machines may perform the same function while using different tag names across vendors, lines, and plants.

ContextFabric AI creates a reusable context layer so future AI applications can consume trusted operational meaning instead of repeatedly starting from raw tags.

## What it does

ContextFabric AI creates:

- Asset context
- Semantic context
- Canonical operational parameters
- Unified Namespace-style paths
- Data quality and trust scores
- AI readiness decisions
- Human-in-the-loop exception handling
- SQLite-backed memory for validated mappings
- Per-tag agent decision traces
- Executive value and reuse view

## Architecture

```text
Raw operational signals
        ↓
Memory Agent
        ↓
Context Agent
        ↓
Semantic Reasoning Agent
        ↓
LLM Semantic Reasoning Agent optional fallback
        ↓
Data Quality + AI Readiness Agent
        ↓
Human-in-the-loop governance
        ↓
Reusable AI-ready context layer
```

## Key capabilities

- Evidence-based semantic scoring instead of hardcoded confidence numbers
- Optional LLM fallback using local Ollama or OpenAI
- SQLite-backed memory agent for validated mappings
- Per-tag trace showing which agents fired and why
- Human validation workflow for uncertain mappings
- Executive value view for business outcomes
- AI consumption readiness matrix

## Tech stack

- Python
- Streamlit
- Pandas
- SQLite
- Ollama with Llama 3.1 for local LLM reasoning
- Optional OpenAI API integration

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Optional: enable local LLM reasoning

```bash
ollama serve
ollama pull llama3.1
streamlit run app.py
```

## Optional: enable OpenAI reasoning

```bash
export OPENAI_API_KEY=sk-your-key
streamlit run app.py
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="sk-your-key"
streamlit run app.py
```

## Sample data

Use `sample_context_tags.csv` or upload your own CSV with this structure:

```csv
Line,Machine,Vendor,TagName,Value
Line1,Filler01,VendorA,FIL01_SPEED_RPM,1450
```

## Repository description

Use this as the GitHub repository description:

> Multi-agent Context Fabric platform that transforms raw industrial signals into trusted operational understanding for AI, Digital Twins, Copilots, and Agentic Operations.

## Roadmap

- OPC-UA and MQTT ingestion
- External taxonomy configuration
- UNS publishing
- Knowledge graph persistence
- Plant template onboarding
- Human approval audit workflow
- Multi-site context templates

## License

Personal prototype. Add your preferred open-source license before publishing.
