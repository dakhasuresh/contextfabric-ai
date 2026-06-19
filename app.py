"""
app.py
------
ContextFabric AI -- v1

What changed from the original prototype, and why:

1. Real agent pipeline (agents.py) replaces the single hardcoded if/elif
   function. Confidence is computed from evidence, not copy-pasted per
   branch. See agents.rule_based_semantic_agent().

2. A real LLM agent (OpenAI Chat Completions) is wired in as a genuine
   fallback step for tags the rule engine can't confidently resolve.
   It is feature-flagged via the OPENAI_API_KEY environment variable --
   works with zero key today (falls back to honest "Human Review"),
   and activates automatically the moment a key is present. No code
   changes needed to go from offline demo to live LLM-in-the-loop demo.

3. A real Memory Agent (memory_store.py, SQLite) persists validated
   mappings so the SAME tag string is recognized instantly on a
   different plant/line/vendor in a later run -- this is demonstrated
   live in the "Memory & Reuse" tab.

4. A visible per-tag agent trace -- you can expand any tag and see the
   actual sequence of agent decisions (memory miss -> context -> rule
   score -> escalate/skip -> readiness decision), not just a final number.

5. A human-in-the-loop resolution workflow for "Human Review Required"
   tags -- resolving one writes it to memory immediately.

Honesty notes kept visible in the UI on purpose:
- The LLM status indicator always shows whether the key is configured,
  so nobody can mistake "rules-only mode" for "agentic AI" during Q&A.
- Confidence scores are never silently rounded up to look more
  impressive than the underlying evidence.
"""

import streamlit as st
import pandas as pd
import json
import time

import agents
import memory_store

memory_store.init_db()

st.set_page_config(
    page_title="ContextFabric AI",
    page_icon="🏭",
    layout="wide"
)

REQUIRED_COLUMNS = ["Line", "Machine", "Vendor", "TagName", "Value"]

st.title("🏭 ContextFabric AI")
st.subheader("Industrial semantic context layer for AI-ready operations")
st.success("Prepare industrial data for AI before AI consumes it.")

# ---------------------------------------------------------------------------
# LLM status banner -- always visible, always honest about current mode.
# ---------------------------------------------------------------------------
if agents.LLM_ENABLED:
    backend_label = {"openai": f"OpenAI (`{agents.OPENAI_MODEL}`)", "ollama": f"Ollama (`{agents.OLLAMA_MODEL}`, local)"}[agents.LLM_BACKEND]
    st.info(f"🟢 **LLM Semantic Reasoning Agent: ACTIVE** — backend: {backend_label}. Escalated tags will be resolved live.")
else:
    st.warning(
        "🟡 **LLM Semantic Reasoning Agent: OFFLINE** — no model backend reachable. "
        "Ambiguous tags will be routed to Human Review instead of guessed. "
        "Either start Ollama locally (`ollama serve`, then `ollama pull llama3.1`) "
        "or set `export OPENAI_API_KEY=sk-...`, then re-run."
    )

with st.expander("ℹ️ How this differs from a keyword-matching demo (read this before judging)"):
    st.markdown("""
- **Confidence is computed, not hardcoded.** Every score below is derived from how many independent
  semantic cues matched the tag name, and is penalized when cues from competing categories collide.
- **Memory is real.** Validated mappings are stored in a local SQLite file (`context_fabric_memory.db`)
  and are reused across plants, lines, and vendors in later runs — see the *Memory & Reuse* tab.
- **The LLM agent is a real, working API integration**, feature-flagged off until a key is provided —
  it is not a placeholder.
- **Every tag has a visible agent trace** — expand any row in *Agent Trace* to see exactly which agents
  fired and why, not just a final label.
""")

st.markdown("""
### Scenario
A manufacturer operates similar machines across vendors, lines and plants. Each machine performs the same operational function, but every vendor uses different tag names.

ContextFabric AI converts raw industrial signals into trusted operational understanding:
- Asset Context · Semantic Context · Canonical Manufacturing Model · Unified Namespace Paths
- Data Quality & Trust Scoring · Exception Investigation · AI Consumption Readiness · Business Impact View

The output is reusable by Copilots, Agentic AI, Digital Twins, Reliability Analytics, Production Intelligence, Energy AI and Quality Systems.
""")


# ---------------------------------------------------------------------------
# Sample data (same as before, plus one deliberately ambiguous tag to
# demonstrate the conflict-detection behavior live)
# ---------------------------------------------------------------------------
def create_sample_data():
    return pd.DataFrame([
        ["Line1", "Filler01", "VendorA", "FIL01_RUN_FB", 1],
        ["Line1", "Filler01", "VendorA", "FIL01_SPEED_RPM", 1450],
        ["Line1", "Filler01", "VendorA", "FIL01_MOTOR_TEMP", 68],
        ["Line1", "Filler01", "VendorA", "FIL01_FAULT_CODE", 0],
        ["Line1", "Filler01", "VendorA", "FIL01_CYCLE_COUNT", 12450],
        ["Line1", "Filler01", "VendorA", "FIL01_AIR_PRESS", 6.8],
        ["Line1", "Filler01", "VendorA", "FIL01_ENERGY_KWH", 22.5],
        ["Line1", "Filler01", "VendorA", "FIL01_GOOD_COUNT", 850],
        ["Line1", "Filler02", "VendorB", "RUNNING_STS", 1],
        ["Line1", "Filler02", "VendorB", "DRV_ACTSPD", 1442],
        ["Line1", "Filler02", "VendorB", "MTR_TMP_ACT", 70],
        ["Line1", "Filler02", "VendorB", "ALM_ACTIVE_CODE", 0],
        ["Line1", "Filler02", "VendorB", "CNT_TOTAL_CYC", 11980],
        ["Line1", "Filler02", "VendorB", "PNEU_AIR_BAR", 6.7],
        ["Line1", "Filler02", "VendorB", "PWR_KWH_TOT", 23.1],
        ["Line1", "Filler02", "VendorB", "GOOD_PARTS_CNT", 842],
        ["Line1", "Filler03", "VendorC", "MC3_RUNNING", 1],
        ["Line1", "Filler03", "VendorC", "MC3_SPEED_ACTUAL", 1438],
        ["Line1", "Filler03", "VendorC", "MC3_TEMP_MOTOR", 69],
        ["Line1", "Filler03", "VendorC", "MC3_PART_COUNT", 839],
        ["Line1", "Filler02", "VendorB", "XYZ99_UNKNOWN", 123],
        ["Line1", "Filler02", "VendorB", "HR_EMPLOYEE_ID", 7845],
    ], columns=REQUIRED_COLUMNS)


def validate_df(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error("Invalid CSV. Required columns: " + ", ".join(REQUIRED_COLUMNS))
        st.stop()


# ---------------------------------------------------------------------------
# Upload / sample selection
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload Context Fabric CSV", type=["csv"])

sample_df = create_sample_data()
col_a, col_b = st.columns(2)
with col_a:
    st.download_button(
        "⬇️ Download Sample Context Fabric Tags",
        sample_df.to_csv(index=False),
        "sample_context_tags.csv",
        "text/csv"
    )
with col_b:
    if st.button("🧹 Reset Memory (clear learned mappings)"):
        memory_store.clear_memory()
        st.success("Memory cleared. Next run starts fresh.")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    validate_df(df)
else:
    st.info("Using built-in sample dataset. You can also upload sample_context_tags.csv.")
    df = sample_df

st.markdown("---")
st.subheader("1. Raw Multi-Vendor Machine Tags")
st.dataframe(df, use_container_width=True)

mem_count = memory_store.get_memory_count()
st.caption(f"🧠 Memory Agent currently holds **{mem_count}** validated tag mapping(s) from prior runs.")


# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
if st.button("🚀 Build AI-Ready Context View", type="primary"):
    progress = st.progress(0)
    log = st.empty()
    results = []

    for idx, row in df.iterrows():
        log.info(f"Supervisor Agent dispatching {row['Vendor']} tag from {row['Machine']}: {row['TagName']}")
        result = agents.run_pipeline_for_tag(row.to_dict())
        results.append(result)

        # Auto-commit high-confidence LLM/rule resolutions to memory so the
        # "reuse across onboarding waves" story is demonstrably true on
        # the very next run, not just in theory.
        if result.mapping_status == "Mapped" and result.source in ("llm", "rule") and result.confidence >= 85:
            agents.commit_to_memory(result, validated_by="auto_high_confidence")

        progress.progress((idx + 1) / len(df))
        time.sleep(0.02)

    log.success("ContextFabric AI view generated successfully.")
    st.session_state["results"] = results
    st.session_state["source_df"] = df

# ---------------------------------------------------------------------------
# Render results (persisted in session_state so human-review actions below
# don't wipe the view on rerun)
# ---------------------------------------------------------------------------
if "results" in st.session_state:
    results = st.session_state["results"]
    df = st.session_state["source_df"]

    rows = []
    for r in results:
        rows.append({
            "Enterprise": "Enterprise",
            "Site": "Site01",
            "Area": "Manufacturing",
            "Line": r.line,
            "Machine": r.machine,
            "Vendor": r.vendor,
            "Raw Tag": r.raw_tag,
            "Asset Class": r.asset_class,
            "Canonical Parameter": r.canonical_parameter,
            "Standard Signal Name": r.standard_signal_name,
            "Unit": r.unit,
            "Business Meaning": r.business_meaning,
            "UNS Path": r.uns_path,
            "Mapping Status": r.mapping_status,
            "AI Readiness": "Ready" if r.mapping_status == "Mapped" else "Not Ready",
            "Context Completeness": "Complete" if r.mapping_status == "Mapped" else "Incomplete",
            "Data Quality Status": r.quality_status,
            "Trust Score %": r.trust_score,
            "Confidence %": r.confidence,
            "Source": r.source,
            "Issue Detected": r.issue,
            "Recommended Action": r.action,
            "Semantic Reasoning Trace": r.reasoning_trace,
            "Recommended AI Use Cases": r.usecases,
            "Publish Decision": "Publish" if r.mapping_status == "Mapped" else "Block",
        })
    context_df = pd.DataFrame(rows)

    total = len(context_df)
    mapped = len(context_df[context_df["Mapping Status"] == "Mapped"])
    review = len(context_df[context_df["Mapping Status"] == "Human Review Required"])
    rejected = len(context_df[context_df["Mapping Status"] == "Rejected"])

    readiness = int((mapped / total) * 100) if total else 0
    asset_coverage = int((len(context_df[context_df["Asset Class"] != "Unknown"]) / total) * 100) if total else 0
    semantic_coverage = int((len(context_df[context_df["Standard Signal Name"] != "Unknown"]) / total) * 100) if total else 0
    uns_coverage = int((len(context_df[context_df["UNS Path"].astype(str).str.len() > 0]) / total) * 100) if total else 0
    dq_score = int((len(context_df[context_df["Data Quality Status"] == "Valid"]) / total) * 100) if total else 0
    avg_trust = context_df["Trust Score %"].mean() if total else 0
    ai_grounding_score = int((semantic_coverage + dq_score + avg_trust) / 3) if total else 0
    enterprise_standardization = int((readiness + uns_coverage + asset_coverage) / 3) if total else 0
    overall_ai_readiness = int((asset_coverage + semantic_coverage + uns_coverage + dq_score + ai_grounding_score + enterprise_standardization) / 6) if total else 0

    solution_tab, trace_tab, memory_tab, executive_tab = st.tabs([
        "Solution View — Industrial Context Fabric",
        "🔍 Agent Trace",
        "🧠 Memory & Reuse",
        "Executive Value Realization",
    ])

    # =======================================================================
    with solution_tab:
        st.markdown("---")
        st.subheader("2. Platform Outcome")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Tags Processed", total)
        k2.metric("Published to Context Fabric", mapped)
        k3.metric("Human Review", review)
        k4.metric("Rejected", rejected)

        st.metric("Overall Industrial AI Readiness", f"{overall_ai_readiness}%")
        st.progress(overall_ai_readiness / 100)

        s1, s2, s3, s4, s5, s6 = st.columns(6)
        s1.metric("Asset Context", f"{asset_coverage}%")
        s2.metric("Semantic Context", f"{semantic_coverage}%")
        s3.metric("UNS Readiness", f"{uns_coverage}%")
        s4.metric("Data Quality", f"{dq_score}%")
        s5.metric("AI Grounding", f"{ai_grounding_score}%")
        s6.metric("Enterprise Standardization", f"{enterprise_standardization}%")

        st.markdown("---")
        st.subheader("3. Before vs After: Raw Tags to Operational Understanding")
        left, right = st.columns(2)
        with left:
            st.markdown("### Before: Vendor-Specific Tags")
            st.code("\n".join(df["Vendor"].astype(str) + " | " + df["Machine"].astype(str) + " | " + df["TagName"].astype(str)))
        with right:
            st.markdown("### After: Enterprise Canonical Context")
            canonical_summary = context_df["Canonical Parameter"].value_counts().to_dict()
            st.success("\n".join([f"- {k}: {v}" for k, v in canonical_summary.items()]))

        st.markdown("---")
        st.subheader("4. Industrial Context Fabric")
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Asset Context", "Semantic Reasoning", "UNS Context",
            "Trust & Quality", "AI Consumption Readiness", "AI Consumption Record"
        ])
        with tab1:
            st.dataframe(context_df[["Enterprise", "Site", "Area", "Line", "Machine", "Vendor", "Raw Tag", "Asset Class"]], use_container_width=True)
        with tab2:
            st.dataframe(context_df[["Raw Tag", "Canonical Parameter", "Standard Signal Name", "Unit", "Business Meaning", "Semantic Reasoning Trace", "Confidence %", "Source"]], use_container_width=True)
        with tab3:
            st.dataframe(context_df[["Machine", "Vendor", "Raw Tag", "Canonical Parameter", "UNS Path", "Publish Decision"]], use_container_width=True)
            st.markdown("**UNS Pattern:** `Enterprise/Site/Area/Line/Machine/Canonical Parameter`")
        with tab4:
            st.dataframe(context_df[["Raw Tag", "Mapping Status", "AI Readiness", "Context Completeness", "Data Quality Status", "Trust Score %", "Publish Decision"]], use_container_width=True)
        with tab5:
            def build_ai_consumption_matrix(context_df):
                mapped_params = set(context_df[context_df["Mapping Status"] == "Mapped"]["Canonical Parameter"])
                rows = [
                    ["Copilot / Natural Language Query", {"machine.status.running", "machine.fault.code"}, "Factory Q&A and operator assistant"],
                    ["Agentic AI Operations", {"machine.status.running", "machine.fault.code", "machine.speed.actual"}, "Autonomous investigation and guided actions"],
                    ["Digital Twin", {"machine.speed.actual", "machine.status.running", "machine.production.cycle_count"}, "Operational state and machine behaviour model"],
                    ["Reliability Analytics", {"machine.motor.temperature", "machine.fault.code", "machine.speed.actual"}, "Asset health and failure indicators"],
                    ["Production Intelligence", {"machine.status.running", "machine.speed.actual", "machine.production.good_count"}, "Throughput, OEE and production performance"],
                    ["Energy Optimization", {"machine.energy.consumption", "machine.utility.air_pressure"}, "Energy usage and compressed-air efficiency"],
                    ["Quality Systems", {"machine.production.good_count", "machine.production.cycle_count"}, "Yield, reject correlation and quality trend analysis"],
                ]
                output = []
                for capability, required, value in rows:
                    present = len(required.intersection(mapped_params))
                    score = int((present / len(required)) * 100)
                    status_label = "Ready" if score >= 90 else ("Needs Minor Review" if score >= 75 else "Not Ready")
                    output.append({"AI Capability": capability, "Readiness %": score, "Status": status_label, "Signals Available": f"{present}/{len(required)}", "Business Value": value})
                return pd.DataFrame(output)
            st.dataframe(build_ai_consumption_matrix(context_df), use_container_width=True)
            st.success("The same context foundation can be consumed by multiple AI applications without repeated engineering effort.")
        with tab6:
            grounding_df = context_df[["Machine", "Vendor", "Raw Tag", "Canonical Parameter", "Business Meaning", "Unit", "UNS Path", "Data Quality Status", "AI Readiness", "Recommended AI Use Cases"]].copy()
            st.dataframe(grounding_df, use_container_width=True)
            sample_record = grounding_df[grounding_df["AI Readiness"] == "Ready"].head(1).to_dict(orient="records")
            if sample_record:
                st.markdown("### Example AI Consumption Record")
                st.code(json.dumps(sample_record[0], indent=2), language="json")

        st.markdown("---")
        st.subheader("5. Industrial Data Governance — Human-in-the-Loop Review")
        exceptions = context_df[context_df["Mapping Status"] != "Mapped"]
        if len(exceptions) > 0:
            st.warning("Some tags should not be published to the AI-ready context fabric yet.")
            st.dataframe(exceptions[["Machine", "Vendor", "Raw Tag", "Mapping Status", "Issue Detected", "Recommended Action", "Confidence %", "Publish Decision"]], use_container_width=True)

            review_items = [r for r in results if r.mapping_status == "Human Review Required"]
            for r in review_items:
                with st.expander(f"🔎 Investigation: {r.raw_tag}  (confidence {r.confidence}%)"):
                    st.write(f"**Issue:** {r.issue}")
                    st.write(f"**Agent reasoning:** {r.reasoning_trace}")
                    st.write(f"**Current best guess:** {r.canonical_parameter}")

                    options = ["-- select correct mapping --"] + list(agents.CANONICAL_TAXONOMY.keys()) + ["not_factory_relevant"]
                    choice = st.selectbox("Engineer validation — correct canonical parameter:", options, key=f"resolve_{r.raw_tag}")
                    if st.button(f"✅ Confirm mapping for {r.raw_tag}", key=f"confirm_{r.raw_tag}"):
                        if choice != options[0]:
                            meta = agents.CANONICAL_TAXONOMY.get(choice, {})
                            r.canonical_parameter = choice
                            r.standard_signal_name = meta.get("signal", "Non-factory enterprise data")
                            r.unit = meta.get("unit", "N/A")
                            r.business_meaning = meta.get("business_meaning", "Enterprise/HR data.")
                            r.asset_class = meta.get("asset_class", "Non-factory data")
                            r.mapping_status = "Rejected" if choice == "not_factory_relevant" else "Mapped"
                            r.confidence = 98
                            r.trust_score = 98 if r.mapping_status == "Mapped" else 0
                            r.source = "human"
                            agents.commit_to_memory(r, validated_by="engineer_ui")
                            st.success(f"Saved to Memory Agent. '{r.raw_tag}' will be recognized automatically on future onboarding waves.")
                            st.rerun()
        else:
            st.success("No exceptions detected. All tags are AI-ready.")

        st.markdown("---")
        st.subheader("6. Agentic Workflow (as actually executed)")
        st.markdown("""
**Supervisor Agent** dispatches each tag through, in order:
1. **Memory Agent** — checks SQLite-backed memory for an exact prior-validated match
2. **Context Agent** — resolves enterprise/site/area/line/machine hierarchy
3. **Semantic Reasoning Agent (rules)** — scores canonical candidates from token evidence, computing confidence from evidence count and category conflicts
4. **LLM Semantic Reasoning Agent** — real OpenAI API call, fires only when rule confidence falls below threshold and a key is configured
5. **AI Readiness Agent** — decides Publish / Human Review / Reject
6. **Memory Write** — high-confidence resolutions and human validations are persisted for reuse

Expand any tag in the **Agent Trace** tab to see this sequence rendered exactly as it executed for that tag.
""")

        st.markdown("---")
        st.subheader("7. Final Solution Outcome")
        st.info(f"""
Industrial Context Fabric Created

✓ Asset Context Established
✓ Semantic Meaning Established
✓ Canonical Manufacturing Model Generated
✓ Unified Namespace Hierarchy Generated
✓ Data Quality Validation Completed (computed, not hardcoded)
✓ Exception Investigation Completed
✓ AI Consumption Readiness Assessed
✓ LLM Semantic Reasoning Agent: {f"Active ({agents.LLM_BACKEND})" if agents.LLM_ENABLED else "Configured but inactive (no model backend reachable)"}
✓ Output Ready For Enterprise AI Consumption

The AI application changes. The context foundation remains the same.
""")
        st.dataframe(context_df, use_container_width=True)

    # =======================================================================
    with trace_tab:
        st.subheader("🔍 Per-Tag Agent Decision Trace")
        st.caption("Exactly which agents fired, in what order, and why — for every tag in this run.")
        for r in results:
            badge = {"Mapped": "🟢", "Human Review Required": "🟡", "Rejected": "🔴"}.get(r.mapping_status, "⚪")
            with st.expander(f"{badge} {r.raw_tag}  →  {r.canonical_parameter}  ({r.mapping_status}, {r.confidence}% confidence, source: {r.source})"):
                for step in r.trace:
                    st.markdown(f"- **{step.agent}** · `{step.action}` — {step.detail}")

    # =======================================================================
    with memory_tab:
        st.subheader("🧠 Memory Agent — Learned Mappings")
        st.caption("Validated mappings persisted in context_fabric_memory.db. These are recognized automatically on future uploads, regardless of plant, line, or vendor.")
        mem_rows = memory_store.get_all_memory()
        if mem_rows:
            mem_df = pd.DataFrame(mem_rows)[["tag_pattern", "canonical_parameter", "confidence", "source", "validated_by", "raw_examples"]]
            st.dataframe(mem_df, use_container_width=True)
            st.info(f"**{len(mem_rows)} tag pattern(s)** will be resolved instantly from memory on the next onboarding wave, skipping rule and LLM steps entirely.")
        else:
            st.write("No learned mappings yet. Resolve a Human Review item or run a high-confidence batch to populate memory.")

        st.markdown("---")
        st.subheader("Recent Memory Activity Log")
        log_rows = memory_store.get_recent_log(15)
        if log_rows:
            st.dataframe(pd.DataFrame(log_rows)[["tag_pattern", "action", "details"]], use_container_width=True)

    # =======================================================================
    with executive_tab:
        st.markdown("---")
        st.subheader("Executive Value Realization")
        st.caption("This tab explains why the solution matters to CTOs, transformation leaders and business sponsors.")

        st.success("""
Most Industrial AI projects spend significant time preparing data before AI creates value.
ContextFabric AI removes that bottleneck by transforming raw industrial signals into trusted operational understanding that any AI application can consume.
""")

        st.markdown("### 1. Current State")
        c1, c2, c3, c4 = st.columns(4)
        c1.error("Multiple vendor naming standards")
        c2.error("Repeated engineering effort")
        c3.error("Slow AI onboarding")
        c4.error("Knowledge trapped in experts")

        st.markdown("### 2. Transformation Achieved")
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Vendor Standards Unified", context_df["Vendor"].nunique())
        t2.metric("Operational Signals Standardized", mapped)
        t3.metric("AI Readiness Achieved", f"{overall_ai_readiness}%")
        t4.metric("Enterprise Context Models", context_df["Canonical Parameter"].nunique())

        st.markdown("### 3. Executive Business Outcomes")
        manual_minutes_per_tag = 18
        agent_minutes_per_tag = 4
        manual_hours = round((total * manual_minutes_per_tag) / 60, 1)
        agent_hours = round((total * agent_minutes_per_tag + review * 12) / 60, 1)
        effort_reduction = int(max(0, ((manual_hours - agent_hours) / manual_hours) * 100)) if manual_hours else 0
        business_df = pd.DataFrame([
            ["Manual engineering mapping effort", f"~{manual_hours} hrs", "Baseline spreadsheet-driven interpretation"],
            ["Agent-assisted context creation effort", f"~{agent_hours} hrs", "Includes review effort for exceptions"],
            ["Estimated effort reduction", f"{effort_reduction}%", "Illustrative demo estimate"],
            ["Repeated AI integration work", "Reduced", "Reusable context fabric consumed by multiple AI apps"],
            ["Invalid data entering AI layer", f"Blocked: {review + rejected}", "Unknown and non-manufacturing tags not published"],
            ["Tags resolved instantly from memory", f"{sum(1 for r in results if r.source == 'memory')}", "Zero re-engineering cost on repeat onboarding"],
            ["AI onboarding acceleration", "Weeks → Days/Hours", "For repeatable machine families and plant rollouts"],
        ], columns=["Business Outcome", "Result", "Explanation"])
        st.dataframe(business_df, use_container_width=True)

        st.markdown("### 4. Enterprise Transformation View")
        before, middle, after = st.columns(3)
        with before:
            st.markdown("#### Before")
            st.warning("Plant A → Vendor A tags\nPlant B → Vendor B tags\nPlant C → Vendor C tags\n\nEvery AI project starts with data discovery and manual context building.")
        with middle:
            st.markdown("#### Industrial Context Fabric")
            st.info("Asset Context\nSemantic Context\nCanonical Model\nUNS Hierarchy\nTrust Scoring\nAI Readiness")
        with after:
            st.markdown("#### After")
            st.success("One enterprise context layer\nMany AI applications\nReusable context\nReduced repeated engineering")

        st.markdown("### 5. Future AI Consumers")
        consumers = pd.DataFrame([
            ["Copilot", "Natural language operational query and assistant experience"],
            ["Agentic AI", "Investigation, recommendation and guided action workflows"],
            ["Digital Twin", "Consistent machine and signal context for twin models"],
            ["Reliability AI", "Asset health, fault patterns and failure indicators"],
            ["Production Intelligence", "OEE, throughput and line performance insight"],
            ["Energy AI", "Energy consumption, compressed air and sustainability analysis"],
            ["Quality Systems", "Yield, good count and process-quality correlation"],
        ], columns=["AI Consumer", "Value Enabled"])
        st.dataframe(consumers, use_container_width=True)

        st.markdown("### 6. Funding Message")
        st.info("""
This is not a single AI use case.
It is an enterprise industrial intelligence foundation.

The platform prepares industrial data once, then reuses the trusted context across many future AI initiatives.
""")

    st.markdown("---")
    st.download_button(
        "⬇️ Download AI-Ready Industrial Context Graph",
        context_df.to_csv(index=False),
        "ai_ready_industrial_context_graph.csv",
        "text/csv"
    )
