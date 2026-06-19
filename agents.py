"""
agents.py
---------
The actual agent pipeline. This replaces the single hardcoded if/elif
canonical_mapping() function from the original prototype with a sequence
of discrete, individually-callable agent functions that each take a
typed input and return a typed output -- so the pipeline can be shown
as a real trace in the UI, and so the LLM agent can be inserted as a
genuine fallback step rather than a cosmetic label.

Pipeline order per tag:
    1. memory_agent_lookup      -> has a human/LLM already validated this exact tag?
    2. context_agent            -> derive site/area/asset context (deterministic, this part
                                    genuinely doesn't need an LLM -- it's structural)
    3. rule_based_semantic_agent-> token/pattern matching, returns a CONFIDENCE
                                    SCORE COMPUTED FROM EVIDENCE COUNT, not a constant
    4. llm_semantic_agent       -> ONLY called if rule-based confidence is below
                                    threshold AND a model backend is available.
                                    Supports Ollama (local, default) or OpenAI
                                    (cloud, if OPENAI_API_KEY is set). Real API
                                    call, real structured output parsing.
    5. data_quality_agent       -> validates completeness/range, assigns trust score
    6. ai_readiness_agent       -> publish / human-review / reject decision
    7. memory_write             -> persist validated outcomes for reuse

Design choice on confidence:
The original code assigned confidence = 94, 92, 91... as fixed numbers per
branch with no relationship to the actual ambiguity of the tag. Here,
confidence is a function of (a) how many independent semantic cues fired,
(b) whether cues from DIFFERENT categories conflicted, and (c) whether the
match came from memory (highest trust), rules (medium), or LLM (explicit
self-reported confidence from the model, clamped and sanity-checked).
"""

import os
import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import memory_store

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Feature flag: the LLM agent activates automatically if EITHER backend is
# reachable. This lets the demo run fully offline today on rules alone,
# light up immediately once Ollama is running locally (zero config), and
# "just work" later with OpenAI too if a key is exported -- no code changes
# needed at any point. OpenAI is preferred over Ollama only if both happen
# to be configured at once (cloud key takes priority as the "production"
# path); otherwise Ollama is used whenever it's reachable.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")


def _ollama_is_reachable() -> bool:
    """Quick liveness check against the local Ollama server, with a short
    timeout so a missing/stopped Ollama doesn't stall app startup."""
    try:
        import urllib.request
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


OLLAMA_AVAILABLE = _ollama_is_reachable()

if OPENAI_API_KEY:
    LLM_BACKEND = "openai"
elif OLLAMA_AVAILABLE:
    LLM_BACKEND = "ollama"
else:
    LLM_BACKEND = None

LLM_ENABLED = LLM_BACKEND is not None

# Confidence threshold below which we escalate from rules to the LLM agent.
# Tuned so clean single-cue matches (e.g. "TEMP") pass through rules alone,
# but genuinely ambiguous or conflicting tags get escalated.
LLM_ESCALATION_THRESHOLD = 65


# ---------------------------------------------------------------------------
# Canonical taxonomy
# Centralized here (not scattered across if/elif branches) so it is the
# single source of truth consumed by rules, the LLM prompt, and the UI.
# ---------------------------------------------------------------------------

CANONICAL_TAXONOMY = {
    "machine.status.running": {
        "signal": "Machine Running Status",
        "unit": "Boolean",
        "asset_class": "Production Machine",
        "business_meaning": "Indicates whether the machine is running.",
        "cues": ["RUN", "RUNNING", "STS", "STATUS", "FB"],
        "usecases": "Copilot, Production Intelligence, OEE, Agentic Operations",
    },
    "machine.speed.actual": {
        "signal": "Actual Machine Speed",
        "unit": "RPM",
        "asset_class": "Production Machine",
        "business_meaning": "Represents actual operating speed of the machine.",
        "cues": ["SPEED", "SPD", "ACTSPD", "RPM"],
        "usecases": "Digital Twin, Production Intelligence, Reliability Analytics",
    },
    "machine.motor.temperature": {
        "signal": "Motor Temperature",
        "unit": "°C",
        "asset_class": "Motor / Drive System",
        "business_meaning": "Represents motor thermal condition.",
        "cues": ["TEMP", "TMP", "THERMAL"],
        "usecases": "Reliability Analytics, Predictive Maintenance, Asset Health",
    },
    "machine.fault.code": {
        "signal": "Fault / Alarm Code",
        "unit": "Code",
        "asset_class": "Control System",
        "business_meaning": "Represents machine fault or alarm state.",
        "cues": ["FAULT", "ALM", "ALARM", "ERR"],
        "usecases": "Incident Copilot, Root Cause Analysis, Agentic Maintenance",
    },
    "machine.production.cycle_count": {
        "signal": "Cycle Count",
        "unit": "Count",
        "asset_class": "Production Machine",
        "business_meaning": "Represents machine production cycle count.",
        "cues": ["CYCLE", "CYC"],
        "usecases": "OEE, Throughput Analytics, Production Planning",
    },
    "machine.utility.air_pressure": {
        "signal": "Compressed Air Pressure",
        "unit": "bar",
        "asset_class": "Utility System",
        "business_meaning": "Represents pneumatic or compressed air supply condition.",
        "cues": ["AIR", "PNEU", "PRESS", "BAR"],
        "usecases": "Energy Optimization, Utility Health, Reliability Analytics",
    },
    "machine.energy.consumption": {
        "signal": "Energy Consumption",
        "unit": "kWh",
        "asset_class": "Energy Metering",
        "business_meaning": "Represents machine energy consumption.",
        "cues": ["KWH", "PWR", "ENERGY"],
        "usecases": "Energy Optimization, Sustainability Analytics, Cost Intelligence",
    },
    "machine.production.good_count": {
        "signal": "Good Product Count",
        "unit": "Count",
        "asset_class": "Production Counter",
        "business_meaning": "Represents good output count from the machine.",
        "cues": ["GOOD_COUNT", "GOOD_PARTS", "PARTS_CNT", "PART_COUNT"],
        "usecases": "Quality Systems, Production Intelligence, Yield Analytics",
    },
}

NON_FACTORY_CUES = ["HR", "EMPLOYEE", "PAYROLL", "FINANCE", "EMAIL", "SALARY", "INVOICE"]


# ---------------------------------------------------------------------------
# Data structures passed between agent steps -- this is what makes the
# pipeline traceable: every step returns a typed record you can log/display.
# ---------------------------------------------------------------------------

@dataclass
class AgentStepTrace:
    agent: str
    action: str
    detail: str


@dataclass
class TagResult:
    raw_tag: str
    vendor: str
    machine: str
    line: str
    canonical_parameter: str
    standard_signal_name: str
    unit: str
    business_meaning: str
    asset_class: str
    mapping_status: str          # Mapped / Human Review Required / Rejected
    confidence: float
    trust_score: float
    quality_status: str
    issue: str
    action: str
    uns_path: str
    source: str                  # memory / rule / llm / reject-rule
    reasoning_trace: str
    usecases: str
    trace: List[AgentStepTrace] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent 1: Memory lookup
# ---------------------------------------------------------------------------

def memory_agent_lookup(tag: str, vendor: str) -> Optional[Dict[str, Any]]:
    return memory_store.lookup(tag, vendor)


# ---------------------------------------------------------------------------
# Agent 2: Context agent (structural -- genuinely deterministic, no LLM needed)
# ---------------------------------------------------------------------------

def context_agent(line: str, machine: str, vendor: str) -> Dict[str, str]:
    return {
        "enterprise": "Enterprise",
        "site": "Site01",
        "area": "Manufacturing",
        "line": line,
        "machine": machine,
        "vendor": vendor,
    }


# ---------------------------------------------------------------------------
# Agent 3: Rule-based semantic agent with COMPUTED confidence
# ---------------------------------------------------------------------------

def _tokenize(tag: str) -> List[str]:
    """Splits a tag like 'FIL01_MOTOR_TEMP' into meaningful tokens."""
    return [t for t in re.split(r"[_\-\.\d]+", tag.upper()) if t]


def rule_based_semantic_agent(tag: str) -> Dict[str, Any]:
    """
    Scores every canonical parameter by counting how many of its cue tokens
    appear in the raw tag, then converts the evidence into a confidence score:

        confidence = 50 + (matched_cues * 12) - (competing_categories * 15)

    - A single clean cue match with no competition lands around 62-70%
      (below the LLM escalation threshold -- intentionally, since a single
      weak cue genuinely IS ambiguous and should be allowed to escalate).
    - Multiple cues for the same category, or the same canonical match
      reinforced from different token positions, push confidence up.
    - If tokens match cues from TWO DIFFERENT canonical parameters (a real
      conflict, e.g. a tag containing both "TEMP" and "SPEED" substrings),
      confidence is penalized and the case is flagged for escalation rather
      than silently picking one.

    This is still a rules engine -- it is NOT a model -- but unlike the
    original prototype, the number it outputs is actually derived from
    the tag's content rather than copy-pasted per branch.
    """
    upper = tag.upper()
    tokens = _tokenize(tag)

    # Reject path checked first and separately -- non-factory data should
    # never compete on confidence with manufacturing categories.
    reject_hits = [c for c in NON_FACTORY_CUES if c in upper or c in tokens]
    if reject_hits:
        return {
            "matched_canonical": "not_factory_relevant",
            "confidence": 96,
            "evidence": reject_hits,
            "competing": [],
            "is_reject": True,
        }

    scores: Dict[str, List[str]] = {}
    for canonical, meta in CANONICAL_TAXONOMY.items():
        hits = [cue for cue in meta["cues"] if cue in upper]
        if hits:
            scores[canonical] = hits

    if not scores:
        return {
            "matched_canonical": None,
            "confidence": 35,
            "evidence": [],
            "competing": [],
            "is_reject": False,
        }

    # Best candidate = most cue hits; ties broken by longest cue match
    # (longer cue strings are less likely to be coincidental substrings).
    best_canonical = max(
        scores,
        key=lambda c: (len(scores[c]), max(len(h) for h in scores[c]))
    )
    best_hits = scores[best_canonical]
    competing = [c for c in scores if c != best_canonical]

    confidence = 50 + (len(best_hits) * 12) - (len(competing) * 15)
    confidence = max(20, min(99, confidence))

    return {
        "matched_canonical": best_canonical,
        "confidence": confidence,
        "evidence": best_hits,
        "competing": competing,
        "is_reject": False,
    }


# ---------------------------------------------------------------------------
# Agent 4: LLM semantic agent -- REAL API call, only fires when escalated
# ---------------------------------------------------------------------------

def _build_taxonomy_prompt(tag: str, vendor: str, machine: str, sample_value: Any) -> tuple:
    """Shared prompt construction for both backends, so the reasoning quality
    is identical regardless of which model answers it."""
    taxonomy_desc = "\n".join(
        f"- {k}: {v['signal']} (unit: {v['unit']})" for k, v in CANONICAL_TAXONOMY.items()
    )

    system_prompt = (
        "You are a semantic reasoning agent for an ContextFabric AI. "
        "Your job is to map a raw, vendor-specific PLC/SCADA tag name to ONE "
        "canonical manufacturing parameter from a fixed taxonomy, or to say "
        "it does not match any of them. Be conservative: if you are not "
        "genuinely confident, say so with a lower confidence score rather "
        "than forcing a match. Respond with strict JSON only, no prose, "
        "no markdown fences, no explanation outside the JSON object."
    )

    user_prompt = f"""Canonical taxonomy:
{taxonomy_desc}

Raw tag to classify:
- Tag name: {tag}
- Vendor: {vendor}
- Machine: {machine}
- Sample value observed: {sample_value}

Return JSON with exactly these keys:
{{
  "canonical_parameter": "<one of the taxonomy keys above, or 'unclassified' or 'not_factory_relevant'>",
  "confidence": <integer 0-100, your genuine confidence in this mapping>,
  "reasoning": "<one or two sentences explaining the semantic evidence you used>"
}}"""
    return system_prompt, user_prompt


def _extract_json_object(text: str) -> dict:
    """Local models (via Ollama) are less reliable than hosted APIs about
    respecting 'JSON only' instructions -- they sometimes wrap the object in
    markdown fences or add a stray sentence before/after it. This pulls the
    first {...} block out of whatever text comes back instead of assuming
    the whole response is clean JSON."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    text = text.strip()
    # Find the first balanced-looking {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


def _call_openai(system_prompt: str, user_prompt: str) -> str:
    import urllib.request
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def _call_ollama(system_prompt: str, user_prompt: str) -> str:
    import urllib.request
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Local models on CPU can be slow on first load -- generous timeout.
    with urllib.request.urlopen(req, timeout=90) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["message"]["content"]


def llm_semantic_agent(tag: str, vendor: str, machine: str, sample_value: Any) -> Optional[Dict[str, Any]]:
    """
    Resolves a tag the rule-based agent could not confidently classify, by
    calling whichever model backend is active (Ollama locally by default,
    or OpenAI if an API key is configured). Returns a structured result with
    an 'error' flag rather than raising -- the pipeline always degrades
    gracefully to "Human Review Required" rather than guessing silently or
    crashing the whole run over one bad tag.
    """
    if not LLM_ENABLED:
        return None

    system_prompt, user_prompt = _build_taxonomy_prompt(tag, vendor, machine, sample_value)

    try:
        if LLM_BACKEND == "openai":
            content = _call_openai(system_prompt, user_prompt)
        elif LLM_BACKEND == "ollama":
            content = _call_ollama(system_prompt, user_prompt)
        else:
            return None

        parsed = _extract_json_object(content)

        canonical = parsed.get("canonical_parameter", "unclassified")
        confidence = parsed.get("confidence", 50)
        reasoning = parsed.get("reasoning", "")

        # Sanity-check the model's own output against the taxonomy --
        # never trust an LLM field value blindly, even your own prompt's.
        # This matters MORE for local models, which are more prone to
        # inventing a plausible-sounding key that isn't in the taxonomy.
        if canonical not in CANONICAL_TAXONOMY and canonical not in (
            "unclassified", "not_factory_relevant"
        ):
            canonical = "unclassified"
            confidence = min(confidence, 50)

        try:
            confidence = max(0, min(100, int(confidence)))
        except (TypeError, ValueError):
            confidence = 50

        return {
            "matched_canonical": canonical,
            "confidence": confidence,
            "reasoning": reasoning,
            "raw_response": content,
            "backend": LLM_BACKEND,
        }

    except Exception as e:
        # Graceful degradation -- a network hiccup, a model that isn't
        # pulled yet, or malformed JSON should route the tag to human
        # review, never crash the pipeline or fabricate a confident answer.
        return {
            "matched_canonical": None,
            "confidence": 0,
            "reasoning": f"LLM call failed ({LLM_BACKEND}): {e}",
            "raw_response": None,
            "backend": LLM_BACKEND,
            "error": True,
        }


# ---------------------------------------------------------------------------
# Agent 5 & 6: Data quality + AI readiness decision
# ---------------------------------------------------------------------------

def decide_status(canonical: Optional[str], confidence: float, is_reject: bool) -> str:
    if is_reject or canonical == "not_factory_relevant":
        return "Rejected"
    if canonical is None or canonical == "unclassified" or confidence < LLM_ESCALATION_THRESHOLD:
        return "Human Review Required"
    return "Mapped"


def build_reasoning_trace(tag: str, source: str, evidence: Any, status: str, llm_reasoning: str = "") -> str:
    if status == "Rejected":
        return (f"Non-manufacturing identifier detected via rule match on: "
                f"{', '.join(evidence) if isinstance(evidence, list) else evidence}. "
                f"Blocked from ContextFabric AI.")
    if source == "memory":
        return f"Resolved from validated memory (previously confirmed mapping for this exact tag string)."
    if source == "llm":
        return f"LLM Semantic Reasoning Agent: {llm_reasoning}"
    if status == "Human Review Required":
        return ("No confident rule-based or memory match found. "
                f"Evidence collected: {evidence if evidence else 'none'}. "
                "Routed for human or LLM validation before AI consumption.")
    return (f"Rule-based match. Evidence tokens matched: {', '.join(evidence)}.")


# ---------------------------------------------------------------------------
# Top-level orchestration: this is the "Supervisor Agent"
# ---------------------------------------------------------------------------

def run_pipeline_for_tag(row: Dict[str, Any]) -> TagResult:
    tag = str(row["TagName"])
    vendor = str(row["Vendor"])
    machine = str(row["Machine"])
    line = str(row["Line"])
    value = row.get("Value", "")

    trace: List[AgentStepTrace] = []

    # --- Step 1: Memory ---
    mem_hit = memory_agent_lookup(tag, vendor)
    if mem_hit:
        trace.append(AgentStepTrace(
            "Memory Agent", "lookup_hit",
            f"Found validated mapping for '{tag}' from prior onboarding wave."
        ))
        ctx = context_agent(line, machine, vendor)
        canonical = mem_hit["canonical_parameter"]
        meta = CANONICAL_TAXONOMY.get(canonical, {})
        signal = mem_hit.get("standard_signal_name") or meta.get("signal", "Unknown")
        unit = mem_hit.get("unit") or meta.get("unit", "Unknown")
        business_meaning = mem_hit.get("business_meaning") or meta.get("business_meaning", "")
        asset_class = mem_hit.get("asset_class") or meta.get("asset_class", "Unknown")
        confidence = mem_hit.get("confidence", 95)
        status = decide_status(canonical, confidence, canonical == "not_factory_relevant")
        usecases = meta.get("usecases", "Copilot, Analytics, Agentic AI")

        return _finalize(
            tag, vendor, machine, line, ctx, canonical, signal, unit,
            business_meaning, asset_class, status, confidence, "memory",
            evidence=mem_hit.get("source", "memory"), trace=trace, llm_reasoning="",
        )

    trace.append(AgentStepTrace("Memory Agent", "lookup_miss", f"No prior validated mapping for '{tag}'."))

    # --- Step 2: Context ---
    ctx = context_agent(line, machine, vendor)
    trace.append(AgentStepTrace(
        "Context Agent", "context_built",
        f"Resolved {ctx['enterprise']}/{ctx['site']}/{ctx['area']}/{line}/{machine}."
    ))

    # --- Step 3: Rule-based semantic agent ---
    rule_result = rule_based_semantic_agent(tag)
    trace.append(AgentStepTrace(
        "Semantic Reasoning Agent (rules)", "scored",
        f"Best candidate: {rule_result['matched_canonical']} "
        f"(confidence {rule_result['confidence']}%, evidence {rule_result['evidence']}, "
        f"competing {rule_result['competing']})."
    ))

    canonical = rule_result["matched_canonical"]
    confidence = rule_result["confidence"]
    source = "rule"
    llm_reasoning = ""
    evidence = rule_result["evidence"]

    # Track whether the rule engine found ANY evidence at all. A tag with
    # zero matched cues is a genuinely blind guess for any classifier --
    # human or machine. We use this as a hard ceiling on what the LLM is
    # ALLOWED to do downstream, independent of whatever confidence number
    # it reports. Small local models in particular tend to produce a
    # plausible-sounding, confidently-wrong answer on pure noise input
    # rather than admitting uncertainty -- this guardrail does not rely on
    # the model accurately self-assessing that; it removes the model's
    # ability to single-handedly authorize publication when there was
    # nothing for it to actually reason from in the first place.
    zero_evidence = (rule_result["matched_canonical"] is None)

    # --- Step 4: Escalate to LLM if rules weren't confident, and not a reject ---
    if not rule_result["is_reject"] and confidence < LLM_ESCALATION_THRESHOLD:
        if LLM_ENABLED:
            trace.append(AgentStepTrace(
                "Semantic Reasoning Agent (rules)", "escalate",
                f"Confidence {confidence}% below threshold ({LLM_ESCALATION_THRESHOLD}%). Escalating to LLM agent."
            ))
            llm_result = llm_semantic_agent(tag, vendor, machine, value)
            if llm_result and not llm_result.get("error"):
                llm_canonical = llm_result["matched_canonical"]
                llm_confidence = llm_result["confidence"]
                llm_reasoning = llm_result["reasoning"]

                if zero_evidence:
                    # Hard guardrail: zero rule evidence means the LLM is
                    # pattern-completing on noise, not reasoning from real
                    # signal. Cap confidence below the publish threshold no
                    # matter what the model claims, and force human review.
                    canonical = llm_canonical if llm_canonical else "unclassified"
                    confidence = min(llm_confidence, LLM_ESCALATION_THRESHOLD - 1)
                    source = "llm"
                    trace.append(AgentStepTrace(
                        "LLM Semantic Reasoning Agent", "capped",
                        f"Model suggested {llm_canonical} at self-reported {llm_confidence}% confidence, "
                        f"but ZERO rule-based evidence tokens were found for this tag -- this is a blind "
                        f"guess on noise, not grounded reasoning. Confidence capped at {confidence}% and "
                        f"forced to human review regardless of the model's self-rating. Reasoning offered "
                        f"as a hint for the reviewing engineer only: {llm_reasoning}"
                    ))
                else:
                    canonical = llm_canonical
                    confidence = llm_confidence
                    source = "llm"
                    trace.append(AgentStepTrace(
                        "LLM Semantic Reasoning Agent", "resolved",
                        f"Model mapped to {canonical} at {confidence}% confidence: {llm_reasoning}"
                    ))
            else:
                err = llm_result.get("reasoning", "unknown error") if llm_result else "no response"
                trace.append(AgentStepTrace(
                    "LLM Semantic Reasoning Agent", "failed",
                    f"LLM call did not return a usable result ({err}). Falling back to human review."
                ))
                canonical = "unclassified"
                confidence = rule_result["confidence"]
        else:
            trace.append(AgentStepTrace(
                "LLM Semantic Reasoning Agent", "skipped",
                "No model backend reachable (Ollama not running and no OPENAI_API_KEY set). "
                "Routing to human review instead of guessing."
            ))

    is_reject = rule_result["is_reject"]
    status = decide_status(canonical, confidence, is_reject)

    meta = CANONICAL_TAXONOMY.get(canonical, {})
    signal = meta.get("signal", "Unknown" if status != "Rejected" else "Non-factory enterprise data")
    unit = meta.get("unit", "N/A" if status == "Rejected" else "Unknown")
    business_meaning = meta.get(
        "business_meaning",
        "Enterprise/HR data. It must not be processed as manufacturing context." if status == "Rejected"
        else "The tag cannot be safely interpreted without human validation."
    )
    asset_class = meta.get("asset_class", "Non-factory data" if status == "Rejected" else "Unknown")
    usecases = meta.get("usecases", "Not eligible until validated")

    return _finalize(
        tag, vendor, machine, line, ctx, canonical, signal, unit,
        business_meaning, asset_class, status, confidence, source,
        evidence=evidence, trace=trace, llm_reasoning=llm_reasoning,
    )


def _finalize(tag, vendor, machine, line, ctx, canonical, signal, unit,
              business_meaning, asset_class, status, confidence, source,
              evidence, trace, llm_reasoning) -> TagResult:

    canonical_label = canonical or "unclassified"
    uns_path = f"{ctx['enterprise']}/{ctx['site']}/{ctx['area']}/{line}/{machine}/{canonical_label}"

    # Trust score: for Mapped items it mirrors confidence (since trust IS the
    # confidence we have in correctness here); for review/reject it reflects
    # data quality state rather than a copy of an unrelated number.
    if status == "Mapped":
        trust_score = confidence
        quality_status = "Valid"
        issue = "None"
        action = "Publish to AI-ready context fabric"
    elif status == "Human Review Required":
        trust_score = confidence
        quality_status = "Exception"
        issue = "No confident rule, memory, or LLM match found"
        action = "Route to controls/manufacturing engineer for validation before AI consumption"
    else:  # Rejected
        trust_score = 0
        quality_status = "Rejected"
        issue = "Tag is not relevant to factory operations"
        action = "Reject and exclude from manufacturing context fabric"

    reasoning_trace = build_reasoning_trace(tag, source, evidence, status, llm_reasoning)
    meta = CANONICAL_TAXONOMY.get(canonical_label, {})
    usecases = meta.get("usecases", "Not eligible until validated") if status == "Mapped" else "Not eligible until validated"

    trace.append(AgentStepTrace(
        "AI Readiness Agent", "decision",
        f"Status: {status}. Trust score: {trust_score}%. Action: {action}"
    ))

    return TagResult(
        raw_tag=tag,
        vendor=vendor,
        machine=machine,
        line=line,
        canonical_parameter=canonical_label,
        standard_signal_name=signal,
        unit=unit,
        business_meaning=business_meaning,
        asset_class=asset_class,
        mapping_status=status,
        confidence=confidence,
        trust_score=trust_score,
        quality_status=quality_status,
        issue=issue,
        action=action,
        uns_path=uns_path,
        source=source,
        reasoning_trace=reasoning_trace,
        usecases=usecases,
        trace=trace,
    )


def commit_to_memory(result: TagResult, validated_by: str = "system") -> None:
    """
    Writes a Mapped result into long-term memory so future runs (any plant,
    any onboarding wave) skip the rule/LLM steps for this exact tag string.
    Called automatically for high-confidence LLM/rule resolutions, and
    explicitly when a human resolves a review item in the UI.
    """
    memory_store.store_validated_mapping(
        tag=result.raw_tag,
        vendor=result.vendor,
        canonical_parameter=result.canonical_parameter,
        standard_signal_name=result.standard_signal_name,
        unit=result.unit,
        business_meaning=result.business_meaning,
        asset_class=result.asset_class,
        source=result.source if result.source in ("llm", "rule") else "human_validated",
        confidence=result.confidence,
        validated_by=validated_by,
    )
