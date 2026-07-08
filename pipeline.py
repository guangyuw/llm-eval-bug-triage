"""
LLM Evaluation on Bug Triage (Mozilla Bugzilla).

An end-to-end NLP / LLM-evaluation pipeline that answers three questions
on 13,019 public Mozilla Bugzilla bug reports:

  1. What themes do bugs repeatedly report? (unsupervised clustering)
  2. Can a new bug automatically find its duplicate? (semantic dedup retrieval
     with hard labels — hit@k / MRR)
  3. Are the structured fields (area / severity / kind) extracted by an LLM
     trustworthy? (self-correcting extraction agent + LLM-as-judge calibrated
     with Cohen's κ against human labels)

Pipeline (8 steps):
  load → embed → cluster topics → Wilson CI stats
  → extraction agent (LangGraph supervisor + real MCP grounding)
  → auto-rater × κ → dedup retrieval → readout

Run:
    python pipeline.py

Data:
    bugs.csv        — 13,019 Mozilla bugs (id, component, severity, summary, description, role)
    dup_pairs.csv   — 4,987 (dup_id, master_id) hard-label pairs
    human_labels.csv — human-annotated area / severity / kind for the extraction eval sample

Costs:
    Steps 1–4 and 7 are free (local / cached).
    Steps 5–6 call the OpenAI API (gpt-4o + text-embedding-3-small).
    Set OPENAI_API_KEY in a .env file before running.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import difflib
from collections import Counter
from operator import add
from typing import Annotated, TypedDict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

EMBED_MODEL        = "text-embedding-3-small"
CHAT_MODEL         = "gpt-4o"
CHROMA_PATH        = "chroma_db"
CHROMA_COLLECTION  = "bugzilla"
EXTRACT_N          = 50    # number of bugs for extraction eval
SEED               = 42
MAX_ATTEMPTS       = 3
TEXT_MAX_CHARS     = 8000
DUP_SIM            = 0.75  # similarity threshold for duplicate flagging
MAX_STEPS          = 10    # supervisor decision budget
RESEARCH_MAX_CALLS = 4     # max tool rounds per researcher invocation

pd.set_option("display.max_colwidth", 100)


# ===========================================================================
# Step 1 · Load corpus
# ===========================================================================

def step1_load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load bugs.csv and dup_pairs.csv, build a text field, and return both frames.

    bugs.csv columns (minimum required): id, component, severity, summary,
        description, role  (dup | master | corpus)
    dup_pairs.csv columns: dup_id, master_id
    """
    bugs = pd.read_csv("bugs.csv")
    pairs = pd.read_csv("dup_pairs.csv")

    def make_text(r):
        comp = str(r.get("component") or "").strip()
        summ = str(r.get("summary") or "").strip()
        desc = str(r.get("description") or "").strip()
        head = f"[{comp}] {summ}" if comp else summ
        return (head + "\n" + desc).strip() if desc and desc != "nan" else head

    bugs["text"] = bugs.apply(make_text, axis=1)
    bugs["desc_words"] = (bugs["description"].fillna("").astype(str)
                          .str.split().str.len())
    bugs = bugs.reset_index(drop=True)

    print(f"[load] bugs={len(bugs):,}  dup_pairs={len(pairs):,}"
          f"  masters={pairs['master_id'].nunique():,}")
    print(f"[load] roles: {bugs['role'].value_counts().to_dict()}")
    print(f"[load] description coverage: {(bugs.desc_words > 0).mean():.1%}"
          f"  median desc words: "
          f"{bugs.loc[bugs.desc_words > 0, 'desc_words'].median():.0f}")
    return bugs, pairs


# ===========================================================================
# Step 2 · Embed all bugs into ChromaDB (cached)
# ===========================================================================

def step2_embed(bugs: pd.DataFrame) -> "chromadb.Collection":
    """Embed all bug texts with text-embedding-3-small into a local ChromaDB.

    Embeddings are cached — re-running costs nothing once the collection exists.
    Returns the Chroma collection (used by the dedup retrieval and the agent).
    """
    import chromadb
    from openai import OpenAI

    oai = OpenAI()
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_or_create_collection(CHROMA_COLLECTION,
                                          metadata={"hnsw:space": "cosine"})

    existing = set(col.get()["ids"])
    to_embed = bugs[~bugs["id"].astype(str).isin(existing)].reset_index(drop=True)

    if len(to_embed) == 0:
        print(f"[embed] all {len(bugs):,} bugs already cached in ChromaDB — skipping")
        return col

    print(f"[embed] embedding {len(to_embed):,} new bugs …")
    BATCH = 500
    for start in range(0, len(to_embed), BATCH):
        batch = to_embed.iloc[start: start + BATCH]
        texts = [t[:TEXT_MAX_CHARS] for t in batch["text"]]
        resp = oai.embeddings.create(model=EMBED_MODEL, input=texts)
        vecs = [d.embedding for d in resp.data]
        col.add(ids=batch["id"].astype(str).tolist(), embeddings=vecs,
                documents=texts,
                metadatas=[{"component": str(r["component"]),
                            "summary":   str(r["summary"])[:200]}
                           for _, r in batch.iterrows()])
        print(f"[embed]   {min(start + BATCH, len(to_embed)):,}/{len(to_embed):,}")

    print(f"[embed] done — {col.count():,} bugs in collection")
    return col


# ===========================================================================
# Step 3 · Cluster topics (KMeans + silhouette-based k selection)
# ===========================================================================

def step3_cluster(bugs: pd.DataFrame, col: "chromadb.Collection") -> pd.DataFrame:
    """Fit KMeans on all embeddings, select k via silhouette on a 3,000-bug sample,
    and label each cluster with a GPT-4o topic name.

    Returns bugs with a new 'cluster_theme' column.
    """
    from openai import OpenAI
    from sklearn.metrics import silhouette_score

    oai = OpenAI()

    # Fetch all embeddings from Chroma
    result = col.get(include=["embeddings"])
    ids_order = result["ids"]
    vecs = np.array(result["embeddings"], dtype=np.float32)  # match development.ipynb dtype
    # Align order to bugs DataFrame
    id_to_idx = {str(bid): i for i, bid in enumerate(ids_order)}
    idx = [id_to_idx[str(bid)] for bid in bugs["id"].astype(str) if str(bid) in id_to_idx]
    vecs_aligned = vecs[idx]

    # Select k on a sample
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(vecs_aligned), size=min(3000, len(vecs_aligned)), replace=False)
    sample_vecs = vecs_aligned[sample_idx]

    best_k, best_score = 8, -1.0
    for k in range(8, 25, 2):   # same sweep as development.ipynb: 8,10,12,...,24 (step 2)
        km = KMeans(n_clusters=k, random_state=SEED, n_init="auto")
        labels_s = km.fit_predict(sample_vecs)
        # compute silhouette directly on the 3000-bug sample (same as development.ipynb)
        score = silhouette_score(sample_vecs, labels_s)
        print(f"[cluster]   k={k:>2}  silhouette={score:.3f}")
        if score > best_score:
            best_k, best_score = k, score

    print(f"[cluster] best k={best_k}  silhouette={best_score:.4f}")

    km_final = KMeans(n_clusters=best_k, random_state=SEED, n_init="auto")  # match development.ipynb
    labels = km_final.fit_predict(vecs_aligned)

    # Name each cluster with GPT-4o
    bugs_aligned = bugs.iloc[:len(idx)].copy()
    bugs_aligned["_cluster"] = labels
    theme_map = {}
    for c in range(best_k):
        samples = (bugs_aligned[bugs_aligned["_cluster"] == c]["summary"]
                   .dropna().head(10).tolist())
        prompt = ("Name this software-bug cluster in ≤5 words (title case, no quotes).\n"
                  "Summaries:\n" + "\n".join(f"- {s}" for s in samples))
        theme_map[c] = oai.chat.completions.create(
            model=CHAT_MODEL, temperature=0,
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content.strip()

    bugs_aligned["cluster_theme"] = bugs_aligned["_cluster"].map(theme_map)
    print(f"[cluster] themes: {list(theme_map.values())}")
    return bugs_aligned.drop(columns=["_cluster"])


# ===========================================================================
# Step 4 · Topic prevalence with Wilson 95% confidence intervals
# ===========================================================================

def step4_topic_stats(bugs: pd.DataFrame) -> pd.DataFrame:
    """Compute topic prevalence with Wilson confidence intervals and save a chart.

    Returns a DataFrame with columns: theme, count, share, ci_lo, ci_hi.
    """
    from statsmodels.stats.proportion import proportion_confint

    s = bugs["cluster_theme"].dropna()
    N = len(s)
    rows = []
    for theme, c in s.value_counts().head(12).items():
        lo, hi = proportion_confint(c, N, alpha=0.05, method="wilson")
        rows.append({"theme": theme, "count": c, "share": c / N,
                     "ci_lo": lo, "ci_hi": hi})
    df = pd.DataFrame(rows)

    print(f"[topic_stats] N={N:,}  95% Wilson CI:")
    for _, r in df.iterrows():
        print(f"  {r['theme'][:34]:<34} {r['share']:6.2%}"
              f"  [{r['ci_lo']:.2%}, {r['ci_hi']:.2%}]  (n={r['count']})")

    # Chart
    labs = [r[:26] for r in df["theme"]]
    err = [df["share"] - df["ci_lo"], df["ci_hi"] - df["share"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(labs, df["share"], xerr=err, color="#4c72b0", capsize=3)
    ax.invert_yaxis()
    ax.set_xlabel("share of bugs")
    ax.set_title("Theme prevalence (95% Wilson CI)")
    plt.tight_layout()
    fig.savefig("assets/topic_prevalence.png", dpi=150, bbox_inches="tight")
    print("[topic_stats] saved topic_prevalence.png")
    return df


# ===========================================================================
# Step 5 · Extraction agent (LangGraph supervisor + MCP grounding)
# ===========================================================================

# -- MCP helpers (sync wrappers around the async MCP client) --

def _mcp_call_sync(tool: str, args: dict | None = None) -> str:
    """Call a tool on the local MCP server (mcp_server.py) synchronously.

    Works in both plain-script context and inside Jupyter (which already has a
    running event loop).  When an event loop is already running we dispatch the
    coroutine to a brand-new thread that owns its own event loop, sidestepping
    the 'asyncio.run() cannot be called from a running event loop' restriction.
    """
    import asyncio
    import concurrent.futures
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def _inner():
        params = StdioServerParameters(command=sys.executable, args=["mcp_server.py"])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as sess:
                await sess.initialize()
                res = await sess.call_tool(tool, args or {})
                return res.content[0].text

    try:
        asyncio.get_running_loop()
        # Already inside an event loop (e.g. Jupyter) – run in a new thread.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _inner()).result()
    except RuntimeError:
        # No running event loop – plain script context.
        return asyncio.run(_inner())


def _load_mcp_catalog() -> tuple[dict, list[str]]:
    catalog    = json.loads(_mcp_call_sync("get_field_catalog"))
    components = json.loads(_mcp_call_sync("get_component_taxonomy"))
    print(f"[mcp] catalog loaded: {list(catalog['fields'])}"
          f"  |  taxonomy: {len(components)} components")
    return catalog, components


# -- Tool implementations (deterministic, zero LLM cost) --

def _make_tools(bugs: pd.DataFrame, col: "chromadb.Collection",
                components: list[str]):
    """Return the three researcher tools bound to the current corpus."""
    from langchain_core.tools import tool

    _by_id = bugs.assign(_sid=bugs["id"].astype(str)).set_index("_sid")

    _SEV_MAP = {
        "s1": "high", "s2": "high", "critical": "high",
        "blocker": "high", "major": "high",
        "s3": "med", "normal": "med",
        "s4": "low", "minor": "low", "trivial": "low",
    }

    def _nearest_component(area: str) -> str | None:
        m = difflib.get_close_matches(str(area), components, n=1, cutoff=0.0)
        return m[0] if m else None

    def _similar_bugs(bug_id, k=6):
        got = col.get(ids=[str(bug_id)], include=["embeddings"])["embeddings"]
        if not len(got):
            return []
        v = np.asarray(got[0]).tolist()
        res = col.query(query_embeddings=[v], n_results=k + 1)
        out = []
        for i, dist in zip(res["ids"][0], res["distances"][0]):
            if i == str(bug_id) or i not in _by_id.index:
                continue
            r = _by_id.loc[i]
            out.append({"id": i, "sim": round(1.0 - dist, 3),
                        "component": str(r["component"]),
                        "summary": str(r["summary"])[:120]})
        return out[:k]

    def _read_bug(bug_id):
        if str(bug_id) not in _by_id.index:
            return {"error": f"bug {bug_id} not found"}
        r = _by_id.loc[str(bug_id)]
        return {"id": str(bug_id), "component": str(r["component"]),
                "summary": str(r["summary"]),
                "description": str(r.get("description", ""))[:1200]}

    def _map_severity(raw):
        return _SEV_MAP.get(str(raw).strip().lower())

    def _component_prior(component):
        near = _nearest_component(component) or str(component)
        sub = bugs[bugs["component"].astype(str) == near]
        mapped = sub["severity"].map(_map_severity).dropna()
        dist = mapped.value_counts(normalize=True).round(2).to_dict()
        typical = mapped.mode().iloc[0] if not mapped.empty else None
        return {"component": near, "n_bugs": int(len(sub)),
                "n_triaged": int(len(mapped)), "severity_dist": dist,
                "typical_severity": typical,
                "example_summaries": sub["summary"].astype(str).head(3).tolist()}

    @tool
    def search_similar_bugs(bug_id: str) -> str:
        """Return up to 6 historical bugs most similar to the given bug id
        (semantic vector search over cached embeddings): id, similarity,
        official component, short summary."""
        return json.dumps(_similar_bugs(bug_id, k=6))

    @tool
    def read_bug(bug_id: str) -> str:
        """Return the FULL text (component, summary, description) of one bug.
        Use to deep-dive a promising but unclear neighbor."""
        return json.dumps(_read_bug(bug_id))

    @tool
    def component_prior(component: str) -> str:
        """Return historical stats for an official component: total bug count,
        triaged count, severity distribution (low/med/high), typical severity,
        and example summaries."""
        return json.dumps(_component_prior(component))

    tools = [search_similar_bugs, read_bug, component_prior]
    tools_by_name = {t.name: t for t in tools}
    return tools, tools_by_name, _similar_bugs, _nearest_component


def _build_agent(catalog: dict, schema: str,
                 valid_sev: set, valid_kind: set,
                 tools, tools_by_name):
    """Build and compile the LangGraph supervisor + extraction graph."""
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph

    SUPERVISOR_PROMPT = (
        "You are the SUPERVISOR of a bug-field extraction team. "
        "Route to exactly ONE worker per turn.\n"
        "Workers:\n"
        "- research: gather evidence via retrieval tools. "
        "Use when the affected area is unclear or you need precedents.\n"
        "- extract: write the JSON {{area,severity,kind}} from bug text + evidence. "
        "Use when you have enough info, OR to REWRITE after a critic rejection.\n"
        "- critique: check the current extraction. Use right after a new extraction.\n"
        "- FINISH: stop. Use ONLY after the critic accepted it (critic_ok=true), "
        "or to give up near the step budget.\n"
        "Reply ONLY JSON: {{\"next\": \"research|extract|critique|FINISH\", "
        "\"reason\": str}}.\n"
        "Team state:\n{state}"
    )

    RESEARCH_SYS = (
        "You are the RESEARCHER. Collect JUST ENOUGH evidence to classify a bug "
        "into these fields: {schema}\n"
        "Tools: search_similar_bugs(bug_id), read_bug(bug_id), "
        "component_prior(component).\n"
        "When done, reply with a SHORT plain-text evidence summary "
        "(NO JSON, NO tool call)."
    ).format(schema=schema)

    EXTRACT_PROMPT = (
        "Extract structured fields from a software bug report. "
        "Field semantics: {schema}\n"
        "Return ONLY a JSON object "
        "{{\"area\": str, \"severity\": \"low|med|high\", "
        "\"kind\": \"crash|ui|performance|security|data|other\"}}.\n"
        "{feedback}"
        "Bug report:\n```{text}```\nCollected evidence:\n{evidence}"
    )

    CRITIQUE_PROMPT = (
        "You are a reviewer. Is this extraction faithful and well-formed? "
        "Reject (ok=false) ONLY for a clear contradiction, invented facts, "
        "or an off-topic area.\n"
        "Reply ONLY JSON: {{\"ok\": true|false, \"reason\": str}}.\n"
        "Bug report:\n```{text}```\nExtraction: {extraction}"
    )

    def schema_ok(obj):
        if not isinstance(obj, dict):
            return False, "not a JSON object"
        if not isinstance(obj.get("area"), str) or not obj["area"].strip():
            return False, "missing 'area'"
        if obj.get("severity") not in valid_sev:
            return False, "bad 'severity'"
        if obj.get("kind") not in valid_kind:
            return False, "bad 'kind'"
        return True, ""

    def _parse_json(txt):
        try:
            return json.loads(txt)
        except Exception:
            m = re.search(r"\{.*\}", txt or "", re.S)
            try:
                return json.loads(m.group()) if m else None
            except Exception:
                return None

    def _fmt_ev(e):
        if "tool" in e:
            return f"[{e['tool']}({e.get('args', {})})] -> {e['result']}"
        if "note" in e:
            return f"[researcher note] {e['note']}"
        return str(e)

    oai_llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    research_llm = oai_llm.bind_tools(tools)

    class S(TypedDict):
        bug_id:     str
        text:       str
        evidence:   Annotated[list, add]
        extraction: dict | None
        ok:         bool
        critique:   str
        decisions:  Annotated[list, add]
        steps:      int
        n_extract:  int

    def _sup_state(s):
        ev = s["evidence"]
        return json.dumps({
            "bug_text":          str(s["text"])[:600],
            "evidence_count":    len(ev),
            "evidence_digest":   [e.get("tool") or "note" for e in ev][-6:],
            "has_extraction":    s["extraction"] is not None,
            "current_extraction": s["extraction"],
            "critic_ok":         s["ok"],
            "last_critique":     s["critique"],
            "steps_used":        s["steps"],
            "max_steps":         MAX_STEPS,
        }, ensure_ascii=False)

    def _call_llm(prompt: str, *, json_mode: bool = False) -> str:
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        for attempt in range(6):
            try:
                return oai_llm.invoke(
                    [{"role": "user", "content": prompt}], **kwargs
                ).content
            except Exception as exc:
                if getattr(exc, "status_code", None) == 429 or "rate_limit" in str(exc).lower():
                    wait = 2 ** attempt * 5
                    print(f"  [rate limit] waiting {wait}s before retry {attempt + 1}/6 …")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("rate limit retries exhausted")

    def n_supervisor(s):
        allowed = {"research", "extract", "critique", "FINISH"}
        if s["steps"] >= MAX_STEPS:
            dec = {"next": "FINISH", "reason": "step budget exhausted"}
        else:
            raw = _call_llm(SUPERVISOR_PROMPT.format(state=_sup_state(s)),
                            json_mode=True)
            try:
                d = json.loads(raw)
            except Exception:
                d = {}
            nxt = d.get("next")
            if nxt not in allowed:
                nxt = "extract" if s["extraction"] is None else "critique"
            if nxt in ("critique", "FINISH") and s["extraction"] is None:
                nxt = "extract"
            dec = {"next": nxt, "reason": str(d.get("reason", ""))[:200]}
        return {"decisions": [dec], "steps": s["steps"] + 1}

    def n_researcher(s):
        last_reason = (s["decisions"][-1].get("reason", "")
                       if s["decisions"] else "")
        prior = "\n".join(_fmt_ev(e) for e in s["evidence"]) or "(none yet)"
        hist = [SystemMessage(RESEARCH_SYS),
                HumanMessage(
                    f"Bug id: {s['bug_id']}\nBug report:\n"
                    f"```{str(s['text'])[:1500]}```\n"
                    f"Why you were called: {last_reason}\n"
                    f"Evidence already collected "
                    f"(do NOT repeat these same tool calls):\n{prior[:1200]}"
                )]
        collected = []
        for _ in range(RESEARCH_MAX_CALLS):
            for attempt in range(6):
                try:
                    resp = research_llm.invoke(hist)
                    break
                except Exception as exc:
                    if (getattr(exc, "status_code", None) == 429
                            or "rate_limit" in str(exc).lower()):
                        wait = 2 ** attempt * 5
                        print(f"  [rate limit/researcher] waiting {wait}s …")
                        time.sleep(wait)
                    else:
                        raise
            hist.append(resp)
            if not resp.tool_calls:
                if resp.content:
                    collected.append({"note": str(resp.content)[:400]})
                break
            for tc in resp.tool_calls:
                try:
                    out = tools_by_name[tc["name"]].invoke(tc["args"])
                except Exception as exc:
                    out = json.dumps({"error": str(exc)})
                hist.append(ToolMessage(str(out), tool_call_id=tc["id"]))
                collected.append({"tool": tc["name"], "args": tc["args"],
                                  "result": str(out)[:400]})
        return {"evidence": collected}

    def n_extractor(s):
        ev = "\n".join(_fmt_ev(e) for e in s["evidence"]) or "(no evidence)"
        fb = (f"A reviewer previously rejected an extraction: {s['critique']}. "
              f"Fix it.\n") if s["critique"] else ""
        raw = _call_llm(
            EXTRACT_PROMPT.format(schema=schema, feedback=fb,
                                  text=str(s["text"])[:2000],
                                  evidence=ev[:1500]),
            json_mode=True
        )
        return {"extraction": _parse_json(raw), "n_extract": s["n_extract"] + 1,
                "critique": ""}

    def n_critic(s):
        ext = s["extraction"]
        ok, reason = schema_ok(ext)
        if not ok:
            return {"ok": False, "critique": f"schema error: {reason}"}
        raw = _call_llm(CRITIQUE_PROMPT.format(
            text=str(s["text"])[:2000],
            extraction=json.dumps(ext)
        ), json_mode=True)
        try:
            d = json.loads(raw)
        except Exception:
            d = {}
        return {"ok": bool(d.get("ok", False)),
                "critique": str(d.get("reason", ""))[:300]}

    def route_supervisor(s):
        return s["decisions"][-1]["next"] if s["decisions"] else "extract"

    g = StateGraph(S)
    g.add_node("supervisor", n_supervisor)
    g.add_node("researcher",  n_researcher)
    g.add_node("extractor",   n_extractor)
    g.add_node("critic",      n_critic)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", route_supervisor,
                            {"research": "researcher", "extract": "extractor",
                             "critique": "critic", "FINISH": END})
    g.add_edge("researcher", "supervisor")
    g.add_edge("extractor",  "supervisor")
    g.add_edge("critic",     "supervisor")

    app = g.compile()
    print(f"[agent] compiled: "
          f"{' / '.join(n for n in app.get_graph().nodes)}")
    return app, schema_ok


def step5_extract(bugs: pd.DataFrame,
                  col: "chromadb.Collection") -> pd.DataFrame:
    """Run the extraction agent on EXTRACT_N bugs and return a results DataFrame."""
    catalog, components = _load_mcp_catalog()
    schema = ("; ".join(f"{k}: {v}" for k, v in catalog["fields"].items())
              + "; enums: " + json.dumps(catalog["enums"]))
    valid_sev  = set(catalog["enums"]["severity"])
    valid_kind = set(catalog["enums"]["kind"])

    tools, tools_by_name, _similar_bugs, _nearest_component = _make_tools(
        bugs, col, components)
    app, _schema_ok = _build_agent(catalog, schema, valid_sev, valid_kind,
                                   tools, tools_by_name)

    pool = (bugs[bugs["desc_words"] >= 40]
            .sample(min(EXTRACT_N, (bugs.desc_words >= 40).sum()),
                    random_state=SEED)
            .reset_index(drop=True))

    recs, oks, steps_list, n_extracts, trajs, dup_flags = [], [], [], [], [], []
    for i, (bid, t) in enumerate(zip(pool["id"].astype(str), pool["text"]), 1):
        init = {"bug_id": bid, "text": t, "evidence": [], "extraction": None,
                "ok": False, "critique": "", "decisions": [], "steps": 0,
                "n_extract": 0}
        fin = app.invoke(init, {"recursion_limit": 60})

        ext = dict(fin.get("extraction") or {})
        ext["area_component"] = (_nearest_component(ext["area"])
                                 if ext.get("area") else None)
        nbrs = _similar_bugs(bid, k=6)
        dup_flags.append([d for d in nbrs if d["sim"] >= DUP_SIM][:5])

        recs.append(ext)
        oks.append(bool(fin["ok"]))
        steps_list.append(fin["steps"])
        n_extracts.append(fin["n_extract"])
        trajs.append([d["next"] for d in fin["decisions"]])

        if i % 15 == 0 or i == len(pool):
            print(f"[extract]   processed {i}/{len(pool)}")
        time.sleep(2)

    n = len(recs)
    after_retry = sum(oks) / n
    first_pass  = sum(1 for o, e in zip(oks, n_extracts) if o and e == 1) / n
    mean_steps  = sum(steps_list) / n
    zero_res    = sum(1 for tr in trajs if "research" not in tr) / n

    print(f"[extract] valid yield: first-pass {first_pass:.1%}"
          f" -> after critique-retry {after_retry:.1%}"
          f" (self-correction {after_retry - first_pass:+.1%})")
    print(f"[extract] mean supervisor steps {mean_steps:.1f}"
          f" | {zero_res:.0%} solved with ZERO retrieval")
    print(f"[extract] dedup flagged "
          f"{sum(1 for d in dup_flags if d)}/{n} bugs (sim ≥ {DUP_SIM})")

    df = pool.copy()
    df["llm_area"]      = [r.get("area") for r in recs]
    df["llm_severity"]  = [r.get("severity") for r in recs]
    df["llm_kind"]      = [r.get("kind") for r in recs]
    df["llm_area_comp"] = [r.get("area_component") for r in recs]
    df["llm_dup_ids"]   = [",".join(d["id"] for d in ds) for ds in dup_flags]
    df["_ok"]           = oks
    df["_steps"]        = steps_list
    df["_traj"]         = [" → ".join(tr) for tr in trajs]
    df["_trajs_raw"]    = trajs   # keep raw for step 6

    return df


# ===========================================================================
# Step 6 · Auto-rater × κ (LLM-as-judge calibrated with human labels)
# ===========================================================================

def step6_kappa(extract_df: pd.DataFrame) -> dict:
    """LLM-as-judge auto-rater calibrated with human annotations.

    Workflow (mirrors development.ipynb):
      • If human_labels.csv lacks human_good annotations → generate/update the
        template CSV so the analyst can fill it in, return kappa=None.
      • Once annotated → run GPT-4o judge on each row and compute Cohen's κ
        between judge scores and human labels.
      • Always plots the supervisor trajectory distribution.

    Returns a dict with keys: kappa (float|None), eval_df (DataFrame|None).
    """
    from openai import OpenAI
    from sklearn.metrics import confusion_matrix

    oai = OpenAI()

    HUMAN         = "human_labels.csv"
    GUIDE         = "docs/human_labels_GUIDE.md"
    JUDGE_MODEL   = CHAT_MODEL
    JUDGE_THRESH  = 5
    FEW_SHOT_IDS  = {1963368, 1991810, 2041211, 2048096}

    ANNOTATION_GUIDE = (
        "# Annotation guide · human_labels.csv\n\n"
        "Judge whether the LLM-extracted fields faithfully reflect the bug text.\n"
        "For each row, read component/summary/description and llm_area/llm_severity/llm_kind, "
        "then fill human_good:\n\n"
        "  1 (good) : area matches the text topic; severity/kind are plausible\n"
        "  0 (bad)  : area is clearly off-topic, OR kind/severity obviously wrong\n"
        "  (blank)  : text too short/technical to judge — skip; does not affect κ\n\n"
        "20–40 labelled rows give a stable κ estimate.\n"
    )

    JUDGE_FEW_SHOT = (
        "Calibration examples (do NOT score these; use to calibrate your scale):\n\n"
        "[Example A — score 5] Bug: [Graphics: WebRender] Hit MOZ_CRASH(bug: texture not allocated)\n"
        'Extraction: {"area": "WebRender", "severity": "high", "kind": "crash"}\n'
        "→ area matches component exactly; crash+high stated in text. Score: 5\n\n"
        "[Example B — score 1] Bug: [Untriaged] gmail doesn't load in Nightly\n"
        '→ area is a website, not a browser component. Score: 1\n\n'
        "[Example C — score 2] Bug: [Upstream Synchronization] Port CSP baseline enforcement\n"
        '{"area":"security","severity":"high","kind":"crash"} → kind=crash is wrong. Score: 2\n\n'
        "[Example D — score 4] Bug: [Gecko Profiler] Lock-order-inversion (potential deadlock)\n"
        '{"area":"Gecko Profiler","severity":"high","kind":"performance"} → acceptable. Score: 4\n'
    )

    def judge_good(text: str, extraction: str) -> int:
        prompt = (
            "Rate whether the extracted JSON is a FAITHFUL and REASONABLE summary of the bug, score 1-5.\n"
            "ONLY judge whether area/severity/kind are defensible given the text.\n"
            "5=perfectly faithful; 4=faithful with minor issues; 3=acceptable; "
            "2=area or kind clearly off; 1=completely wrong.\n\n"
            + JUDGE_FEW_SHOT
            + "\n--- Now score THIS bug (reply ONLY JSON: {\"score\": 1-5, \"reason\": str}) ---\n"
            f"Bug:\n```{str(text)[:2000]}```\nExtraction: {extraction}"
        )
        try:
            r = oai.chat.completions.create(
                model=JUDGE_MODEL, temperature=0, max_tokens=300,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            ).choices[0].message.content
            return int(int(json.loads(r).get("score", 0)) >= JUDGE_THRESH)
        except Exception:
            return 0

    # ── Trajectory distribution chart (always plotted) ──────────────────────
    trajs_raw = list(extract_df.get("_trajs_raw", []))
    if trajs_raw:
        _abbr = {"research": "R", "extract": "E", "critique": "C", "FINISH": "F"}
        sig = Counter(
            "→".join(_abbr.get(s, s[0].upper()) for s in tr) + f"  ({len(tr)})"
            for tr in trajs_raw
        )
        top = sig.most_common(8)[::-1]
        fig, ax = plt.subplots(figsize=(8, max(3, len(top) * 0.55)))
        ax.barh([k for k, _ in top], [v for _, v in top], color="#4c72b0")
        ax.set_xlabel("number of bugs")
        ax.set_title("Supervisor trajectory distribution")
        plt.tight_layout()
        fig.savefig("assets/agent_trajectories.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("[kappa] saved agent_trajectories.png")

    # ── Check whether human_labels.csv has been annotated ───────────────────
    need_template = True
    if os.path.exists(HUMAN):
        _h = pd.read_csv(HUMAN)
        if ("human_good" in _h.columns and
                not _h["human_good"].astype(str).str.strip()
                      .replace("", np.nan).dropna().empty):
            need_template = False

    if need_template:
        tmpl = extract_df[extract_df["_ok"]].copy()
        tmpl = tmpl[["id", "component", "summary", "description",
                      "llm_area", "llm_severity", "llm_kind"]]
        tmpl["description"] = tmpl["description"].astype(str).str.slice(0, 1200)
        tmpl["human_good"] = ""
        tmpl.to_csv(HUMAN, index=False)
        with open(GUIDE, "w") as f:
            f.write(ANNOTATION_GUIDE)
        print(f"[kappa] generated annotation template {HUMAN} ({len(tmpl)} rows) "
              f"and guide {GUIDE}.")
        print("[kappa] Fill human_good (1/0/blank), save, then re-run step6_kappa.")
        return {"kappa": None, "eval_df": None}

    # ── Annotated — run judge and compute κ ──────────────────────────────────
    h = pd.read_csv(HUMAN)
    h = h[h["human_good"].astype(str).str.strip().isin(["0", "1"])].copy()
    h["human_good"] = h["human_good"].astype(int)
    h["_ext"]  = h.apply(
        lambda r: json.dumps({"area": r["llm_area"],
                              "severity": r["llm_severity"],
                              "kind": r["llm_kind"]}), axis=1)
    h["_text"] = ("[" + h["component"].astype(str) + "] "
                  + h["summary"].astype(str) + "\n"
                  + h["description"].astype(str))

    h_kappa = h[~h["id"].isin(FEW_SHOT_IDS)].copy().reset_index(drop=True)
    judge = [judge_good(t, e)
             for t, e in zip(h_kappa["_text"], h_kappa["_ext"])]
    kappa = float(cohen_kappa_score(h_kappa["human_good"], judge))

    print(f"[kappa] labeled rows: {len(h)} total, {len(h_kappa)} used for κ "
          f"({len(FEW_SHOT_IDS)} held out as few-shot anchors)")
    print(f"[kappa] judge model: {JUDGE_MODEL} | threshold: {JUDGE_THRESH}")
    print(f"[kappa] Cohen's κ (judge vs human) = {kappa:.3f}")
    print("confusion (rows=human, cols=judge):\n",
          confusion_matrix(h_kappa["human_good"], judge))
    print(f"judge good={sum(judge)}/{len(judge)} ({sum(judge)/len(judge):.0%})"
          f" | human good={h_kappa['human_good'].sum()}/{len(h_kappa)}"
          f" ({h_kappa['human_good'].mean():.0%})")

    return {"kappa": kappa, "eval_df": h_kappa}


# ===========================================================================
# Step 7 · Dedup retrieval evaluation (hit@k / MRR)
# ===========================================================================

def step7_dedup(bugs: pd.DataFrame, pairs: pd.DataFrame,
                col: "chromadb.Collection") -> dict:
    """Evaluate duplicate-bug retrieval on the 4,987 hard-label pairs.

    Returns a dict with hit@1, hit@5, hit@10, MRR.
    """
    dup_ids = pairs["dup_id"].astype(str).tolist()
    master_ids = pairs["master_id"].astype(str).tolist()

    # Filter to pairs whose embeddings are in Chroma
    existing = set(col.get()["ids"])
    valid = [(d, m) for d, m in zip(dup_ids, master_ids)
             if d in existing and m in existing]
    print(f"[dedup] evaluating {len(valid):,}/{len(pairs):,} pairs "
          f"(both dup and master embedded)")

    hits_at = {1: 0, 5: 0, 10: 0}
    rr_sum = 0.0
    K = 10

    BATCH = 100
    for start in range(0, len(valid), BATCH):
        batch = valid[start: start + BATCH]
        dup_batch = [d for d, _ in batch]
        master_batch = [m for _, m in batch]

        got = col.get(ids=dup_batch, include=["embeddings"])
        # Chroma may reorder
        id2vec = {i: v for i, v in zip(got["ids"], got["embeddings"])}
        vecs = [id2vec[d] for d in dup_batch if d in id2vec]
        valid_sub = [(d, m) for (d, m) in batch if d in id2vec]
        if not vecs:
            continue

        res = col.query(query_embeddings=vecs, n_results=K + 1)
        for i, (dup, master) in enumerate(valid_sub):
            retrieved = [r for r in res["ids"][i] if r != dup][:K]
            for k_val in (1, 5, 10):
                if master in retrieved[:k_val]:
                    hits_at[k_val] += 1
            if master in retrieved:
                rr_sum += 1.0 / (retrieved.index(master) + 1)

        if (start + BATCH) % 1000 == 0 or start + BATCH >= len(valid):
            print(f"[dedup]   {min(start + BATCH, len(valid)):,}/{len(valid):,}")

    n = len(valid)
    metrics = {
        "hit@1":  hits_at[1]  / n,
        "hit@5":  hits_at[5]  / n,
        "hit@10": hits_at[10] / n,
        "mrr":    rr_sum       / n,
    }
    print(f"[dedup] hit@1={metrics['hit@1']:.3f}"
          f"  hit@5={metrics['hit@5']:.3f}"
          f"  hit@10={metrics['hit@10']:.3f}"
          f"  MRR={metrics['mrr']:.3f}")
    return metrics


# ===========================================================================
# Step 8 · Readout
# ===========================================================================

def step8_readout(topic_df: pd.DataFrame,
                  extract_df: pd.DataFrame,
                  kappa_results: dict,
                  dedup_metrics: dict) -> None:
    """Print a final summary table."""
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)

    print("\n-- Top 5 bug themes --")
    for _, r in topic_df.head(5).iterrows():
        print(f"  {r['theme'][:40]:<40}  {r['share']:.1%}")

    print("\n-- Extraction agent (LLM-as-judge) --")
    if not extract_df.empty:
        oks = extract_df["_ok"].tolist()
        n = len(oks)
        n_extract_vals = []
        for traj in extract_df.get("_trajs_raw", [[]]*n):
            n_extract_vals.append(sum(1 for t in traj if t == "extract"))
        fp = sum(1 for o, ne in zip(oks, n_extract_vals) if o and ne == 1) / n
        print(f"  first-pass yield:   {fp:.1%}")
        print(f"  final valid yield:  {sum(oks)/n:.1%}")
        print(f"  mean steps:         "
              f"{sum(extract_df['_steps']) / n:.1f}")

    if kappa_results:
        kappa = kappa_results.get("kappa")
        if kappa is not None:
            print(f"  Cohen's κ (judge vs human): {kappa:.3f}")
        else:
            print("  Cohen's κ: pending human annotation")

    print("\n-- Dedup retrieval --")
    for k, v in dedup_metrics.items():
        print(f"  {k}: {v:.3f}")

    print("=" * 60)


# ===========================================================================
# main
# ===========================================================================

def main() -> None:
    bugs, pairs = step1_load()
    col = step2_embed(bugs)
    bugs = step3_cluster(bugs, col)
    topic_df = step4_topic_stats(bugs)
    extract_df = step5_extract(bugs, col)
    kappa_results = step6_kappa(extract_df)
    dedup_metrics = step7_dedup(bugs, pairs, col)
    step8_readout(topic_df, extract_df, kappa_results, dedup_metrics)


if __name__ == "__main__":
    main()
