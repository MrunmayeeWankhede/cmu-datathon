# Databricks notebook source
# ============================================================
# STEP 3b: AI layer (LLM on Databricks Foundation Model APIs)
#   1. AI evidence check  - an LLM re-reads each shortlisted facility's quotes and judges them
#   2. Question box       - turns "pregnant woman bleeding near Gaya" into service + place + patient
#   3. Plain summary      - two sentences a health worker can act on
# The rules from Step 3 stay in charge; the AI can only push a facility DOWN, and it only sees the quotes.
# Needs Step 3 to have been run first in this notebook session.
# ============================================================
import json, re, io, contextlib
from concurrent.futures import ThreadPoolExecutor
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

LLM_MODELS = ["databricks-meta-llama-3-3-70b-instruct",      # first choice
              "databricks-llama-4-maverick",                 # backups if the first is busy
              "databricks-qwen3-next-80b-a3b-instruct"]
AI_POOL       = 8     # how many top candidates the AI re-checks per search
AI_PENALTY    = {"confirmed": 0, "likely": 0, "unclear": None, "unlikely": 100}   # None = same as one warning
_w = WorkspaceClient()

def llm(system, user, max_tokens=400):
    """Call the first model that answers. Temperature 0 so the same question gives the same answer."""
    last = None
    for model in LLM_MODELS:
        try:
            r = _w.serving_endpoints.query(
                name=model, max_tokens=max_tokens, temperature=0.0,
                messages=[ChatMessage(role=ChatMessageRole.SYSTEM, content=system),
                          ChatMessage(role=ChatMessageRole.USER, content=user)])
            return r.choices[0].message.content, model
        except Exception as e:
            last = e
    raise RuntimeError(f"No AI model answered: {str(last)[:200]}")

def parse_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}

# ---- 1. AI evidence check -------------------------------------------------------
CHECK_SYSTEM = """You check whether a healthcare facility in India offers a specific medical service.
Use ONLY the numbered quotes given. Do not use outside knowledge about the facility. Quotes were scraped from
websites and directories, so they can be wrong, about another hospital, or about a different procedure.

verdict:
- "confirmed": a quote explicitly says this facility performs or provides this exact service
- "likely": quotes strongly imply it (a dedicated unit, the specific equipment) without saying it outright
- "unclear": only vague, generic or category-level mentions; or the quotes could mean something else
- "unlikely": the quotes are about a different procedure, say patients are referred elsewhere / it is not
  available / planned, or the facility clearly cannot provide it
patient_fit: "any" | "children only" | "adults only" | "unclear"
reason: at most 30 words, plain English, citing quote numbers like [1] [3].
Reply with JSON only: {"verdict": "...", "patient_fit": "...", "reason": "..."}"""

_ai_cache = {}

def ai_check(service_label, facility, quotes):
    key = (service_label, facility, tuple(quotes))
    if key in _ai_cache:
        return _ai_cache[key]
    numbered = "\n".join(f"[{i}] {q}" for i, q in enumerate(quotes, 1)) or "(no quotes)"
    try:
        text, model = llm(CHECK_SYSTEM, f"Service: {service_label}\nFacility name: {facility}\nQuotes:\n{numbered}",
                          max_tokens=200)
        j = parse_json(text)
        verdict = str(j.get("verdict", "unclear")).lower().strip()
        res = {"verdict": verdict if verdict in AI_PENALTY else "unclear",
               "fit": str(j.get("patient_fit", "unclear")).lower().strip(),
               "reason": str(j.get("reason", "")).strip()[:300], "model": model}
    except Exception as e:
        res = {"verdict": "not checked", "fit": "unclear", "reason": f"AI check unavailable ({str(e)[:80]})", "model": ""}
    _ai_cache[key] = res
    return res

def _facility_quotes(svc, name, why):
    """All quotes for this facility (up to 6), found via the evidence table."""
    rows = EV_PD[(EV_PD["service"] == svc) & (EV_PD["name"] == name)]
    for ev in rows["evidence"]:
        if "\n".join(ev[:3]) == why:
            return list(ev)
    return list(rows["evidence"].iloc[0]) if len(rows) else [q for q in why.split("\n") if q]

def _capture(service, location, for_child, top_n):
    """Run Step 3's recommend() quietly and keep its summary lines."""
    printed, real_display = io.StringIO(), globals().get("display")
    globals()["display"] = lambda *a, **k: None
    try:
        with contextlib.redirect_stdout(printed):
            out = recommend(service, location, top_n=top_n, for_child=for_child)
    finally:
        globals()["display"] = real_display
    info = {"notes": [], "where": "", "rule": "", "count": ""}
    for line in printed.getvalue().splitlines():
        if line.startswith("LOCATION:"):
            info["where"] = line.split(":", 1)[1].strip()
        elif line.startswith("NOTE:"):
            info["notes"].append(line.split(":", 1)[1].strip())
        elif line.startswith("HOW WE RANK"):
            info["rule"] = line.split(":", 1)[1].strip()
        elif line.startswith("Facilities with"):
            info["count"] = line.strip()
    return out, info

FILLER = r"(?i)\b(near|nearby|around|close to|outside|village|town|city|district|dist|tehsil|block|in|at|the|of|area|from)\b"

def resolve_place(text):
    """Make free-text places searchable: 'village near Gaya' -> 'Gaya'. Falls back to single words."""
    t = str(text).strip()
    cleaned = re.sub(r"\s+", " ", re.sub(r"[^\w\s,.-]", " ", re.sub(FILLER, " ", t))).strip(" ,.-")
    words = [w for w in re.split(r"[\s,]+", cleaned) if len(w) > 2]
    tries = [t, cleaned] + [" ".join(words[i:i + 2]) for i in range(len(words) - 1)] + words
    for cand in dict.fromkeys(x for x in tries if x):
        try:
            locate(cand)
            return cand
        except ValueError:
            pass
    return t          # nothing matched: locate() will raise its helpful "did you mean" message

def smart_search(service, location, for_child=False, top_n=5, use_ai=True):
    """Step 3 ranking, then an AI re-check of the top AI_POOL candidates. Returns (results, info)."""
    location = resolve_place(location)
    svc = resolve_service(service)
    label, cfg = SERVICE_LABELS[svc], MODES[SERVICE_MODE[svc]]
    out, info = _capture(service, location, for_child, AI_POOL if use_ai else top_n)
    info.update(service=label, mode=SERVICE_MODE[svc], patient="child" if for_child else "adult", ai_model="")
    if out.empty or not use_ai:
        return out.head(top_n), info

    quotes = [_facility_quotes(svc, n, w) for n, w in zip(out["Facility"], out["WHY: quoted from the record"])]
    with ThreadPoolExecutor(max_workers=4) as pool:
        checks = list(pool.map(lambda a: ai_check(label, *a), zip(out["Facility"], quotes)))
    info["ai_model"] = next((c["model"] for c in checks if c["model"]), "")

    ranked_km, how, verdicts, reasons = [], [], [], []
    for (_, r), c in zip(out.iterrows(), checks):
        base = float(re.search(r"ranked as (\d+) km", r["HOW it was ranked"]).group(1))
        pen = AI_PENALTY.get(c["verdict"], 0)
        pen = cfg["per_warning_km"] if pen is None else pen
        wrong_patient = (c["fit"] == "children only" and not for_child) or (c["fit"] == "adults only" and for_child)
        if wrong_patient and "children only" not in r["Warnings"]:
            pen += cfg["per_warning_km"]
        txt = re.sub(r" = ranked as \d+ km \(\w+ settings\)$", "", r["HOW it was ranked"])
        if pen:
            txt += f" + {pen} km (AI check: {c['verdict']}{', ' + c['fit'] if wrong_patient else ''})"
        how.append(txt + f" = ranked as {base + pen:.0f} km ({info['mode']} settings)")
        ranked_km.append(base + pen)
        verdicts.append(c["verdict"] + ("" if c["fit"] in ("any", "unclear") else f" ({c['fit']})"))
        reasons.append(c["reason"])
    out = out.assign(**{"HOW it was ranked": how, "AI check": verdicts, "AI reasoning": reasons, "_km": ranked_km,
                        "All quotes": ["\n".join(q) for q in quotes]})   # every quote the AI read, numbered on the card

    main = out[out["Note"] == ""].sort_values("_km", kind="stable").head(top_n)
    extra = out[out["Note"] != ""]
    out = pd.concat([main, extra]).drop(columns="_km").reset_index(drop=True)
    out["Rank"] = range(1, len(out) + 1)
    return out, info

# ---- 2. Question box ---------------------------------------------------------------
INTAKE_SYSTEM = f"""You turn a health worker's message about a patient into a search.
Services you can choose (pick the single best match, or "unknown"): {", ".join(SERVICE_LABELS.values())}.
Guidance: labour complications, obstructed or high-risk delivery, heavy bleeding in pregnancy -> Emergency C-section;
sick or premature newborn -> NICU; kidney failure -> Dialysis; road accident, fall, injury -> Trauma / emergency care;
fire, scald, acid injury -> Burn care.
location: the place NAME only, e.g. "Gaya" or "800001" or "25.6, 85.1" - drop words like near, village, district; "" if none.
patient: "child" if under 18 or a newborn, else "adult".
Reply with JSON only: {{"service": "...", "location": "...", "patient": "...", "why": "<10 words on the service choice>"}}"""

def understand_question(text):
    raw, model = llm(INTAKE_SYSTEM, text, max_tokens=150)
    j = parse_json(raw)
    service = str(j.get("service", "unknown"))
    if service not in SERVICE_LABELS.values():
        service = "unknown"
    return {"service": service, "location": str(j.get("location", "")).strip(),
            "patient": "child" if str(j.get("patient", "")).lower() == "child" else "adult",
            "why": str(j.get("why", "")), "model": model}

# ---- 3. Plain summary -----------------------------------------------------------------
SUMMARY_SYSTEM = """Write a 2-sentence summary for a community health worker, using ONLY the facts given.
Sentence 1: the best option, how far, and why (evidence level, AI check). Sentence 2: the main caution or the
backup option. Plain words, no jargon, no new facts. Always end with "Call ahead to confirm." """

def summarize(out, info):
    top = out.head(3)
    facts = "\n".join(
        f"{r['Rank']}. {r['Facility']} ({r['Where']}) - {r['Distance']} - evidence {r['Evidence']} - "
        f"AI check {r.get('AI check', 'n/a')} - warnings: {r['Warnings'].replace(chr(10), '; ')}"
        for _, r in top.iterrows())
    try:
        text, _ = llm(SUMMARY_SYSTEM, f"Patient needs: {info['service']} ({info['patient']}) near {info['where']}\n"
                                      f"Notes: {' '.join(info['notes'])}\nTop options:\n{facts}", max_tokens=160)
        return text.strip()
    except Exception:
        return ""

# ---- Try it ---------------------------------------------------------------------------
def show(res, cols=("Rank", "Facility", "Distance", "Evidence", "AI check", "AI reasoning", "HOW it was ranked")):
    if res.empty:
        print("No facilities found.")
    else:
        display(res[[c for c in cols if c in res.columns]])

res, info = smart_search("burn care", "Patna")
print("AI model used:", info["ai_model"] or "none (no candidates)")
show(res)

q = understand_question("Woman in labour for 20 hours, baby not coming out, we are in a village near Gaya")
print("\nQuestion understood as:", q)
res2, info2 = smart_search(q["service"], q["location"], for_child=(q["patient"] == "child"))
print("\nSUMMARY:", summarize(res2, info2) if len(res2) else "No facilities found.")
show(res2)
