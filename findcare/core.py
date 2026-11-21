"""
findcare.core - the ranking + AI logic, usable from a notebook or a Databricks App.

    from findcare import core
    core.load(core.spark_reader(spark))            # in a notebook
    core.load(core.warehouse_reader())             # in a Databricks App
    result = core.search(question="woman in labour, village near Gaya")
    result = core.search(service="Dialysis", location="Jaipur", patient="adult")

`search` returns a dict: {"ok", "error", "understood", "service", "mode", "patient", "where", "count",
"notes", "rule", "summary", "ai_model", "results": [ {card fields...}, ... ]}

The lookup tables are produced by notebooks/05_export_app_tables.py.
"""
import difflib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

# =====================================================================================
# Settings: together these ARE the answer to "how did you decide?"
# =====================================================================================
TABLE_PREFIX = os.environ.get("FINDCARE_TABLE_PREFIX", "workspace.default.app_")

SERVICE_LABELS = {"dialysis": "Dialysis", "emergency_c_section": "Emergency C-section", "nicu": "NICU",
                  "cardiac_surgery": "Cardiac surgery", "mri": "MRI", "ct_scan": "CT scan", "burn_care": "Burn care",
                  "trauma_emergency": "Trauma / emergency care", "icu": "ICU"}
SERVICE_MODE = {"emergency_c_section": "emergency", "trauma_emergency": "emergency", "burn_care": "emergency",
                "nicu": "emergency", "icu": "emergency",
                "dialysis": "planned", "mri": "planned", "ct_scan": "planned", "cardiac_surgery": "planned"}
MODES = {
    "emergency": dict(radius_km=50, penalty={"Strong": 0, "Moderate": 10, "Weak": None}, per_warning_km=10),
    "planned": dict(radius_km=100, penalty={"Strong": 0, "Moderate": 30, "Weak": 75}, per_warning_km=25),
}
WEAK_LAST_RESORT_KM = 75
ROAD_FACTOR, ROAD_SPEED_KMH = 1.4, 40

MAJOR_FLAGS = {
    "No coordinates",
    "Coordinates fall outside India",
    "Name suggests a single-specialty clinic but record claims many specialties (possible merged record)",
    "Unusually many claims for one facility (possible merged record)",
    "Bed count and doctor count are implausible together",
    "Some source URLs are unrelated pages (e.g. login pages)",
}

LLM_MODELS = ["databricks-meta-llama-3-3-70b-instruct", "databricks-llama-4-maverick",
              "databricks-qwen3-next-80b-a3b-instruct"]
AI_POOL = 8
AI_PENALTY = {"confirmed": 0, "likely": 0, "unclear": None, "unlikely": 100}   # None = one warning's worth

SERVICE_ALIASES = {
    "dialysis": "dialysis", "kidney": "dialysis", "hemodialysis": "dialysis", "haemodialysis": "dialysis",
    "c-section": "emergency_c_section", "c section": "emergency_c_section", "csection": "emergency_c_section",
    "caesarean": "emergency_c_section", "cesarean": "emergency_c_section", "emergency c-section": "emergency_c_section",
    "high-risk delivery": "emergency_c_section", "high risk delivery": "emergency_c_section",
    "obstetric": "emergency_c_section",
    "nicu": "nicu", "newborn": "nicu", "neonatal": "nicu", "sncu": "nicu",
    "cardiac surgery": "cardiac_surgery", "heart surgery": "cardiac_surgery", "bypass": "cardiac_surgery",
    "mri": "mri", "ct": "ct_scan", "ct scan": "ct_scan", "cat scan": "ct_scan",
    "burn": "burn_care", "burns": "burn_care", "burn care": "burn_care",
    "trauma": "trauma_emergency", "emergency": "trauma_emergency", "accident": "trauma_emergency",
    "icu": "icu", "intensive care": "icu", "critical care": "icu",
}

JUNK_URL = re.compile(r"(?i)(login|signin|sign-in|creditcard|checkout|cart)")
CHILD_TEXT = re.compile(r"(?i)(p(a)?ediatric|\bchild|\bneonat|\binfant|\bnewborn)")
CHILD_NAME = re.compile(r"(?i)\b(child|children'?s?|p(a)?ediatric\w*|kids|shishu)\b")
CHILD_ONLY_NOTE = "Evidence mentions children only - may not treat adults"
SPECIALIST_NAME = re.compile(r"(?i)\b(eye|netra|nethralaya|ophthalm\w*|dental|dentist|teeth|ayurved\w*|homoeo\w*|"
                             r"homeo\w*|physiotherap\w*|skin|derma\w*|ivf|fertility)\b")
SPECIALIST_OK = {"ivf": {"emergency_c_section", "nicu"}, "fertility": {"emergency_c_section", "nicu"}}
FILLER = r"(?i)\b(near|nearby|around|close to|outside|village|town|city|district|dist|tehsil|block|in|at|the|of|area|from)\b"
LIST_COLS = ["evidence", "caution_mentions", "quality_flags", "loc_flags", "url_list"]

# =====================================================================================
# Data loading
# =====================================================================================
EV = TAG = CITY = PIN = DIST = OFFICE = None


def spark_reader(spark):
    """Read a table in a notebook."""
    return lambda name: spark.table(name).toPandas()


def warehouse_reader(warehouse_id=None):
    """Read a table from a SQL warehouse (used inside a Databricks App)."""
    from databricks import sql
    from databricks.sdk.core import Config
    cfg = Config()
    wid = warehouse_id or os.environ["DATABRICKS_WAREHOUSE_ID"]

    def read(name):
        with sql.connect(server_hostname=cfg.host, http_path=f"/sql/1.0/warehouses/{wid}",
                         credentials_provider=lambda: cfg.authenticate) as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT * FROM {name}")
                return cur.fetchall_arrow().to_pandas()
    return read


def _as_list(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return [v] if v else []
    return [x for x in list(v) if x]


def load(reader, prefix=TABLE_PREFIX):
    """Load the six lookup tables written by the export notebook."""
    global EV, TAG, CITY, PIN, DIST, OFFICE
    EV = reader(prefix + "evidence")
    for c in LIST_COLS:
        EV[c] = EV[c].map(_as_list)
    TAG = reader(prefix + "tag_only")
    CITY = reader(prefix + "city").set_index("k")
    PIN = reader(prefix + "pin").astype({"pin": str}).set_index("pin")
    DIST = reader(prefix + "district").set_index("k")
    OFFICE = reader(prefix + "office").astype({"pin": str}).set_index("k")
    return {"evidence": len(EV), "tag_only": len(TAG), "cities": len(CITY), "pins": len(PIN),
            "districts": len(DIST), "offices": len(OFFICE)}


# =====================================================================================
# Understanding the inputs
# =====================================================================================
def norm_name(s):
    s = re.sub(r"\b(district|city|dist\.?)\b", "", str(s).lower())
    s = re.sub(r"\b[bsh]\.?o\.?$", "", s.strip())
    return re.sub(r"[^a-z]", "", s)


def resolve_service(text):
    t = str(text).strip().lower().replace("_", " ")
    if t.replace(" ", "_") in SERVICE_LABELS:
        return t.replace(" ", "_")
    for key, label in SERVICE_LABELS.items():
        if t == label.lower():
            return key
    if t in SERVICE_ALIASES:
        return SERVICE_ALIASES[t]
    for alias in sorted(SERVICE_ALIASES, key=len, reverse=True):
        if alias in t:
            return SERVICE_ALIASES[alias]
    raise ValueError(f"Unknown service '{text}'. Try one of: {', '.join(SERVICE_LABELS.values())}")


def locate(text):
    """Coordinates / PIN / city / district / post office -> (lat, lon, description)."""
    t = str(text).strip()
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", t)
    if m:
        return float(m.group(1)), float(m.group(2)), f"coordinates {t}"
    if re.fullmatch(r"\d{6}", t):
        if t in PIN.index:
            r = PIN.loc[t]
            return float(r.pin_lat), float(r.pin_lon), f"PIN {t} ({r.pin_district}, {r.pin_state})"
        raise ValueError(f"PIN code {t} was not found in the India Post directory.")
    k = norm_name(t)
    if k in CITY.index:
        r = CITY.loc[k]
        n = int(r.n)
        return float(r.lat), float(r.lon), f"{r.city}, {r.state} (centre of {n} listed facilit{'y' if n == 1 else 'ies'})"
    if k in DIST.index:
        r = DIST.loc[k]
        return float(r.lat), float(r.lon), f"{r.pin_district} district, {r.state} (centre of its PIN codes)"
    if k in OFFICE.index:
        r = OFFICE.loc[k]
        return float(r.lat), float(r.lon), f"{r.office} post office area, {r.district}, {r.state} (PIN {r.pin})"
    names = list(CITY.index) + list(DIST.index)
    close = list(dict.fromkeys(difflib.get_close_matches(k, names, n=5, cutoff=0.75)))[:3]
    hint = f" Did you mean: {', '.join(close)}?" if close else " Try a district name or a 6-digit PIN code."
    raise ValueError(f"Couldn't find '{text}'.{hint}")


def resolve_place(text):
    """'village near Gaya' -> 'Gaya'. Falls back to single words."""
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
    return t


# =====================================================================================
# Ranking helpers
# =====================================================================================
def km_np(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def road_time(km_straight):
    mins = max(5, int(round(km_straight * ROAD_FACTOR / ROAD_SPEED_KMH * 60 / 5.0)) * 5)
    return f"~{mins} min" if mins < 60 else f"~{mins // 60} h {mins % 60:02d} min"


def children_only(name, evidence):
    return bool(CHILD_NAME.search(name or "")) or (len(evidence) > 0 and all(CHILD_TEXT.search(e) for e in evidence))


def specialist_note(name, svc):
    m = SPECIALIST_NAME.search(name or "")
    if not m or svc in SPECIALIST_OK.get(m.group(1).lower(), set()):
        return []
    return [f"Name suggests a specialist clinic ('{m.group(1)}') - unlikely to offer {SERVICE_LABELS[svc]}"]


def alt_distance(r, lat, lon):
    """If a facility's location is in doubt, the distance to the OTHER place its record points to."""
    alts = []
    if any("from its PIN code area" in f for f in r.loc_flags) and pd.notna(r.pin_lat):
        alts.append(float(km_np(lat, lon, r.pin_lat, r.pin_lon)))
    if any("its stated city" in f for f in r.loc_flags):
        k = norm_name(r.address_city)
        if k in CITY.index:
            alts.append(float(km_np(lat, lon, CITY.loc[k, "lat"], CITY.loc[k, "lon"])))
    return max(alts) if alts else np.nan


# =====================================================================================
# Rules-based ranking
# =====================================================================================
def recommend(service, location, top_n=5, for_child=False, radius_km=None):
    """Returns (candidates DataFrame with raw fields, info dict). No printing, no AI."""
    svc = resolve_service(service)
    lat, lon, where = locate(location)
    mode = SERVICE_MODE[svc]
    cfg = MODES[mode]
    radius = radius_km or cfg["radius_km"]

    c = EV[EV["service"] == svc].copy()
    listed = km_np(lat, lon, c["best_lat"].values, c["best_lon"].values)
    alt = np.array([alt_distance(r, lat, lon) for r in c.itertuples()]) if len(c) else np.array([])
    c["distance_lo"] = np.fmin(listed, alt) if len(c) else listed
    c["distance_km"] = np.fmax(listed, alt) if len(c) else listed
    c["uncertain"] = (c["distance_km"] - c["distance_lo"]) > 10
    child = [(svc != "nicu") and (not for_child) and children_only(n, e) for n, e in zip(c["name"], c["evidence"])]
    c["warnings"] = [[f for f in q if f in MAJOR_FLAGS] + list(l) + ([CHILD_ONLY_NOTE] if co else [])
                     + specialist_note(n, svc)
                     for q, l, co, n in zip(c["quality_flags"], c["loc_flags"], child, c["name"])]
    c["eligible"] = c["evidence_strength"].map(lambda s: cfg["penalty"][s] is not None)
    c["evidence_penalty"] = c["evidence_strength"].map(
        lambda s: cfg["penalty"][s] if cfg["penalty"][s] is not None else WEAK_LAST_RESORT_KM)
    c["warning_penalty"] = c["warnings"].map(len) * cfg["per_warning_km"]
    c["effective_km"] = c["distance_km"] + c["evidence_penalty"] + c["warning_penalty"]
    c["note"] = ""
    order = ["effective_km", "distance_km"]

    near = c[c["distance_km"] <= radius]
    within = near[near["eligible"]]
    notes = []
    if len(within):
        ranked = within.sort_values(order).head(top_n)
    elif len(near):
        notes.append(f"No facility within {radius} km has Strong or Moderate evidence. Showing nearby places with "
                     f"only weak evidence - phone first to confirm before travelling.")
        ranked = near.sort_values(order).head(top_n)
    else:
        notes.append(f"No facility with evidence for {SERVICE_LABELS[svc]} within {radius} km. "
                     f"Showing the best options further away.")
        pool = c[c["eligible"]] if c["eligible"].any() else c
        ranked = pool.sort_values(order).head(top_n)
    if len(c) and not (ranked["evidence_strength"] == "Strong").any():
        strong = c[c["evidence_strength"] == "Strong"].sort_values(order).head(1)
        if len(strong):
            s = strong.iloc[0]
            notes.append(f"None of these has Strong evidence. The best option with Strong evidence is "
                         f"{s['name']}, {s['distance_km']:.0f} km away (added at the bottom).")
            ranked = pd.concat([ranked, strong.assign(note="Outside search area: best Strong option")])
        else:
            notes.append(f"No facility in the dataset has Strong evidence for {SERVICE_LABELS[svc]}.")
    if len(within) < top_n and TAG is not None and len(TAG):
        t = TAG[TAG["service"] == svc].copy()
        t["d"] = km_np(lat, lon, t["best_lat"].values, t["best_lon"].values)
        t = t[(t["d"] <= radius) & ~t["name"].isin(ranked["name"])].sort_values("d")
        if len(t):
            one = len(t) == 1
            names = ", ".join(f"{n} ({d:.0f} km)" for n, d in zip(t["name"].head(3), t["d"].head(3)))
            notes.append(f"{len(t)} more facilit{'y' if one else 'ies'} within {radius} km list{'s' if one else ''} a "
                         f"related specialty but {'its record says' if one else 'their records say'} nothing specific "
                         f"about {SERVICE_LABELS[svc]}: {names}. Worth a phone call if the options below are too far.")
    if len(ranked) and ranked["uncertain"].any():
        notes.append("Some facilities have conflicting locations in their records; for those we show a distance "
                     "range and rank them on the longer distance.")
    notes.append("Distances are straight-line; road times are rough estimates. Always call ahead to confirm "
                 "the service is available today.")

    p = cfg["penalty"]
    weak_txt = f"{p['Weak']} km if Weak" if p["Weak"] is not None else "Weak evidence only as a last resort"
    info = {"service_key": svc, "service": SERVICE_LABELS[svc], "mode": mode, "radius_km": radius,
            "patient": "child" if for_child else "adult", "where": where,
            "count": f"Facilities with Strong/Moderate evidence within {radius} km: {len(within)}",
            "notes": notes, "ai_model": "",
            "rule": (f"distance, plus {p['Moderate']} km if evidence is only Moderate, {weak_txt}, and "
                     f"{cfg['per_warning_km']} km per warning. Lowest total first.")}
    return ranked.reset_index(drop=True), info


def _card(r, mode):
    nw = len(r.warnings)
    dist_txt = (f"{r.distance_km:.0f} km away (the longer of two conflicting locations)" if r.uncertain
                else f"{r.distance_km:.0f} km away")
    how = (dist_txt
           + (f" + {r.evidence_penalty} km (evidence only {r.evidence_strength})" if r.evidence_penalty else "")
           + (f" + {r.warning_penalty} km ({nw} warning{'s' if nw > 1 else ''})" if r.warning_penalty else ""))
    size = ", ".join(x for x in [f"{int(r.beds)} beds" if pd.notna(r.beds) else "",
                                 f"{int(r.doctors)} doctors" if pd.notna(r.doctors) else ""] if x)
    return {
        "facility": r["name"],
        "where": ", ".join(x for x in [r.address_city, r.address_stateOrRegion] if isinstance(x, str) and x),
        "distance": ((f"{r.distance_lo:.0f}-{r.distance_km:.0f} km straight-line (location uncertain), "
                      f"{road_time(r.distance_lo)} to {road_time(r.distance_km)} by road") if r.uncertain else
                     f"{r.distance_km:.0f} km straight-line, {road_time(r.distance_km)} by road"
                     + (" (approx. location)" if r.location_source != "listed coordinates" else "")),
        "distance_km": round(float(r.distance_km), 1),
        "lat": float(r.best_lat), "lon": float(r.best_lon),
        "evidence_level": r.evidence_strength,
        "evidence_label": r.evidence_strength + (" (2+ separate procedure statements)" if r.upgraded else ""),
        "quotes": list(r.evidence),
        "warnings": list(r.warnings),
        "cautions": ["Mentioned as referral / unavailable: " + m for m in r.caution_mentions],
        "how_parts": how, "ranked_km": float(r.effective_km),
        "size": size or "Not listed",
        "sources": [u for u in dict.fromkeys(r.url_list) if not JUNK_URL.search(u)][:3],
        "note": r.note,
        "ai_verdict": "", "ai_fit": "", "ai_reason": "",
    }


# =====================================================================================
# AI layer
# =====================================================================================
_client = None


def llm(system, user, max_tokens=400):
    global _client
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
    if _client is None:
        _client = WorkspaceClient()
    last = None
    for model in LLM_MODELS:
        try:
            r = _client.serving_endpoints.query(
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
    if key not in _ai_cache:
        numbered = "\n".join(f"[{i}] {q}" for i, q in enumerate(quotes, 1)) or "(no quotes)"
        try:
            text, model = llm(CHECK_SYSTEM, f"Service: {service_label}\nFacility name: {facility}\nQuotes:\n{numbered}",
                              max_tokens=200)
            j = parse_json(text)
            v = str(j.get("verdict", "unclear")).lower().strip()
            _ai_cache[key] = {"verdict": v if v in AI_PENALTY else "unclear",
                              "fit": str(j.get("patient_fit", "unclear")).lower().strip(),
                              "reason": str(j.get("reason", "")).strip()[:300], "model": model}
        except Exception as e:
            _ai_cache[key] = {"verdict": "not checked", "fit": "unclear",
                              "reason": f"AI check unavailable ({str(e)[:80]})", "model": ""}
    return _ai_cache[key]


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
    return {"service": service if service in SERVICE_LABELS.values() else "unknown",
            "location": str(j.get("location", "")).strip(),
            "patient": "child" if str(j.get("patient", "")).lower() == "child" else "adult",
            "why": str(j.get("why", "")), "model": model}


SUMMARY_SYSTEM = """Write a 2-sentence summary for a community health worker, using ONLY the facts given.
Sentence 1: the best option, how far, and why (evidence level, AI check). Sentence 2: the main caution or the
backup option. Plain words, no jargon, no new facts. Always end with "Call ahead to confirm." """


def summarize(cards, info):
    facts = "\n".join(
        f"{i}. {c['facility']} ({c['where']}) - {c['distance']} - evidence {c['evidence_label']} - "
        f"AI check {c['ai_verdict'] or 'n/a'} - warnings: {'; '.join(c['warnings'] + c['cautions']) or 'none'}"
        for i, c in enumerate(cards[:3], 1))
    try:
        text, _ = llm(SUMMARY_SYSTEM, f"Patient needs: {info['service']} ({info['patient']}) near {info['where']}\n"
                                      f"Notes: {' '.join(info['notes'])}\nTop options:\n{facts}", max_tokens=160)
        return text.strip()
    except Exception:
        return ""


# =====================================================================================
# The one function the app calls
# =====================================================================================
def search(question="", service="", location="", patient="adult", top_n=5, use_ai=True):
    out = {"ok": False, "error": "", "understood": None, "results": [], "summary": ""}
    try:
        if question.strip():
            q = understand_question(question)
            out["understood"] = q
            if q["service"] == "unknown" or not q["location"]:
                raise ValueError("Couldn't tell the service or the place from the description. "
                                 "Please add them, or pick them from the menus instead.")
            service, location, patient = q["service"], q["location"], q["patient"]
        for_child = str(patient).lower() == "child"
        ranked, info = recommend(service, resolve_place(location), top_n=AI_POOL if use_ai else top_n,
                                 for_child=for_child)
    except ValueError as e:
        out["error"] = str(e)
        return out

    cards = [_card(r, info["mode"]) for _, r in ranked.iterrows()]
    if use_ai and cards:
        cfg = MODES[info["mode"]]
        with ThreadPoolExecutor(max_workers=4) as pool:
            checks = list(pool.map(lambda c: ai_check(info["service"], c["facility"], c["quotes"]), cards))
        info["ai_model"] = next((c["model"] for c in checks if c["model"]), "")
        for c, chk in zip(cards, checks):
            pen = AI_PENALTY.get(chk["verdict"], 0)
            pen = cfg["per_warning_km"] if pen is None else pen
            wrong = (chk["fit"] == "children only" and not for_child) or (chk["fit"] == "adults only" and for_child)
            if wrong and CHILD_ONLY_NOTE not in c["warnings"]:
                pen += cfg["per_warning_km"]
            if pen:
                c["how_parts"] += f" + {pen} km (AI check: {chk['verdict']}{', ' + chk['fit'] if wrong else ''})"
            c["ranked_km"] += pen
            c["ai_verdict"], c["ai_fit"], c["ai_reason"] = chk["verdict"], chk["fit"], chk["reason"]
        main = sorted([c for c in cards if not c["note"]], key=lambda c: c["ranked_km"])[:top_n]
        cards = main + [c for c in cards if c["note"]]
    else:
        cards = cards[:top_n] + [c for c in cards[top_n:] if c["note"]]
    for i, c in enumerate(cards, 1):
        c["rank"] = i
        c["how"] = f"{c['how_parts']} = ranked as {c['ranked_km']:.0f} km ({info['mode']} settings)"
    out.update(info)
    out["results"] = cards
    out["ok"] = True
    if use_ai and cards:
        out["summary"] = summarize(cards, info)
    return out
