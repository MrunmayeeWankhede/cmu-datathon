# Databricks notebook source
# ============================================================
# STEP 4: App for health workers (notebook widgets + result cards)
# Input boxes appear at the TOP of the notebook. Fill them in, then re-run this cell.
# Needs Step 3 and Step 3b (AI layer) to have been run first in this session.
# ============================================================
import html, re

# ---- 1. Input boxes ---------------------------------------------------------
dbutils.widgets.text("question", "", "0. Describe the case in your own words (optional, overrides 1-3)")
dbutils.widgets.dropdown("service", "Dialysis", list(SERVICE_LABELS.values()), "1. Service needed")
dbutils.widgets.text("location", "Jaipur", "2. Where is the patient? (city, district, PIN or lat, lon)")
dbutils.widgets.dropdown("patient", "Adult", ["Adult", "Child"], "3. Patient")
dbutils.widgets.dropdown("ai", "On", ["On", "Off"], "4. AI check")

# ---- 2. Card design ---------------------------------------------------------
E = html.escape
BADGE = {"Strong": ("#e6f4ea", "#1e7e34"), "Moderate": ("#fff4e0", "#a15c00"), "Weak": ("#eeeeee", "#555555")}
AI_BADGE = {"confirmed": ("#e6f4ea", "#1e7e34"), "likely": ("#e3f4f4", "#0f6e6e"), "unclear": ("#fff4e0", "#a15c00"),
            "unlikely": ("#fdeceb", "#a3221a"), "not checked": ("#eeeeee", "#555555")}
FIELD_LABEL = {"procedure": "Procedure", "capability": "Capability", "equipment": "Equipment",
               "description": "Description", "specialties": "Specialty tag"}

def quote_html(i, q):
    m = re.match(r"\[(\w+)\]\s*(.*)", q, re.S)
    field, text = (m.group(1), m.group(2)) if m else ("", q)
    return (f'<li><span class="num">[{i}]</span> <span class="tag">{E(FIELD_LABEL.get(field, field))}</span> '
            f'<span class="q">&ldquo;{E(text)}&rdquo;</span></li>')

def card_html(r, ai_on):
    level = r["Evidence"].split(" ")[0]
    bg, fg = BADGE.get(level, BADGE["Weak"])
    quotes = "".join(quote_html(i, q) for i, q in
                     enumerate([q for q in str(r.get("All quotes") or r["WHY: quoted from the record"]).split("\n")
                                if q.strip()], 1))
    ai = ""
    if ai_on and "AI check" in r:
        v = str(r["AI check"]).split(" ")[0]
        abg, afg = AI_BADGE.get(v, AI_BADGE["not checked"])
        ai = (f'<div class="ai"><span class="badge" style="background:{abg};color:{afg}">AI check: {E(str(r["AI check"]))}</span>'
              f' <span class="air">{E(str(r["AI reasoning"]))}</span></div>')
    warn = "" if r["Warnings"] == "None" else (
        '<div class="warn"><b>Check before sending a patient:</b><ul>'
        + "".join(f"<li>{E(w)}</li>" for w in str(r["Warnings"]).split("\n") if w.strip()) + "</ul></div>")
    links = [u for u in str(r["Sources"]).split("\n") if u.startswith("http")]
    src = (" &middot; ".join(f'<a href="{E(u)}" target="_blank" rel="noopener">{E(re.sub(r"^https?://(www\\.)?", "", u)[:60])}</a>'
                             for u in links) or "None listed")
    note = f'<div class="note">{E(r["Note"])}</div>' if r["Note"] else ""
    return f"""
    <div class="card">
      {note}
      <div class="top">
        <div class="rank">{int(r["Rank"])}</div>
        <div class="title"><div class="name">{E(r["Facility"])}</div><div class="where">{E(r["Where"])}</div></div>
        <div class="badge" style="background:{bg};color:{fg}">{E(r["Evidence"])} evidence</div>
      </div>
      <div class="dist">{E(r["Distance"])}</div>
      <div class="sec">Why we think they offer this (quoted from the record)</div><ul class="quotes">{quotes}</ul>
      {ai}
      {warn}
      <div class="meta"><b>How it was ranked:</b> {E(r["HOW it was ranked"])}</div>
      <div class="meta"><b>Size (as listed):</b> {E(r["Size (as listed)"])}</div>
      <div class="meta"><b>Sources:</b> {src}</div>
    </div>"""

CSS = """
<style>
 .app{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#1d1d1f;background:#fafafa;
      padding:20px;border-radius:12px;max-width:920px}
 .app h2{margin:0 0 4px;font-size:22px} .app .sub{color:#555;margin-bottom:12px;font-size:14px}
 .understood{background:#f3f0ff;border:1px solid #d8cffc;border-radius:8px;padding:10px 14px;margin:10px 0;font-size:14px}
 .summary{background:#fff;border-left:4px solid #1d1d1f;border-radius:6px;padding:10px 14px;margin:12px 0;font-size:15px}
 .notes{background:#eef4ff;border:1px solid #c9dafc;border-radius:8px;padding:10px 14px 10px 30px;margin:10px 0;font-size:14px}
 .notes li{margin:3px 0}
 .rule{font-size:13px;color:#444;margin:8px 0 16px}
 .card{background:#fff;border:1px solid #e2e2e2;border-radius:10px;padding:16px 18px;margin:12px 0;
       box-shadow:0 1px 2px rgba(0,0,0,.04)}
 .top{display:flex;align-items:center;gap:12px}
 .rank{flex:none;width:32px;height:32px;border-radius:50%;background:#1d1d1f;color:#fff;display:flex;
       align-items:center;justify-content:center;font-weight:600}
 .title{flex:1} .name{font-size:17px;font-weight:600} .where{color:#666;font-size:13px}
 .badge{flex:none;padding:4px 10px;border-radius:999px;font-size:12px;font-weight:600;display:inline-block}
 .dist{margin:10px 0 6px;font-size:14px;font-weight:500}
 .sec{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#777;margin-top:8px}
 .quotes{margin:6px 0 8px;padding-left:18px;font-size:14px;list-style:none} .quotes li{margin:4px 0}
 .num{color:#888;font-size:12px}
 .tag{display:inline-block;font-size:11px;background:#f0f0f0;color:#555;border-radius:4px;padding:1px 6px;margin-right:4px}
 .q{color:#222}
 .ai{margin:6px 0 8px;font-size:13px} .air{color:#333}
 .warn{background:#fff6f5;border:1px solid #f3c5c0;border-radius:8px;padding:8px 12px;margin:8px 0;font-size:13px;color:#8a1c12}
 .warn ul{margin:4px 0 0;padding-left:18px}
 .note{background:#fff4e0;color:#7a4500;border-radius:6px;padding:4px 10px;font-size:12px;font-weight:600;
       display:inline-block;margin-bottom:8px}
 .meta{font-size:13px;color:#444;margin-top:4px} .meta a{color:#0b57d0}
 .err{background:#fff6f5;border:1px solid #f3c5c0;border-radius:10px;padding:16px;color:#8a1c12;font-size:15px}
 .foot{font-size:12px;color:#777;margin-top:14px}
</style>"""

# ---- 3. Draw the page ---------------------------------------------------------
def render(question, service, location, patient, ai_on=True):
    understood = ""
    try:
        if question.strip():
            q = understand_question(question)
            understood = (f'<div class="understood"><b>We understood:</b> {E(q["service"])} for {"a child" if q["patient"] == "child" else "an adult"} patient '
                          f'near <b>{E(q["location"] or "?")}</b> &middot; <i>{E(q["why"])}</i><br>'
                          f'<span style="font-size:12px;color:#666">Not right? Clear box 0 and use boxes 1-3 instead.</span></div>')
            if q["service"] == "unknown" or not q["location"]:
                raise ValueError("Couldn't tell the service or the place from the description. "
                                 "Please add them, or use boxes 1-3.")
            service, location, patient = q["service"], q["location"], q["patient"].title()
        res, info = smart_search(service, location, for_child=(patient == "Child"), use_ai=ai_on)
    except ValueError as e:
        return CSS + f'<div class="app">{understood}<div class="err"><b>Couldn\'t run the search.</b><br>{E(str(e))}</div></div>'
    summary = summarize(res, info) if (ai_on and len(res)) else ""
    notes = "".join(f"<li>{E(n)}</li>" for n in info["notes"])
    cards = "".join(card_html(r, ai_on) for _, r in res.iterrows()) if len(res) else \
        '<div class="err">No facility in the dataset has evidence for this service.</div>'
    ai_line = (f" AI check by {E(info['ai_model'])}: it read only the quotes shown on each card and can only move a "
               f"facility down, never up." if ai_on and info.get("ai_model") else "")
    return CSS + f"""
    <div class="app">
      {understood}
      <h2>{E(info["service"])} &middot; {E(patient.lower())} patient &middot; {E(info["mode"])} service</h2>
      <div class="sub">Near <b>{E(info["where"])}</b> &middot; {E(info["count"])}</div>
      {f'<div class="summary">{E(summary)}</div>' if summary else ''}
      <ul class="notes">{notes}</ul>
      <div class="rule"><b>How we rank:</b> {E(info["rule"])}{ai_line}</div>
      {cards}
      <div class="foot">Evidence is quoted from the Virtue Foundation facility records, which were extracted from
      websites and may be out of date or wrong. Treat every result as a lead to confirm by phone, not a guarantee.</div>
    </div>"""

displayHTML(render(dbutils.widgets.get("question"), dbutils.widgets.get("service"), dbutils.widgets.get("location"),
                   dbutils.widgets.get("patient"), ai_on=(dbutils.widgets.get("ai") == "On")))
