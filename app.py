"""
FindCare India - Databricks App (Streamlit).
A health worker describes a case (or picks a service + place) and gets ranked facilities,
each with the quoted evidence, an AI check, warnings and an explanation of the ranking.
All logic lives in findcare/core.py; this file only draws the screen.
"""
import html
import re

import pandas as pd
import streamlit as st

from findcare import core

st.set_page_config(page_title="FindCare India", page_icon=":hospital:", layout="centered")


@st.cache_resource(show_spinner="Loading facility data...")
def load_data():
    return core.load(core.warehouse_reader())


load_data()
E = html.escape

CSS = """
<style>
 .fc-card{background:#fff;border:1px solid #e2e2e2;border-radius:10px;padding:14px 16px;margin:12px 0;color:#1d1d1f}
 .fc-top{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
 .fc-rank{flex:none;width:30px;height:30px;border-radius:50%;background:#1d1d1f;color:#fff;display:flex;
          align-items:center;justify-content:center;font-weight:600}
 .fc-title{flex:1;min-width:160px} .fc-name{font-size:17px;font-weight:600} .fc-where{color:#666;font-size:13px}
 .fc-badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600}
 .fc-dist{margin:8px 0 4px;font-size:14px;font-weight:500}
 .fc-sec{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#777;margin-top:8px}
 .fc-quotes{margin:6px 0;padding-left:0;font-size:14px;list-style:none} .fc-quotes li{margin:4px 0}
 .fc-num{color:#888;font-size:12px}
 .fc-tag{display:inline-block;font-size:11px;background:#f0f0f0;color:#555;border-radius:4px;padding:1px 6px;margin-right:4px}
 .fc-ai{margin:6px 0;font-size:13px}
 .fc-warn{background:#fff6f5;border:1px solid #f3c5c0;border-radius:8px;padding:8px 12px;margin:8px 0;font-size:13px;color:#8a1c12}
 .fc-warn ul{margin:4px 0 0;padding-left:18px}
 .fc-note{background:#fff4e0;color:#7a4500;border-radius:6px;padding:3px 10px;font-size:12px;font-weight:600;
          display:inline-block;margin-bottom:6px}
 .fc-meta{font-size:13px;color:#444;margin-top:3px} .fc-meta a{color:#0b57d0}
</style>"""
EVIDENCE_COLORS = {"Strong": ("#e6f4ea", "#1e7e34"), "Moderate": ("#fff4e0", "#a15c00"), "Weak": ("#eeeeee", "#555")}
AI_COLORS = {"confirmed": ("#e6f4ea", "#1e7e34"), "likely": ("#e3f4f4", "#0f6e6e"), "unclear": ("#fff4e0", "#a15c00"),
             "unlikely": ("#fdeceb", "#a3221a")}
FIELD_LABEL = {"procedure": "Procedure", "capability": "Capability", "equipment": "Equipment",
               "description": "Description", "specialties": "Specialty tag"}


def badge(text, colors):
    bg, fg = colors
    return f'<span class="fc-badge" style="background:{bg};color:{fg}">{E(text)}</span>'


def card_html(c):
    quotes = []
    for i, q in enumerate(c["quotes"], 1):
        m = re.match(r"\[(\w+)\]\s*(.*)", q, re.S)
        field, text = (m.group(1), m.group(2)) if m else ("", q)
        quotes.append(f'<li><span class="fc-num">[{i}]</span> <span class="fc-tag">{E(FIELD_LABEL.get(field, field))}'
                      f'</span> &ldquo;{E(text)}&rdquo;</li>')
    ai = ""
    if c["ai_verdict"]:
        label = c["ai_verdict"] + ("" if c["ai_fit"] in ("any", "unclear", "") else f" ({c['ai_fit']})")
        ai = (f'<div class="fc-ai">{badge("AI check: " + label, AI_COLORS.get(c["ai_verdict"], EVIDENCE_COLORS["Weak"]))}'
              f' {E(c["ai_reason"])}</div>')
    warns = c["warnings"] + c["cautions"]
    warn = ('<div class="fc-warn"><b>Check before sending a patient:</b><ul>'
            + "".join(f"<li>{E(w)}</li>" for w in warns) + "</ul></div>") if warns else ""
    src = " &middot; ".join(f'<a href="{E(u)}" target="_blank" rel="noopener">'
                            f'{E(re.sub(r"^https?://(www[.])?", "", u)[:50])}</a>' for u in c["sources"]) or "None listed"
    note = f'<div class="fc-note">{E(c["note"])}</div>' if c["note"] else ""
    level = c["evidence_level"]
    return f"""<div class="fc-card">{note}
      <div class="fc-top"><div class="fc-rank">{c['rank']}</div>
        <div class="fc-title"><div class="fc-name">{E(c['facility'])}</div><div class="fc-where">{E(c['where'])}</div></div>
        {badge(c['evidence_label'] + ' evidence', EVIDENCE_COLORS.get(level, EVIDENCE_COLORS['Weak']))}</div>
      <div class="fc-dist">{E(c['distance'])}</div>
      <div class="fc-sec">Why we think they offer this (quoted from the record)</div>
      <ul class="fc-quotes">{''.join(quotes)}</ul>{ai}{warn}
      <div class="fc-meta"><b>How it was ranked:</b> {E(c['how'])}</div>
      <div class="fc-meta"><b>Size (as listed):</b> {E(c['size'])}</div>
      <div class="fc-meta"><b>Sources:</b> {src}</div></div>"""


# ---------------------------------------------------------------- screen
st.title("FindCare India")
st.caption("Find the nearest facility that can really provide the care a patient needs, with the evidence behind "
           "every suggestion.")

mode = st.radio("How do you want to search?", ["Describe the case", "Pick service and place"], horizontal=True)
question, service, location, patient = "", "", "", "adult"
if mode == "Describe the case":
    question = st.text_area("What does the patient need, and where are they?",
                            placeholder="e.g. Woman in labour for 20 hours, baby not coming out, village near Gaya",
                            height=90)
else:
    col1, col2 = st.columns(2)
    service = col1.selectbox("Service needed", list(core.SERVICE_LABELS.values()))
    location = col2.text_input("Where is the patient?", placeholder="City, district, PIN code or lat, lon")
    patient = st.radio("Patient", ["adult", "child"], horizontal=True)
use_ai = st.toggle("AI check (an AI re-reads each facility's quotes)", value=True)

if st.button("Find care", type="primary", use_container_width=True):
    if not (question.strip() or location.strip()):
        st.warning("Please describe the case, or enter where the patient is.")
        st.stop()
    with st.spinner("Searching facilities and checking evidence..."):
        r = core.search(question=question, service=service, location=location, patient=patient, use_ai=use_ai)

    if r.get("understood"):
        u = r["understood"]
        st.info(f"**We understood:** {u['service']} for {'a child' if u['patient'] == 'child' else 'an adult'} "
                f"near **{u['location'] or '?'}** ({u['why']}). Not right? Use *Pick service and place* instead.")
    if not r["ok"]:
        st.error(r["error"])
        st.stop()

    st.subheader(f"{r['service']} near {r['where']}")
    st.caption(f"{r['mode'].title()} service · {r['count']}")
    if r["summary"]:
        st.success(r["summary"])
    for n in r["notes"]:
        st.markdown(f"- {n}")
    with st.expander("How we rank"):
        st.write(f"Ranked distance = {r['rule']}")
        if r["ai_model"]:
            st.write(f"AI check by `{r['ai_model']}`: it reads only the quotes on each card and can only move a "
                     f"facility down, never up.")
        st.write("Evidence points: procedure statement +3, capability or equipment +2, description or specialty "
                 "tag +1. Strong = 5+ points from 2+ fields (or 2+ procedure statements); Moderate = 3+; Weak = 1-2.")

    if r["results"]:
        pts = [{"lat": c["lat"], "lon": c["lon"]} for c in r["results"]]
        st.map(pd.DataFrame(pts), size=200, zoom=7)
        st.markdown(CSS + "".join(card_html(c) for c in r["results"]), unsafe_allow_html=True)
    else:
        st.warning("No facility in the dataset has evidence for this service.")

st.divider()
st.caption("Evidence is quoted from the Virtue Foundation facility records, extracted from websites; it may be out of "
           "date or wrong. Treat every result as a lead to confirm by phone, not a guarantee.")
