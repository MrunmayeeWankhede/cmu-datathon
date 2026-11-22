"""
FindCare India - Databricks App (Streamlit).
A health worker describes a case (or picks a service + place) and gets ranked facilities,
each with the quoted evidence, an AI check, warnings and an explanation of the ranking.
All logic lives in findcare/core.py; this file only draws the screen.
"""
import html
import re

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import streamlit.components.v1  # noqa: F401  (st.components.v1.iframe)

from findcare import core, coverage

st.set_page_config(page_title="FindCare India", page_icon=":hospital:", layout="centered")

# ---------------------------------------------------------------- look and feel (Apple-style, visual only)
LOOK = """
<style>
 html, body, [class*="css"], .stApp, button, input, textarea, select {
   font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Helvetica Neue", Helvetica,
                Arial, sans-serif !important;
   -webkit-font-smoothing: antialiased; }
 .stApp { background: #000; }
 .block-container, [data-testid="stMainBlockContainer"] { max-width: 980px; padding-top: 3rem; }
 h1 { font-size: 48px !important; font-weight: 600 !important; letter-spacing: -0.025em; color: #f5f5f7; }
 h2, h3, h4, h5 { font-weight: 600 !important; letter-spacing: -0.015em; color: #f5f5f7; }
 [data-testid="stCaptionContainer"], .stCaption { color: #98989d !important; }
 .fc-hero-sub { font-size: 21px; line-height: 1.4; color: #98989d; margin: -6px 0 22px; max-width: 760px; }
 /* buttons: pills */
 .stButton > button, .stDownloadButton > button { border-radius: 980px !important; font-weight: 500;
   padding: 0.55rem 1.4rem; border: 1px solid #38383a; background: #1c1c1e; color: #0a84ff; }
 .stButton > button[kind="primary"], [data-testid="baseButton-primary"], [data-testid="stBaseButton-primary"] {
   background: #0a84ff !important; border-color: #0a84ff !important; color: #fff !important; }
 .stButton > button[kind="primary"]:hover, [data-testid="baseButton-primary"]:hover,
 [data-testid="stBaseButton-primary"]:hover { background: #409cff !important; }
 /* inputs */
 [data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"], .stTextArea textarea {
   border-radius: 12px !important; background: #1c1c1e !important; border-color: #38383a !important; }
 /* tabs as a segmented control */
 [data-baseweb="tab-list"], .stTabs [role="tablist"] { background: #1c1c1e !important; border-radius: 10px;
   padding: 3px; gap: 2px !important; width: fit-content; border: none !important; box-shadow: none !important; }
 [data-baseweb="tab"], .stTabs [role="tab"] { border-radius: 8px !important; padding: 6px 18px !important;
   height: auto !important; background: transparent; color: #f5f5f7 !important; border: none !important;
   margin: 0 !important; }
 [data-baseweb="tab"][aria-selected="true"], .stTabs [role="tab"][aria-selected="true"] {
   background: #3a3a3c !important; box-shadow: 0 1px 3px rgba(0,0,0,.4); }
 [data-baseweb="tab-highlight"], [data-baseweb="tab-border"], .stTabs .react-aria-SelectionIndicator {
   display: none !important; }
 .stTabs [role="tab"] p { font-size: 14px !important; font-weight: 500; }
 /* tiles, expanders, alerts, tables: dark rounded cards */
 [data-testid="stMetric"] { background: #1c1c1e; border-radius: 18px; padding: 16px 18px; }
 [data-testid="stMetricValue"] { font-weight: 600; letter-spacing: -0.02em; }
 [data-testid="stExpander"] details, [data-testid="stExpander"] { border-radius: 14px !important;
   border-color: #2c2c2e !important; background: #1c1c1e; }
 [data-testid="stAlert"] { border-radius: 14px !important; }
 [data-testid="stDataFrame"], [data-testid="stDeckGlJsonChart"], [data-testid="stVegaLiteChart"] {
   border-radius: 18px; overflow: hidden; }
 hr { border-color: #2c2c2e !important; }
</style>"""
st.markdown(LOOK, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading facility data...")
def load_data():
    return core.load(core.warehouse_reader())


load_data()
E = html.escape

CSS = """
<style>
 .fc-card{background:#1c1c1e;border:none;border-radius:18px;padding:18px 20px;margin:14px 0;color:#f5f5f7}
 .fc-top{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
 .fc-rank{flex:none;width:30px;height:30px;border-radius:50%;background:#0a84ff;color:#fff;display:flex;
          align-items:center;justify-content:center;font-weight:600}
 .fc-title{flex:1;min-width:160px} .fc-name{font-size:19px;font-weight:600;letter-spacing:-0.01em} .fc-where{color:#98989d;font-size:13px}
 .fc-badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600}
 .fc-dist{margin:8px 0 4px;font-size:14px;font-weight:500}
 .fc-sec{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#8e8e93;margin-top:10px}
 .fc-quotes{margin:6px 0;padding-left:0;font-size:14px;list-style:none} .fc-quotes li{margin:4px 0}
 .fc-num{color:#8e8e93;font-size:12px}
 .fc-tag{display:inline-block;font-size:11px;background:#2c2c2e;color:#c7c7cc;border-radius:6px;padding:1px 6px;margin-right:4px}
 .fc-ai{margin:6px 0;font-size:13px}
 .fc-warn{background:rgba(255,69,58,.12);border:1px solid rgba(255,69,58,.35);border-radius:12px;padding:8px 12px;margin:8px 0;font-size:13px;color:#ff6961}
 .fc-warn ul{margin:4px 0 0;padding-left:18px}
 .fc-note{background:rgba(255,159,10,.16);color:#ffb340;border-radius:980px;padding:3px 10px;font-size:12px;font-weight:600;
          display:inline-block;margin-bottom:6px}
 .fc-meta{font-size:13px;color:#c7c7cc;margin-top:3px} .fc-meta a{color:#0a84ff}
</style>"""
EVIDENCE_COLORS = {"Strong": ("rgba(48,209,88,.16)", "#30d158"), "Moderate": ("rgba(255,159,10,.16)", "#ff9f0a"),
                   "Weak": ("rgba(142,142,147,.2)", "#aeaeb2")}
AI_COLORS = {"confirmed": ("rgba(48,209,88,.16)", "#30d158"), "likely": ("rgba(100,210,255,.16)", "#64d2ff"),
             "unclear": ("rgba(255,159,10,.16)", "#ff9f0a"), "unlikely": ("rgba(255,69,58,.16)", "#ff453a")}
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
st.markdown('<div class="fc-hero-sub">Find the nearest facility that can really provide the care a patient needs, '
            'with the evidence behind every suggestion.</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- maps and dashboard
DARK_MAP = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"   # free dark base map


def _hex_rgb(h, alpha=220):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [alpha]


def _zoom_for(span_km):
    for z, limit in [(11, 4), (10, 10), (9, 25), (8, 60), (7, 130), (6, 280), (5, 600)]:
        if span_km <= limit:
            return z
    return 4


def results_map(results):
    """Numbered result pins, zoomed to fit the results (st.map ignored the zoom setting)."""
    df = pd.DataFrame([{"rank": str(c["rank"]), "facility": c["facility"], "lat": c["lat"], "lon": c["lon"],
                        "distance": c["distance"]} for c in reversed(results)]).dropna(subset=["lat", "lon"])  # #1 drawn last = on top
    if df.empty:
        return
    span = float(core.km_np(df["lat"].min(), df["lon"].min(), df["lat"].max(), df["lon"].max()))
    view = pdk.ViewState(latitude=float(df["lat"].mean()), longitude=float(df["lon"].mean()), zoom=_zoom_for(span))
    pins = pdk.Layer("ScatterplotLayer", df, get_position=["lon", "lat"], get_fill_color=[10, 132, 255, 240],
                     get_line_color=[255, 255, 255, 255], stroked=True, line_width_min_pixels=2,
                     radius_min_pixels=11, radius_max_pixels=11, pickable=True)
    labels = pdk.Layer("TextLayer", df, get_position=["lon", "lat"], get_text="rank", get_size=13,
                       get_color=[255, 255, 255, 255], get_alignment_baseline="'center'")
    st.pydeck_chart(pdk.Deck(layers=[pins, labels], initial_view_state=view, map_style=DARK_MAP,
                             tooltip={"html": "<b>{rank}. {facility}</b><br/>{distance}"}))


@st.cache_data(show_spinner=False)
def _access(service_key, exclude_flagged):
    return coverage.district_access(service_key, exclude_flagged)


@st.cache_data(show_spinner=False)
def _evidence():
    return coverage.evidence_by_service()


@st.cache_data(show_spinner=False)
def _warnings():
    return coverage.warning_counts()


@st.cache_data(show_spinner=False)
def _headline():
    return coverage.headline()


def _legend(counts):
    items = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:18px">'
        f'<span style="width:12px;height:12px;border-radius:50%;background:{coverage.STATUS_COLOR[s]};'
        f'box-shadow:0 0 0 2px #000, 0 0 0 3px rgba(255,255,255,.15)"></span>{coverage.STATUS_LABEL[s]} '
        f'<b>{counts.get(s, 0)}</b></span>' for s in coverage.STATUS_ORDER)
    st.markdown(f'<div style="font-size:14px;margin:4px 0 8px">{items}</div>', unsafe_allow_html=True)


TEAM_DASHBOARD = ("https://dbc-f2aceaa2-f39e.cloud.databricks.com/dashboardsv3/01f1e6837b141b5aa704224faeaa68b9/published?o=7474655528822479")


def coverage_tab():
    st.markdown("#### Where is trustworthy care missing?")
    st.caption("For district planners and NGOs: the same evidence ratings, warnings and search radius as "
               "*Find care*, applied to every district in India at once.")
    st.link_button("Open the team's full planner dashboard ↗", TEAM_DASHBOARD, type="primary")
    return
    with st.expander("Show the team dashboard here"):
        st.components.v1.iframe(TEAM_DASHBOARD.replace("/dashboardsv3/", "/embed/dashboardsv3/"), height=900,
                                scrolling=True)
        st.caption("If this area stays empty, use the button above; embedding has to be allowed in workspace settings.")
    h = _headline()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Facilities", f"{h['facilities']:,}")
    k2.metric("Services checked", h["services"])
    k3.metric("Districts mapped", f"{h['districts']:,}")
    k4.metric("Facilities with a warning", f"{h['flagged_share']:.0%}")

    c1, c2 = st.columns([3, 2])
    label = c1.selectbox("Service", list(core.SERVICE_LABELS.values()), key="cov_service")
    svc = core.resolve_service(label)
    exclude = c2.toggle("Ignore facilities with warnings", value=False, key="cov_exclude",
                        help="Only count facilities whose records have no warning shown to users.")
    radius = coverage.radius_for(svc)
    mode = core.SERVICE_MODE[svc]
    a = _access(svc, exclude)
    counts = a["status"].value_counts().to_dict()
    n = len(a)

    gap = counts.get("gap", 0)
    st.markdown(f"**{gap} of {n} districts ({gap / n:.0%})** have no facility with Strong or Moderate evidence for "
                f"{label} within {radius} km ({mode} services use a {radius} km search radius).")
    _legend(counts)

    a = a.assign(color=a["status"].map(lambda s: _hex_rgb(coverage.STATUS_COLOR[s])),
                 near_txt=np.where(a["nearest_strong_km"].notna(),
                                   a["nearest_strong"].astype(str) + " (" +
                                   a["nearest_strong_km"].round(0).astype("Int64").astype(str) + " km)",
                                   "none in the dataset"))
    layer = pdk.Layer("ScatterplotLayer", a, get_position=["lon", "lat"], get_fill_color="color",
                      get_line_color=[0, 0, 0, 255], stroked=True, line_width_min_pixels=1,
                      radius_min_pixels=4, radius_max_pixels=9, get_radius=9000, pickable=True)
    st.pydeck_chart(pdk.Deck(layers=[layer], map_style=DARK_MAP,
                             initial_view_state=pdk.ViewState(latitude=22.6, longitude=80.0, zoom=3.6),
                             tooltip={"html": "<b>{District}</b>, {State}<br/>{status_label}<br/>"
                                              "Nearest Strong: {near_txt}<br/>"
                                              "Strong/Moderate within reach: {within_radius}"}))
    st.caption("Each dot is a district, placed at the centre of its PIN codes. Distances are straight-line.")

    st.markdown("##### Biggest gaps")
    worst = (a.sort_values("nearest_strong_km", ascending=False, na_position="first")
             .loc[:, ["District", "State", "status_label", "nearest_strong", "nearest_strong_km", "within_radius"]]
             .rename(columns={"status_label": "Status", "nearest_strong": "Nearest Strong facility",
                              "nearest_strong_km": "Distance (km)", "within_radius": "Strong/Moderate within reach"}))
    st.dataframe(worst.head(15), hide_index=True, use_container_width=True,
                 column_config={"Distance (km)": st.column_config.NumberColumn(format="%d")})
    st.download_button(f"Download all {n} districts (CSV)", worst.to_csv(index=False).encode(),
                       file_name=f"findcare_coverage_{svc}.csv", mime="text/csv")

    st.divider()
    st.markdown("##### How strong is the evidence for each service?")
    st.caption("Facilities per service by evidence level. *Specialty tag only* means the record lists a related "
               "specialty but says nothing specific about the service, so it is never recommended as if confirmed.")
    ev = _evidence()
    order = ev.groupby("Service")["facilities"].sum().sort_values(ascending=False).index.tolist()
    bars = alt.Chart(ev).mark_bar(cornerRadiusEnd=4, stroke="#000000", strokeWidth=2).encode(
        y=alt.Y("Service:N", sort=order, title=None, axis=alt.Axis(labelLimit=220, labelOverlap=False)),
        x=alt.X("facilities:Q", title="Facilities", stack="zero"),
        color=alt.Color("evidence:N", title="Evidence",
                        scale=alt.Scale(domain=coverage.EVIDENCE_ORDER, range=coverage.EVIDENCE_COLOR),
                        legend=alt.Legend(orient="top")),
        order=alt.Order("rank:Q"),
        tooltip=["Service", alt.Tooltip("evidence:N", title="Evidence"),
                 alt.Tooltip("facilities:Q", title="Facilities", format=",")],
    ).transform_calculate(
        rank="indexof(['Strong','Moderate','Weak','Specialty tag only'], datum.evidence)"
    ).properties(height=44 * len(order))
    st.altair_chart(bars, use_container_width=True)
    with st.expander("Show as a table"):
        st.dataframe(ev.pivot_table(index="Service", columns="evidence", values="facilities", fill_value=0)
                     .reindex(columns=[c for c in coverage.EVIDENCE_ORDER if c in set(ev["evidence"])])
                     .reindex(order), use_container_width=True)
    return
    st.divider()
    st.markdown("##### How much can we trust the records?")
    st.caption("Facilities carrying each warning. Warnings marked *shown to users* appear on result cards and push "
               "a facility down the ranking; background signals fire too often to say much about one facility.")
    w = _warnings()
    if len(w):
        w = w.assign(label=w["Warning"].str.slice(0, 70))
        wbars = alt.Chart(w).mark_bar(cornerRadiusEnd=4, color="#3987e5").encode(
            y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=420, labelOverlap=False)),
            x=alt.X("facilities:Q", title="Facilities"),
            opacity=alt.condition(alt.datum["Shown to users"] == "Yes", alt.value(1.0), alt.value(0.45)),
            tooltip=[alt.Tooltip("Warning:N"), alt.Tooltip("facilities:Q", title="Facilities", format=","),
                     alt.Tooltip("share:Q", title="Share of facilities", format=".0%"), "Shown to users"],
        ).properties(height=40 * len(w))
        st.altair_chart(wbars, use_container_width=True)
        st.caption("Solid bars are shown to users; faded bars are background signals.")
        with st.expander("Show as a table"):
            st.dataframe(w.assign(share=w["share"] * 100)[["Warning", "facilities", "share", "Shown to users"]], hide_index=True,
                         use_container_width=True,
                         column_config={"share": st.column_config.NumberColumn("Share", format="%.1f%%")})

tab_find, tab_cov = st.tabs(["Find care", "Coverage dashboard"])

# The dashboard is drawn first so a st.stop() in the search tab can't hide it.
with tab_cov:
    coverage_tab()

with tab_find:
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
            results_map(r["results"])
            cards = CSS + "".join(card_html(c) for c in r["results"])
            st.markdown(re.sub(r"\n\s*", " ", cards), unsafe_allow_html=True)
        else:
            st.warning("No facility in the dataset has evidence for this service.")

st.divider()
st.caption("Evidence is quoted from the Virtue Foundation facility records, extracted from websites; it may be out of "
           "date or wrong. Treat every result as a lead to confirm by phone, not a guarantee.")