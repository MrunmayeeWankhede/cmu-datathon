"""
findcare.baseline - how a naive search compares with FindCare.

The naive search is what a quick keyword lookup does: take every facility whose record mentions the service
(even only as a specialty tag) and pick the nearest one by its listed coordinates. FindCare's pick is #1 from
core.recommend() (rules only, no AI, so the comparison is fast and repeatable).

Both picks are judged by the same yardstick, `problems()`, which lists what a health worker would want to know
before sending a patient there.

    from findcare import core, baseline
    core.load(core.spark_reader(spark))
    runs = baseline.compare()                 # one row per (service, city)
    baseline.summary(runs)                    # headline numbers
"""
import re

import numpy as np
import pandas as pd

from findcare import core

PROBLEM_TYPES = [
    "Only a specialty tag, no statement about the service",
    "Weak evidence only",
    "Evidence mentions children only",
    "Specialist hospital, unlikely to offer it",
    "Record looks merged or has impossible numbers",
    "Location in the record is inconsistent",
]


def problems(row, svc, for_child=False):
    """What is wrong with sending a patient to this facility for this service. Empty list = nothing found."""
    out = []
    if row.get("tag_only"):
        out.append(PROBLEM_TYPES[0])
    elif row.get("evidence_strength") == "Weak":
        out.append(PROBLEM_TYPES[1])
    ev = list(row.get("evidence") or [])
    if svc != "nicu" and not for_child and core.children_only(row.get("name"), ev):
        out.append(PROBLEM_TYPES[2])
    if core.specialist_note(row.get("name"), svc):
        out.append(PROBLEM_TYPES[3])
    if any(f in core.MAJOR_FLAGS for f in (row.get("quality_flags") or [])):
        out.append(PROBLEM_TYPES[4])
    if row.get("loc_flags"):
        out.append(PROBLEM_TYPES[5])
    return out


def _pool(svc):
    """Everything a keyword search would match: evidence rows of any strength plus specialty-tag-only rows."""
    ev = core.EV[core.EV["service"] == svc].dropna(subset=["best_lat", "best_lon"]).copy()
    ev["tag_only"] = False
    tag = core.TAG[core.TAG["service"] == svc].dropna(subset=["best_lat", "best_lon"]).copy() \
        if core.TAG is not None else pd.DataFrame()
    if len(tag):
        # borrow the record's flags from any evidence row of the same facility (another service), if there is one
        flags = (core.EV.drop_duplicates(["name", "best_lat", "best_lon"])
                 .set_index(["name", "best_lat", "best_lon"])[["quality_flags", "loc_flags"]])
        tag = tag.join(flags, on=["name", "best_lat", "best_lon"])
        tag["quality_flags"] = tag["quality_flags"].map(lambda v: v if isinstance(v, list) else [])
        tag["loc_flags"] = tag["loc_flags"].map(lambda v: v if isinstance(v, list) else [])
        tag["evidence"] = [[] for _ in range(len(tag))]
        tag["evidence_strength"] = "Specialty tag only"
        tag["tag_only"] = True
        tag = tag[~tag.set_index(["name"]).index.isin(ev["name"])]     # facility already in the evidence pool
    cols = ["name", "address_city", "best_lat", "best_lon", "evidence_strength", "evidence",
            "quality_flags", "loc_flags", "tag_only"]
    return pd.concat([ev[cols], tag[cols]] if len(tag) else [ev[cols]], ignore_index=True)


def top_cities(n=50):
    """The n cities with the most listed facilities (the places most searches come from)."""
    c = core.CITY.sort_values("n", ascending=False).head(n)
    return [(str(r.city), str(r.state), float(r.lat), float(r.lon)) for r in c.itertuples()]


def compare(cities=None, services=None):
    """Run the naive search and FindCare for every service in every city. One row per pair."""
    rows = []
    for svc in services or list(core.SERVICE_LABELS):
        pool = _pool(svc)
        if pool.empty:
            continue
        for city, state, lat, lon in cities or top_cities():
            d = core.km_np(lat, lon, pool["best_lat"].values, pool["best_lon"].values)
            naive = pool.iloc[int(np.argmin(d))].to_dict()
            naive_km = float(d.min())
            try:
                ranked, _ = core.recommend(svc, f"{lat}, {lon}", top_n=1)
            except ValueError:
                continue
            if ranked.empty:
                continue
            fc = ranked.iloc[0].to_dict()
            fc["tag_only"] = False
            n_p, f_p = problems(naive, svc), problems(fc, svc)
            rows.append({
                "Service": core.SERVICE_LABELS[svc], "City": city, "State": state,
                "Naive pick": naive["name"], "Naive km": round(naive_km, 1),
                "Naive evidence": naive["evidence_strength"], "Naive problems": n_p,
                "FindCare pick": fc["name"], "FindCare km": round(float(fc["distance_km"]), 1),
                "FindCare evidence": fc["evidence_strength"], "FindCare problems": f_p,
                "Same pick": naive["name"] == fc["name"],
            })
    out = pd.DataFrame(rows)
    if len(out):
        out["Naive has problem"] = out["Naive problems"].map(bool)
        out["FindCare has problem"] = out["FindCare problems"].map(bool)
    return out


def summary(runs):
    """Headline numbers for a slide."""
    n = len(runs)
    changed = runs[~runs["Same pick"]]
    fixed = runs[runs["Naive has problem"] & ~runs["FindCare has problem"]]
    return {
        "searches": n,
        "naive_problem_share": runs["Naive has problem"].mean(),
        "findcare_problem_share": runs["FindCare has problem"].mean(),
        "findcare_changed_pick_share": len(changed) / n if n else 0.0,
        "fixed": len(fixed),
        "median_extra_km_when_fixed": float((fixed["FindCare km"] - fixed["Naive km"]).median()) if len(fixed) else 0.0,
    }


def headline(runs):
    s = summary(runs)
    return (f"Across {s['searches']} searches ({runs['Service'].nunique()} services x the {runs['City'].nunique()} cities with the most "
            f"facilities), the nearest keyword match had a problem a health worker should know about "
            f"{s['naive_problem_share']:.0%} of the time. FindCare's #1 had one {s['findcare_problem_share']:.0%} "
            f"of the time, at a median of {s['median_extra_km_when_fixed']:.0f} km further when it avoided one.")


def by_service(runs):
    t = (runs.groupby("Service")[["Naive has problem", "FindCare has problem"]].mean() * 100).round(0)
    t.columns = ["Naive search: % with a problem", "FindCare: % with a problem"]
    return t.sort_values("Naive search: % with a problem", ascending=False).reset_index()


def problem_counts(runs):
    rows = []
    for p in PROBLEM_TYPES:
        rows.append({"Problem": p,
                     "Naive picks": int(runs["Naive problems"].map(lambda l: p in l).sum()),
                     "FindCare picks": int(runs["FindCare problems"].map(lambda l: p in l).sum())})
    return pd.DataFrame(rows)


def examples(runs, k=10):
    """Searches where the naive pick had a problem and FindCare's did not, worst first: good slide material."""
    ex = runs[runs["Naive has problem"] & ~runs["FindCare has problem"]].copy()
    ex["n"] = ex["Naive problems"].map(len)
    ex = ex.sort_values(["n", "Naive km"], ascending=[False, True]).head(k)
    ex["Why the naive pick is risky"] = ex["Naive problems"].map("; ".join)
    return ex[["Service", "City", "Naive pick", "Naive km", "Why the naive pick is risky",
               "FindCare pick", "FindCare km", "FindCare evidence"]]


def for_tables(runs):
    """The results in shapes a Databricks dashboard can chart: (all searches, per service, per problem type).
    Lists become text and column names become snake_case, which Delta tables need."""
    def clean(df):
        df = df.copy()
        df.columns = [re.sub(r"[^0-9a-zA-Z]+", "_", c).strip("_").lower() for c in df.columns]
        return df
    all_runs = runs.copy()
    for c in ["Naive problems", "FindCare problems"]:
        all_runs[c] = all_runs[c].map("; ".join)
    per_service = by_service(runs)
    per_service.columns = ["Service", "Naive pct with problem", "FindCare pct with problem"]
    return clean(all_runs), clean(per_service), clean(problem_counts(runs))
