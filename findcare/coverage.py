"""
findcare.coverage - the planner view: where in India is trustworthy care missing?

Uses the same tables, evidence ratings, warnings and search radii as the ranking in core.py,
so the dashboard and the "Find care" results always agree. Call core.load(...) first.

    from findcare import core, coverage
    core.load(core.spark_reader(spark))
    coverage.district_access("dialysis")     # one row per district
    coverage.evidence_by_service()           # facilities per service and evidence level
    coverage.warning_counts()                # how many facilities carry each warning
"""
import numpy as np
import pandas as pd

from findcare import core

STATUS_ORDER = ["covered", "moderate_only", "gap"]
STATUS_LABEL = {
    "covered": "Strong evidence within reach",
    "moderate_only": "Only Moderate evidence within reach",
    "gap": "Nothing trustworthy within reach",
}
STATUS_COLOR = {"covered": "#0ca30c", "moderate_only": "#fab219", "gap": "#d03b3b"}
EVIDENCE_ORDER = ["Strong", "Moderate", "Weak", "Specialty tag only"]
EVIDENCE_COLOR = ["#1c5cab", "#5598e7", "#9ec5f4", "#8a8a85"]


def _facilities():
    """One row per facility (EV has one row per facility and service)."""
    return core.EV.drop_duplicates(["name", "best_lat", "best_lon"])


def _flagged(quality_flags, loc_flags):
    """True if the facility carries a warning that is shown to users and penalised in ranking."""
    return any(f in core.MAJOR_FLAGS for f in quality_flags) or len(loc_flags) > 0


def radius_for(service_key):
    return core.MODES[core.SERVICE_MODE[service_key]]["radius_km"]


def district_access(service_key, exclude_flagged=False):
    """For every district: the nearest facility with Strong evidence, the nearest with Strong or Moderate
    evidence, how many such facilities are within the service's search radius, and a status."""
    radius = radius_for(service_key)
    c = core.EV[(core.EV["service"] == service_key)
                & core.EV["evidence_strength"].isin(["Strong", "Moderate"])].dropna(subset=["best_lat", "best_lon"])
    if exclude_flagged and len(c):
        keep = [not _flagged(q, l) for q, l in zip(c["quality_flags"], c["loc_flags"])]
        c = c[keep]
    d = core.DIST.reset_index().dropna(subset=["lat", "lon"]).reset_index(drop=True)

    out = pd.DataFrame({
        "District": d["pin_district"].astype(str).str.title(),
        "State": d["state"].astype(str).str.title(),
        "lat": d["lat"].astype(float),
        "lon": d["lon"].astype(float),
    })
    if len(c) == 0:
        out["nearest_strong_km"] = np.nan
        out["nearest_strong"] = ""
        out["nearest_any_km"] = np.nan
        out["within_radius"] = 0
    else:
        dist = core.km_np(d["lat"].values[:, None], d["lon"].values[:, None],
                          c["best_lat"].values[None, :], c["best_lon"].values[None, :])
        strong = (c["evidence_strength"] == "Strong").values
        names = c["name"].fillna("(unnamed)").values
        out["nearest_any_km"] = dist.min(axis=1)
        out["within_radius"] = (dist <= radius).sum(axis=1)
        if strong.any():
            ds = dist[:, strong]
            idx = ds.argmin(axis=1)
            out["nearest_strong_km"] = ds[np.arange(len(d)), idx]
            out["nearest_strong"] = names[strong][idx]
        else:
            out["nearest_strong_km"] = np.nan
            out["nearest_strong"] = ""

    out["status"] = np.where(out["nearest_strong_km"] <= radius, "covered",
                             np.where(out["nearest_any_km"] <= radius, "moderate_only", "gap"))
    out["status_label"] = out["status"].map(STATUS_LABEL)
    return out


def evidence_by_service():
    """Facilities per service and evidence level, plus facilities that only list a related specialty."""
    ev = core.EV.groupby(["service", "evidence_strength"]).size().rename("facilities").reset_index()
    ev = ev[ev["evidence_strength"].isin(["Strong", "Moderate", "Weak"])]
    parts = [ev.rename(columns={"evidence_strength": "evidence"})]
    if core.TAG is not None and len(core.TAG):
        tag = core.TAG.groupby("service").size().rename("facilities").reset_index()
        tag["evidence"] = "Specialty tag only"
        parts.append(tag)
    out = pd.concat(parts, ignore_index=True)
    out = out[out["service"].isin(core.SERVICE_LABELS)]
    out["Service"] = out["service"].map(core.SERVICE_LABELS)
    return out[["Service", "evidence", "facilities"]]


def warning_counts():
    """How many facilities carry each warning, and whether users see it on the result card."""
    f = _facilities()
    rows = []
    for q, l in zip(f["quality_flags"], f["loc_flags"]):
        for flag in dict.fromkeys(list(q) + list(l)):
            rows.append(flag)
    if not rows:
        return pd.DataFrame(columns=["Warning", "facilities", "share", "Shown to users"])
    s = pd.Series(rows).value_counts().rename_axis("Warning").rename("facilities").reset_index()
    s["share"] = s["facilities"] / len(f)
    loc = set(x for l in f["loc_flags"] for x in l)
    s["Shown to users"] = s["Warning"].map(lambda w: "Yes" if (w in core.MAJOR_FLAGS or w in loc) else
                                            "No (background signal)")
    return s


def headline():
    """Numbers for the stat tiles at the top of the dashboard."""
    f = _facilities()
    flagged = sum(_flagged(q, l) for q, l in zip(f["quality_flags"], f["loc_flags"]))
    return {
        "facilities": len(f),
        "services": int(core.EV["service"].isin(core.SERVICE_LABELS).groupby(core.EV["service"]).any().sum()),
        "districts": int(core.DIST.dropna(subset=["lat", "lon"]).shape[0]),
        "flagged_share": flagged / len(f) if len(f) else 0.0,
    }
