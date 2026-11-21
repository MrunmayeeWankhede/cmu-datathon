# Databricks notebook source
# ============================================================
# STEP 3: Location + ranking
# Builds:  workspace.default.pin_geo            (PIN code -> centre point, district, state)
#          workspace.default.facility_location  (best location per facility + location red flags)
# Defines: recommend(service, location)  -> ranked facilities with WHY / HOW / warnings
# ============================================================
import re, difflib
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

BASE    = "databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset"
PINS    = f"{BASE}.india_post_pincode_directory"
CLEAN   = "workspace.default.facilities_clean"
EVID    = "workspace.default.facility_service_evidence"
PIN_GEO = "workspace.default.pin_geo"
FAC_LOC = "workspace.default.facility_location"

# ---- RANKING KNOBS: tune these. Together they ARE the answer to "how did you decide?" ----
# Emergency: every minute counts, so closeness matters most -- but only places with a real claim (Strong/Moderate).
# Planned:   the trip can be organised, so stronger evidence is worth more distance.
SERVICE_MODE = {"emergency_c_section": "emergency", "trauma_emergency": "emergency", "burn_care": "emergency",
                "nicu": "emergency", "icu": "emergency",
                "dialysis": "planned", "mri": "planned", "ct_scan": "planned", "cardiac_surgery": "planned"}
MODES = {
    #            search radius   extra km if evidence is...                          extra km per data warning
    "emergency": dict(radius_km=50,  penalty={"Strong": 0, "Moderate": 10, "Weak": None}, per_warning_km=10),
    "planned":   dict(radius_km=100, penalty={"Strong": 0, "Moderate": 30, "Weak": 75},   per_warning_km=25),
}
WEAK_LAST_RESORT_KM = 75     # in emergency mode, Weak is only shown if nothing better is nearby, with this penalty
ROAD_FACTOR         = 1.4    # roads are ~1.4x longer than a straight line (rough estimate)
ROAD_SPEED_KMH      = 40     # rough average road speed for the travel-time estimate
PIN_MISMATCH_KM     = 50     # coordinates this far from their PIN area get flagged
CITY_MISMATCH_KM    = 50     # coordinates this far from the centre of their stated city get flagged

# Step 1 flags serious enough to show users. The others fire on most records (see report) and are ignored here.
MAJOR_FLAGS = {
    "No coordinates",
    "Coordinates fall outside India",
    "Name suggests a single-specialty clinic but record claims many specialties (possible merged record)",
    "Unusually many claims for one facility (possible merged record)",
    "Bed count and doctor count are implausible together",
    "Some source URLs are unrelated pages (e.g. login pages)",
}

# ---- Spark helpers ----------------------------------------------
def km_between(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two points (haversine)."""
    dlat, dlon = F.radians(lat2 - lat1), F.radians(lon2 - lon1)
    a = F.pow(F.sin(dlat / 2), 2) + F.cos(F.radians(lat1)) * F.cos(F.radians(lat2)) * F.pow(F.sin(dlon / 2), 2)
    return F.lit(2 * 6371.0) * F.asin(F.sqrt(a))

def in_india(lat, lon):
    return lat.isNotNull() & lon.isNotNull() & lat.between(6, 37.5) & lon.between(68, 97.5)

STATE_ALIASES = {"orissa": "odisha", "pondicherry": "puducherry", "nctofdelhi": "delhi", "newdelhi": "delhi",
                 "uttaranchal": "uttarakhand", "andamanandnicobar": "andamanandnicobarislands"}
def norm_state(c):
    s = F.lower(F.coalesce(c, F.lit("")))
    s = F.regexp_replace(s, "&", "and")
    s = F.regexp_replace(s, "[^a-z]", "")
    s = F.regexp_replace(s, "^the", "")
    for k, v in STATE_ALIASES.items():
        s = F.when(s == k, F.lit(v)).otherwise(s)
    return s

# ---- 1. PIN code -> centre point ---------------------------------
offices = (spark.table(PINS)
           .select(F.regexp_extract(F.col("pincode").cast("string"), r"(\d{6})", 1).alias("pin"),
                   F.expr("try_cast(latitude as double)").alias("lat"),
                   F.expr("try_cast(longitude as double)").alias("lon"),
                   F.regexp_replace(F.initcap(F.trim("officename")), r"(?i)\s+[bsh]\.?o\.?$", "").alias("office"),
                   F.initcap(F.trim("district")).alias("district"),
                   F.initcap(F.trim("statename")).alias("state"))
           .filter(F.col("pin") != "")
           .withColumn("lat", F.when(in_india(F.col("lat"), F.col("lon")), F.col("lat")))
           .withColumn("lon", F.when(F.col("lat").isNotNull(), F.col("lon"))))

pin_geo = (offices.groupBy("pin")
           .agg(F.percentile_approx("lat", 0.5).alias("pin_lat"),
                F.percentile_approx("lon", 0.5).alias("pin_lon"),
                F.first("district", ignorenulls=True).alias("pin_district"),
                F.first("state", ignorenulls=True).alias("pin_state")))
pin_geo.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(PIN_GEO)
pin_geo = spark.table(PIN_GEO)
print(f"PIN codes with a location: {pin_geo.filter(F.col('pin_lat').isNotNull()).count()} of {pin_geo.count()}")

# ---- 2. Best location per facility + location red flags -------------
fl = (spark.table(CLEAN)
      .select("unique_id", "pin", "latitude", "longitude", "address_city", "address_stateOrRegion")
      .join(pin_geo, "pin", "left")
      .withColumn("coords_ok", in_india(F.col("latitude"), F.col("longitude")))
      .withColumn("dist_to_pin_km", F.when(F.col("coords_ok") & F.col("pin_lat").isNotNull(),
                  km_between(F.col("latitude"), F.col("longitude"), F.col("pin_lat"), F.col("pin_lon"))))
      .withColumn("fs", norm_state(F.col("address_stateOrRegion")))
      .withColumn("ps", norm_state(F.col("pin_state")))
      .withColumn("state_mismatch", (F.col("fs") != "") & (F.col("ps") != "")
                  & ~F.expr("instr(fs, ps) > 0 OR instr(ps, fs) > 0")))

fl = (fl
      .withColumn("loc_flags", F.filter(F.array(
          F.when(F.col("dist_to_pin_km") > PIN_MISMATCH_KM,
                 F.concat(F.lit("Listed coordinates are "), F.round("dist_to_pin_km").cast("int").cast("string"),
                          F.lit(" km from its PIN code area ("), F.col("pin"), F.lit(", "),
                          F.coalesce(F.col("pin_district"), F.lit("unknown district")), F.lit(")"))),
          F.when(F.col("state_mismatch"),
                 F.concat(F.lit("PIN code "), F.col("pin"), F.lit(" is in "), F.col("pin_state"),
                          F.lit(" but the address says "), F.col("address_stateOrRegion")))),
          lambda x: x.isNotNull()))
      .withColumn("best_lat", F.when(F.col("coords_ok"), F.col("latitude")).otherwise(F.col("pin_lat")))
      .withColumn("best_lon", F.when(F.col("coords_ok"), F.col("longitude")).otherwise(F.col("pin_lon")))
      .withColumn("location_source", F.when(F.col("coords_ok"), F.lit("listed coordinates"))
                                       .when(F.col("pin_lat").isNotNull(), F.lit("approximate (PIN code area)")))
      .select("unique_id", "pin_district", "pin_state", "dist_to_pin_km", "loc_flags",
              "best_lat", "best_lon", "location_source", "pin_lat", "pin_lon"))
fl.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(FAC_LOC)
fl = spark.table(FAC_LOC)

print("\nLocation checks:")
display(fl.select(F.explode("loc_flags").alias("f"))
          .select(F.regexp_replace("f", r"\d+|\(.*\)| is in .*", "…").alias("location flag"))
          .groupBy("location flag").count())

# ---- 3. Lookup tables for typed locations (small -> pandas) ---------
def norm_name(s):
    s = re.sub(r"\b(district|city|dist\.?)\b", "", str(s).lower())
    s = re.sub(r"\b[bsh]\.?o\.?$", "", s.strip())          # 'Kothimir B.O' -> 'kothimir'
    return re.sub(r"[^a-z]", "", s)

trusted = (spark.table(CLEAN).select("unique_id", "address_city", "address_stateOrRegion")
           .join(fl, "unique_id")
           .filter((F.col("location_source") == "listed coordinates")
                   & (F.coalesce(F.col("dist_to_pin_km"), F.lit(0.0)) <= PIN_MISMATCH_KM)))
CITY_PD = (trusted.filter(F.col("address_city").isNotNull())
           .groupBy(F.lower(F.trim("address_city")).alias("k"))
           .agg(F.first("address_city").alias("city"), F.first("address_stateOrRegion", ignorenulls=True).alias("state"),
                F.percentile_approx("best_lat", 0.5).alias("lat"), F.percentile_approx("best_lon", 0.5).alias("lon"),
                F.count("*").alias("n")).toPandas())
CITY_PD["k"] = CITY_PD["k"].map(norm_name)
CITY_PD = CITY_PD.sort_values("n", ascending=False).drop_duplicates("k").set_index("k")

PIN_PD = pin_geo.filter(F.col("pin_lat").isNotNull()).toPandas().set_index("pin")
DIST_PD = (pin_geo.filter(F.col("pin_lat").isNotNull() & F.col("pin_district").isNotNull())
           .groupBy("pin_district").agg(F.first("pin_state").alias("state"),
                F.percentile_approx("pin_lat", 0.5).alias("lat"), F.percentile_approx("pin_lon", 0.5).alias("lon"))
           .toPandas())
DIST_PD["k"] = DIST_PD["pin_district"].map(norm_name)
DIST_PD = DIST_PD.drop_duplicates("k").set_index("k")
OFFICE_PD = (offices.filter(F.col("lat").isNotNull())
             .select("office", "pin", "district", "state", "lat", "lon").toPandas())
OFFICE_PD["k"] = OFFICE_PD["office"].map(norm_name)
OFFICE_PD = OFFICE_PD.drop_duplicates("k").set_index("k")

def km_np(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))

# ---- 4. Evidence table joined with locations (pandas, for fast queries) ----
EV_PD = (spark.table(EVID)
         .filter(F.col("evidence_strength").isin("Strong", "Moderate", "Weak")
                 & (F.col("points_breakdown") != "specialties (+1)"))    # a bare specialty tag is not evidence
         .join(fl.select("unique_id", "loc_flags", "best_lat", "best_lon", "location_source", "pin_lat", "pin_lon"),
               "unique_id")
         .filter(F.col("best_lat").isNotNull())
         .select("unique_id", "service", "name", "address_city", "address_stateOrRegion", "evidence_strength",
                 "evidence_points", "evidence", "caution_mentions", "quality_flags", "loc_flags", "url_list",
                 "best_lat", "best_lon", "location_source", "beds", "doctors", "pin_lat", "pin_lon")
         .toPandas())
for c in ["evidence", "caution_mentions", "quality_flags", "loc_flags", "url_list"]:
    EV_PD[c] = EV_PD[c].map(lambda v: [x for x in v if x] if v is not None else [])   # drop empty entries
# Coordinates far from the centre of the facility's own stated city (e.g. "Kanpur" but located in Lucknow)
def _city_flag(city, lat, lon):
    k = norm_name(city) if city else ""
    if k in CITY_PD.index and CITY_PD.loc[k, "n"] >= 3:
        d = float(km_np(lat, lon, CITY_PD.loc[k, "lat"], CITY_PD.loc[k, "lon"]))
        if d > CITY_MISMATCH_KM:
            return [f"Listed coordinates are {d:.0f} km from the centre of {city}, its stated city"]
    return []
EV_PD["loc_flags"] = [l + _city_flag(c, a, o) for l, c, a, o in
                      zip(EV_PD["loc_flags"], EV_PD["address_city"], EV_PD["best_lat"], EV_PD["best_lon"])]

# Facilities that only carry a related specialty tag (not evidence, but worth a phone call when nothing else is near)
TAG_PD = (spark.table(EVID).filter(F.col("points_breakdown") == "specialties (+1)")
          .join(fl.select("unique_id", "best_lat", "best_lon"), "unique_id")
          .filter(F.col("best_lat").isNotNull())
          .select("service", "name", "address_city", "best_lat", "best_lon").toPandas())

# Show the most convincing quotes first
FIELD_ORDER = {"[procedure]": 0, "[capability]": 1, "[equipment]": 2, "[description]": 3, "[specialties]": 4}
EV_PD["evidence"] = EV_PD["evidence"].map(lambda ev: sorted(ev, key=lambda e: FIELD_ORDER.get(e.split(" ")[0], 9)))
# Two or more separate procedure statements are as convincing as mentions in two fields -> Strong
n_proc = EV_PD["evidence"].map(lambda ev: sum(e.startswith("[procedure]") for e in ev))
EV_PD["upgraded"] = (EV_PD["evidence_strength"] == "Moderate") & (n_proc >= 2)
EV_PD.loc[EV_PD["upgraded"], "evidence_strength"] = "Strong"
print(f"\nFacility-service pairs available for recommendations: {len(EV_PD)} "
      f"({int(EV_PD['upgraded'].sum())} upgraded to Strong for 2+ procedure statements)")

# ---- 5. Understanding what the user typed ---------------------------
SERVICE_ALIASES = {
    "dialysis": "dialysis", "kidney": "dialysis", "hemodialysis": "dialysis", "haemodialysis": "dialysis",
    "c-section": "emergency_c_section", "c section": "emergency_c_section", "csection": "emergency_c_section",
    "caesarean": "emergency_c_section", "cesarean": "emergency_c_section", "emergency c-section": "emergency_c_section",
    "high-risk delivery": "emergency_c_section", "high risk delivery": "emergency_c_section", "obstetric": "emergency_c_section",
    "nicu": "nicu", "newborn": "nicu", "neonatal": "nicu", "sncu": "nicu",
    "cardiac surgery": "cardiac_surgery", "heart surgery": "cardiac_surgery", "bypass": "cardiac_surgery",
    "mri": "mri", "ct": "ct_scan", "ct scan": "ct_scan", "cat scan": "ct_scan",
    "burn": "burn_care", "burns": "burn_care", "burn care": "burn_care",
    "trauma": "trauma_emergency", "emergency": "trauma_emergency", "accident": "trauma_emergency",
    "icu": "icu", "intensive care": "icu", "critical care": "icu",
}
SERVICE_LABELS = {"dialysis": "Dialysis", "emergency_c_section": "Emergency C-section", "nicu": "NICU",
                  "cardiac_surgery": "Cardiac surgery", "mri": "MRI", "ct_scan": "CT scan", "burn_care": "Burn care",
                  "trauma_emergency": "Trauma / emergency care", "icu": "ICU"}

def resolve_service(text):
    t = str(text).strip().lower().replace("_", " ")
    if t.replace(" ", "_") in SERVICE_LABELS:
        return t.replace(" ", "_")
    if t in SERVICE_ALIASES:
        return SERVICE_ALIASES[t]
    for alias in sorted(SERVICE_ALIASES, key=len, reverse=True):
        if alias in t:
            return SERVICE_ALIASES[alias]
    raise ValueError(f"Unknown service '{text}'. Try one of: {', '.join(SERVICE_LABELS.values())}")

def locate(text):
    """Turn coordinates / PIN / city / district / post-office name into (lat, lon, description)."""
    t = str(text).strip()
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", t)
    if m:
        return float(m.group(1)), float(m.group(2)), f"coordinates {t}"
    if re.fullmatch(r"\d{6}", t):
        if t in PIN_PD.index:
            r = PIN_PD.loc[t]
            return r.pin_lat, r.pin_lon, f"PIN {t} ({r.pin_district}, {r.pin_state})"
        raise ValueError(f"PIN code {t} was not found in the India Post directory.")
    k = norm_name(t)
    if k in CITY_PD.index:
        r = CITY_PD.loc[k]
        return r.lat, r.lon, f"{r.city}, {r.state} (centre of {int(r.n)} listed facilit{'y' if int(r.n) == 1 else 'ies'})"
    if k in DIST_PD.index:
        r = DIST_PD.loc[k]
        return r.lat, r.lon, f"{r.pin_district} district, {r.state} (centre of its PIN codes)"
    if k in OFFICE_PD.index:
        r = OFFICE_PD.loc[k]
        return r.lat, r.lon, f"{r.office} post office area, {r.district}, {r.state} (PIN {r.pin})"
    names = list(CITY_PD.index) + list(DIST_PD.index)
    close = list(dict.fromkeys(difflib.get_close_matches(k, names, n=5, cutoff=0.75)))[:3]
    hint = f" Did you mean: {', '.join(close)}?" if close else " Try a district name or a 6-digit PIN code."
    raise ValueError(f"Couldn't find '{text}'.{hint}")

JUNK_URL   = re.compile(r"(?i)(login|signin|sign-in|creditcard|checkout|cart)")
CHILD_TEXT = re.compile(r"(?i)(p(a)?ediatric|\bchild|\bneonat|\binfant|\bnewborn)")
CHILD_NAME = re.compile(r"(?i)\b(child|children'?s?|p(a)?ediatric\w*|kids|shishu)\b")
CHILD_ONLY_NOTE = "Evidence mentions children only - may not treat adults"
# Specialist clinics (by name) that are unlikely to offer the services we cover
SPECIALIST_NAME = re.compile(r"(?i)\b(eye|netra|nethralaya|ophthalm\w*|dental|dentist|teeth|ayurved\w*|homoeo\w*|"
                             r"homeo\w*|physiotherap\w*|skin|derma\w*|ivf|fertility)\b")
SPECIALIST_OK = {"ivf": {"emergency_c_section", "nicu"}, "fertility": {"emergency_c_section", "nicu"}}

def specialist_note(name, svc):
    m = SPECIALIST_NAME.search(name or "")
    if not m:
        return []
    word = m.group(1).lower()
    if svc in SPECIALIST_OK.get(word, set()):
        return []
    return [f"Name suggests a specialist clinic ('{m.group(1)}') - unlikely to offer {SERVICE_LABELS[svc]}"]

def children_only(name, evidence):
    return bool(CHILD_NAME.search(name or "")) or (len(evidence) > 0 and all(CHILD_TEXT.search(e) for e in evidence))

def alt_distance(r, lat, lon):
    """If a facility's location is in doubt, the distance to the OTHER place its record points to."""
    alts = []
    if any("from its PIN code area" in f for f in r.loc_flags) and pd.notna(r.pin_lat):
        alts.append(float(km_np(lat, lon, r.pin_lat, r.pin_lon)))
    if any("its stated city" in f for f in r.loc_flags):
        k = norm_name(r.address_city)
        if k in CITY_PD.index:
            alts.append(float(km_np(lat, lon, CITY_PD.loc[k, "lat"], CITY_PD.loc[k, "lon"])))
    return max(alts) if alts else np.nan

def road_time(km_straight):
    mins = km_straight * ROAD_FACTOR / ROAD_SPEED_KMH * 60
    mins = max(5, int(round(mins / 5.0)) * 5)
    return f"~{mins} min" if mins < 60 else f"~{mins // 60} h {mins % 60:02d} min"

# ---- 6. The recommender ---------------------------------------------
def recommend(service, location, top_n=5, for_child=False, radius_km=None, show=True):
    """service: e.g. 'dialysis', 'c-section', 'NICU'.  location: city, district, PIN, post office or 'lat, lon'.
    for_child=True if the patient is a child (children's hospitals are then not marked down)."""
    svc = resolve_service(service)
    lat, lon, where = locate(location)
    mode = SERVICE_MODE[svc]
    cfg = MODES[mode]
    radius = radius_km or cfg["radius_km"]

    c = EV_PD[EV_PD["service"] == svc].copy()
    listed = km_np(lat, lon, c["best_lat"].values, c["best_lon"].values)
    alt = np.array([alt_distance(r, lat, lon) for r in c.itertuples()]) if len(c) else np.array([])
    c["distance_lo"] = np.fmin(listed, alt) if len(c) else listed
    c["distance_km"] = np.fmax(listed, alt) if len(c) else listed      # rank on the LONGER distance when in doubt
    c["uncertain"] = (c["distance_km"] - c["distance_lo"]) > 10
    child_only = [(svc != "nicu") and (not for_child) and children_only(n, e) for n, e in zip(c["name"], c["evidence"])]
    c["warnings"] = [[f for f in q if f in MAJOR_FLAGS] + list(l) + ([CHILD_ONLY_NOTE] if co else [])
                     + specialist_note(n, svc)
                     for q, l, co, n in zip(c["quality_flags"], c["loc_flags"], child_only, c["name"])]
    c["eligible"] = c["evidence_strength"].map(lambda s: cfg["penalty"][s] is not None)
    c["evidence_penalty"] = c["evidence_strength"].map(lambda s: cfg["penalty"][s] if cfg["penalty"][s] is not None
                                                        else WEAK_LAST_RESORT_KM)
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
    if not (ranked["evidence_strength"] == "Strong").any():
        strong = c[c["evidence_strength"] == "Strong"].sort_values(order).head(1)
        if len(strong):
            s = strong.iloc[0]
            notes.append(f"None of these has Strong evidence. The best option with Strong evidence is "
                         f"{s['name']}, {s['distance_km']:.0f} km away (added at the bottom).")
            ranked = pd.concat([ranked, strong.assign(note="Outside search area: best Strong option")])
        else:
            notes.append(f"No facility in the dataset has Strong evidence for {SERVICE_LABELS[svc]}.")
    if len(within) < top_n and len(TAG_PD):
        t = TAG_PD[TAG_PD["service"] == svc].copy()
        t["d"] = km_np(lat, lon, t["best_lat"].values, t["best_lon"].values)
        t = t[(t["d"] <= radius) & ~t["name"].isin(ranked["name"])].sort_values("d")
        if len(t):
            names = ", ".join(f"{n} ({d:.0f} km)" for n, d in zip(t["name"].head(3), t["d"].head(3)))
            one = len(t) == 1
            notes.append(f"{len(t)} more facilit{'y' if one else 'ies'} within {radius} km list{'s' if one else ''} a "
                         f"related specialty but {'its record says' if one else 'their records say'} nothing specific "
                         f"about {SERVICE_LABELS[svc]}: {names}. "
                         f"Worth a phone call if the options below are too far.")
    if ranked["uncertain"].any():
        notes.append("Some facilities have conflicting locations in their records; for those we show a distance "
                     "range and rank them on the longer distance.")
    notes.append("Distances are straight-line; road times are rough estimates. Always call ahead to confirm "
                 "the service is available today.")

    rows = []
    for i, (_, r) in enumerate(ranked.iterrows(), start=1):
        nw = len(r.warnings)
        how = ((f"{r.distance_km:.0f} km away (the longer of two conflicting locations)" if r.uncertain
                else f"{r.distance_km:.0f} km away")
               + (f" + {r.evidence_penalty} km (evidence only {r.evidence_strength})" if r.evidence_penalty else "")
               + (f" + {r.warning_penalty} km ({nw} warning{'s' if nw > 1 else ''})" if r.warning_penalty else "")
               + f" = ranked as {r.effective_km:.0f} km ({mode} settings)")
        cautions = ["Mentioned as referral / unavailable: " + m for m in r.caution_mentions]
        urls = [u for u in dict.fromkeys(r.url_list) if not JUNK_URL.search(u)][:2]
        size = ", ".join([x for x in [f"{int(r.beds)} beds" if pd.notna(r.beds) else "",
                                      f"{int(r.doctors)} doctors" if pd.notna(r.doctors) else ""] if x])
        rows.append({
            "Rank": i,
            "Facility": r["name"],
            "Where": ", ".join([x for x in [r.address_city, r.address_stateOrRegion] if x]),
            "Distance": ((f"{r.distance_lo:.0f}-{r.distance_km:.0f} km straight-line (location uncertain), "
                          f"{road_time(r.distance_lo)} to {road_time(r.distance_km)} by road") if r.uncertain else
                         (f"{r.distance_km:.0f} km straight-line, {road_time(r.distance_km)} by road"
                          + (" (approx. location)" if r.location_source != "listed coordinates" else ""))),
            "Evidence": r.evidence_strength + (" (2+ separate procedure statements)" if r.upgraded else ""),
            "WHY: quoted from the record": "\n".join(r.evidence[:3]),
            "HOW it was ranked": how,
            "Warnings": "\n".join(r.warnings + cautions) or "None",
            "Size (as listed)": size or "Not listed",
            "Sources": "\n".join(urls) or "None listed",
            "Note": r.note,
        })
    out = pd.DataFrame(rows)

    if show:
        p = cfg["penalty"]
        print(f"SERVICE:  {SERVICE_LABELS[svc]}  ({mode} service)")
        print(f"LOCATION: {where}")
        print(f"Facilities with Strong/Moderate evidence within {radius} km: {len(within)}")
        for n in notes:
            print("NOTE: " + n)
        weak_txt = (f"{p['Weak']} km if Weak" if p["Weak"] is not None else "Weak evidence only as a last resort")
        print(f"HOW WE RANK ({mode}): distance, plus {p['Moderate']} km if evidence is only Moderate, {weak_txt}, "
              f"and {cfg['per_warning_km']} km per warning. Lowest total first.")
        if out.empty:
            print("No facilities to show for this service.")
        else:
            display(out)
    return out

# ---- 7. Try it: the challenge's own example questions --------------------
_ = recommend("dialysis", "Jaipur")
_ = recommend("emergency c-section", "Lucknow")
_ = recommend("burn care", "Patna")
