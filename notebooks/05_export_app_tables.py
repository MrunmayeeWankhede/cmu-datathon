# Databricks notebook source
# ============================================================
# STEP 5: Export the lookup tables the app needs
# Run AFTER Step 3 in the same notebook session (it uses EV_PD, TAG_PD, CITY_PD, PIN_PD, DIST_PD, OFFICE_PD).
# Writes six small tables: workspace.default.app_evidence, app_tag_only, app_city, app_pin, app_district, app_office
# ============================================================
import json

PREFIX = "workspace.default.app_"

def _save(pdf, name):
    spark.createDataFrame(pdf).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(PREFIX + name)
    print(f"{PREFIX + name}: {len(pdf)} rows")

ev = EV_PD[["service", "name", "address_city", "address_stateOrRegion", "evidence_strength", "upgraded",
            "evidence", "caution_mentions", "quality_flags", "loc_flags", "url_list",
            "best_lat", "best_lon", "location_source", "beds", "doctors", "pin_lat", "pin_lon"]].copy()
for c in ["evidence", "caution_mentions", "quality_flags", "loc_flags", "url_list"]:
    ev[c] = ev[c].map(lambda v: json.dumps(list(v)))
for c in ["beds", "doctors", "pin_lat", "pin_lon", "best_lat", "best_lon"]:
    ev[c] = ev[c].astype("float64")
ev["upgraded"] = ev["upgraded"].astype(bool)
for c in ["name", "address_city", "address_stateOrRegion", "location_source"]:
    ev[c] = ev[c].astype(object).where(ev[c].notna(), None)
_save(ev, "evidence")

_save(TAG_PD[["service", "name", "address_city", "best_lat", "best_lon"]].copy(), "tag_only")
_save(CITY_PD.reset_index()[["k", "city", "state", "lat", "lon", "n"]], "city")
_save(PIN_PD.reset_index()[["pin", "pin_lat", "pin_lon", "pin_district", "pin_state"]], "pin")
_save(DIST_PD.reset_index()[["k", "pin_district", "state", "lat", "lon"]], "district")
_save(OFFICE_PD.reset_index()[["k", "office", "pin", "district", "state", "lat", "lon"]], "office")

# COMMAND ----------
# Quick check: load the shared logic from the repo and run one search
import os, sys
sys.path.insert(0, os.path.abspath(".."))   # repo root, so `findcare` can be imported
from findcare import core
print(core.load(core.spark_reader(spark)))
r = core.search(service="Dialysis", location="Jaipur")
print(r["where"], "|", r["summary"])
for c in r["results"]:
    print(c["rank"], c["facility"], "|", c["distance"], "|", c["evidence_label"], "| AI:", c["ai_verdict"])
