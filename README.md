# FindCare India - CMU Datathon Fall 2026

**A patient needs a specific kind of care. Where should they go, and why should anyone trust that answer?**

FindCare takes a **service** (dialysis, emergency C-section, NICU, cardiac surgery, MRI, CT, burn care, trauma /
emergency care, ICU) and a **location** (city, district, PIN code, village or coordinates), or a plain-language
description of the case, and returns the best facilities from the Virtue Foundation's ~10,000 Indian healthcare
facility records. Every suggestion shows the quoted evidence behind it, an AI second opinion, any warnings about the
record, and the exact arithmetic that ranked it.

Our one principle: **NEVER recommend anything we can't show evidence for.**

Built entirely on **Databricks Free Edition**: PySpark notebooks on serverless compute, Unity Catalog tables,
Llama 3.3 70B through Foundation Model APIs, a Databricks App (Streamlit) and an AI/BI dashboard.

## Who it's for

| User | What they use | What they get |
| --- | --- | --- |
| Community health worker, NGO coordinator | **FindCare app**, *Find care* tab | Describe the case or pick a service and place, and get ranked facilities as cards with quotes, warnings, road time and sources |
| District planner, health official | **Coverage dashboard** (app tab + AI/BI dashboard) | Where trustworthy care is missing, how much evidence backs each service, and how reliable the records are |
| Judges / reviewers | **Notebooks** + this README | The full, transparent pipeline from raw records to ranking |

## How it decides

**Ranked distance = straight-line distance + a penalty for weaker evidence + a penalty for each warning. Lowest wins.**

1. **Trust layer**: every record gets plain-English warnings, such as impossible or missing coordinates, a
   PIN code or city that conflicts with the coordinates, a single-specialty name claiming many specialties
   (merged records), impossible bed-to-doctor ratios, or junk source URLs. Flags that fire on most records
   (e.g. data pulled from many websites) are kept as background signals and don't affect ranking.
2. **Evidence layer**: for each facility and service, the exact supporting sentences are pulled out and scored:
   procedure statement **+3**, capability or equipment **+2**, description or specialty tag **+1**.
   **Strong** = 5+ points from 2+ different fields, or 2+ separate procedure statements · **Moderate** = 3+ ·
   **Weak** = 1–2. A bare specialty tag is never treated as evidence; "referred elsewhere" or "not available"
   mentions become cautions.
3. **Ranking**: the place typed is geocoded with the India Post PIN directory. Urgency changes the trade-off:

   | | Emergency (C-section, trauma, burns, NICU, ICU) | Planned (dialysis, MRI, CT, cardiac surgery) |
   | --- | --- | --- |
   | Search radius | 50 km | 100 km |
   | Moderate instead of Strong | +10 km | +30 km |
   | Weak evidence | last resort only | +75 km |
   | Each warning | +10 km | +25 km |

   Facilities with warnings are pushed down, never hidden. Beds and doctors are shown but not scored, because
   they're missing for 64–75% of records.
4. **AI layer**: Llama 3.3 70B (temperature 0) re-reads the top 8 candidates. It sees **only** that facility's
   numbered quotes plus a plain definition of the service (e.g. trauma = injuries from accidents, falls or
   violence; a specialty hospital's "24×7 emergency" doesn't count). It returns *confirmed / likely / unclear /
   unlikely*, whether the facility suits the patient, and a reason citing quote numbers. **The AI can only push a
   facility down.** The same model turns free text into service + place + patient, and writes a 2-sentence
   summary from the card facts only.

## Design choices we made on purpose

**Why not train a model?** There is no ground truth in this data: nobody has verified which facilities really
offer each service. A trained model would learn to copy the same messy claims, and it couldn't tell a health
worker *why*. A transparent rule, where every number traces back to a quoted sentence, is checkable, which matters
more when a patient's time is at stake.

**Why FindCare doesn't diagnose.** The health worker decides what care is needed; FindCare finds where to get it,
and proves it. If someone types only symptoms ("pounding headache"), the app doesn't guess. It says to start
at a clinic or primary health centre, or go to an emergency department if symptoms are sudden or severe.

**Honest uncertainty instead of confident answers:**
- "No facility with Strong or Moderate evidence within 50 km", plus the nearest Strong option further away.
- Nearby facilities that only list a related specialty appear as "worth a phone call", never as recommendations.
- When a record's coordinates and PIN code disagree, the stated city breaks the tie. If the city agrees with the
  coordinates, they are trusted and the card says why. Otherwise the card shows a distance *range* and ranks
  on the longer distance.
- Specialist hospitals (urology, eye, cancer, maternity…) get an "unlikely to offer" warning for unrelated services.
- Distances are straight-line with a rough road-time estimate, and every card says to call ahead.

## How we tested it

We hand-checked real queries (`docs/test_queries.md`) and fixed every wrong answer at its cause:

| Query | Problem found | Fix |
| --- | --- | --- |
| Trauma · Jaipur | A urology institute ranked #1 ("24×7 emergency" matched) | Specialty names checked per service; AI given service definitions |
| Dialysis · PIN 401504 (Palghar) | The local hospital (13 km, Strong) ranked #24: its PIN pointed to Mumbai, plus a false "wrong state" warning | City breaks coordinate-vs-PIN ties; state warnings need a genuinely different state |
| C-section · village near Gaya | A facility with coordinates 79 km from its PIN area showed as 65 km away | Conflicting locations shown as a range and ranked on the longer distance |
| Burn care · Patna | An eye hospital matched "skin grafts for scar correction" | Burn keywords tightened; specialist-clinic warning |
| "Pounding headache" | Turned into a trauma search | Symptoms alone map to no service, and the app gives clinic / emergency guidance |

`notebooks/07_naive_vs_findcare.py` compares FindCare's pick with a naive keyword search (nearest facility whose
record mentions the service) across the 50 cities with the most facilities, judging both by the same checks
(`findcare/baseline.py`).

## Repo layout

```
app.py, app.yaml, requirements.txt   Databricks App (Streamlit): "Find care" and "Coverage dashboard" tabs
.streamlit/config.toml               dark, Apple-style theme
findcare/core.py                     ranking + AI logic shared by notebooks and app: core.search(...)
findcare/coverage.py                 planner view: district gaps, evidence by service, warning counts
findcare/baseline.py                 naive keyword search vs FindCare comparison
notebooks/01 ... 07                  the pipeline, run in order on Databricks Free Edition (serverless)
docs/test_queries.md                 hand-checked test queries and results
```

## Running it

**Notebooks** (Databricks Free Edition, serverless). Add this repo with *Workspace → Create → Git folder*, then run
`01_trust_layer` → `02_evidence_layer` → `03_location_ranking` → `03b_ai_layer` → `05_export_app_tables`
(and optionally `07_naive_vs_findcare`). Output tables go to `workspace.default`. `04_app_widgets` is a
notebook-only version of the app.

**App**
1. Run notebook 05 so the six `workspace.default.app_*` tables exist.
2. Open the app switcher (9-dot grid, top right) → **Databricks Apps** → create a custom app pointed at this Git
   folder (repo root).
3. Add resources: **SQL warehouse** (key `sql-warehouse`, *Can use*) and **serving endpoint**
   `databricks-meta-llama-3-3-70b-instruct` (*Can query*).
4. Give the app's service principal read access to the tables:
   ```sql
   GRANT USE CATALOG ON CATALOG workspace TO `<app service principal ID>`;
   GRANT USE SCHEMA ON SCHEMA workspace.default TO `<app service principal ID>`;
   GRANT SELECT ON SCHEMA workspace.default TO `<app service principal ID>`;
   ```
5. Deploy. Free Edition stops apps after 24 hours, so press **Start**, then **Deploy** before demoing.

To update after a code change: Git folder → **Pull** → app → **Deploy**. No secrets are stored in this repo;
Databricks handles authentication inside notebooks and apps.

## Limits and next steps

- Straight-line distances with a rough road-time estimate (×1.4, 40 km/h); next: real road travel times.
- Records were scraped from websites and may be out of date; next: feedback from health workers to correct them.
- FindCare covers 9 services; next: a "general hospital / primary care" option and emergency red-flag guidance.
- Next: add NFHS-5 district health indicators to the planner view, to weigh gaps by local need.

## Data

Virtue Foundation India facilities dataset (Databricks Marketplace: `databricks_virtue_foundation_dataset_dais_2026`),
India Post PIN code directory, NFHS-5 district indicators. Treat every result as a lead to confirm by phone.
