# FindCare India · CMU Datathon Fall 2026

**Where should a patient go for a specific kind of care, and why should we trust that answer?**

FindCare takes a **service** (dialysis, emergency C-section, NICU, cardiac surgery, MRI, CT, burn care, trauma, ICU)
and a **location** (city, district, PIN code, village or coordinates), or a plain-language description of the case,
and returns the best facilities from the Virtue Foundation dataset of ~10,000 Indian healthcare facilities. Every
suggestion shows the quoted evidence behind it, a second opinion from an AI, any warnings about the record, and the
exact arithmetic that ranked it.

## Who it's for

| User | What they get |
| --- | --- |
| Community health worker, NGO coordinator | The **app**: describe the case, get ranked facilities as cards with quotes, warnings and road time |
| District planner | The pipeline tables (and dashboard, if built): where coverage is missing |
| Judges / reviewers | The **notebooks**: the full, transparent pipeline from raw records to ranking |

## How it decides (the "HOW")

**Ranked distance = straight-line distance + a penalty for weaker evidence + a penalty for each warning. Lowest wins.**

1. **Trust layer** (`notebooks/01_trust_layer.py`): every record gets plain-English red flags, such as no or
   impossible coordinates, a single-specialty name claiming many specialties (merged records), impossible
   bed-to-doctor ratios, or junk source URLs.
2. **Evidence layer** (`02_evidence_layer.py`): for each facility and service, the exact supporting sentences are
   pulled out and scored: procedure statement **+3**, capability or equipment **+2**, description or specialty
   tag **+1**.
   **Strong** = 5+ points from 2+ different fields, or 2+ separate procedure statements · **Moderate** = 3+ ·
   **Weak** = 1–2. A bare specialty tag is never treated as evidence. "Referred elsewhere" or "not available"
   mentions become cautions, not evidence.
3. **Location + ranking** (`03_location_ranking.py`): the input is geocoded with the India Post PIN directory.
   Facilities whose coordinates conflict with their PIN area or stated city get a distance *range* and are ranked
   on the longer distance.

   | | Emergency (C-section, trauma, burns, NICU, ICU) | Planned (dialysis, MRI, CT, cardiac surgery) |
   | --- | --- | --- |
   | Search radius | 50 km | 100 km |
   | Moderate instead of Strong | +10 km | +30 km |
   | Weak evidence | last resort only | +75 km |
   | Each warning | +10 km | +25 km |

   Facilities with warnings are pushed down, never hidden. Beds and doctors are shown but not scored, because they
   are missing for 64–75% of records.
4. **AI layer** (`03b_ai_layer.py`): Llama 3.3 70B (Databricks Foundation Model APIs, temperature 0) re-reads
   the top 8 candidates. It sees **only** that facility's numbered quotes and returns *confirmed / likely /
   unclear / unlikely*, whether it suits the patient (adults/children), and a reason citing quote numbers. The AI
   can only push a facility **down**. The same model turns free text ("woman in labour 20 hours, village near
   Gaya") into service + place + patient, and writes a 2-sentence summary from the card facts only.

Uncertainty is shown rather than hidden: distance ranges, "no Strong option within 50 km" notes, nearby
specialty-only facilities listed as "worth a phone call", and "call ahead to confirm" on every answer.

## Repo layout

```
app.py, app.yaml, requirements.txt   Databricks App (Streamlit), deployed from the repo root
findcare/core.py                     ranking + AI logic shared by notebooks and app: core.search(...)
notebooks/01 ... 05                  the pipeline, run in order on Databricks Free Edition (serverless)
docs/test_queries.md                 hand-checked test queries and results
```

## Running it

**Notebooks** (Databricks Free Edition, serverless). Add this repo with *New → Git folder*, then run in order:
`01_trust_layer` → `02_evidence_layer` → `03_location_ranking` → `03b_ai_layer` → `05_export_app_tables`.
They can also be run as cells of one notebook in that order. `04_app_widgets` is a notebook-only version of the
app using widgets. Output tables go to `workspace.default`.

**App**
1. Run notebook 05 so the `workspace.default.app_*` tables exist.
2. *Compute → Apps → Create app → Custom*, point it at this Git folder (repo root).
3. Add resources: **SQL warehouse** (key `sql-warehouse`, *Can use*) and **serving endpoint**
   `databricks-meta-llama-3-3-70b-instruct` (*Can query*).
4. Give the app's service principal read access to the tables, in a SQL editor:
   ```sql
   GRANT USE CATALOG ON CATALOG workspace TO `<app service principal>`;
   GRANT USE SCHEMA ON SCHEMA workspace.default TO `<app service principal>`;
   GRANT SELECT ON SCHEMA workspace.default TO `<app service principal>`;
   ```
5. Deploy. Free Edition apps stop after 24 hours, so restart before demoing.

No secrets are stored in this repo: Databricks handles authentication inside notebooks and apps.

## Data

Virtue Foundation India facilities dataset (Databricks Marketplace: `databricks_virtue_foundation_dataset_dais_2026`),
India Post PIN code directory, NFHS-5 district indicators. Records were extracted from websites; treat every result
as a lead to confirm by phone.
