# Databricks notebook source
# ============================================================
# STEP 2: Evidence layer — for every facility x service, find the
#         exact supporting sentences and score evidence strength.
# Input:  workspace.default.facilities_clean   (from Step 1)
# Output: workspace.default.facility_service_evidence
# ============================================================
from pyspark.sql import functions as F

CLEAN = "workspace.default.facilities_clean"
OUT   = "workspace.default.facility_service_evidence"

# --- 1. The service dictionary --------------------------------
# For each service: what counts as a mention in free text, in equipment, and in the specialty list.
# (?i) = case-insensitive, \b = whole-word match. Add services/terms here as you go.
SERVICES = {
    "dialysis": dict(
        text  = r"(?i)\b(dialysis|ha?emodialysis|peritoneal dialysis|CRRT)\b",
        equip = r"(?i)\b(dialysis|ha?emodialysis|RO plant)\b",
        spec  = r"(?i)nephrology"),
    "emergency_c_section": dict(
        text  = r"(?i)\b(c-?sections?|caesarean|cesarean|LSCS|CEmONC|EmOC|emergency obstetric\w*)\b",
        equip = r"(?i)\b(labou?r (room|ward|theatre)|obstetric \w+|fetal monitor\w*|CTG|cardiotocograph\w*)\b",
        spec  = r"(?i)(gynecologyAndObstetrics|obstetric)"),
    "nicu": dict(
        text  = r"(?i)\b(NICU|SNCU|neonatal intensive|neonatal ICU|newborn intensive|special newborn care)\b",
        equip = r"(?i)\b(incubators?|radiant warmers?|phototherapy|neonatal ventilators?|bubble CPAP)\b",
        spec  = r"(?i)neonatology"),
    "cardiac_surgery": dict(
        text  = r"(?i)\b(cardiac surgery|cardiothoracic|CTVS|bypass surgery|CABG|open[- ]heart|valve replacement)\b",
        equip = r"(?i)\b(cath(eterization)? ?lab|heart[- ]lung machine|cardiopulmonary bypass)\b",
        spec  = r"(?i)(cardiacSurgery|cardiothoracic)"),
    "mri": dict(
        text  = r"(?i)\b(MRI|magnetic resonance)\b",
        equip = r"(?i)\b(MRI|magnetic resonance)\b",
        spec  = r"(?i)radiology"),
    "ct_scan": dict(
        text  = r"(?i)\b(CT scans?|CT scanners?|computed tomography|HRCT|CECT|multi-?slice CT)\b",
        equip = r"(?i)\b(CT scanners?|CT scan|computed tomography|multi-?slice CT)\b",
        spec  = r"(?i)radiology"),
    "burn_care": dict(
        text  = r"(?i)\b(burns? (unit|ward|care|centre|center|ICU)|burn injur\w*|burns? (patients?|management|treatment|surgery))\b",
        equip = r"(?i)\b(burns? (unit|ward|ICU|dressing|bath))\b",
        spec  = r"(?i)(burn|plasticSurgery)"),
    "trauma_emergency": dict(
        text  = r"(?i)\b(trauma (centre|center|care|unit|ward)|casualty|accident (and|&) emergency|emergency (department|services?|ward|room)|24\s*[x×/]\s*7 emergency)\b",
        equip = r"(?i)\b(defibrillators?|crash cart|trauma \w+)\b",
        spec  = r"(?i)(emergencyMedicine|trauma)"),
    "icu": dict(
        text  = r"(?i)\b(ICU|intensive care|critical care|MICU|SICU|CCU)\b",
        equip = r"(?i)\b(ventilators?|multi-?para(meter)? monitors?)\b",
        spec  = r"(?i)criticalCare"),
}

# How much each kind of mention is worth. A specific procedure statement is the
# strongest claim; a bare specialty tag is the weakest.
FIELD_POINTS = {"procedure": 3, "capability": 2, "equipment": 2, "description": 1, "specialties": 1}
FIELD_PATTERN = {"procedure": "text", "capability": "text", "description": "text",
                 "equipment": "equip", "specialties": "spec"}

# Mentions like "patients are referred elsewhere for dialysis" are NOT evidence of the service.
NEGATION = r"(?i)\b(refer(s|red|ral)?|not available|unavailable|no facility|does not|do not have|not have|lack\w*)\b"

rules = spark.createDataFrame(
    [(svc, field, pats[FIELD_PATTERN[field]], pts)
     for svc, pats in SERVICES.items() for field, pts in FIELD_POINTS.items()],
    "service string, field string, pattern string, points int")

# --- 2. One row per (facility, field, sentence) ---------------
clean = spark.table(CLEAN)
def items(list_col, field):
    return clean.select("unique_id", F.lit(field).alias("field"), F.explode(list_col).alias("text"))

all_items = (items("proc_list", "procedure")
             .unionByName(items("cap_list", "capability"))
             .unionByName(items("equip_list", "equipment"))
             .unionByName(items("spec_list", "specialties"))
             .unionByName(items(F.array("description"), "description"))
             .filter(F.col("text").isNotNull() & (F.trim("text") != "")))

# --- 3. Match every sentence against every service rule -------
matches = (all_items.join(F.broadcast(rules), "field")
           .filter(F.expr("text rlike pattern"))
           .withColumn("negated", F.col("text").rlike(NEGATION))
           .withColumn("snippet", F.concat(F.lit("["), F.col("field"), F.lit("] "),
                                           F.substring("text", 1, 300))))

# --- 4. Roll up per facility x field, then per facility x service --
per_field = (matches.groupBy("unique_id", "service", "field", "points")
             .agg(F.max(~F.col("negated")).alias("has_positive"),
                  F.collect_set(F.when(~F.col("negated"), F.col("snippet"))).alias("pos_snips"),
                  F.collect_set(F.when(F.col("negated"), F.col("snippet"))).alias("neg_snips")))

ev = (per_field.groupBy("unique_id", "service")
      .agg(F.sum(F.when(F.col("has_positive"), F.col("points")).otherwise(0)).alias("evidence_points"),
           F.sum(F.when(F.col("has_positive") & (F.col("field") != "specialties"), 1).otherwise(0)).alias("n_independent_fields"),
           F.concat_ws(", ", F.sort_array(F.collect_list(F.when(F.col("has_positive"),
                F.concat(F.col("field"), F.lit(" (+"), F.col("points").cast("string"), F.lit(")")))))).alias("points_breakdown"),
           F.flatten(F.collect_list("pos_snips")).alias("evidence"),
           F.flatten(F.collect_list("neg_snips")).alias("caution_mentions")))

ev = (ev
      .withColumn("evidence", F.slice("evidence", 1, 6))
      .withColumn("caution_mentions", F.slice("caution_mentions", 1, 3))
      .withColumn("evidence_strength",
          F.when((F.col("evidence_points") >= 5) & (F.col("n_independent_fields") >= 2), "Strong")
           .when(F.col("evidence_points") >= 3, "Moderate")
           .when(F.col("evidence_points") >= 1, "Weak")
           .otherwise("Unclear (only mentioned as referral / not available)"))
      .withColumn("strength_explanation", F.concat(
          F.lit("Evidence found in: "), F.when(F.col("points_breakdown") == "", F.lit("none"))
                                         .otherwise(F.col("points_breakdown")),
          F.lit(" = "), F.col("evidence_points").cast("string"), F.lit(" pts across "),
          F.col("n_independent_fields").cast("string"), F.lit(" independent field(s). "),
          F.lit("Rule: Strong = 5+ pts in 2+ fields; Moderate = 3+ pts; Weak = 1-2 pts."))))

# attach facility info + trust flags from Step 1 so the next step has everything in one place
ev = ev.join(clean.select("unique_id", "name", "address_city", "address_stateOrRegion", "pin",
                          "latitude", "longitude", "beds", "doctors", "url_list",
                          "quality_flags", "n_flags"), "unique_id")

ev.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(OUT)
ev = spark.table(OUT)
print(f"Saved {ev.count()} facility-service rows to {OUT}")

# --- 5. Report --------------------------------------------------
print("\nFacilities per service, by evidence strength:")
display(ev.groupBy("service").pivot("evidence_strength").count().orderBy("service"))

print("\nExample: strongest dialysis evidence")
display(ev.filter(F.col("service") == "dialysis")
          .orderBy(F.desc("evidence_points"), "n_flags")
          .select("name", "address_city", "evidence_strength", "strength_explanation",
                  "evidence", "caution_mentions", "quality_flags")
          .limit(10))
