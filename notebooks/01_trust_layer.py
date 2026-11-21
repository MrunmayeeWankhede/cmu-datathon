# Databricks notebook source
# ============================================================
# STEP 1: Clean the facilities table + add data-quality red flags
# Output: workspace.default.facilities_clean
# ============================================================
from pyspark.sql import functions as F
from pyspark.sql.window import Window

SRC = "databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.facilities"
OUT = "workspace.default.facilities_clean"

raw = spark.table(SRC)

# --- helpers -------------------------------------------------
def as_list(col_name):
    """Turn a JSON-array string like '["a","b"]' into a real list. Plain text -> 1-item list."""
    c = F.trim(F.col(col_name))
    parsed = F.from_json(c, "array<string>")
    return (F.when(c.isNull() | (c == "") | (F.lower(c) == "null"), F.array().cast("array<string>"))
             .when(c.startswith("["), F.coalesce(parsed, F.array().cast("array<string>")))
             .otherwise(F.array(c)))

def as_int(col_name):
    """Safely turn '650' / '650.0' / 'unknown' into 650 / 650 / null."""
    return F.expr(f"try_cast(try_cast(`{col_name}` as double) as int)")

# --- 1. parse fields ------------------------------------------
df = (raw
      .withColumn("spec_list",  as_list("specialties"))
      .withColumn("proc_list",  as_list("procedure"))
      .withColumn("equip_list", as_list("equipment"))
      .withColumn("cap_list",   as_list("capability"))
      .withColumn("url_list",   as_list("source_urls"))
      .withColumn("beds",       as_int("capacity"))
      .withColumn("doctors",    as_int("numberDoctors"))
      .withColumn("year_est",   as_int("yearEstablished"))
      .withColumn("pin",        F.regexp_extract(F.coalesce(F.col("address_zipOrPostcode"), F.lit("")), r"(\d{6})", 1))
     )

# --- 2. measurements used by the red flags -------------------
df = (df
      .withColumn("n_specs",          F.size("spec_list"))
      .withColumn("n_specs_distinct", F.size(F.array_distinct("spec_list")))
      .withColumn("n_claims", F.size("proc_list") + F.size("equip_list") + F.size("cap_list"))
      .withColumn("source_domains", F.array_distinct(F.transform(
            "url_list", lambda u: F.regexp_extract(u, r"^https?://(?:www\.)?([^/]+)", 1))))
      .withColumn("n_source_domains", F.size("source_domains"))
      .withColumn("has_junk_url", F.exists(
            "url_list", lambda u: u.rlike(r"(?i)(login|signin|sign-in|creditcard|checkout|cart)")))
      .withColumn("n_listing_claims", F.size(F.filter(
            "cap_list", lambda x: x.rlike(r"(?i)\b(is listed|listed (as|under|in))\b"))))
      .withColumn("cluster_size", F.when(F.col("cluster_id").isNull(), F.lit(1))
            .otherwise(F.count("*").over(Window.partitionBy("cluster_id"))))
     )

# records with far more claims than normal are likely several facilities merged together
p99_claims = df.approxQuantile("n_claims", [0.99], 0.01)[0]
print(f"99th percentile of claims per record: {p99_claims:.0f}")

# --- 3. human-readable red flags ------------------------------
single_specialty_name = F.col("name").rlike(
    r"(?i)\b(eye|netra|nethralaya|dental|dentist|teeth|ayurved\w*|homoeo\w*|homeo\w*|skin|derma\w*|ivf|fertility|physiotherap\w*)\b")

checks = [
    (F.col("latitude").isNull() | F.col("longitude").isNull(),
        "No coordinates"),
    (~F.col("latitude").between(6, 37.5) | ~F.col("longitude").between(68, 97.5),
        "Coordinates fall outside India"),
    ((F.col("pin") == "") | F.col("pin").startswith("0"),
        "No valid 6-digit PIN code"),
    (single_specialty_name & (F.col("n_specs_distinct") > 6),
        "Name suggests a single-specialty clinic but record claims many specialties (possible merged record)"),
    ((F.col("n_specs") >= 6) & (F.col("n_specs_distinct") / F.col("n_specs") < 0.5),
        "Specialties heavily repeated (sign of data merged from multiple sources)"),
    (F.col("n_claims") > F.lit(p99_claims),
        "Unusually many claims for one facility (possible merged record)"),
    ((F.col("beds") >= 100) & (F.col("doctors") <= 2),
        "Bed count and doctor count are implausible together"),
    (F.col("has_junk_url"),
        "Some source URLs are unrelated pages (e.g. login pages)"),
    (F.col("n_source_domains") > 5,
        "Information pulled from many unrelated websites"),
    (F.col("n_listing_claims") >= 3,
        "Several capabilities are directory listings that mention other hospitals"),
    (F.col("cluster_size") > 1,
        "Shares a cluster ID with other records (possible duplicate)"),
]

flags = F.array(*[F.when(cond, F.lit(msg)) for cond, msg in checks])
df = (df
      .withColumn("quality_flags", F.filter(flags, lambda x: x.isNotNull()))
      .withColumn("n_flags", F.size("quality_flags")))

# --- 4. save ---------------------------------------------------
(df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(OUT))
clean = spark.table(OUT)
print(f"Saved {clean.count()} rows to {OUT}")

# --- 5. quick report -------------------------------------------
print("\nHow many records trip each red flag:")
display(clean.select(F.explode("quality_flags").alias("flag"))
             .groupBy("flag").count().orderBy(F.desc("count")))

print("\nRecords by number of red flags:")
display(clean.groupBy("n_flags").count().orderBy("n_flags"))

print("\nThe Aravind example record:")
display(clean.filter(F.col("name") == "Aravind Eye Hospital")
             .select("name", "address_city", "pin", "latitude", "longitude",
                     "beds", "doctors", "n_claims", "n_source_domains", "cluster_size", "quality_flags"))
