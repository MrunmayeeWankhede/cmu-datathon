# Databricks notebook source
# ============================================================
# STEP 7: Naive search vs FindCare
# For every service in the 50 cities with the most facilities, compare:
#   naive    = the nearest facility whose record mentions the service at all (a keyword search)
#   FindCare = #1 from our ranking (rules only, no AI, so it is fast and repeatable)
# Both picks are judged by the same checks (findcare/baseline.py -> problems()).
# Runs in 1-3 minutes. Needs the six app_ tables (Step 5).
# ============================================================
import os, sys
sys.path.insert(0, os.path.abspath(".."))      # repo root, so `findcare` can be imported

import importlib
from findcare import core, baseline
importlib.reload(core); importlib.reload(baseline)

print(core.load(core.spark_reader(spark)))
runs = baseline.compare()
print(len(runs), "searches compared")

# COMMAND ----------
# The headline for the slide
print(baseline.headline(runs))
s = baseline.summary(runs)
print(f"\nFindCare chose a different facility than the naive search in {s['findcare_changed_pick_share']:.0%} of searches.")
print(f"It avoided a problem pick in {s['fixed']} searches.")

# COMMAND ----------
# By service: share of picks with at least one problem (use the chart button under the table: bar chart,
# Service on the x-axis, both % columns as values)
display(baseline.by_service(runs))

# COMMAND ----------
# Which problems the naive search walks into
display(baseline.problem_counts(runs))

# COMMAND ----------
# Concrete examples for the slides: naive pick was risky, FindCare's was not
display(baseline.examples(runs, k=15))

# COMMAND ----------
# Everything, for anyone who wants to dig in
display(runs)

# COMMAND ----------
# Save the results as tables so they can go on the Databricks dashboard:
#   workspace.default.naive_vs_findcare               one row per search
#   workspace.default.naive_vs_findcare_by_service    % of picks with a problem, per service (bar chart)
#   workspace.default.naive_vs_findcare_problems      how often each problem type appears (bar chart)
all_runs, per_service, per_problem = baseline.for_tables(runs)
for name, df in [("naive_vs_findcare", all_runs), ("naive_vs_findcare_by_service", per_service),
                 ("naive_vs_findcare_problems", per_problem)]:
    spark.createDataFrame(df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"workspace.default.{name}")
    print(f"workspace.default.{name}: {len(df)} rows")
