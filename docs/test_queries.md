# Hand-checked test queries

For each query: run it, look at the top 3, and judge whether a sensible health worker would agree.
Verdict: ✅ good · ⚠️ acceptable but questionable · ❌ wrong. Write what's wrong in Notes; that is what we fix.

| # | Query (service · place · patient) | Top result | Evidence / AI | Verdict | Notes |
|---|---|---|---|---|---|
| 1 | Dialysis · Jaipur · adult | Vaishali Hospital & Surgical Research Center (5 km) | Strong | ✅ | Surya (pediatric dialysis) correctly no longer #1 |
| 2 | Emergency C-section · Lucknow · adult | BNK Hospital (2 km) | Strong (2+ procedures) | ✅ | Surman (Kanpur listing, Lucknow coords) dropped after city check |
| 3 | Burn care · Patna · adult | Palika Vinayak Hospital (2 km) | Strong / AI confirmed | ✅ | Eye hospital removed after keyword fix |
| 4 | "Woman in labour 20 h, village near Gaya" | Medizone Hospitals, Patna (92 km) | Strong / AI confirmed | ✅ | No C-section evidence within 50 km of Gaya; 3 local hospitals suggested to phone |
| 5 | NICU · ? · newborn | | | | |
| 6 | Trauma · highway town · adult | | | | |
| 7 | MRI · a PIN code in a small town | | | | |
| 8 | Cardiac surgery · tier-2 city | | | | |
| 9 | Dialysis · rural district HQ | | | | |
| 10 | ICU · coordinates of a village | | | | |
| 11 | Burn care · a city with no burn unit | | | | |
| 12 | C-section · a north-east state district | | | | |
| 13 | Dialysis · child | | | | |
| 14 | Free text: "child hit by a motorbike near ..." | | | | |
| 15 | Free text: "kidney patient needs dialysis 3x a week, lives in ..." | | | | |
