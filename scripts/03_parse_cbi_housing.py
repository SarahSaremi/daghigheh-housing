"""Parse CBI monthly 'Tehran housing market' PDFs (Data/CBI_housing_raw/pdf) -> monthly price per m2.
Each report's Table 2 row 'متوسط قیمت هر متر مربع' gives: %yoy, %mom, current month, previous month,
same month previous year. Older reports are in THOUSAND rials, newer in MILLION rials -> unified to
million rials. Every triple is validated against its own % changes; garbled extractions are dropped.
Each month can be observed in up to 3 reports; we keep the 'current' observation and cross-check."""
import glob, os, re
import pdfplumber, pandas as pd, numpy as np

ROW = "عبرم رتم ره تميق"           # reversed 'قیمت هر متر مربع' as pdfplumber extracts it
NUM = re.compile(r"-?\d+(?:\.\d+)?")
obs, issues = [], []

def shift(y, m, k):
    t = y * 12 + (m - 1) + k
    return t // 12, t % 12 + 1

for f in sorted(glob.glob("Data/CBI_housing_raw/pdf/cbi_tehran_*.pdf")):
    y, m = map(int, re.search(r"_(\d{4})_(\d{2})_", f).groups())
    with pdfplumber.open(f) as p:
        lines = "\n".join((pg.extract_text() or "") for pg in p.pages).split("\n")
    found = False
    for i, l in enumerate(lines):
        if ROW not in l:
            continue
        cand = NUM.findall(l)
        if len(cand) < 3 and i + 1 < len(lines):
            cand = NUM.findall(lines[i + 1])
        vals = list(map(float, cand))
        if len(vals) >= 5:            # yoy%, mom%, current, prev month, same month last year
            yoy, mom, cur, prev, ly = vals[:5]
            mom_ok = prev > 0 and abs((cur / prev - 1) * 100 - mom) < 0.6
            yoy_ok = ly > 0 and abs((cur / ly - 1) * 100 - yoy) < 0.6
            check = "validated_by_pct"
        elif len(vals) == 3 and min(vals) > 1000:   # 1396-style table: current, prev month, same month last year
            cur, prev, ly = vals; mom_ok = yoy_ok = True; check = "3col_unvalidated(cross-checked)"
        else:
            continue
        if not (mom_ok or yoy_ok):
            issues.append((os.path.basename(f), "validation failed / garbled", l[:80]))
            continue
        if not (mom_ok and yoy_ok):
            issues.append((os.path.basename(f), "one cell garbled; kept only the validated pair", l[:80]))
        unit = "thousand_rial" if cur > 5000 else "million_rial"
        k = 1000 if unit == "thousand_rial" else 1
        trip = [((y, m), cur, "current")]
        if mom_ok: trip.append((shift(y, m, -1), prev, "prev_month"))
        if yoy_ok: trip.append(((y - 1, m), ly, "same_month_last_year"))
        for (yy, mm), v, role in trip:
            obs.append(dict(year=yy, month=mm, price_mrial_m2=v / k, role=role, unit_in_pdf=unit,
                            check=check, source=os.path.basename(f)))
        found = True
        break   # first matching row = monthly table (later rows are cumulative YTD tables)
    if not found:
        issues.append((os.path.basename(f), "price row not found/validated", ""))

o = pd.DataFrame(obs)
# reports whose digits are image glyphs (not extractable) were transcribed by hand from the page image
o = pd.concat([o, pd.read_csv("Data/CBI_housing_raw/manual_transcriptions.csv")], ignore_index=True)
o.to_csv("Data/CBI_housing_raw/cbi_tehran_price_observations.csv", index=False, encoding="utf-8-sig")
# Revision policy: the value from the most recently published report wins (CBI revises earlier months);
# the first-release value is kept for transparency.
o["report_t"] = o.source.str.extract(r"_(\d{4})_(\d{2})_").astype(int).pipe(lambda d: d[0] * 12 + d[1])
def pick(d):
    d = d.sort_values("report_t")
    return pd.Series({"price_million_rial_m2": d.price_mrial_m2.iloc[-1],
                      "first_release_million_rial_m2": d.price_mrial_m2.iloc[0],
                      "source": f"{d.source.iloc[-1]} ({d.role.iloc[-1]})", "n_obs": len(d),
                      "max_rel_diff_across_reports": d.price_mrial_m2.max() / d.price_mrial_m2.min() - 1})
monthly = o.groupby(["year", "month"]).apply(pick).reset_index()
monthly.to_csv("Data/processed/cbi_tehran_price_monthly.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(issues, columns=["file", "issue", "line"]).to_csv("Data/processed/cbi_parse_issues.csv", index=False, encoding="utf-8-sig")
print(monthly.to_string()); print("ISSUES:", issues)
